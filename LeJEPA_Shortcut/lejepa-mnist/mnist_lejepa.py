import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import pytorch_lightning as pl
import numpy as np
import os

import timm
import wandb
import tqdm
from omegaconf import DictConfig, OmegaConf
import lejepa
from torchvision.ops import MLP

# ==========================================
# 1. Custom Augmentation: Gaussian Noise
# ==========================================
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        # tensor: float 타입이라고 가정
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class OnlineCovariance:
    def __init__(self, device="cuda"):
        self.device = device
        self.sum_x = None
        self.sum_xtx = None
        self.n = 0

    def update(self, features):
        x = features.detach().float()
        batch_size = x.size(0)
        
        if self.sum_x is None:
            d = x.size(1)
            self.sum_x = torch.zeros(d, device=self.device)
            self.sum_xtx = torch.zeros(d, d, device=self.device)
            
        self.sum_x += x.sum(dim=0)
        self.sum_xtx += x.T @ x
        self.n += batch_size

    def compute_spectrum(self, return_tensors=True):
        if self.n <= 1: return None, None
        
        mean_x = self.sum_x / self.n
        mean_xtx = self.sum_xtx / self.n
        cov = mean_xtx - (mean_x.unsqueeze(1) @ mean_x.unsqueeze(0))
        
        eigvals, eigvecs = torch.linalg.eigh(cov)
        idx = torch.argsort(eigvals, descending=True)
        
        sorted_eigvals = eigvals[idx].clamp(min=0) 
        sorted_eigvecs = eigvecs[:, idx] 

        if return_tensors:
            return sorted_eigvals, sorted_eigvecs
        else:
            return sorted_eigvals.cpu().numpy(), sorted_eigvecs.cpu().numpy()

class ColoredMNIST(Dataset):
    def __init__(self, root='./data', train=True, transform=None, V=1):
        super().__init__()
        self.transform = transform
        self.V = V
        split = 'train' if train else 'test'
        data_path = os.path.join(root, f'colored_mnist_{split}.pt')
        
        if not os.path.exists(data_path):
            raise RuntimeError(f"Dataset not found at {data_path}. Run generate_dataset.py first.")
            
        dataset = torch.load(data_path)
        self.data = dataset['images']
        self.targets = dataset['targets']
        self.colors = dataset['colors']
        
        # ==========================================
        # 2. Augmentation 적용 (Colored MNIST 맞춤)
        # ==========================================
        if self.V > 1 and self.transform is None:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(28, scale=(0.7, 1.0)),
                transforms.RandomApply([AddGaussianNoise(mean=0., std=0.1)], p=0.5),
                # 0~1 사이의 float 텐서라고 가정하고 ImageNet과 유사하거나 대칭적인 Normalize 적용
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
            ])

    def __getitem__(self, index):
        img, target, color = self.data[index], self.targets[index], self.colors[index]

        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        
        if self.V > 1:
            views = []
            for _ in range(self.V):
                v = self.transform(img) if self.transform else img
                views.append(v)
            img = torch.stack(views) # [V, 3, 28, 28]
        else:
            if self.transform:
                img = self.transform(img)

        return img, target, color

    def __len__(self):
        return len(self.data)

class LitViT(pl.LightningModule):
    def __init__(self, lr=1e-3, proj_dim=128, lamb=0.01):
        super().__init__()
        self.save_hyperparameters()
        
        # ==========================================
        # 3. Backbone 교체: ResNet18 -> ViT-Tiny/4
        # ==========================================
        # patch_size=4, img_size=28 적용. num_classes=0으로 설정해 classifier를 제외하고 특징만 추출
        self.backbone = timm.create_model(
            'vit_tiny_patch4_32', 
            pretrained=False, 
            num_classes=0, 
            img_size=28
        )
        num_ftrs = self.backbone.num_features # 일반적으로 ViT-Tiny는 192
        
        # Projection Head for LeJEPA
        self.proj = MLP(num_ftrs, [512, 512, proj_dim], norm_layer=nn.BatchNorm1d)
        
        # Linear Probe for Classification
        self.probe = nn.Sequential(nn.LayerNorm(num_ftrs), nn.Linear(num_ftrs, 10))
        
        # SigReg Loss (EppsPulley)
        univariate_test = lejepa.univariate.EppsPulley(n_points=17)
        self.sigreg = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=univariate_test,
            num_slices=128
        )
        
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        if x.ndim == 5:
            B, V, C, H, W = x.shape
            x_flat = x.flatten(0, 1) 
            emb = self.backbone(x_flat)
            proj = self.proj(emb)
            
            emb = emb.view(B, V, -1).transpose(0, 1)
            proj = proj.view(B, V, -1).transpose(0, 1)
        else:
            emb = self.backbone(x)
            proj = self.proj(emb)
            
        return emb, proj

    def training_step(self, batch, batch_idx):
        x, y, color = batch 
        emb, proj = self(x)
        
        # LeJEPA Loss
        sigreg_loss = self.sigreg(proj.flatten(0, 1))
        grad_penalty = (proj.mean(0).unsqueeze(0) - proj).square().mean()
        lejepa_loss = sigreg_loss * self.hparams.lamb + grad_penalty * (1 - self.hparams.lamb)
        
        # Probe Loss (Detach 적용하여 Backbone에 그래디언트 흐름 방지)
        emb_mean = emb.mean(0).detach() 
        logits = self.probe(emb_mean)
        probe_loss = self.criterion(logits, y)
        
        loss = lejepa_loss + probe_loss
        
        acc = (logits.argmax(dim=1) == y).float().mean()
        acc_spurious = (logits.argmax(dim=1) == color).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_lejepa', lejepa_loss)
        self.log('train_sigreg', sigreg_loss)
        self.log('train_inv', grad_penalty)
        self.log('train_probe', probe_loss)
        self.log('train_acc', acc, prog_bar=True)
        self.log('train_acc_spurious', acc_spurious, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, color = batch
        emb, proj = self(x)
        
        emb_mean = emb.mean(0)
        logits = self.probe(emb_mean)
        loss = self.criterion(logits, y)
        
        acc = (logits.argmax(dim=1) == y).float().mean()
        acc_spurious = (logits.argmax(dim=1) == color).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_acc_spurious', acc_spurious, prog_bar=True)
        return loss

    def configure_optimizers(self):
        params = [
            {"params": self.backbone.parameters(), "lr": self.hparams.lr},
            {"params": self.proj.parameters(), "lr": self.hparams.lr},
            {"params": self.probe.parameters(), "lr": 1e-3}
        ]
        return torch.optim.Adam(params)

def main(cfg: DictConfig):
    pl.seed_everything(42)

    wandb_logger = pl.loggers.WandbLogger(
        project="LeJEPA-MNIST",
        config=OmegaConf.to_container(cfg, resolve=True),
        resume="allow"
    )
    
    V = cfg.get("V", 2)
    train_ds = ColoredMNIST(root='./data', train=True, V=V)
    val_ds = ColoredMNIST(root='./data', train=False, V=V)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg.bs, shuffle=False, num_workers=4)
    
    # 변경된 모델 사용
    model = LitViT(lr=cfg.lr)
    
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator='auto',
        devices=1,
        log_every_n_steps=10,
        logger=wandb_logger
    )
    
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    # ==========================================
    # 4. 단일 스크립트 실행을 위한 더미 Config
    # ==========================================
    # Hydra 없이 스크립트를 직접 실행(python script.py)할 수 있도록 구조 수정
    default_config = OmegaConf.create({
        "lr": 1e-3,
        "bs": 256,
        "epochs": 100,
        "V": 2
    })
    
    main(default_config)