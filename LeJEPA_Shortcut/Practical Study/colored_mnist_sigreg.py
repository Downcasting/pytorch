import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import torchvision

import lejepa

import os
import wandb
from pytorch_lightning.loggers import WandbLogger

from callback import OnlineLinearEvaluation

class ColorMNIST(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=1e-3, ssl=True, lambd = 10, offdiag_lambd=1e-1):
        super().__init__()

        self.backbone = torchvision.models.resnet18(pretrained=False)
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        def init_weights(m):
            # Conv2d 또는 Linear 레이어만 골라서 초기화
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1e-1)
                # Bias가 있다면 0으로 초기화
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            # (옵션) BatchNorm은 보통 weight=1, bias=0으로 둡니다.
            # 만약 BN까지 10^-5로 만들면 신호가 아예 죽을 수 있어 주의해야 합니다.
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # [적용] Backbone 전체에 재귀적으로 적용
        self.backbone.apply(init_weights)

        self.lambd = lambd
        self.offdiag_lambd = offdiag_lambd
        self.save_hyperparameters()

    def forward(self, x):
        z = self.feature_extractor(x)
        z = torch.flatten(z, 1)
        return z

    # 수정된 training_step (Online Linear Probing 추가)
    def training_step(self, batch, batch_idx):
        ((x1, x2), x_clean), y = batch
        
        # 1. SSL Forward & Loss
        z1 = self(x1)
        z2 = self(x2)
        loss = barlow_twins_loss(z1, z2, self.lambd, self.offdiag_lambd)
        
        # 로그 기록
        self.log("ssl_loss", loss)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-6)

class ThreeWayColoredMNIST(Dataset):
    def __init__(self, base_dataset, ssl_transform, linear_transform):
        self.dataset = base_dataset
        self.ssl_transform = ssl_transform       # Augmentation (SSL용)
        self.linear_transform = linear_transform # Normalize Only (Linear용)

    def __getitem__(self, idx):
        # Raw Tensor Image (0~1)
        x, y = self.dataset[idx]
        
        # 1. SSL용: 강한 증강 (Augmentation)
        x1 = self.ssl_transform(x)
        x2 = self.ssl_transform(x)
        
        # 2. Linear Probe용: 단순 정규화 (Clean)
        x_clean = self.linear_transform(x)
        
        # 리턴: ((SSL뷰1, SSL뷰2), 리니어뷰), 라벨
        return ((x1, x2), x_clean), y

    def __len__(self):
        return len(self.dataset)

class SavedColoredMNIST(Dataset):
    def __init__(self, root="./colored_mnist", split="train"):
        assert split in ["train", "test"]

        self.images = torch.load(os.path.join(root, f"{split}_images.pt"))
        self.labels = torch.load(os.path.join(root, f"{split}_labels.pt"))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class GaussianNoise(object):
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, x):
        return x + torch.randn_like(x) * self.sigma
    
def barlow_twins_loss(z1, z2, lambd=5e-3, offdiag_lambd=1e-3):
    N, D = z1.size()
    
    # Normalize the representations along the batch dimension
    z1_norm = (z1 - z1.mean(0)) / z1.std(0)
    z2_norm = (z2 - z2.mean(0)) / z2.std(0)
    
    # Cross-correlation matrix
    c = torch.mm(z1_norm.T, z2_norm) / N

    univariate_test = lejepa.univariate.EppsPulley(n_points=17)
    loss_fn = lejepa.multivariate.SlicingUnivariateTest(
        univariate_test=univariate_test,
        num_slices=128
    ).to(z1.device)
    loss_sigreg = (loss_fn(z1_norm) + loss_fn(z2_norm)) / 2
    
    # Loss computation
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = (c - torch.diag(torch.diagonal(c))).pow_(2).sum()
    
    loss = on_diag + lambd * loss_sigreg + offdiag_lambd * off_diag
    return loss

class ValWrapper(Dataset):
    def __init__(self, ds, tf): self.ds = ds; self.tf = tf
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx): 
        x, y = self.ds[idx]
        return self.tf(x), y
    


def main():

    wandb.init(project="colored-mnist")
    wandb_logger = WandbLogger(project="colored-mnist")

    lr = 7e-5
    epoch = 70
    momentum = 0.9
    weight_decay = 1e-6
    scaling_factor = 0.1

    # Data transforms
    transform = transforms.Compose([
        transforms.RandomResizedCrop((48, 48), scale=(0.7, 1.0)),
        transforms.RandomApply([GaussianNoise(0.1)],p=0.5),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    val_transform = transforms.Compose([
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    

    # Load datasets
    train_dataset = SavedColoredMNIST(root="./colored_mnist", split="train")
    train_dataset = ThreeWayColoredMNIST(train_dataset, transform, val_transform)

    val_dataset = SavedColoredMNIST(root="./colored_mnist", split="test")
    val_dataset = ValWrapper(val_dataset, val_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4, persistent_workers=True, drop_last=True)

    # Initialize model
    model = ColorMNIST(num_classes=10, learning_rate=lr)

    online_eval_callback = OnlineLinearEvaluation(
        num_classes=10,
        feature_dim=512,      # ResNet18 output dim
        learning_rate=1e-3    # Probe용 LR (보통 SSL LR보다 약간 높게 잡음)
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=epoch,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        logger=wandb_logger,
        callbacks=[online_eval_callback]
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Test the model
    trainer.test(model, val_loader)


if __name__ == "__main__":
    main()
