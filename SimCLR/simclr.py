import cv2
import numpy as np

import os
import datetime

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD
from torch_optimizer import LARS
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR, ReduceLROnPlateau

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from torch.nn import Flatten


class SimCLRTrainDataTransform(object):
    
    def __init__(
        self,
        input_height: int = 32,
        gaussian_blur: bool = False,
        jitter_strength: float = 1.,
        normalize: Optional[transforms.Normalize] = None
    ) -> None:

        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur

        self.normalize = normalize or transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength
        )

        data_transforms = [
            transforms.RandomResizedCrop(size=self.input_height),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ]

        if self.gaussian_blur:
            kernel_size = max(3, int(0.1 * self.input_height))
            kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
            data_transforms.append(GaussianBlur(kernel_size=kernel_size, p=0.5))

        data_transforms.append(transforms.ToTensor())

        if self.normalize:
            data_transforms.append(self.normalize)

        self.train_transform = transforms.Compose(data_transforms)

    def __call__(self, sample):
        transform = self.train_transform

        if isinstance(sample, tuple):
            image, _ = sample
        else:
            image = sample

        xi = transform(image)
        xj = transform(image)

        return xi, xj

class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, p=0.5, min=0.1, max=2.0):
        self.min = min
        self.max = max

        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < self.p:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample
    
def nt_xent_loss(out_1, out_2, temperature=0.5):
    loss_batch_size = out_1.size(0)
    out = torch.cat([out_1, out_2], dim=0)  # [2N, D]

    # Cosine similarity matrix
    sim_matrix = torch.matmul(out, out.T) / temperature  # [2N, 2N]
    sim_matrix = sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0]  # for stability
    sim_matrix = torch.exp(sim_matrix)

    # Mask out self-similarity
    mask = (~torch.eye(2 * loss_batch_size, device=out.device).bool()).float()
    sim_matrix = sim_matrix * mask

    # Positive similarity (i와 i+N은 양의 쌍)
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

    # Denominator: sum over all except self
    denom = sim_matrix.sum(dim=1)

    loss = -torch.log(pos_sim / denom)
    return loss.mean()


class Projection(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False))

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)
    
class SimCLR(pl.LightningModule):
    def __init__(self,
                 lr=0.3,
                 loss_temperature=0.5,
                 resnet18=True,
                 max_epochs=500,
                 warmup_epochs=5,
                 batch_size=256,
                 use_scheduler=False,
                 use_warmup=False,
                 use_cosine=False,
                 use_reduceonplateau=False,
                 **kwargs):
        
        super().__init__()
        self.save_hyperparameters()

        self.nt_xent_loss = nt_xent_loss
        self.encoder = self.init_encoder()

        # h -> || -> z
        if resnet18:
            self.projection = Projection()
        else:
            self.projection = Projection(input_dim=2048, hidden_dim=2048)

    def init_encoder(self):
        encoder = models.resnet18() if self.hparams.resnet18 else models.resnet50()
        encoder = nn.Sequential(*list(encoder.children())[:-1])  # 마지막 FC Layer 제거
        return encoder

    def configure_optimizers(self):
        # optimizer = SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=1e-4)

        optimizer = LARS(self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=1e-6, trust_coefficient=0.001, eps=1e-8)

        schedulers = []
        # 1. Warmup
        if use_warmup:
            warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
            schedulers.append(("warmup", warmup))

        # 2. Cosine
        if use_cosine:
            cosine = CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs, eta_min=0.0)
            schedulers.append(("cosine", cosine))

        # 3. ReduceLROnPlateau
        if use_reduceonplateau:
            reduceonplateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, min_lr=1e-6)
            schedulers.append(("reduce", reduceonplateau))

        # 스케줄러 설정
        if use_scheduler:
            if use_warmup and use_cosine:
                # warmup + cosine은 SequentialLR로 묶기
                scheduler = {"scheduler": SequentialLR(optimizer, schedulers=[s[1] for s in schedulers if s[0] in ["warmup", "cosine"]], 
                                                       milestones=[warmup_epochs]), "interval": "epoch", "frequency": 1}
                return {"optimizer": optimizer, "lr_scheduler": scheduler}
            
            elif use_reduceonplateau:
                # ReduceLROnPlateau는 별도로 리턴해야 함 (monitor 필요)
                return {"optimizer": optimizer, "lr_scheduler": {"scheduler": reduceonplateau, "monitor": "train_loss_end_of_epoch", "interval": "epoch", "frequency": 1}}

            else:
                # 나머지 일반적인 경우 (cosine만 쓰는 등)
                sched_list = [{"scheduler": sched[1], "interval": "epoch", "frequency": 1} for sched in schedulers]
                return {"optimizer": optimizer, "lr_scheduler": sched_list}
        
        else:
            return optimizer


    def forward(self, x):
        if isinstance(x, list):
            x = x[0]

        result = self.encoder(x)
        if isinstance(result, list):
            result = result[-1]
        return result

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        return {"loss": loss}  # ✅ 최신 버전 호환 코드

    def shared_step(self, batch, batch_idx):
        (img1, img2), y = batch

        # ENCODE
        # encode -> representations
        # (b, 3, 32, 32) -> (b, 512, 2, 2)
        h1 = self.encoder(img1)
        h2 = self.encoder(img2)

        # the bolts resnets return a list of feature maps
        if isinstance(h1, list):
            h1 = h1[-1]
            h2 = h2[-1]

        # PROJECT
        # img -> E -> h -> || -> z
        # (b, 512, 2, 2) -> (b, 128)
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        loss = self.nt_xent_loss(z1, z2, self.hparams.loss_temperature)
        return loss
    
    def train_dataloader(self):
        transform = SimCLRTrainDataTransform(input_height=32)
        train_dataset = CIFAR10(
            root='./../data',
            train=True,
            transform=transform, 
            download=True
        )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            persistent_workers=True,
            shuffle=True)
        return train_loader

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_loss_end_of_epoch", avg_loss, on_epoch=True, prog_bar=True, logger=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("current_lr", current_lr, prog_bar=True, logger=True)

    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % 50 == 0:
            with open(f"version info.txt", "a") as f:
                f.write(f"v{version} has reached epoch {self.current_epoch+1}.\n")
            self.trainer.save_checkpoint(f"v{version}.ckpt")
            torch.save(self.encoder.state_dict(), f"v{version}_encoder.pth")

def version_exist(version):
    # Check if the version folder already exists
    base_path = "tb_logs/SimCLR"
    version_path = f"{base_path}/v{version}"
    return os.path.exists(version_path)

def save_version_info():
    # Save the version information to a text file
    with open(f"version info.txt", "a") as f:
        f.write(f"----------------------------------------\n")
        f.write(f"[Version: {version}]\n\n")
        f.write(f"Date: {datetime.datetime.now()}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Max Epochs: {max_epochs}\n")
        f.write(f"Temperature: {temperature}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Warmup Epochs: {warmup_epochs}\n")
        f.write(f"Using Model: {"ResNet18" if usingResNet18 else "ResNet50"}\n")
        f.write(f"Using scheduler: {"All" if use_scheduler else "Only SGD"}\n")
        f.write(f"----------------------------------------\n\n")

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')  # 또는 'medium'
    # pick data
    cifar_height = 32

    ######################################## HYPERPARAMETERS ########################################
    #################################################################################################
    
    # optimizer
    use_scheduler = True  # True: use all optimizers, False: only SGD
    use_warmup = True
    use_cosine = True
    use_reduceonplateau = False

    # real Hyperparameters
    batch_size = 512
    max_epochs = 1000
    temperature = 0.5
    learning_rate = 0.3 * (batch_size / 256)
    warmup_epochs = 5

    # using model
    usingResNet18 = True

    # continue training?
    continue_training = False  # True: continue training, False: start from scratch
    version = 8 # Version of the mode, increment if you start a new training session!!

    #################################################################################################
    #################################################################################################


    if continue_training:
        checkpoint_path = f"v{version}.ckpt"
        model = SimCLR.load_from_checkpoint(
            checkpoint_path, 
            batch_size=batch_size, 
            max_epochs=max_epochs,
            warmup_epochs=warmup_epochs,
            loss_temperature=temperature, 
            lr = learning_rate, 
            resnet18=usingResNet18,
            use_scheduler=use_scheduler,
            use_warmup=use_warmup,
            use_cosine=use_cosine,
            use_reduceonplateau=use_reduceonplateau
        )
    else:
        model = SimCLR(
            batch_size=batch_size, 
            max_epochs=max_epochs,
            warmup_epochs=warmup_epochs,
            loss_temperature=temperature, 
            lr = learning_rate, 
            resnet18=usingResNet18,
            use_scheduler=use_scheduler,
            use_warmup=use_warmup,
            use_cosine=use_cosine,
            use_reduceonplateau=use_reduceonplateau
        )
        while version_exist(version):
            print(f"Version v{version} already exists. Automatically incrementing version.")
            version += 1
        save_version_info()


    logger = TensorBoardLogger("tb_logs", name="SimCLR", version=f"v{version}")
    
    trainer = pl.Trainer(
        max_epochs=max_epochs, 
        enable_progress_bar=True, 
        devices=1, 
        accelerator="gpu", 
        resume_from_checkpoint=checkpoint_path if continue_training else None,
        logger=logger)
    trainer.fit(model)

