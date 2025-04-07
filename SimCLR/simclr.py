import cv2
import numpy as np

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from torch.nn import Flatten

logger = TensorBoardLogger("tb_logs", name="SimCLR")

class SimCLRTrainDataTransform(object):
    def __init__(
        self,
        input_height: int = 224,
        gaussian_blur: bool = False,
        jitter_strength: float = 1.,
        normalize: Optional[transforms.Normalize] = None
    ) -> None:

        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.normalize = normalize

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
            data_transforms.append(GaussianBlur(kernel_size=int(0.1 * self.input_height), p=0.5))

        data_transforms.append(transforms.ToTensor())

        if self.normalize:
            data_transforms.append(normalize)

        self.train_transform = transforms.Compose(data_transforms)

    def __call__(self, sample):
        transform = self.train_transform

        xi = transform(sample)
        xj = transform(sample)

        return xi, xj


class SimCLREvalDataTransform(object):
    def __init__(
        self,
        input_height: int = 224,
        normalize: Optional[transforms.Normalize] = None
    ):
        self.input_height = input_height
        self.normalize = normalize

        data_transforms = [
            transforms.Resize(self.input_height),
            transforms.ToTensor()
        ]

        if self.normalize:
            data_transforms.append(normalize)

        self.test_transform = transforms.Compose(data_transforms)

    def __call__(self, sample):
        transform = self.test_transform

        xi = transform(sample)
        xj = transform(sample)

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
    
def nt_xent_loss(out_1, out_2, temperature):
    out = torch.cat([out_1, out_2], dim=0)
    n_samples = len(out)

    # Full similarity matrix
    cov = torch.mm(out, out.t().contiguous())
    sim = torch.exp(cov / temperature)

    mask = ~torch.eye(n_samples, device=sim.device).bool()
    neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)
    
    # Positive similarity
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    loss = -torch.log(pos / neg).mean()
    return loss

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
                 lr=1e-4,
                 loss_temperature=0.5,
                 resnet18=True,
                 use_optimizer=False,
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
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)

        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=0.0)
        scheduler = {
            'scheduler': SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[10]),
            'interval': 'epoch',
            'monitor': 'train_loss',
        }
        if self.hparams.use_optimizer:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
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
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}  # ✅ 최신 버전 호환 코드

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
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
            batch_size=batch_size,
            num_workers=4,
            persistent_workers=True,
            shuffle=True)
        return train_loader
    
    def val_dataloader(self):
        transform = SimCLREvalDataTransform(input_height=32)
        val_dataset = CIFAR10(
            root='./../data',
            train=False,
            transform=transform
        )

        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=4,
            persistent_workers=True,
            shuffle=False)
        return val_loader
    
    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % 25 == 0:
            self.trainer.save_checkpoint(f"epoch_{self.current_epoch+201}.ckpt")
            torch.save(self.encoder.state_dict(), f"encoder_epoch_{self.current_epoch+201}.pth")


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')  # 또는 'medium'
    # pick data
    cifar_height = 32

    ### HYPERPARAMETERS ###

    batch_size = 256
    max_epochs = 200
    temperature = 0.15
    learning_rate = 2e-4

    usingResNet18 = False
    continue_training = True  # True: continue training, False: start from scratch

    warmup_epochs = 10

    #######################


    if continue_training:
        checkpoint_path = "checkpoint_1_200_ResNet50.ckpt"
        model = SimCLR.load_from_checkpoint(checkpoint_path, batch_size=batch_size, loss_temperature=temperature, lr = learning_rate, resnet18=usingResNet18)
    else:
        model = SimCLR(batch_size=batch_size, loss_temperature=temperature, lr=learning_rate, resnet18=usingResNet18, use_optimizer=True)

    
    trainer = pl.Trainer(max_epochs=max_epochs, enable_progress_bar=True, devices=1, accelerator="gpu", logger=logger)
    trainer.fit(model)

'''
# init data
dm = CIFAR10DataModule('./../data', num_workers=0, batch_size=batch_size)
dm.train_transforms = SimCLRTrainDataTransform(cifar_height)
dm.val_transforms = SimCLREvalDataTransform(cifar_height)

# realize the data
dm.prepare_data()
dm.setup()

train_samples = len(dm.train_dataloader())
'''