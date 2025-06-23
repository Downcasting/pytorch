# 표준 라이브러리
import os
import datetime
from typing import Optional

# 서드파티 라이브러리
import cv2
import yaml
import numpy as np

# PyTorch 및 관련 패키지
import torch
from torch import nn
from torch.nn import functional as F, Flatten
from torch.optim import SGD
from torch_optimizer import LARS  # pip install torch-optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

# torchvision 관련
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, STL10, SVHN

# 로컬 모듈
import datasets
from datasets.factory import get_dataset



class SimCLRTrainDataTransform(object):
    
    def __init__(
        self,
        input_height: int = 32,
        gaussian_blur: bool = False,
        jitter_strength: float = 0.5,
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
    norm_out = torch.norm(out, dim=-1, keepdim=True)
    sim_matrix = torch.matmul(out, out.T) / (norm_out * norm_out.T + 1e-8) # [2N, 2N]
    sim_matrix = sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0]  # for stability
    sim_matrix = torch.exp(sim_matrix / temperature)

    # Mask out self-similarity
    mask = (~torch.eye(2 * loss_batch_size, device=out.device).bool()).float()
    sim_matrix = sim_matrix * mask

    # Positive similarity (i와 i+N은 양의 쌍)
    norm_out1 = torch.norm(out_1, dim=-1, keepdim=True)
    norm_out2 = torch.norm(out_2, dim=-1, keepdim=True)
    pos_sim = torch.sum(out_1 * out_2, dim=-1) / (norm_out1.squeeze(-1) * norm_out2.squeeze(-1) + 1e-8)
    pos_sim = torch.exp(pos_sim / temperature)
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
                 optimization=None,
                 dataset=None,
                 training=None,
                 model=None,
                 **kwargs):
        
        super().__init__()
        self.save_hyperparameters()

        self.nt_xent_loss = nt_xent_loss
        self.encoder = self.init_encoder()

        # h -> || -> z
        dimension = self.hparams.model['projection_dim']
        self.projection = Projection(input_dim=dimension, hidden_dim=dimension)

    def init_encoder(self):
        encoder = models.resnet18() if self.hparams.resnet18 else models.resnet50()

        # CIFAR-10, CIFAR-100용 ResNet Stem 조정
        # 첫 번째 7x7 Conv of stride 2 -> 3x3 Conv of stride 1
        # 첫 번째 max pooling operation 제거
        if using_data == "CIFAR10" or using_data == "CIFAR100":
            encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            encoder.maxpool = nn.Identity() 

        encoder = nn.Sequential(*list(encoder.children())[:-1])  # 마지막 FC Layer 제거
        return encoder

    def configure_optimizers(self):
        learning_rate = self.hparams.training['learning_rate']
        warmup_epochs = self.hparams.training['warmup_epochs']
        max_epochs = self.hparams.training['epochs']

        optimizer_type = self.hparams.optimization['optimizer']
        use_scheduler = self.hparams.optimization['scheduler']
        use_warmup = self.hparams.optimization['warmup']
        use_cosine = self.hparams.optimization['cosine']

        if optimizer_type == "SGD":
            optimizer = SGD(self.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        elif optimizer_type == "LARS":
            optimizer = LARS(self.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-6, trust_coefficient=0.001, eps=1e-8)
        else:
            print(f"Optimizer {optimizer_type} is not supported. Using SGD as default.")
            optimizer = SGD(self.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

        schedulers = []
        # 1. Warmup
        if use_warmup:
            warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
            schedulers.append(("warmup", warmup))

        # 2. Cosine
        if use_cosine:
            cosine = CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs, eta_min=0.0)
            schedulers.append(("cosine", cosine))


        # 스케줄러 설정
        if use_scheduler:
            if use_warmup and use_cosine:
                # warmup + cosine은 SequentialLR로 묶기
                scheduler = {"scheduler": SequentialLR(optimizer, schedulers=[s[1] for s in schedulers if s[0] in ["warmup", "cosine"]], 
                                                       milestones=[warmup_epochs]), "interval": "epoch", "frequency": 1}
                return {"optimizer": optimizer, "lr_scheduler": scheduler}
            
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

        loss = self.nt_xent_loss(z1, z2, self.hparams.training['temperature'])
        return loss
    
    def train_dataloader(self):
        transform = SimCLRTrainDataTransform(input_height=self.hparams.dataset['input_size'], gaussian_blur=False, jitter_strength=0.5)
        train_dataset = get_dataset(name=self.hparams.dataset, transform=transform, root="./data")

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=num_workers,
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

def version_exist(version_num):
    # Check if the version folder already exists
    base_path = "tb_logs/SimCLR"
    version_path = f"{base_path}/v{version_num}"
    return os.path.exists(version_path)

def save_version_info():
    # Save the version information to a text file
    with open(f"version info.txt", "a") as f:
        f.write(f"----------------------------------------\n")
        f.write(f"[Version: {version}]\n\n")
        f.write(f"Date: {datetime.datetime.now()}\n")
        f.write(f"Dataset: {dataset_config['name']}\n")
        f.write(f"Batch Size: {training_config['batch_size']}\n")
        f.write(f"Max Epochs: {training_config['epochs']}\n")
        f.write(f"Temperature: {training_config['temperature']}\n")
        f.write(f"Learning Rate: {optimization_config['learning_rate']}\n")
        f.write(f"Warmup Epochs: {training_config['warmup_epochs']}\n")
        f.write(f"Using Model: {model_config['backbone']}\n")
        f.write(f"Using scheduler: {'Yes' if optimization_config['scheduler'] else 'No'}\n")
        f.write(f"Using optimizer: {optimization_config['optimizer']}\n")
        f.write(f"----------------------------------------\n\n")

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')  # 또는 'medium'
    # pick data

    ######################################## HYPERPARAMETERS ########################################
    #################################################################################################

    # Choose your dataset here
    # Supported datasets: "CIFAR10", "CIFAR100", "STL10", "SVHN"
    using_data = "CIFAR10"  

    # Number of workers for DataLoader
    num_workers = 4  
    
    # Version of the mode
    version = 21

    #################################################################################################
    #################################################################################################

    # Load hyperparameters from the selected dataset's YAML file
    config = yaml.safe_load(open(f"{using_data.lower()}.yaml", "r"))

    # Load hyperparameters from config (e.g., cifar10.yaml)
    dataset_config = config["dataset"]
    training_config = config["training"]
    optimization_config = config["optimization"]
    model_config = config["model"]

    # Assume continuation of training is not needed initially
    continue_training = False
    while version_exist(version):
        print(f"Version v{version} already exists. Do you want to continue training from this version? (y/n/q)")
        user_input = input().strip().lower()
        if user_input == 'y':
            continue_training = True
            break
        elif user_input == 'n':
            continue_training = False
            version += 1
        elif user_input == 'q':
            print("Exiting the program.")
            exit()
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


    print(f"Starting training with version v{version}...")

    if continue_training:
        checkpoint_path = f"v{version}.ckpt"
        model = SimCLR.load_from_checkpoint(
            checkpoint_path, 
            dataset = dataset_config,
            training = training_config,
            optimization = optimization_config,
            model = model_config
        )
    else:
        model = SimCLR(
            dataset = dataset_config,
            training = training_config,
            optimization = optimization_config,
            model = model_config
        )
        save_version_info()


    logger = TensorBoardLogger("tb_logs", name="SimCLR", version=f"v{version}")
    
    trainer = pl.Trainer(
        max_epochs=training_config["max_epochs"], 
        enable_progress_bar=True, 
        devices=1, 
        accelerator="gpu", 
        resume_from_checkpoint=checkpoint_path if continue_training else None,
        logger=logger)
    trainer.fit(model)


