import torch
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam

import matplotlib.pyplot as plt
import numpy as np


torch.set_float32_matmul_precision("high")

# Hyperparameters
IMG_SIZE = 64
BATCH_SIZE = 128
T = 300  # Diffusion timesteps
LR = 0.001
EPOCHS = 100

def normalize(t):
    return (t*2)-1

# Dataset preparation
def load_transformed_dataset():
    data_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(normalize)  # Normalize to [-1, 1]
    ])
    
    train = torchvision.datasets.StanfordCars(root="./../data/", download=False, transform=data_transforms, split='train')
    test = torchvision.datasets.StanfordCars(root="./../data/", download=False, transform=data_transforms, split='test')
    
    return torch.utils.data.ConcatDataset([train, test])

# Linear Beta Schedule
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

betas = linear_beta_schedule(timesteps=T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

if __name__ == "__main__":
    data = load_transformed_dataset()
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=8, persistent_workers=True)

    # Model Definition (U-Net)
    class SimpleUnet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 3, 3, padding=1)
            )

        def forward(self, x, t):
            return self.conv(x)
        
        # Diffusion Model using PyTorch Lightning
    class DiffusionModel(pl.LightningModule):
        def __init__(self, model, timesteps=T):
            super().__init__()
            self.model = model
            self.timesteps = timesteps

        def forward_diffusion_sample(self, x_0, t):
            noise = torch.randn_like(x_0)
            sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape).to(t.device)
            sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape).to(t.device)
            return sqrt_alphas_cumprod_t * x_0.to(t.device) + sqrt_one_minus_alphas_cumprod_t * noise.to(t.device), noise.to(t.device)

        def get_loss(self, x_0, t):
            x_noisy, noise = self.forward_diffusion_sample(x_0, t)
            noise_pred = self.model(x_noisy, t)
            return F.l1_loss(noise, noise_pred)

        def training_step(self, batch, batch_idx):
            x_0 = batch[0]
            t = torch.randint(0, self.timesteps, (BATCH_SIZE,), device=self.device).long()
            loss = self.get_loss(x_0, t)
            self.log("train_loss", loss)
            return loss

        def configure_optimizers(self):
            return Adam(self.model.parameters(), lr=LR)
        
        def on_train_epoch_end(self):
            epoch = self.current_epoch
            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), f"diffusion_model_{epoch}.pth")
                batch = next(iter(dataloader))  # 첫 번째 배치 선택
                x_0 = batch[0]  # 배치에서 첫 번째 이미지를 가져옴
                t = torch.randint(0, self.timesteps, (x_0.size(0),), device=self.device).long()  # 랜덤 timestep 선택

                # Noisy image 생성 (디퓨전 샘플링)
                x_noisy, _ = self.forward_diffusion_sample(x_0, t)
                
                # 모델로 복원된 이미지 생성
                noise_pred = self.model(x_noisy, t)

                # 이미지를 matplotlib로 출력
                self.show_images(x_0, noise_pred)
    
        def show_images(self, x_0, noise_pred):
            # Reverse transforms to prepare the image for displaying
            reverse_transforms = transforms.Compose([
                transforms.Lambda(lambda t: (t + 1) / 2),  # Normalize to [0, 1]
                transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
                transforms.Lambda(lambda t: t * 255.),  # Scale to [0, 255]
                transforms.Lambda(lambda t: t.detach().numpy().astype(np.uint8)),  # Convert to numpy (move to CPU)
                transforms.ToPILImage(),  # Convert to PIL Image
            ])

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # 첫 번째 이미지: 원본 이미지
            img = reverse_transforms(x_0[0].cpu())  # CPU로 이동 후 변환
            axes[0].imshow(img)
            axes[0].set_title("Original Image")

            # 두 번째 이미지: 모델이 복원한 이미지
            img_pred = reverse_transforms(noise_pred[0].cpu())  # CPU로 이동 후 변환
            axes[1].imshow(img_pred)
            axes[1].set_title("Restored Image")

            plt.axis('off')
            plt.show()


    # Training
    model = SimpleUnet()
    diffusion_module = DiffusionModel(model)
    trainer = pl.Trainer(max_epochs=EPOCHS, accelerator="gpu" if torch.cuda.is_available() else "cpu")
    trainer.fit(diffusion_module, dataloader)
