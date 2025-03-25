import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning import Trainer

# hyper parameters
num_classes = 10
num_epochs = 10
batch_size = 512
learning_rate = 0.001
temperature = 0.5


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print("using device: " + "gpu" if torch.cuda.is_available() else "cpu")

class NT_Xent(torch.nn.Module):
    def __init__(self, temperature):
        super(NT_Xent, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]

        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Compute similarity matrix
        z = torch.cat([z_i, z_j], dim=0)  # (2N, feature_dim)
        sim_matrix = torch.mm(z, z.T)  # (2N, 2N)

        # Apply temperature scaling
        sim_matrix /= self.temperature

        # Mask diagonal elements (self-similarity 제거)
        mask = torch.eye(2 * batch_size, device=sim_matrix.device)
        sim_matrix = sim_matrix * (1 - mask)

        # Positive pairs similarity
        pos_sim = torch.cat([torch.diag(sim_matrix, batch_size), torch.diag(sim_matrix, -batch_size)], dim=0)

        # NT-Xent Loss 계산
        labels = torch.arange(2 * batch_size, device=z.device)
        labels = (labels + batch_size) % (2 * batch_size)  # positive pair index

        loss = -torch.log(torch.exp(pos_sim) / torch.sum(torch.exp(sim_matrix), dim=1))
        loss = loss.mean()
        
        return loss

class SimCLR(pl.LightningModule):
    def __init__(self, model, temperature):
        super(SimCLR, self).__init__()
        self.model = model
        self.temperature = temperature

        self.encoder = nn.Sequential(*list(model.children())[:-1])
        
        self.proj_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        out = self.proj_head(x)
        return out

    def training_step(self, batch, batch_idx):
        (x_i, x_j), _ = batch  # Augmentation 된 두 개의 이미지
        z_i = self(x_i)
        z_j = self(x_j)
        loss = self.nt_xent_loss(z_i, z_j)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer
    
    def train_dataloader(self):
 
        train_dataset = torchvision.datasets.CIFAR10(root='./../data', 
                                             train=True, 
                                             transform=SimCLRTransform(),
                                             download=True)

        train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size=batch_size, 
                                                   num_workers=4,
                                                   persistent_workers=True,
                                                   shuffle=True)
        return train_loader
    
    def validation_step(self, batch):    
        images, labels = batch
        images = images.reshape(-1, 28*28)

        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss
    
    def val_dataloader(self):
        val_dataset = torchvision.datasets.CIFAR10(root='./../data',
                                            train=False, 
                                            transform=SimCLRTransform())

        val_loader = torch.utils.data.DataLoader(val_dataset, 
                                            batch_size=batch_size, 
                                            num_workers=4,
                                            persistent_workers=True,
                                            shuffle=False)
        return val_loader
    
    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get('val_loss')
        if val_loss is not None:
            self.log("val_loss", val_loss, prog_bar=True)
    
class SimCLRTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)
    
if __name__ == '__main__':

    # Training
    model = models.resnet50(pretrained=True)
    SimCLR_module = SimCLR(model, temperature)
    trainer = pl.Trainer(max_epochs=num_epochs, accelerator="gpu" if torch.cuda.is_available() else "cpu")
    trainer.fit(SimCLR_module)
