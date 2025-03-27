import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning import Trainer

torch.set_float32_matmul_precision("high")

# hyper parameters
batch_size = 512
num_epochs = 10
temperature = 0.5
learning_rate = 0.001


class SimCLR(pl.LightningModule):
    def __init__(self, backbone, temperature):
        super(SimCLR, self).__init__()
        self.backbone = backbone
        self.temperature = temperature

        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 128)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        z = torch.flatten(h, 1)
        return self.projection_head(z)

    def nt_xent_loss(self, z_i, z_j):
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        batch_size = z_i.shape[0]

        z = torch.cat([z_i, z_j], dim=0)
        sim_matrix = torch.mm(z, z.T) / self.temperature

        mask = torch.eye(2 * batch_size, device=sim_matrix.device)
        sim_matrix = sim_matrix * (1 - mask)

        pos_sim = torch.cat([torch.diag(sim_matrix, batch_size), torch.diag(sim_matrix, -batch_size)], dim=0)
        loss = -torch.log(torch.exp(pos_sim) / torch.sum(torch.exp(sim_matrix), dim=1))
        return loss.mean()
    
    def training_step(self, batch, batch_idx):
        (x_i, x_j), _ = batch
        z_i = self(x_i)
        z_j = self(x_j)
        loss = self.nt_xent_loss(z_i, z_j)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)
    def train_dataloader(self):
 
        train_dataset = torchvision.datasets.CIFAR10(root='./../data', 
                                             train=True, 
                                             transform = SimCLRTransform(),
                                             download=True)

        train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size=batch_size, 
                                                   num_workers=4,
                                                   persistent_workers=True,
                                                   shuffle=True)
        return train_loader
    
    def val_dataloader(self):
        val_dataset = torchvision.datasets.CIFAR10(root='./../data',
                                            train=False, 
                                            transform = SimCLRTransform()
                                            )

        val_loader = torch.utils.data.DataLoader(val_dataset, 
                                            batch_size=batch_size, 
                                            num_workers=4,
                                            persistent_workers=True,
                                            shuffle=False)
        return val_loader
    def on_train_end(self):
        torch.save(SimCLR_module.encoder.state_dict(), "simclr_encoder.pth")
    
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
    model = models.resnet18(pretrained=True)
    SimCLR_module = SimCLR(model, temperature)
    trainer = pl.Trainer(max_epochs=num_epochs, accelerator="gpu" if torch.cuda.is_available() else "cpu")
    trainer.fit(SimCLR_module)
