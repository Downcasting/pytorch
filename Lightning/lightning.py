import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning import Trainer

# hyper parameters
input_size = 784 # 28x28
hidden_size = 500
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001


class LitNeuralNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LitNeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no softmax here
        return out

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, 28*28)

        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)
    
    def train_dataloader(self):
        train_dataset = torchvision.datasets.MNIST(root='./../data',
                                           train=True,
                                           transform=transforms.ToTensor(), 
                                           download=True)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           num_workers=4,
                                           persistent_workers=True,
                                           shuffle=True)
        
        return train_loader
    
    def validation_step(self, batch, batch_idx):    
        images, labels = batch
        images = images.reshape(-1, 28*28)

        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss
    
    def val_dataloader(self):
        val_dataset = torchvision.datasets.MNIST(root='./../data',
                                          train=False,
                                          transform=transforms.ToTensor())

        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                          batch_size=batch_size,
                                          num_workers=4,
                                          persistent_workers=True,
                                          shuffle=False)
        return val_loader
    
    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get('val_loss')
        if val_loss is not None:
            self.log("val_loss", val_loss, prog_bar=True)
    
if __name__ == '__main__':
    trainer = Trainer(max_epochs = num_epochs,
                      fast_dev_run=False)
    model = LitNeuralNet(input_size, hidden_size, num_classes)
    # trainer = pl.Trainer(max_epochs=num_epochs)
    trainer.fit(model)