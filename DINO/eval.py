import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import yaml

import torch.nn as nn
import torch.nn.functional as F

from models.backbone import get_backbone
from dino import DINO
from datasets.get_dataset import DINODataModule

import datetime


class LinearClassifier(pl.LightningModule):
    def __init__(self, encoder_location, num_classes=10):
        super().__init__()
        self.encoder = self.load_encoder(encoder_location)
        feature_dim = model_config["projection_dim"]  # feature_dim은 config에서 가져옴
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        with torch.no.grad():
            features = self.encoder(x)
            features = torch.flatten(features, start_dim=1)
        return self.fc(features)
    
    def load_encoder(self, encoder_location):
        checkpoint = torch.load(encoder_location, map_location='cuda')
        encoder = get_backbone(backbone_name, image_size=image_size)
        missing_keys, unexpected_keys = encoder.load_state_dict(checkpoint, strict=False)
    
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            print("Warning: Some keys were not loaded correctly!")
            print("Do you want to continue? (y/n)")
            while True:
                response = input().strip().lower()
                if response == 'n':
                    print("Exiting due to missing/unexpected keys.")
                    exit(1)
                elif response == 'y':
                    print("Continuing despite missing/unexpected keys.")
                    break
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")

        encoder = encoder.cuda()
        for param in encoder.parameters():
            param.requires_grad = False

        return encoder

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images, labels = images.cuda(), labels.cuda()

        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)


    
    def validation_step(self, batch, batch_idx):
        

    def configure_optimizers(self):
    

    def train_dataloader(self):
        
    
    def val_dataloader(self):
        val_dataset = DINODataModule(name=dataset_config['name'], transform=transform, root="./../data")

        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True,
            shuffle=False)
        
        return val_loader
    
    def on_train_epoch_end(self):
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("current_lr", current_lr, prog_bar=True, logger=True)
    
    def on_fit_end(self):
        with open(f"log/{using_data}_eval info.txt", "a") as f:
            f.write(f"----------------------------------------\n")
            f.write(f"[Version: {version}]\n\n")
            f.write(f"Date: {datetime.datetime.now()}\n")
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Max Epochs: {max_epochs}\n")
            f.write(f"Total Accuracy: {self.trainer.callback_metrics['val_acc']*100:.2f}%\n")
            f.write(f"----------------------------------------\n\n")

if __name__ == "__main__":

    ### HYPERPARAMETERS ###
    using_data = "CIFAR10"  # 사용할 데이터셋 이름 (예: CIFAR10, CIFAR100 등)
    using_data = using_data.upper()

    batch_size = 128
    max_epochs = 100
    num_workers = 4  # 데이터 로더의 워커 수
    version = 1
    #######################

    config = yaml.safe_load(open(f"config/{using_data}.yaml", "r"))

    model_config = config['model']
    dataset_config = config['dataset']

    backbone_name = model_config['student_backbone']
    image_size = dataset_config['input_size']
    encoder_location = model_config['encoder_location']

    
