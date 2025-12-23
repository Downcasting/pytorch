import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import datetime

from collections import OrderedDict

import yaml

from datasets.get_dataset import get_dataset, get_test_dataset
from models.backbone import get_backbone


class LinearClassifier(pl.LightningModule):
    def __init__(self, encoder_location, num_classes=10):
        super().__init__()
        self.encoder = self.load_encoder(encoder_location)
        feature_dim = model_config["projection_dim"]
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
            features = torch.flatten(features, start_dim=1)  # Flatten the features
        return self.fc(features)
    
    def load_encoder(self, encoder_location):
        checkpoint = torch.load(encoder_location, map_location='cuda')
        encoder = get_backbone(model_config["backbone"], using_data=dataset_config["name"])

        # encoder.load_state_dict(new_state_dict, strict=False)

        missing, unexpected = encoder.load_state_dict(checkpoint, strict=False)

        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

        if len(missing) > 0 or len(unexpected) > 0:
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
            param.requires_grad = False  # Freeze encoder parameters

        return encoder
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        images, labels = images.cuda(), labels.cuda()

        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()

        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_acc", acc, prog_bar=True, logger=True)

        # Log the loss value
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images, labels = images.cuda(), labels.cuda()

        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()

        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, logger=True)

        # Log the loss value
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.fc.parameters(), lr=0.001)
        scheduler = {
            "scheduler" : optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3),
            "monitor" : "val_loss",
            "interval" : "epoch",
            "frequency" : 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    

    def train_dataloader(self):
        train_dataset = get_dataset(name=dataset_config['name'], transform=transform, root="./../data", pretrain=False)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True,
            shuffle=True)
        
        return train_loader
    
    def val_dataloader(self):
        val_dataset = get_test_dataset(name=dataset_config['name'], transform=transform, root="./../data")

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

    ### ResNet-18 or ResNet-50 ###

    using_data = "cifar10"
    using_data = using_data.upper()
    batch_size = 128
    num_workers = 4 # DataLoader의 num_workers
    version = 24 # version
    max_epochs = 100 # Maximum epochs
    ##############################

    config = yaml.safe_load(open(f"config/{using_data.lower()}.yaml", "r"))

    dataset_config = config["dataset"]
    model_config = config["model"]
    num_of_classes = dataset_config["classes"]
    transform_config = config["transform"]
    normalize = transform_config['normalize']

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=normalize['mean'],
            std=normalize['std']
        )
    ])

    logger = TensorBoardLogger("tb_logs", name=f"SimCLR Eval_{using_data}", version=f"v{version}")

    torch.set_float32_matmul_precision('medium')
    # Model Initialization
    model = LinearClassifier(encoder_location=f"{using_data}_v{version}_encoder.pth", num_classes=num_of_classes)
    # Trainer 설정
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu", devices=1, logger=logger)
    # Model Training
    trainer.fit(model)
    # Model Saving
    # trainer.save_checkpoint("linear_classifier_300.ckpt") 
