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

# 로컬 모듈
from datasets.get_dataset import get_dataset, get_test_dataset
from models.backbone import get_backbone

# 1️⃣ CIFAR-10 데이터셋 로드
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], 
        std=[0.2023, 0.1994, 0.2010]
    )
])

batch_size = 128

# 새로운 키를 적용한 state_dict 로드

class LinearClassifier(pl.LightningModule):
    def __init__(self, encoder_location, num_classes=10):
        super().__init__()
        self.encoder = self.load_encoder(encoder_location)
        feature_dim = model_config["projection_dim"]  # feature_dim은 config에서 가져옴
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():  # Encoder 부분은 gradient 계산 안 함
            features = self.encoder(x)
            features = torch.flatten(features, start_dim=1)  # Flatten the features
        return self.fc(features)
    
    def load_encoder(self, encoder_location):
        checkpoint = torch.load(encoder_location, map_location='cuda')
        '''
        new_state_dict = OrderedDict()
        key_map = {
            "0.": "conv1.",
            "1.": "bn1.",
            "4.": "layer1.",
            "5.": "layer2.",
            "6.": "layer3.",
            "7.": "layer4."
        }

        for k, v in checkpoint.items():
            new_key = k
            for old, new in key_map.items():
                if k.startswith(old):
                    new_key = k.replace(old, new, 1)
                    break
            new_state_dict[new_key] = v
        '''
        '''
        # 1️⃣ ResNet-18 / ResNet-50 불러오기 (pretrained=False 명시)
        if self.resnet18:
            encoder = torchvision.models.resnet18(weights=None)  # ResNet-18 사용
        else:
            encoder = torchvision.models.resnet50(weights=None)  # ResNet-50 사용

        # !!! ResNet-18/50의 conv1 레이어를 CIFAR-10에 맞게 수정
        encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # 첫 번째 max pooling operation 제거
        encoder.maxpool = nn.Identity() 

        # 3️⃣ Encoder의 마지막 fc 레이어 제거 (feature extractor만 사용)
        encoder = nn.Sequential(*list(encoder.children())[:-1])  # 🔥 마지막 FC 제거!
        '''

        encoder = get_backbone(model_config["backbone"], using_data=dataset_config["name"])

        # 2️⃣ SimCLR Encoder 불러오기
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
            

        # 4️⃣ Encoder freeze (학습 X)
        encoder = encoder.cuda()  # GPU로 이동
        for param in encoder.parameters():
            param.requires_grad = False  # encoder의 가중치는 학습하지 않음 (freeze)

        return encoder
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        images, labels = images.cuda(), labels.cuda()

        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        preds = torch.argmax(outputs, dim=1)  # 🔥 가장 확률 높은 class 선택
        acc = (preds == labels).float().mean()  # 🔥 Accuracy 계산

        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_acc", acc, prog_bar=True, logger=True)  # 🔥 Accuracy 로그 추가!

        # Log the loss value
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images, labels = images.cuda(), labels.cuda()

        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        preds = torch.argmax(outputs, dim=1)  # 🔥 가장 확률 높은 class 선택
        acc = (preds == labels).float().mean()  # 🔥 Accuracy 계산

        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, logger=True)  # 🔥 Accuracy 로그 추가!

        # Log the loss value
        return loss

    def configure_optimizers(self):
        # 4️⃣ Linear Classifier 학습을 위한 optimizer 설정
        optimizer = optim.Adam(self.fc.parameters(), lr=0.001)
        scheduler = {
            "scheduler" : optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True),
            "monitor" : "val_loss",
            "interval" : "epoch",
            "frequency" : 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    

    def train_dataloader(self):
        train_dataset = get_dataset(name=dataset_config['name'], transform=transform, root="./../data")

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
    
    def training_epoch_end(self, outputs):
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("current_lr", current_lr, prog_bar=True, logger=True)
    
    def on_fit_end(self):
        with open(f"/log/{using_data}_eval info.txt", "a") as f:
            f.write(f"----------------------------------------\n")
            f.write(f"[Version: {version}]\n\n")
            f.write(f"Date: {datetime.datetime.now()}\n")
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Max Epochs: {max_epochs}\n")
            f.write(f"Total Accuracy: {self.trainer.callback_metrics['val_acc']*100:.2f}%\n")
            f.write(f"----------------------------------------\n\n")

if __name__ == "__main__":

    ### ResNet-18 or ResNet-50 ###

    using_data = "stl10"
    using_data = using_data.upper()  # 소문자로 변환
    num_workers = 4 # DataLoader의 num_workers 설정
    version = 1 # 버전
    max_epochs = 100 # 최대 에폭
    ##############################

    config = yaml.safe_load(open(f"config/{using_data.lower()}.yaml", "r"))

    dataset_config = config["dataset"]
    model_config = config["model"]
    num_of_classes = dataset_config["classes"]

    logger = TensorBoardLogger("tb_logs", name=f"SimCLR Eval_{using_data}", version=f"v{version}")

    torch.set_float32_matmul_precision('medium')
    # 1️⃣ 모델 초기화
    model = LinearClassifier(encoder_location=f"{using_data}_v{version}_encoder.pth", num_classes=num_of_classes)
    # 2️⃣ Trainer 설정
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu", devices=1, logger=logger)
    # 3️⃣ 모델 학습
    trainer.fit(model)
    # 4️⃣ 모델 저장
    # trainer.save_checkpoint("linear_classifier_300.ckpt")  # ✅ Lightning 권장 방식
