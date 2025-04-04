import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

import pytorch_lightning as pl

from collections import OrderedDict

from torchvision.datasets import CIFAR10

# 1️⃣ CIFAR-10 데이터셋 로드
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

batch_size = 128

# 새로운 키를 적용한 state_dict 로드

class LinearClassifier(pl.LightningModule):
    def __init__(self, encoder_location, num_classes=10):
        super().__init__()
        self.encoder = self.load_encoder(encoder_location)
        self.fc = nn.Linear(512, num_classes)  # ResNet18의 feature_dim=512

    def forward(self, x):
        with torch.no_grad():  # Encoder 부분은 gradient 계산 안 함
            features = self.encoder(x)
            features = torch.flatten(features, start_dim=1)  # Flatten the features
        return self.fc(features)
    
    def load_encoder(self, encoder_location):
        checkpoint = torch.load(encoder_location, map_location='cuda')
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
        
        # 1️⃣ ResNet-18 불러오기 (pretrained=False 명시)
        encoder = torchvision.models.resnet18(pretrained=False)  # 네가 사용한 encoder 구조로 변경해야 함

        # 2️⃣ SimCLR Encoder 불러오기
        encoder.load_state_dict(new_state_dict, strict=False)

        # 3️⃣ Encoder의 마지막 fc 레이어 제거 (feature extractor만 사용)
        encoder = nn.Sequential(*list(encoder.children())[:-1])  # 🔥 마지막 FC 제거!

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

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)  # 🔥 Accuracy 로그 추가!

        # Log the loss value
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images, labels = images.cuda(), labels.cuda()

        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        preds = torch.argmax(outputs, dim=1)  # 🔥 가장 확률 높은 class 선택
        acc = (preds == labels).float().mean()  # 🔥 Accuracy 계산

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)  # 🔥 Accuracy 로그 추가!

        # Log the loss value
        return loss

    def configure_optimizers(self):
        # 4️⃣ Linear Classifier 학습을 위한 optimizer 설정
        optimizer = optim.Adam(self.fc.parameters(), lr=0.001)
        return optimizer
    
    def train_dataloader(self):
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

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    # 1️⃣ 모델 초기화
    model = LinearClassifier(encoder_location="encoder_5_500.pth")
    # 2️⃣ Trainer 설정
    trainer = pl.Trainer(max_epochs=50, accelerator="gpu", devices=1)
    # 3️⃣ 모델 학습
    trainer.fit(model)
    # 4️⃣ 모델 저장
    trainer.save_checkpoint("linear_classifier_5_500.ckpt")  # ✅ Lightning 권장 방식
