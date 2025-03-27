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

class LinearEvaluation(pl.LightningModule):
    def __init__(self, encoder, num_classes=10, learning_rate=0.001):
        super(LinearEvaluation, self).__init__()
        self.encoder = encoder
        self.encoder.requires_grad_(False)  # 🧊 Encoder freezing (가중치 고정)

        self.classifier = nn.Linear(512, num_classes)  # CIFAR-10 분류기

        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        with torch.no_grad():  # ❄️ Encoder는 gradient 계산 X
            x = self.encoder(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

# Pretrained Encoder 불러오기
encoder = models.resnet18(pretrained=False)
encoder.load_state_dict(torch.load("simclr_encoder.pth"))

# 선형 분류기 학습
linear_eval = LinearEvaluation(encoder)
trainer = pl.Trainer(max_epochs=10, accelerator="gpu" if torch.cuda.is_available() else "cpu")
trainer.fit(linear_eval)