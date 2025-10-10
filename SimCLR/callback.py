import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms as transforms

import copy

# 로컬 모듈 import (경로에 맞게 수정 필요 시 수정)
from datasets.get_dataset import get_dataset, get_test_dataset

# Linear Probe를 위한 간단한 LightningModule
class LinearProbe(pl.LightningModule):
    def __init__(self, encoder, num_classes, feature_dim, learning_rate=1e-3):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.feature_dim = feature_dim

        self.fc = nn.Linear(feature_dim, num_classes)

        # Encoder는 얼려서 가중치 업데이트 방지
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
            features = torch.flatten(features, start_dim=1)
        return self.fc(features)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        self.log("probe_train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log("probe_val_loss", loss)
        self.log("probe_val_acc", acc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.fc.parameters(), lr=self.learning_rate)

# 주기적으로 선형 평가를 수행할 콜백
class OnlineLinearEvaluation(pl.Callback):
    def __init__(self, dataset_config, transform_config, batch_size=128, num_workers=4, probe_epochs=5, eval_every_n_epochs=25, feature_dim=512):
        super().__init__()
        self.dataset_config = dataset_config
        self.transform_config = transform_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.feature_dim = feature_dim
        self.probe_epochs = probe_epochs
        self.eval_every_n_epochs = eval_every_n_epochs
        
        # 평가 시 사용할 데이터 변환 정의
        normalize = self.transform_config['normalize']
        self.eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize['mean'], std=normalize['std'])
        ])


    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch + 1
        if current_epoch % self.eval_every_n_epochs != 0:
            return

        print(f"\nEpoch {current_epoch}: --- Running Online Linear Evaluation ---")

        # 평가용 데이터로더 준비
        train_dataset = get_dataset(name=self.dataset_config['name'], transform=self.eval_transform, root="./../data", pretrain=False)
        val_dataset = get_test_dataset(name=self.dataset_config['name'], transform=self.eval_transform, root="./../data")

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, persistent_workers=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, persistent_workers=True)

        # !!! Encoder를 복사하지 않고 Deep Copy로 변경 !!!
        encoder_copy = copy.deepcopy(pl_module.encoder)

        # Linear Probe 모델 초기화
        probe_model = LinearProbe(
            encoder=encoder_copy,
            num_classes=self.dataset_config['classes'],
            feature_dim=self.feature_dim,
        ).to(pl_module.device)

        # 평가용 Trainer 설정 및 실행
        eval_trainer = pl_trainer = pl.Trainer(
            max_epochs=self.probe_epochs,
            devices=1,
            accelerator='gpu',
            enable_progress_bar=False,
            logger=False
        )
        eval_trainer.fit(probe_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # 평가 결과를 메인 모듈의 로그에 기록
        final_val_acc = eval_trainer.callback_metrics.get("probe_val_acc", 0.0)
        pl_module.log("online_val_acc", final_val_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        print(f"Epoch {current_epoch}: --- Online Linear Evaluation Finished --- Val Acc: {final_val_acc:.4f}\n")

        pl_module.train()  # 메인 모듈을 다시 훈련 모드로 전환

        pl_module.to(pl_module.device)  # 메인 모듈이 올바른 장치에 있는지 확인