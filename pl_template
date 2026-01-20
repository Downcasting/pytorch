import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

class BaseLitModule(pl.LightningModule):
    def __init__(
        self, 
        model: nn.Module, 
        criterion: nn.Module = nn.CrossEntropyLoss(), 
        lr: float = 1e-3,
        optimizer_cls: type = torch.optim.Adam,
        scheduler_cls: type = None
    ):
        """
        Args:
            model (nn.Module): 학습할 실제 Pytorch 모델
            criterion (nn.Module): 손실 함수 (기본: CrossEntropyLoss)
            lr (float): Learning Rate
            optimizer_cls (type): 사용할 Optimizer 클래스 (기본: Adam)
            scheduler_cls (type): 사용할 Scheduler 클래스 (옵션)
        """
        super().__init__()
        # model을 hparams에 저장하지 않기 위해 ignore 처리 (모델 구조가 너무 크면 로그가 지저분해짐)
        self.save_hyperparameters(ignore=['model', 'criterion'])
        
        self.model = model
        self.criterion = criterion
        self.optimizer_cls = optimizer_cls
        self.scheduler_cls = scheduler_cls

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # 로그 기록 (prog_bar=True로 설정하면 진행바에 표시됨)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # 검증 손실 기록
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # 필요하다면 여기에 Accuracy 같은 메트릭 계산 로직 추가
        # preds = torch.argmax(logits, dim=1)
        # acc = (preds == y).float().mean()
        # self.log('val_acc', acc, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(self.parameters(), lr=self.hparams.lr)
        
        if self.scheduler_cls:
            scheduler = self.scheduler_cls(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",  # 스케줄러가 모니터링할 메트릭
                },
            }
        return optimizer

# -------------------------------------------------------------------------
# 사용 예시 (Usage Example)
# -------------------------------------------------------------------------
if __name__ == '__main__':
    from torch.utils.data import DataLoader, TensorDataset

    # 1. 데이터 준비 (임의의 더미 데이터)
    # 실제 사용 시에는 본인의 DataLoader로 대체하세요.
    train_x, train_y = torch.randn(100, 1, 28, 28), torch.randint(0, 10, (100,))
    val_x, val_y = torch.randn(20, 1, 28, 28), torch.randint(0, 10, (20,))
    
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=32)

    # 2. 모델 정의 (어떤 nn.Module이든 가능)
    # 예: 간단한 CNN
    backbone = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    # 3. Lightning Module 래핑 (여기서 모델을 주입)
    model = BaseLitModule(
        model=backbone,
        criterion=nn.CrossEntropyLoss(),
        lr=1e-3
    )

    # 4. 콜백 설정 (옵션: 체크포인트 저장, 조기 종료 등)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./checkpoints',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min'
    )

    # 5. Trainer 실행
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",    # GPU/MPS/CPU 자동 감지
        devices="auto",        # 가용 장치 자동 할당
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=True            # TensorBoard 기본 사용 (원하면 WandB 등으로 교체 가능)
    )

    # 학습 시작
    trainer.fit(model, train_loader, val_loader)