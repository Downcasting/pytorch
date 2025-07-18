import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch

from utils.config import load_config
from backbone import get_backbone
from dino_module import DINO
from datamodule import DINODataModule

def main():
    # 🔧 설정 불러오기
    cfg = load_config("config.yaml")

    # 🧠 백본 모델 로딩
    student_backbone, feature_dim = get_backbone(cfg.model.backbone, cfg.dataset.name)
    teacher_backbone, _ = get_backbone(cfg.model.backbone, cfg.dataset.name)

    # 🧪 DINO 모델 생성
    model = DINO(student_backbone, teacher_backbone, feature_dim, cfg)

    # 🧳 DataModule 준비
    datamodule = DINODataModule(cfg)
    
    # 🧚‍♀️ 콜백 설정
    checkpoint_cb = ModelCheckpoint(
        monitor="train_loss",
        save_top_k=1,
        mode="min",
        filename="{epoch}-{train_loss:.4f}"
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # 🚀 Trainer 실행
    trainer = pl.Trainer(
        accelerator="gpu" if cfg.training.use_gpu and torch.cuda.is_available() else "cpu",
        max_epochs=cfg.training.max_epochs,
        callbacks=[checkpoint_cb, lr_monitor],
        log_every_n_steps=10,
        precision=16 if cfg.training.use_gpu else 32,
    )

    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
