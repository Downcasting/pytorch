import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import yaml

from models.backbone import get_backbone
from dino import DINO
from datasets.get_dataset import DINODataModule
def main():

    torch.set_float32_matmul_precision('medium')

    # 🔧 설정 불러오기
    cfg = yaml.safe_load(open(f"config/{using_data}.yaml", "r"))

    # 🧠 백본 모델 로딩
    student_backbone, feature_dim = get_backbone(cfg['model']['student_backbone'], cfg['dataset']['input_size'])
    teacher_backbone, _ = get_backbone(cfg['model']['teacher_backbone'], cfg['dataset']['input_size'])

    # 🧪 DINO 모델 생성
    model = DINO(student_backbone, teacher_backbone, feature_dim, cfg)

    # 🧳 DataModule 준비
    datamodule = DINODataModule(
        name=using_data,
        root='./../data',
        batch_size=cfg['training']['batch_size'],
        num_workers=num_workers,
        cfg=cfg
    )
    
    # 🧚‍♀️ 콜백 설정
    checkpoint_cb = ModelCheckpoint(
        monitor="train_loss",
        save_top_k=1,
        mode="min",
        filename="{epoch}-{train_loss:.4f}"
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # TensorBoard Logger 설정
    tb_logger = TensorBoardLogger(
        save_dir="tb_logs",
        name=f"{using_data}_dino_v{version}",
        default_hp_metric=False
    )

    # 🚀 Trainer 실행
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=cfg['training']['max_epochs'],
        callbacks=[checkpoint_cb, lr_monitor],
        log_every_n_steps=10,
        logger=tb_logger,
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":

    ### HYPERPARAMETERS ###
    using_data = "CIFAR10"  # 사용할 데이터셋 이름 (예: CIFAR10, CIFAR100 등)
    using_data = using_data.upper()
    num_workers = 4  # 데이터 로더의 워커 수

    version = 1
    #######################

    main()


