import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import yaml
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import get_backbone
from datasets.get_dataset import DINOEvalDataModule
import glob
import os

class DINOEval(pl.LightningModule):
    def __init__(self, feature_dim, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=["student_backbone", "teacher_backbone"])
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for p in self.teacher.parameters():
            p.requires_grad = False  # teacher는 학습하지 않음

        self.model = cfg['model']
        self.dataset = cfg['dataset']

        self.using_data = cfg['dataset']['name'].upper()

        self.encoder = self.load_encoder()
        self.fc = nn.Linear(feature_dim, self.dataset['classes'])

    def forward(self, x):
        with torch.no_grad():
            z = self.encoder(x)
            features = torch.flatten(z, start_dim=1)
        return self.fc(features)

    def load_encoder(self):
        ckpt_list = glob.glob(f"tb_logs/{self.using_data}_dino_v{version}/checkpoints/*.ckpt")
        checkpoint_location = ckpt_list[0] if ckpt_list else None

        ckpt = torch.load(checkpoint_location, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = ckpt['state_dict']

        encoder, _ = get_backbone(self.model['student_backbone'], self.dataset['input_size'])
        encoder.load_state_dict(state_dict)

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

def main():
    global version, using_data, num_workers

    torch.set_float32_matmul_precision('medium')

    # 🔧 설정 불러오기
    cfg = yaml.safe_load(open(f"config/{using_data}.yaml", "r"))

    # 🧳 DataModule 준비
    datamodule = DINOEvalDataModule(
        name=using_data,
        root='./../data',
        batch_size=128,
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
        name=f"{using_data}_dinoeval_v{version}",
        default_hp_metric=False
    )
        
    trainer = pl.Trainer(
        accumulate_grad_batches=4,
        precision='16-mixed',
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=cfg['training']['max_epochs'],
        callbacks=[checkpoint_cb, lr_monitor],
        log_every_n_steps=10,
        logger=tb_logger,
        enable_progress_bar=True,
        )

    model = DINOEval(feature_dim=224, cfg=cfg)
    trainer.fit(model, datamodule=datamodule)


    
if __name__ == "__main__":
    global using_data, num_workers

    ### HYPERPARAMETERS ###
    using_data = "CIFAR10"  # 사용할 데이터셋 이름 (예: CIFAR10, CIFAR100 등)
    using_data = using_data.upper()
    num_workers = 8  # 데이터 로더의 워커 수

    version = 1
    #######################

    main()


