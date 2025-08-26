import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import yaml
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.backbone import get_backbone
from datasets.get_dataset import DINOEvalDataModule
import glob
import os

class DINOEval(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = cfg['model']
        self.dataset = cfg['dataset']

        self.using_data = cfg['dataset']['name'].upper()

        self.encoder, feature_dim = self.load_encoder()
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

        # student 부분만 뽑기
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("student.0.vit"):  # student backbone만
                new_k = k.replace("student.0.", "")  # "vit.xxx" 형태로 바꿔줌
                new_state_dict[new_k] = v

        encoder, num_features = get_backbone(self.model['student_backbone'], self.dataset['input_size'])
        missing, unexpected = encoder.load_state_dict(new_state_dict, strict=False)

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

        return encoder, num_features
    
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
            "scheduler" : optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3),
            "monitor" : "val_loss",
            "interval" : "epoch",
            "frequency" : 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

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

    model = DINOEval(cfg=cfg)
    trainer.fit(model, datamodule=datamodule)


    
if __name__ == "__main__":
    global using_data, num_workers

    ### HYPERPARAMETERS ###
    using_data = "CIFAR10"  # 사용할 데이터셋 이름 (예: CIFAR10, CIFAR100 등)
    using_data = using_data.upper()
    num_workers = 4  # 데이터 로더의 워커 수

    version = 1
    #######################

    main()


