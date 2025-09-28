import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import yaml
import re

from models.backbone import get_backbone
from dino import DINO
from datasets.get_dataset import DINODataModule
import glob
import os

def main():
    global version, using_data, num_workers

    torch.set_float32_matmul_precision('medium')

    # 🔧 설정 불러오기
    cfg = yaml.safe_load(open(f"config/{using_data}.yaml", "r"))

    # 🧠 백본 모델 로딩
    student_backbone, feature_dim = get_backbone(cfg['model']['student_backbone'], cfg['dataset']['input_size'])
    teacher_backbone, _ = get_backbone(cfg['model']['teacher_backbone'], cfg['dataset']['input_size'])

    # 🧪 DINO 모델 생성
    model = DINO(student_backbone, teacher_backbone, feature_dim, cfg, version)

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
        name=f"DINO_{using_data}",
        version=f"v{version}",
        default_hp_metric=False
    )

    continue_training = False
    folder_path = f"tb_logs/DINO_{using_data}/v{version}"
    
    while os.path.exists(folder_path):

        ckpt_list = glob.glob(f"{folder_path}/checkpoints/*.ckpt")
        ckpt_list = sorted(ckpt_list, key=os.path.getmtime, reverse=True)  # 가장 최근 파일이 앞으로 오도록 정렬
        resume_ckpt = ckpt_list[0] if ckpt_list else None

        if resume_ckpt is None:
            print(f"No checkpoints found in {folder_path}. Starting a new run.")
            break

        epoch_num = -1
        match = re.search(r"epoch=(\d+)", os.path.basename(resume_ckpt))
        if match:
            epoch_num = int(match.group(1))

        print(f"Previous run (Epoch: {epoch_num}) exists. Continue from previous checkpoint?")
        user_input = input("Enter 'y' to continue or 'n' to start a new run: ")
        if user_input.lower() == 'y':
            continue_training = True
            break
        elif user_input.lower() == 'n':
            version += 1
        elif user_input.lower() == 'q':
            print("Exiting the program.")
            return
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
        folder_path = f"tb_logs/DINO_{using_data}/v{version}"

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
    # 🚀 Trainer 실행
    if continue_training:
        ckpt_list = glob.glob(f"{folder_path}/checkpoints/*.ckpt")
        ckpt_list = sorted(ckpt_list, key=os.path.getmtime, reverse=True)  # 가장 최근 파일이 앞으로 오도록 정렬
        resume_ckpt = ckpt_list[0] if ckpt_list else None
        trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt)

    else:
        trainer.fit(model, datamodule=datamodule)


    
if __name__ == "__main__":
    global version, using_data, num_workers

    ### HYPERPARAMETERS ###
    using_data = "CIFAR10"  # 사용할 데이터셋 이름 (예: CIFAR10, CIFAR100 등)
    using_data = using_data.upper()
    num_workers = 8  # 데이터 로더의 워커 수

    version = 5
    #######################

    main()


