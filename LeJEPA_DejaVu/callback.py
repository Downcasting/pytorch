import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback

from LeJEPA_DejaVu.dejavu import search_kNN
import numpy as np

class OnlineLinearEvaluation(Callback):
    def __init__(self, num_classes, feature_dim, every_n_epochs = 100, **kwargs):
        """
        SSL 학습과 동시에 Linear Probe를 진행하는 Callback
        Args:
            num_classes: 분류할 클래스 개수 (MNIST=10)
            feature_dim: Backbone의 출력 차원 (ResNet18=512)
            learning_rate: Linear Probe의 학습률
            **kwargs: probe_epochs 등 호환성을 위해 남겨둠 (사용 안 함)
        """
        super().__init__()
        self.num_classes = num_classes
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        
        current_epoch = trainer.current_epoch + 1
        if current_epoch % self.every_n_epochs != 0:
            return

        pl_module.eval()

        accuracy = 0
        representations = []
        labels = []
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(trainer.datamodule.val_dataloader()):
                data = data.to(pl_module.device)
                outputs = pl_module(data)
                representations.append(outputs.cpu())
                labels.append(label.cpu())


        for _, (data, labels) in enumerate(trainer.datamodule.train_dataloader()):
            data = data.to(pl_module.device)
            outputs_a = pl_module(data)

            answer_a = search_kNN(outputs_a.cpu(), representations, labels, k=100)
            accuracy += np.sum(answer_a == labels.numpy())
            
        accuracy = accuracy / len(trainer.datamodule.train_dataloader().dataset)

        # Logging (Epoch 단위 평균 기록)
        pl_module.log("online/val_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

