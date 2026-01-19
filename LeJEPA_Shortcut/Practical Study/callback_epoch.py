import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from tqdm import tqdm # 진행 상황 표시용 (선택)

class PeriodicLinearEvaluation(Callback):
    def __init__(self, num_classes, feature_dim, probe_epochs=3, learning_rate=1e-3):
        """
        매 에폭마다 Linear Probe를 초기화하고, probe_epochs 만큼 새로 학습시켜 평가하는 콜백
        """
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.probe_epochs = probe_epochs
        self.learning_rate = learning_rate

    def on_train_epoch_end(self, trainer, pl_module):
        device = pl_module.device
        
        # 1. Probe & Optimizer 초기화
        probe = nn.Linear(self.feature_dim, self.num_classes).to(device)
        optimizer = torch.optim.SGD(
            probe.parameters(), 
            lr=self.learning_rate, 
            momentum=0.9
        )

        train_loader = trainer.train_dataloader

        # 정확도 누적 변수
        final_acc = 0.0

        # 2. Probe 학습 루프 (예: 10 Epochs)
        for epoch in range(self.probe_epochs):
            
            # 마지막 에폭인지 확인 (이때만 정확도를 계산)
            is_last_epoch = (epoch == self.probe_epochs - 1)
            
            epoch_correct = 0
            epoch_total = 0

            for batch in train_loader:
                try:
                    ((x1, x2), x_clean), y = batch
                except:
                    continue

                x = x_clean.to(device)
                y = y.to(device)

                # Backbone Forward (No Grad)
                with torch.no_grad():
                    features = pl_module(x)
                features = features.detach()

                # Probe Forward
                logits = probe(features)
                loss = F.cross_entropy(logits, y)

                # Probe Update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # [핵심] 마지막 에폭인 경우에만 정확도 누적 계산
                if is_last_epoch:
                    pred = logits.argmax(dim=1)
                    epoch_correct += (pred == y).float().sum()
                    epoch_total += y.size(0)

            # 마지막 에폭이 끝나면 정확도 확정
            if is_last_epoch:
                if epoch_total > 0:
                    final_acc = epoch_correct / epoch_total
                else:
                    final_acc = 0.0

        # 3. 로깅 (Train Accuracy)
        # 주의: train_loader로 측정했으므로 'train_acc'가 정확한 표현입니다.
        pl_module.log("periodic/train_acc", final_acc, on_step=False, on_epoch=True)
        
        print(f"Epoch {trainer.current_epoch} | Periodic Train Acc: {final_acc:.4f}")