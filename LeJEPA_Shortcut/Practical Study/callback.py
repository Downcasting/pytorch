import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback

class OnlineLinearEvaluation(Callback):
    def __init__(self, num_classes, feature_dim, learning_rate=1e-3, **kwargs):
        """
        SSL 학습과 동시에 Linear Probe를 학습시키는 Callback
        Args:
            num_classes: 분류할 클래스 개수 (MNIST=10)
            feature_dim: Backbone의 출력 차원 (ResNet18=512)
            learning_rate: Linear Probe의 학습률
            **kwargs: probe_epochs 등 호환성을 위해 남겨둠 (사용 안 함)
        """
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.learning_rate = learning_rate
        
        self.probe_epochs = 10
        self.optimizer = None

    def on_fit_start(self, trainer, pl_module):
        # 학습 시작 시 Probe 초기화 및 GPU 이동
        self.probe = nn.Linear(self.feature_dim, self.num_classes).to(pl_module.device)
        self.optimizer = torch.optim.SGD(
            self.probe.parameters(), 
            lr=self.learning_rate, 
            momentum=0.9
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 1. 데이터 가져오기 (3-Way 구조: ((x1, x2), x_clean), y)
        ((x1, x2), x_clean), y = batch
        
        # GPU 이동
        x = x_clean.to(pl_module.device)
        y = y.to(pl_module.device)

        # 2. Feature Extraction (Backbone Gradient 차단)
        with torch.no_grad():
            features = pl_module(x)
        features = features.detach() # Gradient Flow 차단

        # 3. Probe Forward
        logits = self.probe(features)
        
        # --- [추가됨] Loss 및 Accuracy 계산 ---
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean() # 정확도 계산 로직 추가
        # -----------------------------------

        # 4. Probe Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 5. Logging (Train Loss와 Train Acc 모두 기록)
        # on_step=True로 설정하여 매 스텝마다 그래프를 그립니다.
        pl_module.log("online/train_loss", loss, on_step=True, on_epoch=False)
        pl_module.log("online/train_acc", acc, on_step=True, on_epoch=False)

    def on_train_epoch_end(self, trainer, pl_module):
        """
        [검증 단계] - 수동 구현
        Validation Loop이 없으므로, 여기서 직접 Validation Set을 가져와서 평가합니다.
        """
        # 1. Validation Dataloader 가져오기
        # 우선순위: 1. __init__에서 받은 것 -> 2. Trainer에 있는 것 -> 3. DataModule에 있는 것
        val_loader = self.val_dataloader
        if val_loader is None:
            # Trainer에 val_dataloaders가 세팅되어 있는지 확인
            if trainer.val_dataloaders:
                val_loader = trainer.val_dataloaders
            # 혹은 DataModule에서 가져오기
            elif hasattr(trainer, 'datamodule') and trainer.datamodule is not None:
                val_loader = trainer.datamodule.val_dataloader()
        
        if val_loader is None:
            print("[OnlineEval] Warning: Validation Dataloader not found. Skipping evaluation.")
            return

        # 리스트로 감싸져 있는 경우 처리 (Lightning 특성상 가끔 list로 옴)
        if isinstance(val_loader, list):
            val_loader = val_loader[0]

        # 2. 평가 시작 (학습 X, 평가만 O)
        total_correct = 0
        total_count = 0
        device = pl_module.device
        
        # 모델을 평가 모드로 전환할 필요는 없지만(Backbone은 어차피 eval 모드일 것이고 Probe는 Linear라 상관없음),
        # 명시적으로 gradient 계산을 끕니다.
        with torch.no_grad():
            for batch in val_loader:
                try:
                    x, y = batch
                except:
                    continue # 데이터 구조 안 맞으면 패스
                
                x = x.to(device)
                y = y.to(device)

                # Backbone & Probe Forward
                features = pl_module(x)
                logits = self.probe(features)
                
                pred = logits.argmax(dim=1)
                total_correct += (pred == y).float().sum()
                total_count += y.size(0)

        # 3. 결과 계산 및 로깅
        if total_count > 0:
            final_acc = total_correct / total_count
            pl_module.log("online/val_acc", final_acc, on_step=False, on_epoch=True, prog_bar=True)
            print(f"\n[Online Eval] Epoch {trainer.current_epoch} Val Acc: {final_acc:.4f}")
        else:
            print("\n[Online Eval] Warning: No validation data processed.")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # 1. 데이터 가져오기 (Validation 데이터셋은 (x, y) 구조)
        x, y = batch
        x = x.to(pl_module.device)
        y = y.to(pl_module.device)

        # 2. Forward
        with torch.no_grad():
            features = pl_module(x)
            logits = self.probe(features)
            loss = F.cross_entropy(logits, y)
            acc = (logits.argmax(dim=1) == y).float().mean()

        # 3. Logging (매 Epoch마다 정확도 기록)
        pl_module.log("online/val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        pl_module.log("online/val_loss", loss, on_step=False, on_epoch=True)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # 체크포인트 저장 시 Probe 상태도 같이 저장
        checkpoint['online_eval_probe'] = self.probe.state_dict()
        checkpoint['online_eval_optimizer'] = self.optimizer.state_dict()

    def on_load_checkpoint(self, trainer, pl_module, callback_state):
        # 체크포인트 로드 시 Probe 상태 복구
        if 'online_eval_probe' in callback_state:
            self.probe.load_state_dict(callback_state['online_eval_probe'])
            self.optimizer.load_state_dict(callback_state['online_eval_optimizer'])