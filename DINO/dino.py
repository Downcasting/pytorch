import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.optim import SGD, AdamW
import torch_optimizer


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=65536, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        # MLP (3-layer)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 논문에 맞게 추가
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim)
        )
        
        # L2 Normalization + WeightNorm on last layer
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)  # weight norm scale 초기화
        self.last_layer.weight_g.requires_grad = False  # 고정
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)  # L2 normalization
        x = self.last_layer(x)
        return x

class DINO(pl.LightningModule):
    def __init__(self, student_backbone, teacher_backbone, feature_dim, cfg, version):
        super().__init__()
        self.save_hyperparameters(ignore=["student_backbone", "teacher_backbone"])
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.version = version

        # 학생 / 교사 모델
        self.student = nn.Sequential(
            student_backbone,
            DINOHead(feature_dim)  # 앞서 수정한 논문 버전 DINOHead 적용 가능
        )

        self.teacher = nn.Sequential(
            teacher_backbone,
            DINOHead(feature_dim)
        )

        for p in self.teacher.parameters():
            p.requires_grad = False  # teacher는 학습하지 않음

        # 하이퍼파라미터
        self.teacher_temperature = cfg['training']['teacher_temperature']
        self.student_temperature = cfg['training']['student_temperature']
        self.center_momentum = cfg['training'].get('center_momentum', 0.9)
        self.center = torch.zeros(1, 65536).to(self._device)  # teacher 출력 평균값 추적용
        self.dataset = cfg['dataset']['name'].upper()

        self.max_epochs = cfg['training']['max_epochs']

    def forward(self, x):
        return self.student(x)


    @torch.no_grad()
    def _update_teacher(self, current_step):
        m_base = 0.996
        m_final = 0.999
        # cosine schedule로 momentum 증가
        momentum = m_base + (m_final - m_base) * (current_step / self.max_epochs)
        for (name_s, param_s), (name_t, param_t) in zip(self.student.named_parameters(), self.teacher.named_parameters()):
            param_t.data = param_t.data * momentum + param_s.data * (1. - momentum)

    @torch.no_grad()
    def _update_center(self, teacher_output):
        """teacher 출력의 평균(center) 업데이트"""
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def dino_loss(self, student_out, teacher_out):
        student_out = F.log_softmax(student_out / self.student_temperature, dim=-1)

        # Center 적용
        teacher_out = teacher_out - self.center
        teacher_out = F.softmax(teacher_out / self.teacher_temperature, dim=-1).detach()

        loss = F.kl_div(student_out, teacher_out, reduction='batchmean')
        return loss

    def training_step(self, batch, batch_idx):
        # batch: ([global_views...], [local_views...], labels)
        views, labels = batch
        # print(f"Views: {len(views)}, Labels: {len(labels)}")
        global_views = views[:2]
        local_views = views[2:]
        # print(f"Global Views: {len(global_views)}, Local Views: {len(local_views)}")

        # 보장: global_views, local_views는 Tensor 리스트임
        # global 먼저
        global_view_1, global_view_2 = global_views  # 예: 2개의 글로벌 뷰
        student_g1 = self.student(global_view_1)
        student_g2 = self.student(global_view_2)

        # print("Student's global views processed:", student_g1.shape, student_g2.shape)
        
        with torch.no_grad():
            teacher_g1 = self.teacher(global_view_1)
            teacher_g2 = self.teacher(global_view_2)

        # 필요 시 local views 처리 (옵션)
        student_locals = []
        for lv in local_views:
            student_locals.append(self.student(lv))

        # print("Student's local views processed:", [sl.shape for sl in student_locals])

        # DINO loss 계산
        # 예시: global ↔ global, global ↔ local
        loss = 0
        loss += self.dino_loss(student_g1, teacher_g2)
        loss += self.dino_loss(student_g2, teacher_g1)

        # local loss도 쓰고 싶으면 추가
        for sl in student_locals:
            loss += self.dino_loss(sl, teacher_g1)
            loss += self.dino_loss(sl, teacher_g2)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {
        'loss': loss,
        'teacher_g1': teacher_g1.detach(),
        'teacher_g2': teacher_g2.detach()
    }


    def configure_optimizers(self):
        
        optimizer_type = self.hparams.cfg['optimization']['optimizer']
        learning_rate = self.hparams.cfg['training']['learning_rate']

        if optimizer_type == "AdamW":
            optimizer = AdamW(self.student.parameters(), lr=learning_rate)
        elif optimizer_type == "SGD":
            optimizer = SGD(self.student.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        elif optimizer_type == "LARS":
            optimizer = torch_optimizer.LARS(self.student.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-6, trust_coefficient=0.001, eps=1e-8)
        else:
            print(f"Optimizer {optimizer_type} is not supported. Using AdamW as default.")
            optimizer = AdamW(self.student.parameters(), lr=learning_rate)

        warmup = self.hparams.cfg['optimization']['warmup']
        cosine = self.hparams.cfg['optimization']['cosine']

        warmup_epochs = self.hparams.cfg['training']['warmup_epochs']
        max_epochs = self.hparams.cfg['training']['max_epochs']

        # Use both warmup and cosine annealing
        if warmup and cosine:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs)
            ],
            milestones=[warmup_epochs]
            )
        elif warmup:
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        elif cosine:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        else:
            scheduler = None
        return [optimizer], [scheduler]

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        # ✅ teacher는 batch마다 업데이트
        self._update_teacher(current_step=self.global_step)

        # ✅ teacher 출력 평균만 epoch 동안 누적
        teacher_g1, teacher_g2 = outputs['teacher_g1'], outputs['teacher_g2']
        batch_center = torch.cat([teacher_g1, teacher_g2], dim=0).mean(dim=0, keepdim=True)

        if not hasattr(self, "center_sum"):
            self.center_sum = torch.zeros_like(batch_center)
            self.center_count = 0

        self.center_sum += batch_center.detach()
        self.center_count += 1

    def on_train_epoch_end(self):
        # ✅ epoch 끝날 때 한 번만 center 업데이트
        if hasattr(self, "center_sum") and self.center_count > 0:
            epoch_center = self.center_sum / self.center_count  # 평균만 사용

            if not hasattr(self, "center"):
                self.register_buffer('center', torch.zeros_like(epoch_center))
                self.center_momentum = 0.99  # 논문 기준 0.9~0.99

            # moving average update
            self.center = self.center * self.center_momentum + epoch_center.to(self.center.device) * (1 - self.center_momentum)

            # epoch 통계 초기화
            self.center_sum.zero_()
            self.center_count = 0

        # ✅ checkpoint 저장
        if (self.current_epoch + 1) % 10 == 0:
            self.trainer.save_checkpoint(f"{self.dataset}_v{self.version}_epoch={self.current_epoch + 1}.ckpt")
