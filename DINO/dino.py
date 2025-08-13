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
    def __init__(self, student_backbone, teacher_backbone, feature_dim, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=["student_backbone", "teacher_backbone"])
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        self.temperature = cfg['training']['temperature']
        self.momentum = 0.996
        self.center_momentum = cfg['training'].get('center_momentum', 0.9)
        self.center = torch.zeros(1, 65536).to(self._device)  # teacher 출력 평균값 추적용

    def forward(self, x):
        return self.student(x)


    @torch.no_grad()
    def _update_teacher(self):
        for (name_s, param_s), (name_t, param_t) in zip(self.student.named_parameters(), self.teacher.named_parameters()):
            param_t.data = param_t.data * self.momentum + param_s.data * (1. - self.momentum)

    @torch.no_grad()
    def _update_center(self, teacher_output):
        """teacher 출력의 평균(center) 업데이트"""
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def dino_loss(self, student_out, teacher_out):
        # student_out: [2B, D], teacher_out: [B, D]
        student_out = student_out.chunk(6)
        student_outs = [F.log_softmax(out / self.temperature, dim=-1) for out in student_out]

        # Center 적용
        teacher_out = (teacher_out - self.center)
        teacher_out = F.softmax(teacher_out / self.temperature, dim=-1)
        teacher_out = teacher_out.detach()

        loss = 0
        for s_out in student_outs:
            loss += F.kl_div(s_out, teacher_out, reduction='batchmean')
        loss /= len(student_outs)
        return loss

    def training_step(self, batch, batch_idx):
        global_view, local_view = batch  # 두 개의 augmented view
        
        student_out1 = self.student(global_view)
        student_out2 = self.student(local_view)

        student_out = torch.cat([student_out1, student_out2], dim=0)

        with torch.no_grad():
            teacher_out = self.teacher(global_view)

        loss = self.dino_loss(student_out, teacher_out)
        self._update_teacher()
        self._update_center(teacher_out)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        
        optimizer_type = self.hparams.cfg['optimization']['optimizer']
        learning_rate = self.hparams.cfg['training']['learning_rate']

        if optimizer_type == "AdamW":
            optimizer = AdamW(self.student.parameters(), lr=learning_rate)
        elif optimizer_type == "SGD":
            optimizer = SGD(self.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        elif optimizer_type == "LARS":
            optimizer = torch_optimizer.LARS(self.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-6, trust_coefficient=0.001, eps=1e-8)
        else:
            print(f"Optimizer {optimizer_type} is not supported. Using AdamW as default.")
            optimizer = AdamW(self.parameters(), lr=learning_rate)

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