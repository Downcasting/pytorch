import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.optim import SGD, AdamW
import torch_optimizer


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=65536, use_bn=False):
        super().__init__()
        hidden_dim = 2048
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        return self.mlp(x)

class DINO(pl.LightningModule):
    def __init__(self, student_backbone, teacher_backbone, feature_dim, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=["student_backbone", "teacher_backbone"])
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 학생/교사 모델
        self.student = nn.Sequential(
            student_backbone,
            DINOHead(feature_dim, use_bn=True)
        )

        self.teacher = nn.Sequential(
            teacher_backbone,
            DINOHead(feature_dim, use_bn=True)
        )
        # EMA teacher는 학습하지 않음
        for p in self.teacher.parameters():
            p.requires_grad = False

        # 하이퍼파라미터
        self.temperature = cfg['training']['temperature']
        self.momentum = 0.996  # 보통 고정 or cosine schedule 가능
        self.center = torch.zeros(1, 65536).to(self._device)

    def forward(self, x):
        return self.student(x)

    @torch.no_grad()
    def _update_teacher(self):
        for s, t in zip(self.student.parameters(), self.teacher.parameters()):
            t.data = t.data * self.momentum + s.data * (1. - self.momentum)

    def dino_loss(self, student_out, teacher_out):
        # student_out: [2B, D], teacher_out: [B, D]
        student_out = student_out.chunk(2)
        student_out1, student_out2 = student_out

        student_out1 = F.log_softmax(student_out1 / self.temperature, dim=-1)
        student_out2 = F.log_softmax(student_out2 / self.temperature, dim=-1)

        teacher_out = (teacher_out - self.center)
        teacher_out = F.softmax(teacher_out / self.temperature, dim=-1)
        teacher_out = teacher_out.detach()

        loss = 0.5 * (F.kl_div(student_out1, teacher_out, reduction='batchmean') +
                      F.kl_div(student_out2, teacher_out, reduction='batchmean'))
        return loss

    def training_step(self, batch, batch_idx):
        (view1, view2), _ = batch  # 두 개의 augmented view
        student_input = torch.cat([view1, view2], dim=0)
        student_out = self.student(student_input)

        with torch.no_grad():
            teacher_out = self.teacher(view1)

        loss = self.dino_loss(student_out, teacher_out)
        self._update_teacher()

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