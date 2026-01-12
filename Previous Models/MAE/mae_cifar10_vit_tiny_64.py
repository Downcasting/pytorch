#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# MAE pretraining + finetuning on CIFAR-10 with ViT-Tiny backbone (64x64).
# - Pretrain for 100 epochs using MAE with mask ratios in {0.25, 0.50, 0.75}
# - Then finetune for 50 epochs with a mean-pooling classifier head
# - Image size is 64x64; patch size is 8 (=> 8x8 = 64 tokens)
# - Logs final test accuracy for each mask ratio and plots a comparison bar chart.
# This file is standalone (no external repo code) and loosely follows MAE (He et al., 2022).
#
# Usage example:
#   python mae_cifar10_vit_tiny_64.py --device cuda --epochs_pre 100 --epochs_ft 50 --batch_size 128
# For a quick sanity check:
#   python mae_cifar10_vit_tiny_64.py --device cuda --epochs_pre 2 --epochs_ft 2 --mask_ratios 0.5

import math
import argparse
import os
import random
from dataclasses import dataclass
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False):
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing='ij')
    grid = torch.stack(grid, dim=0)  # 2, Gw, Gh
    grid = grid.reshape(2, 1, grid_size, grid_size)
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        cls_embed = torch.zeros([1, embed_dim], dtype=torch.float32)
        pos_embed = torch.cat([cls_embed, pos_embed], dim=0)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.
    omega = 1. / (10000 ** omega)
    pos = pos.reshape(-1)
    out = torch.einsum('m,d->md', pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0].reshape(-1))
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1].reshape(-1))
    emb = torch.cat([emb_h, emb_w], dim=1)
    return emb

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class MAEViT(nn.Module):
    def __init__(
        self,
        img_size=64,
        patch_size=8,
        in_chans=3,
        embed_dim=192,
        depth=12,
        num_heads=3,
        decoder_embed_dim=512,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4.0,
        mask_token_std=0.02
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.img_size = img_size

        enc_pos = get_2d_sincos_pos_embed(embed_dim, int(math.sqrt(num_patches)))
        self.register_buffer("pos_embed_enc", enc_pos.unsqueeze(0), persistent=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=True) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        dec_pos = get_2d_sincos_pos_embed(decoder_embed_dim, int(math.sqrt(num_patches)))
        self.register_buffer("pos_embed_dec", dec_pos.unsqueeze(0), persistent=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        nn.init.normal_(self.mask_token, std=mask_token_std)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True) for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * patch_size * in_chans, bias=True)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        B, C, H, W = imgs.shape
        assert H == W == self.img_size and H % p == 0 and W % p == 0
        h = w = H // p
        x = imgs.reshape(B, C, h, p, w, p).permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, p * p * C)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        B, L, D = x.shape
        h = w = int(math.sqrt(L))
        assert h * w == L
        x = x.reshape(B, h, w, p, p, self.in_chans).permute(0, 5, 1, 3, 2, 4).reshape(B, self.in_chans, h * p, w * p)
        return x

    def random_masking(self, x: torch.Tensor, mask_ratio: float):
        B, N, C = x.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))

        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed_enc.to(x.device)

        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)

        B, N_keep, C = x.shape
        N = self.num_patches
        mask_tokens = self.mask_token.repeat(B, N - N_keep, 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, C))

        x_ = x_ + self.pos_embed_dec.to(x_.device)

        for blk in self.decoder_blocks:
            x_ = blk(x_)
        x_ = self.decoder_norm(x_)
        x_ = self.decoder_pred(x_)
        return x_

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

class ViTClassifier(nn.Module):
    def __init__(self, mae: MAEViT, num_classes: int = 10):
        super().__init__()
        self.patch_embed = mae.patch_embed
        self.pos_embed_enc = mae.pos_embed_enc
        self.blocks = mae.blocks
        self.norm = mae.norm
        embed_dim = mae.patch_embed.proj.out_channels
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed_enc.to(x.device)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)

@dataclass
class TrainConfig:
    img_size: int = 64
    patch_size: int = 8
    batch_size: int = 128
    num_workers: int = 4
    epochs_pre: int = 100
    epochs_ft: int = 50
    lr_pre: float = 1.5e-3
    lr_ft: float = 5e-4
    wd_pre: float = 0.05
    wd_ft: float = 0.05
    warmup_epochs_pre: int = 5
    warmup_epochs_ft: int = 3
    device: str = "cuda"
    amp: bool = False
    seed: int = 42

def get_dataloaders(cfg: TrainConfig):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    ft_tf = transforms.Compose([
        transforms.Resize(cfg.img_size, antialias=True),
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = datasets.CIFAR10(root=".././data", train=True, download=True, transform=train_tf)
    ft_train_ds = datasets.CIFAR10(root=".././data", train=True, download=False, transform=ft_tf)
    test_ds = datasets.CIFAR10(root=".././data", train=False, download=True, transform=ft_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    ft_train_loader = DataLoader(ft_train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, ft_train_loader, test_loader

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0.0):
    warmup_schedule = []
    if warmup_epochs > 0:
        warmup_iters = warmup_epochs * niter_per_ep
        for it in range(warmup_iters):
            warmup_schedule.append(start_warmup_value + (base_value - start_warmup_value) * it / max(1, warmup_iters - 1))

    iters = epochs * niter_per_ep
    schedule = []
    for it in range(iters - len(warmup_schedule)):
        t = it / (iters - len(warmup_schedule))
        schedule.append(final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * t)))
    schedule = warmup_schedule + schedule
    assert len(schedule) == iters
    return schedule

def train_mae_epoch(model, dataloader, optimizer, scaler, lr_schedule, wd, epoch, device, mask_ratio):
    model.train()
    total_loss = 0.0
    n = 0
    for it, (imgs, _) in enumerate(dataloader):
        step = epoch * len(dataloader) + it
        for pg in optimizer.param_groups:
            pg["lr"] = lr_schedule[step]
            pg["weight_decay"] = wd

        imgs = imgs.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss, _, _ = model(imgs, mask_ratio=mask_ratio)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, _, _ = model(imgs, mask_ratio=mask_ratio)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total_loss / n

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    for imgs, targets in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(imgs)
        pred = logits.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)
    return correct / total

def train_classifier(model, train_loader, test_loader, cfg: TrainConfig):
    device = cfg.device
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_ft, weight_decay=cfg.wd_ft, betas=(0.9, 0.999))
    scaler = torch.cuda.amp.GradScaler() if (cfg.amp and device.startswith("cuda")) else None
    niter = len(train_loader)
    lr_schedule = cosine_scheduler(cfg.lr_ft, cfg.lr_ft * 0.01, cfg.epochs_ft, niter, warmup_epochs=cfg.warmup_epochs_ft, start_warmup_value=cfg.lr_ft * 0.1)

    ce = nn.CrossEntropyLoss()
    best_acc = 0.0
    for epoch in range(cfg.epochs_ft):
        model.train()
        for it, (imgs, targets) in enumerate(train_loader):
            step = epoch * niter + it
            lr = lr_schedule[step]
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(imgs)
                    loss = ce(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs)
                loss = ce(logits, targets)
                loss.backward()
                optimizer.step()

        acc = evaluate(model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
        print(f"[Finetune] Epoch {epoch+1}/{cfg.epochs_ft} - test acc: {acc*100:.2f}% (best {best_acc*100:.2f}%)")
    return best_acc

def save_ckpt(model: nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def run_experiment(mask_ratios: List[float], cfg: TrainConfig, workdir: str = "./outputs"):
    set_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    cfg.device = device
    print(f"Using device: {device}")
    print(f"Config: {cfg}")

    train_loader, ft_train_loader, test_loader = get_dataloaders(cfg)

    final_accs: Dict[float, float] = {}
    for mr in mask_ratios:
        print(f"\n=== Pretraining with mask_ratio={mr} ===")
        mae = MAEViT(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            embed_dim=192, depth=12, num_heads=3,
            decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=8,
        ).to(device)
        optimizer = torch.optim.AdamW(mae.parameters(), lr=cfg.lr_pre, weight_decay=cfg.wd_pre, betas=(0.9, 0.95))
        scaler = torch.cuda.amp.GradScaler() if (cfg.amp and device.startswith("cuda")) else None

        niter = len(train_loader)
        lr_sched = cosine_scheduler(cfg.lr_pre, cfg.lr_pre * 0.1, cfg.epochs_pre, niter, warmup_epochs=cfg.warmup_epochs_pre, start_warmup_value=cfg.lr_pre * 0.01)

        for ep in range(cfg.epochs_pre):
            loss = train_mae_epoch(mae, train_loader, optimizer, scaler, lr_sched, cfg.wd_pre, ep, device, mask_ratio=mr)
            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"[Pretrain] Epoch {ep+1}/{cfg.epochs_pre} - loss: {loss:.4f}")

        enc_path = os.path.join(workdir, f"mae_encoder_mr{int(mr*100)}.pth")
        save_ckpt(mae, enc_path)

        print(f"=== Finetuning (mask_ratio={mr}) ===")
        clf = ViTClassifier(mae, num_classes=10).to(device)
        best_acc = train_classifier(clf, ft_train_loader, test_loader, cfg)
        final_accs[mr] = best_acc
        save_ckpt(clf, os.path.join(workdir, f"classifier_mr{int(mr*100)}_best.pth"))
        print(f"[Result] mask_ratio={mr}: best test acc = {best_acc*100:.2f}%")

    ratios = [str(r) for r in mask_ratios]
    accs = [final_accs[r] * 100.0 for r in mask_ratios]
    plt.figure(figsize=(6,4))
    plt.bar(ratios, accs)
    plt.xlabel("Mask Ratio")
    plt.ylabel("Best Test Accuracy (%)")
    plt.title("CIFAR-10 Finetune Accuracy vs. MAE Mask Ratio (ViT-Tiny, 64x64)")
    plt.tight_layout()
    os.makedirs(workdir, exist_ok=True)
    fig_path = os.path.join(workdir, "accuracy_vs_mask_ratio.png")
    plt.savefig(fig_path)
    print(f"Saved comparison plot to: {fig_path}")
    # print("Final accuracies:", {k: f\"{v*100:.2f}%\" for k,v in final_accs.items()})

def parse_args():
    p = argparse.ArgumentParser(description="MAE ViT-Tiny on CIFAR-10 (64x64) pretrain+finetune")
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--patch_size", type=int, default=8, help="8 => 8x8 patches for 64x64 image (64 tokens)")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs_pre", type=int, default=100)
    p.add_argument("--epochs_ft", type=int, default=50)
    p.add_argument("--lr_pre", type=float, default=1.5e-3)
    p.add_argument("--lr_ft", type=float, default=5e-4)
    p.add_argument("--wd_pre", type=float, default=0.05)
    p.add_argument("--wd_ft", type=float, default=0.05)
    p.add_argument("--warmup_epochs_pre", type=int, default=5)
    p.add_argument("--warmup_epochs_ft", type=int, default=3)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true", help="use mixed precision (recommended on CUDA)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workdir", type=str, default="./outputs")
    p.add_argument("--mask_ratios", type=float, nargs="+", default=[0.25, 0.5, 0.75], help="MAE mask ratios to try")
    args = p.parse_args()

    cfg = TrainConfig(
        img_size=args.img_size,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs_pre=args.epochs_pre,
        epochs_ft=args.epochs_ft,
        lr_pre=args.lr_pre,
        lr_ft=args.lr_ft,
        wd_pre=args.wd_pre,
        wd_ft=args.wd_ft,
        warmup_epochs_pre=args.warmup_epochs_pre,
        warmup_epochs_ft=args.warmup_epochs_ft,
        device=args.device,
        amp=args.amp,
        seed=args.seed,
    )
    return cfg, args.workdir, args.mask_ratios

if __name__ == "__main__":
    cfg, workdir, mask_ratios = parse_args()
    run_experiment(mask_ratios, cfg, workdir)
