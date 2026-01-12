#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Masked Autoencoder (MAE) patch-size sweep (single file)
- Backbone: ViT-Tiny (timm)
- Image size: 64x64
- Patch sizes: 4, 8, 16
- Pretrain epochs: (4->150, 8->300, 16->600)
- Mask ratio: 0.5 (fixed)
- Linear probing: 50 epochs (fixed)
- Dataset: CIFAR-10 (train split for pretrain, train for probe with labels, test for eval)
- Outputs: checkpoints and a bar chart comparing accuracy per patch size
"""

import os
import math
import json
import random
import argparse
from pathlib import Path
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import timm
from timm.models.vision_transformer import VisionTransformer
import matplotlib.pyplot as plt

# ------------------ Utils ------------------

def seed_all(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def exists(x):
    return x is not None

def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# ------------------ Data ------------------

def build_cifar10_loaders(data_root: str, img_size: int, batch_size: int, workers: int = 8):
    normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                     std=(0.2023, 0.1994, 0.2010))
    train_transform_ssl = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize,
    ])
    # For linear probe, keep augmentations weak and consistent
    train_transform_sup = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    pretrain_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform_ssl)
    probe_train_set = datasets.CIFAR10(root=data_root, train=True, download=False, transform=train_transform_sup)
    test_set = datasets.CIFAR10(root=data_root, train=False, download=False, transform=test_transform)

    pretrain_loader = DataLoader(pretrain_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
    probe_train_loader = DataLoader(probe_train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    return pretrain_loader, probe_train_loader, test_loader

# ------------------ MAE Components ------------------

class PatchMasker:
    """
    Randomly masks a fixed ratio of patch tokens.
    """
    def __init__(self, mask_ratio: float):
        assert 0.0 < mask_ratio < 1.0
        self.mask_ratio = mask_ratio

    def __call__(self, x_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x_tokens: (B, N, C)
        returns:
            visible_tokens: (B, N_vis, C)
            mask: (B, N) with 1 for masked, 0 for visible
            ids_restore: indices to restore original order (B, N)
        """
        B, N, C = x_tokens.shape
        device = x_tokens.device
        len_keep = int(N * (1.0 - self.mask_ratio))

        noise = torch.rand(B, N, device=device)
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)                 # ascend: small is keep
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        visible_tokens = torch.gather(x_tokens, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))

        mask = torch.ones([B, N], device=device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask in the original order
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return visible_tokens, mask, ids_restore

class TinyDecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = h + x
        h = x
        x = self.norm2(x)
        x = h + self.mlp(x)
        return x

class SimpleMAEDecoder(nn.Module):
    """
    Lightweight decoder that reconstructs pixel patches from encoded tokens.
    """
    def __init__(self, encoder_embed_dim: int, decoder_dim: int, num_patches: int, patch_size: int, depth: int = 2, num_heads: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Linear(encoder_embed_dim, decoder_dim, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))
        self.blocks = nn.ModuleList([TinyDecoderBlock(decoder_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(decoder_dim)
        # predict pixels per patch (3 * P * P)
        self.head = nn.Linear(decoder_dim, 3 * patch_size * patch_size, bias=True)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, tokens_full: torch.Tensor) -> torch.Tensor:
        """
        tokens_full: (B, N, C_enc) includes visible tokens placed back and mask tokens
        returns:
            patch_preds: (B, N, 3*P*P)
        """
        x = self.proj(tokens_full) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        patch_preds = self.head(x)
        return patch_preds

# ------------------ MAE Model Wrapper ------------------

class MAEViT(nn.Module):
    def __init__(self, img_size: int, patch_size: int, mask_ratio: float = 0.5, encoder_name: str = "vit_tiny_patch16_224", embed_dim: int = 192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.masker = PatchMasker(mask_ratio)

        # Build ViT-Tiny with specified patch size & image size using timm factory
        # We'll override patch_size via model cfg
        model_kwargs = dict(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.encoder: VisionTransformer = timm.create_model(
            'vit_tiny_patch16_224', pretrained=False, num_classes=0, **model_kwargs
        )
        # Ensure CLS token exists
        assert exists(self.encoder.cls_token)

        # Number of patches (no cls) = (H/P)*(W/P)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

        # A learnable mask token to fill masked positions for the decoder input
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Simple decoder
        self.decoder = SimpleMAEDecoder(encoder_embed_dim=embed_dim, decoder_dim=embed_dim, num_patches=self.num_patches, patch_size=patch_size, depth=2, num_heads=4)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: (B, 3, H, W)
        return: patches: (B, N, 3*P*P)
        """
        B, C, H, W = imgs.shape
        P = self.patch_size
        assert H == W == self.img_size
        assert H % P == 0 and W % P == 0
        h = H // P
        w = W // P
        x = imgs.reshape(B, C, h, P, w, P).permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, P * P * C)
        return x

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches: (B, N, 3*P*P) -> (B, 3, H, W)
        """
        B, N, _ = patches.shape
        P = self.patch_size
        C = 3
        H = W = self.img_size
        h = H // P
        w = W // P
        assert N == h * w
        x = patches.reshape(B, h, w, P, P, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
        return x

    def forward_encoder_tokens(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        returns: visible_tokens, mask, ids_restore, all_tokens_before_mask (B, N+1, C) for reference
        """
        # timm ViT expects (B, 3, H, W); it will patchify internally and prepend CLS
        x = self.encoder.patch_embed(imgs)  # (B, N, C)
        B, N, C = x.shape
        cls_tokens = self.encoder.cls_token.expand(B, -1, -1)  # (B, 1, C)
        x = x + self.encoder.pos_embed[:, 1:(N+1), :]
        x = torch.cat((cls_tokens + self.encoder.pos_embed[:, :1, :], x), dim=1)  # (B, 1+N, C)

        # Drop CLS for masking (MAE masks only patch tokens)
        patch_tokens = x[:, 1:, :]  # (B, N, C)

        visible_tokens, mask, ids_restore = self.masker(patch_tokens)
        # add back CLS before passing through blocks
        x_vis = torch.cat([x[:, :1, :], visible_tokens], dim=1)

        # Pass through encoder blocks & norm
        for blk in self.encoder.blocks:
            x_vis = blk(x_vis)
        x_vis = self.encoder.norm(x_vis)

        # split back
        cls_rep = x_vis[:, :1, :]
        enc_visible = x_vis[:, 1:, :]

        return enc_visible, mask, ids_restore, cls_rep

    def forward_decoder(self, enc_visible: torch.Tensor, mask: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """
        Compose full-length token sequence for decoder by inserting mask tokens.
        """
        B, N_vis, C = enc_visible.shape
        N = mask.shape[1]
        # Prepare full token sequence (without CLS) for decoder: place visible tokens, fill masked with mask_token (learned)
        mask_tokens = self.mask_token.repeat(B, N - N_vis, 1)
        x_ = torch.cat([enc_visible, mask_tokens], dim=1)  # (B, N, C) in shuffled order
        # Unshuffle to original patch order
        x_full = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, C))
        # Decode to pixel patches
        patch_preds = self.decoder(x_full)  # (B, N, 3*P*P)
        return patch_preds

    def forward(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MAE pretraining forward. Returns reconstruction loss and predictions.
        """
        target_patches = self.patchify(imgs)  # (B, N, 3*P*P)
        enc_visible, mask, ids_restore, _ = self.forward_encoder_tokens(imgs)
        patch_preds = self.forward_decoder(enc_visible, mask, ids_restore)
        # Compute loss only on masked patches
        loss = (patch_preds - target_patches) ** 2
        loss = loss.mean(dim=-1)  # per-patch MSE
        loss = (loss * mask).sum() / mask.sum()  # average over masked positions
        return loss, patch_preds

    @torch.no_grad()
    def encode_cls(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Get CLS representation from the encoder (for linear probing features).
        """
        x = self.encoder.patch_embed(imgs)  # (B, N, C)
        B, N, C = x.shape
        x = x + self.encoder.pos_embed[:, 1:(N+1), :]
        cls_tokens = self.encoder.cls_token.expand(B, -1, -1) + self.encoder.pos_embed[:, :1, :]
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)
        return x[:, 0]  # CLS

# ------------------ Training Loops ------------------

def pretrain_one_model(args, patch_size: int, epochs: int, loaders, out_dir: Path) -> Path:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pretrain_loader, _, _ = loaders

    model = MAEViT(img_size=args.img_size, patch_size=patch_size, mask_ratio=args.mask_ratio, embed_dim=192).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.blr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    global_step = 0
    for epoch in range(1, epochs + 1):
        running = 0.0
        for it, (imgs, _) in enumerate(pretrain_loader):
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                loss, _ = model(imgs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            global_step += 1

        avg_loss = running / len(pretrain_loader)
        if epoch % args.log_interval == 0 or epoch == 1 or epoch == epochs:
            print(f"[Pretrain][P={patch_size}] Epoch {epoch}/{epochs} - recon_loss={avg_loss:.4f}")

        # optional: save intermediate
        if epoch % args.ckpt_interval == 0 or epoch == epochs:
            ckpt_path = out_dir / f"pretrain_p{patch_size}_ep{epoch}.pt"
            torch.save({"epoch": epoch, "model": model.state_dict()}, ckpt_path)

    # Save final
    final_ckpt = out_dir / f"pretrain_p{patch_size}_final.pt"
    torch.save({"epoch": epochs, "model": model.state_dict()}, final_ckpt)
    return final_ckpt

def linear_probe(args, ckpt_path: Path, patch_size: int, loaders, out_dir: Path) -> Dict[str, float]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    _, probe_train_loader, test_loader = loaders

    # Rebuild model with same patch size
    model = MAEViT(img_size=args.img_size, patch_size=patch_size, mask_ratio=args.mask_ratio, embed_dim=192).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Linear head on CLS
    num_classes = 10
    head = nn.Linear(192, num_classes).to(device)
    optimizer = torch.optim.SGD(head.parameters(), lr=args.probe_lr, momentum=0.9, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.probe_epochs)

    best_acc = 0.0
    for epoch in range(1, args.probe_epochs + 1):
        head.train()
        for imgs, y in probe_train_loader:
            imgs = imgs.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.no_grad():
                feats = model.encode_cls(imgs)  # (B, 192)
            logits = head(feats)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # eval
        head.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, y in test_loader:
                imgs = imgs.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                feats = model.encode_cls(imgs)
                logits = head(feats)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total * 100.0
        if epoch % args.log_interval == 0 or epoch == 1 or epoch == args.probe_epochs:
            print(f"[Probe][P={patch_size}] Epoch {epoch}/{args.probe_epochs} - acc={acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save({"epoch": epoch, "head": head.state_dict()}, out_dir / f"probe_head_p{patch_size}_best.pt")

    return {"best_acc": best_acc}

# ------------------ Orchestration ------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='.././data')
    parser.add_argument('--out_dir', type=str, default='./runs_mae_patch')
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--blr', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--amp', action='store_true', help='use AMP mixed precision')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--ckpt_interval', type=int, default=1000)  # large by default (save only final)
    parser.add_argument('--probe_epochs', type=int, default=50)
    parser.add_argument('--probe_lr', type=float, default=0.1)
    parser.add_argument('--no_pretrain', action='store_true', help='skip pretraining and only run probing if ckpts exist')
    args = parser.parse_args()

    seed_all(42)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Build loaders once
    loaders = build_cifar10_loaders(args.data_root, args.img_size, args.batch_size, args.workers)

    # Patch-size plan
    # plan = [(4, 150), (8, 300), (16, 600)]
    plan = [(8, 300), (16, 600)]

    results = {}
    ckpts = {}

    for psize, epochs in plan:
        exp_dir = out_root / f"patch_{psize}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_pretrain:
            ckpt = pretrain_one_model(args, psize, epochs, loaders, exp_dir)
        else:
            # try to find a final ckpt
            ckpt = exp_dir / f"pretrain_p{psize}_final.pt"
            if not ckpt.exists():
                raise FileNotFoundError(f"Checkpoint missing for patch {psize}: {ckpt}")
        ckpts[psize] = ckpt
        probe_dir = exp_dir / "probe"
        probe_dir.mkdir(exist_ok=True)
        res = linear_probe(args, ckpt, psize, loaders, probe_dir)
        results[str(psize)] = res['best_acc']

    # Save results as JSON
    save_json(results, out_root / "results.json")

    # Plot bar chart
    patch_sizes = [str(p) for p, _ in plan]
    accs = [results[str(p)] for p, _ in plan]

    plt.figure(figsize=(6,4))
    plt.bar(patch_sizes, accs)
    plt.xlabel("Patch size")
    plt.ylabel("Linear probe Top-1 (%)")
    plt.title("MAE (mask=0.5, ViT-Tiny, 64x64): Accuracy vs Patch size")
    for i, v in enumerate(accs):
        plt.text(i, v, f"{v:.1f}%", ha='center', va='bottom')
    fig_path = out_root / "accuracy_by_patch_size.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    print(f"Saved summary: {out_root/'results.json'} and chart: {fig_path}")

if __name__ == "__main__":
    main()
