import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import timm, wandb, hydra, tqdm
from omegaconf import DictConfig
from datasets import load_dataset
from datasets import load_from_disk
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.ops import MLP

# [추가됨]
import os 
import io
import matplotlib
matplotlib.use('Agg')  # 백엔드 설정 (GUI 필요 없음)
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from src.utils import OnlineCovariance, compute_effective_rank, plot_combined_spectrum, project_to_nullspace
from src.loss import SIGReg
from src.model_data import ViTEncoder, HFDataset

# 4. Training Loop
@hydra.main(version_base=None)
def main(cfg: DictConfig):

    # [추가됨]
    checkpoint_path = "checkpoint.pth"
    wandb_id = None
    start_epoch = 0

    # 체크포인트가 있으면 ID만 미리 읽어옴
    if os.path.exists(checkpoint_path):
        try:
            temp_ckpt = torch.load(checkpoint_path, map_location='cpu')
            wandb_id = temp_ckpt.get('wandb_run_id', None)
        except:
            pass

    wandb.init(project="LeJEPA-shortcut", config=dict(cfg), id=wandb_id, resume="allow" if wandb_id else None)
    torch.manual_seed(0)

    ### [추가됨] parameter 선택에 따른 pretrain dataset 선택
    pretrain_mode = cfg.get("clean_pretrain", False)

    if pretrain_mode:
        print("Pretrain with clean dataset")
        train_ds = HFDataset("train", V=cfg.V, mode='clean')
    else:
        print("Pretrain with colored dataset")
        train_ds = HFDataset("train", V=cfg.V)

    test_ds = HFDataset("validation", V=1)
    test_ds_clean = HFDataset("validation", V=1, mode='clean')
    # [추가됨 - num workers 8 -> 4, persistent_workers=True, pin_memory=True]
    train = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True, drop_last=True, num_workers=4, persistent_workers=True, pin_memory=True)
    test = DataLoader(test_ds, batch_size=256, num_workers=4, persistent_workers=True, pin_memory=True)
    test_clean = DataLoader(test_ds_clean, batch_size=256, num_workers=4, persistent_workers=True, pin_memory=True)

    # modules and loss
    net = ViTEncoder(proj_dim=cfg.proj_dim).to("cuda")
    probe = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 10)).to("cuda")
    sigreg = SIGReg().to("cuda")
    # Optimizer and scheduler
    g1 = {"params": net.parameters(), "lr": cfg.lr, "weight_decay": 5e-2}
    g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
    opt = torch.optim.AdamW([g1, g2])
    warmup_steps = len(train)
    total_steps = len(train) * cfg.epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-3)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])
    
    scaler = GradScaler(enabled="cuda" == "cuda")

    ### [추가됨] Lambda를 추가로 넣어 상위 eigenvalue 억제
    adaptive_lambda = cfg.get("adaptive_lambda", None)
    if adaptive_lambda:
        print("Using Adaptive Lambda Scheduling")
        top_k = cfg.top_k  # 상위 k개 벡터
    else:
        adaptive_lambda = None
        top_k = None

    adaptive_vecs_proj = None
    adaptive_vecs_emb = None

    # 1. [추가됨] 저장된 모델이 있으면 불러오기 (Resume)
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path} with wandb_id: {wandb_id}")
        checkpoint = torch.load(checkpoint_path)

        net.load_state_dict(checkpoint['net_state_dict'])
        probe.load_state_dict(checkpoint['probe_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        adaptive_vecs_proj = checkpoint.get('adaptive_vectors_proj', None)
        adaptive_vecs_emb = checkpoint.get('adaptive_vectors_emb', None)
        print(f"Resuming from Epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting from scratch.")

    # [추가됨] - basic eigenvectors
    basis_path = cfg.get("basis_path", None)
    fixed_basis_proj = None
    fixed_basis_emb = None

    fixed_basis_proj_clean = None
    fixed_basis_emb_clean = None

    if basis_path and os.path.exists(basis_path):
        print(f"Loading fixed basis from {basis_path}")
        basis_data = torch.load(basis_path, map_location="cuda")
        fixed_basis_proj = basis_data["colored_proj_eigvecs"][:, :16].to("cuda")
        fixed_basis_emb = basis_data["colored_emb_eigvecs"][:, :16].to("cuda")

        fixed_basis_proj_clean = basis_data["clean_proj_eigvecs"][:, :16].to("cuda")
        fixed_basis_emb_clean = basis_data["clean_emb_eigvecs"][:, :16].to("cuda")

        print(f"Loaded fixed basis from {basis_path}")
    else:
        print(f"Basis not found at {basis_path}")

    # [추가됨]
    print(f"Start Training: Epochs= {start_epoch} ~ {cfg.epochs}, BatchSize={cfg.bs}")

    # Training
    for epoch in range(start_epoch, cfg.epochs):
        net.train(), probe.train()
        for batch in tqdm.tqdm(train, total=len(train)):
            if len(batch) == 3:
                vs, y, _ = batch
            else:
                vs, y = batch
            with autocast("cuda", dtype=torch.bfloat16):
                vs = vs.to("cuda", non_blocking=True)
                y = y.to("cuda", non_blocking=True)
                emb, proj = net(vs)
                inv_loss = (proj.mean(0) - proj).square().mean()
                sigreg_loss = sigreg(proj)
                lejepa_loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)
                y_rep, yhat = y.repeat_interleave(cfg.V), probe(emb.detach())
                probe_loss = F.cross_entropy(yhat, y_rep)

                # [추가됨] Adaptive Lambda Scheduling
                adaptive_loss = 0.0
                if adaptive_lambda is not None and adaptive_lambda > 0.0 and adaptive_vecs_proj is not None:
                    adaptive_loss = sigreg(proj, target_vec=adaptive_vecs_proj[:, :top_k]) * adaptive_lambda


                # [추가됨] 아직은 사용 안 하는 걸로
                # if adaptive_vecs_emb is not None:
                #     adaptive_loss += sigreg(emb, target_vec=adaptive_vecs_emb[:, :top_k]) * adaptive_lambda

                loss = lejepa_loss + probe_loss + adaptive_loss

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            wandb.log(
                {
                    "train/probe": probe_loss.item(),
                    "train/lejepa": lejepa_loss.item(),
                    "train/sigreg": sigreg_loss.item(),
                    "train/adaptive": adaptive_loss.item() / adaptive_lambda if isinstance(adaptive_loss, torch.Tensor) else adaptive_loss,
                    "train/inv": inv_loss.item(),
                }
            )

        # 3. [추가됨] 한 epoch이 끝날 때마다 저장하기
        torch.save({
            'epoch': epoch,
            'net_state_dict': net.state_dict(),
            'probe_state_dict': probe.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'wandb_run_id': wandb.run.id,
            'adaptive_vectors_proj': adaptive_vecs_proj,
            'adaptive_vectors_emb': adaptive_vecs_emb,
        }, checkpoint_path)
        print(f"Checkpoint saved at Epoch {epoch}")
        '''
        if epoch % 40 == 0:
            torch.save([
                net.state_dict(),
                probe.state_dict(),
                wandb.run.id,
                adaptive_vecs_proj,
                adaptive_vecs_emb,
            ], f"vit_encoder_epoch{epoch}.pth")
        '''
        # Evaluation
        net.eval(), probe.eval()

        # Test Accuracy (Validation Set)
        correct = 0
        correct_spurious = 0
        correct_clean = 0

        # [추가됨] - variance 측정용
        stepwise_variances = torch.zeros(16, device="cuda") 
        stepwise_variances_clean = torch.zeros(16, device="cuda") 
        stepwise_variances_emb = torch.zeros(16, device="cuda") 
        stepwise_variances_clean_emb = torch.zeros(16, device="cuda") 
        
        total_samples = 0
        total_samples_clean = 0

        # 통계량 계산기 초기화
        cov_colored_proj = OnlineCovariance("cuda")
        cov_clean_proj = OnlineCovariance("cuda")
        cov_colored = OnlineCovariance("cuda")
        cov_clean = OnlineCovariance("cuda")

        with torch.inference_mode():
            for vs, y, y_colored in test:
                vs = vs.to("cuda", non_blocking=True)
                y = y.to("cuda", non_blocking=True)
                y_colored = y_colored.to("cuda", non_blocking=True)

                with autocast("cuda", dtype=torch.bfloat16):
                    emb, proj = net(vs)
                    y_hat = probe(emb).argmax(1)
                    correct += (y_hat == y).sum().item()
                    correct_spurious += (y_hat == y_colored).sum().item()
                    cov_colored.update(emb)
                    cov_colored_proj.update(proj.flatten(0, 1))

                    if basis_path:
                        z_emb = emb.float()
                        z_emb_centered = z_emb - z_emb.mean(dim=0, keepdim=True)
                        
                        coeff_emb = z_emb_centered @ fixed_basis_emb
                        
                        stepwise_variances_emb += coeff_emb.pow(2).sum(dim=0)

                        z = proj.flatten(0, 1).float()
                        z_centered = z - z.mean(dim=0, keepdim=True)
                        
                        coeff = z_centered @ fixed_basis_proj
                        
                        stepwise_variances += coeff.pow(2).sum(dim=0)
                        total_samples += z.size(0)

        acc = correct / len(test_ds)
        acc_spurious = correct_spurious / len(test_ds)

        # 2. Clean Test Accuracy (Original Imagenette)
        with torch.inference_mode():
            for vs, y in test_clean:
                vs = vs.to("cuda", non_blocking=True)
                y = y.to("cuda", non_blocking=True)
                with autocast("cuda", dtype=torch.bfloat16):
                    emb, proj = net(vs)
                    logits = probe(emb)
                    correct_clean += (logits.argmax(1) == y).sum().item()
                    cov_clean.update(emb)
                    cov_clean_proj.update(proj.flatten(0, 1))

                    if basis_path:
                        z_emb_clean = emb.float()
                        z_emb_clean_centered = z_emb_clean - z_emb_clean.mean(dim=0, keepdim=True)
                        
                        coeff_emb_clean = z_emb_clean_centered @ fixed_basis_emb_clean
                        
                        stepwise_variances_clean_emb += coeff_emb_clean.pow(2).sum(dim=0)

                        z_clean = proj.flatten(0, 1).float()
                        z_clean_centered = z_clean - z_clean.mean(dim=0, keepdim=True)
                        
                        coeff_clean = z_clean_centered @ fixed_basis_proj_clean
                        
                        stepwise_variances_clean += coeff_clean.pow(2).sum(dim=0)
                        total_samples_clean += z_clean.size(0)
        
        acc_clean = correct_clean / len(test_ds_clean)

        # [추가됨]
        eig_vals_colored_tensor, eig_vecs_colored_tensor = cov_colored.compute_spectrum(return_tensors=True)
        eig_vals_colored_proj_tensor, eig_vecs_colored_proj_tensor = cov_colored_proj.compute_spectrum(return_tensors=True)
        
        # 다음 Epoch을 위해 Top-k 벡터 저장 (Gradient 전파 안 되게 detach)
        if eig_vecs_colored_tensor is not None and top_k is not None and top_k > 0:
            adaptive_vecs_proj = eig_vecs_colored_proj_tensor[:, :top_k].detach()
            adaptive_vecs_emb = eig_vecs_colored_tensor[:, :top_k].detach()

        # Eigenvalue 계산
        eig_vals_colored = eig_vals_colored_tensor.cpu().numpy()
        eig_vals_clean, _ = cov_clean.compute_spectrum(return_tensors=False)

        eig_vals_colored_proj = eig_vals_colored_proj_tensor.cpu().numpy()
        eig_vals_clean_proj, _ = cov_clean_proj.compute_spectrum(return_tensors=False)
        
        # Effective Rank 계산
        rank_colored = compute_effective_rank(eig_vals_colored)
        rank_clean = compute_effective_rank(eig_vals_clean)

        rank_colored_proj = compute_effective_rank(eig_vals_colored_proj)
        rank_clean_proj = compute_effective_rank(eig_vals_clean_proj)

        

        log_dict = {
            "test/acc": acc,
            "test/acc_clean": acc_clean,
            "test/acc_spurious": acc_spurious,
            "test/epoch": epoch,
            
            # Rank & Top-1 Eigenvalue는 매번 기록 (추세 확인용)
            "analysis/rank_colored": rank_colored,
            "analysis/rank_clean": rank_clean,
            "analysis/rank_diff": rank_colored - rank_clean,
            "analysis/top1_eig_colored": eig_vals_colored[0] if eig_vals_colored is not None else 0,
            "analysis/top1_eig_clean": eig_vals_clean[0] if eig_vals_clean is not None else 0,
            "analysis/rank_colored_proj": rank_colored_proj,
            "analysis/rank_clean_proj": rank_clean_proj,
            "analysis/rank_diff_proj": rank_colored_proj - rank_clean_proj,
            "analysis/top1_eig_colored_proj": eig_vals_colored_proj[0] if eig_vals_colored_proj is not None else 0,
            "analysis/top1_eig_clean_proj": eig_vals_clean_proj[0] if eig_vals_clean_proj is not None else 0,
        }

        log_image_interval = 10 
        
        if epoch % log_image_interval == 0 or epoch == cfg.epochs - 1:
            # 그래프 그리기 (이때만 수행)
            spectrum_plot = plot_combined_spectrum(eig_vals_colored, eig_vals_clean, epoch)
            spectrum_plot_proj = plot_combined_spectrum(eig_vals_colored_proj, eig_vals_clean_proj, epoch)
            
            # 딕셔너리에 이미지 추가
            log_dict["analysis/spectrum_plot"] = wandb.Image(spectrum_plot, caption=f"Spectrum Ep {epoch}")
            log_dict["analysis/spectrum_plot_proj"] = wandb.Image(spectrum_plot_proj, caption=f"Spectrum Ep {epoch}")

        for i in range(20):
            log_dict[f"eigenvalue/colored_rank_{i}"] = eig_vals_colored[i]
            log_dict[f"eigenvalue/clean_rank_{i}"] = eig_vals_clean[i]

        if basis_path:
            def add_stepwise_logs(prefix, variances, n_samples):
                avg_vars = (variances / n_samples).cpu().numpy()
                for rank_idx, val in enumerate(avg_vars):
                    # "stepwise/colored_proj/rank_0" 형태로 저장
                    log_dict[f"stepwise/{prefix}/rank_{rank_idx}"] = val

            add_stepwise_logs("colored_proj", stepwise_variances, total_samples)
            add_stepwise_logs("colored_emb", stepwise_variances_emb, total_samples) # Emb 샘플수는 Proj와 배율 다르지만(V배), 여기선 total_samples 사용해도 무방(또는 V로 나눔)

            add_stepwise_logs("clean_proj", stepwise_variances_clean, total_samples_clean)
            add_stepwise_logs("clean_emb", stepwise_variances_clean_emb, total_samples_clean)

        wandb.log(log_dict)
    wandb.finish()


if __name__ == "__main__":
    main()