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


# -----------------------------------------------------------------------------
# [Helper Class] 온라인 공분산 계산기 (메모리 절약 & 코드 정리용)
# -----------------------------------------------------------------------------
class OnlineCovariance:
    def __init__(self, device="cuda"):
        self.device = device
        self.sum_x = None
        self.sum_xtx = None
        self.n = 0

    def update(self, features):
        """배치 단위 Feature를 받아서 통계량 누적"""
        # features: [Batch, D]
        # BFloat16이면 정밀도 위해 Float32로 변환
        x = features.detach().float()
        batch_size = x.size(0)
        
        if self.sum_x is None:
            d = x.size(1)
            self.sum_x = torch.zeros(d, device=self.device)
            self.sum_xtx = torch.zeros(d, d, device=self.device)
            
        self.sum_x += x.sum(dim=0)
        self.sum_xtx += x.T @ x
        self.n += batch_size

    def compute_spectrum(self, return_tensors=True):
        """최종 Eigenvalue 계산 (내림차순 정렬)"""
        if self.n <= 1: return None, None
        
        # E[X^T X] - E[X]^T E[X] 공식 사용
        mean_x = self.sum_x / self.n
        mean_xtx = self.sum_xtx / self.n
        cov = mean_xtx - (mean_x.unsqueeze(1) @ mean_x.unsqueeze(0))
        
        # Eigen Decomposition (Symmetric)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        
        # 3. 내림차순 정렬 (Shortcut = Top Eigenvalues)
        # 중요: Eigenvalue 순서에 맞춰 Eigenvector도 순서를 바꿔야 함
        idx = torch.argsort(eigvals, descending=True)
        
        sorted_eigvals = eigvals[idx].clamp(min=0) # 음수 방지
        sorted_eigvecs = eigvecs[:, idx] # 열 순서 재배치

        if return_tensors:
            # Loss 계산에 바로 쓸 수 있게 CUDA Tensor 반환
            return sorted_eigvals, sorted_eigvecs
        else:
            # 로깅용 CPU Numpy 반환
            return sorted_eigvals.cpu().numpy(), sorted_eigvecs.cpu().numpy()
'''
def compute_effective_rank_old(eigvals):
    """Effective Rank 계산"""
    if eigvals is None: return 0
    eig_sum = eigvals.sum()
    eig_sq_sum = (eigvals ** 2).sum()
    return (eig_sum ** 2) / eig_sq_sum if eig_sq_sum > 0 else 0
'''
def compute_effective_rank(eigvals):

    if eigvals is None or len(eigvals) == 0:
        return 0
    
    eigvals = np.abs(eigvals)
    eig_sum = np.sum(eigvals)
    
    if eig_sum <= 0:
        return 0
    p = eigvals / eig_sum
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    
    return np.exp(entropy)

def plot_combined_spectrum(eig_colored, eig_clean, epoch):
    """Colored와 Clean 스펙트럼을 비교하는 그래프 생성"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Log-Log Scale Plot
    if eig_colored is not None:
        ax.loglog(eig_colored, label='Colored (Valid)', color='red', alpha=0.7, linewidth=2)
    if eig_clean is not None:
        ax.loglog(eig_clean, label='Clean (OOD)', color='blue', alpha=0.7, linewidth=2, linestyle='--')
        
    ax.set_title(f"Eigenvalue Spectrum (Epoch {epoch})")
    ax.set_xlabel("Rank Index (Log)")
    ax.set_ylabel("Eigenvalue (Log)")
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.3)
    
    # 이미지로 변환
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img

# 1. SIGReg Test Statistic
class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj, target_vec=None):
        if target_vec is None:
            A = torch.randn(proj.size(-1), 256, device="cuda") # [D, K]
            A = A.div_(A.norm(p=2, dim=0))
        else:
            A = target_vec.to(proj.device)  # [D, K]
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

# 2. Model Definition (ViT Small)
class ViTEncoder(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch8_224",
            pretrained=False,
            num_classes=512,
            drop_path_rate=0.1,
            img_size=128,
        )
        self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)

    
# 3. Dataset Definition
class HFDataset(torch.utils.data.Dataset):
    def __init__(self, split, V=1, mode='colored'):
        self.V = V
        if mode == 'clean':
            self.ds = load_dataset("frgfm/imagenette", "160px", split=split)
        else:
            self.ds = load_from_disk("./colored_imagenette")[split]
        self.aug = v2.Compose(
            [
                v2.RandomResizedCrop(128, scale=(0.08, 1.0)),
                v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                v2.RandomGrayscale(p=0.2),
                v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
                v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
                v2.RandomHorizontalFlip(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.test = v2.Compose(
            [
                v2.Resize(128),
                v2.CenterCrop(128),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __getitem__(self, i):
        item = self.ds[i]
        img = item["image"].convert("RGB")
        transform = self.aug if self.V > 1 else self.test
        return torch.stack([transform(img) for _ in range(self.V)]), item["label"]

    def __len__(self):
        return len(self.ds)


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

    train_ds = HFDataset("train", V=cfg.V)
    test_ds = HFDataset("validation", V=1)
    test_ds_clean = HFDataset("validation", V=1, mode='clean')
    # [추가됨 - num workers 8 -> 4, persistent_workers=True, pin_memory=True]
    train = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True, drop_last=True, num_workers=4, persistent_workers=True, pin_memory=True)
    test = DataLoader(test_ds, batch_size=256, num_workers=4, persistent_workers=True, pin_memory=True)
    test_clean = DataLoader(test_ds_clean, batch_size=256, num_workers=4, persistent_workers=True, pin_memory=True)

    # modules and loss
    net = ViTEncoder(proj_dim=cfg.proj_dim).to("cuda")
    probe = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 100)).to("cuda")
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
        for vs, y in tqdm.tqdm(train, total=len(train)):
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

        if epoch % 40 == 0:
            torch.save(net.state_dict(), f"vit_encoder_epoch{epoch}.pth")

        # Evaluation
        net.eval(), probe.eval()

        # Test Accuracy (Validation Set)
        correct = 0
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
            for vs, y in test:
                vs = vs.to("cuda", non_blocking=True)
                y = y.to("cuda", non_blocking=True)
                with autocast("cuda", dtype=torch.bfloat16):
                    emb, proj = net(vs)
                    correct += (probe(emb).argmax(1) == y).sum().item()
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