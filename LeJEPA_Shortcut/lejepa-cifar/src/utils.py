import torch
import numpy as np
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

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

def project_to_nullspace(features, eigvecs):
    """
    features: [Batch, D] - 원본 임베딩 (emb)
    eigvecs: [D, K] - 마스킹할 Top-K Eigenvectors (adaptive_vecs_emb)
    """
    if eigvecs is None or eigvecs.size(1) == 0:
        return features
    
    # 1. 엄밀한 투영을 위해 평균을 0으로 맞춤 (Covariance 기반이므로)
    mean_f = features.mean(dim=0, keepdim=True)
    centered_features = features - mean_f
    
    # 2. X @ V: 특징 벡터를 Top-k 방향으로 투영했을 때의 계수 (Coefficient)
    proj_coeff = centered_features @ eigvecs  # [Batch, K]
    
    # 3. (X @ V) @ V^T: Top-k 방향으로 복원된(reconstructed) 성분
    reconstructed = proj_coeff @ eigvecs.T  # [Batch, D]
    
    # 4. Nullspace: 원본에서 복원된 성분을 제거
    null_features = centered_features - reconstructed
    
    # 5. 평균을 뺐었다면 다시 더해줌 (선택)
    return null_features + mean_f