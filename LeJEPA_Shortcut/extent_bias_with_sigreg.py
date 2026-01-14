import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import lejepa

# --- 1. Settings from Section 4.1 & D.1 ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Dimensions & Parameters
n = 1000        # Dataset size
m = 10          # Total feature dimension
ml = 9          # Size of larger part (background-like)
ms = 1          # Size of smaller part (object-like)
d = 2           # Output dimension

steps = 6000
noise_a = 0.01          # Augmentation noise parameter a
lr = 6e-4               # Learning rate (Section D.1)
bt_lambda = 1e-2        # Scaling factor / Off-diagonal weight (Section D.1)
init_std = 0.01         # Small initialization to start from 0 eigenvalues

# --- 2. Data Generation (Section 4.1) ---
class ExtentBiasDataset:
    def __init__(self, n, ml, ms, noise_a):
        self.n = n
        self.ml = ml
        self.ms = ms
        self.noise_a = noise_a
        
        # Generate x_base: bl, bs ~ Bernoulli(0.5) -> {-1, 1}
        # Shape: (n, 1)
        bl = torch.randint(0, 2, (n, 1)).float() * 2 - 1
        bs = torch.randint(0, 2, (n, 1)).float() * 2 - 1
        
        # Construct base vectors: [bl * 1_ml, bs * 1_ms]
        part_l = bl @ torch.ones(1, ml)
        part_s = bs @ torch.ones(1, ms)
        
        # x_base: (n, m)
        self.x_base = torch.cat([part_l, part_s], dim=1)

    def get_batch(self):
        # Augmentation: x = x_base + epsilon
        noise1 = torch.randn_like(self.x_base) * self.noise_a
        noise2 = torch.randn_like(self.x_base) * self.noise_a
        
        x1 = self.x_base + noise1
        x2 = self.x_base + noise2
        return x1, x2

# --- 3. Feature Definitions for Alignment ---
# Unit vectors for alignment measurement
v_l = torch.cat([torch.ones(ml), torch.zeros(ms)]).float()
v_s = torch.cat([torch.zeros(ml), torch.ones(ms)]).float()
# e_l = v_l / torch.norm(v_l)
# e_s = v_s / torch.norm(v_s)
e_l = v_l
e_s = v_s

# --- 4. Model & Loss (Toy Barlow Twins - No Batch Norm) ---
# Theoretical framework uses Linear network without Bias
model = nn.Linear(m, d, bias=False)
nn.init.normal_(model.weight, mean=0.0, std=init_std)

optimizer = optim.SGD(model.parameters(), lr=lr)

def toy_barlow_twins_loss(z1, z2, lambd):
    # [CRITICAL CHANGE] No Batch Normalization / Standardization
    # Following Eq (1) and (2) in the paper: C = Sum(z_A * z_B.T) / n
    # Since we use batch-based optimization, we approximate over the batch.
    
    batch_size = z1.shape[0]
    
    # Empirical Cross-Correlation Matrix
    # Shape: (d, d)
    c = torch.matmul(z1.T, z2) / batch_size
    
    # Loss: ||C - I||^2
    # on_diag: terms where i == j
    on_diag = torch.diagonal(c).add(-1).pow(2).sum()
    
    # off_diag: terms where i != j
    # We flatten the matrix and remove diagonal elements
    n_dim, m_dim = c.shape
    # off_diag_elements = c.flatten()[:-1].view(n_dim-1, n_dim+1)[:, 1:].flatten()
    # off_diag = off_diag_elements.pow(2).sum()

    # [추가됨: SIGReg]
    univariate_test = lejepa.univariate.EppsPulley(n_points=17)
    loss_fn = lejepa.multivariate.SlicingUnivariateTest(
        univariate_test=univariate_test,
        num_slices=128
    )
    loss_sigreg = (loss_fn(z1) + loss_fn(z2)) / 2
    
    loss = (1 - lambd) * on_diag +  bt_lambda * loss_sigreg
    return loss, c

# --- 5. Training Loop ---
dataset = ExtentBiasDataset(n, ml, ms, noise_a)

log_loss = []
log_eig1 = []
log_eig2 = []
log_align_l = []
log_align_s = []

print("Training started...")
for t in range(steps):
    optimizer.zero_grad()
    
    # Full batch gradient descent (as implied by n=1000 in D.1 context)
    x1, x2 = dataset.get_batch()
    
    z1 = model(x1)
    z2 = model(x2)
    
    loss, C = toy_barlow_twins_loss(z1, z2, bt_lambda)
    
    loss.backward()
    optimizer.step()
    
    # --- Monitoring ---
    with torch.no_grad():
        W = model.weight.detach()
        
        # 1. Eigenvalues of C (Covariance of output)
        # Note: In the paper's theoretical limit, C should approach Identity.
        # We calculate eigenvalues of the current correlation matrix C.
        # Since C is not necessarily symmetric in sample usage (z1.T @ z2), 
        # but theoretically z1 ~ z2. For stability we symmetrize or use SVD.
        # Here we use eigvalsh on (C + C.T)/2 for stability.
        C_sym = (C + C.T) / 2
        eigvals = torch.linalg.eigvalsh(C_sym)
        eigvals = eigvals.sort(descending=True).values
        
        # 2. Feature Alignment ||We||^2
        align_l = torch.norm(W @ e_l).pow(2)
        align_s = torch.norm(W @ e_s).pow(2)
        
        log_loss.append(loss.item())
        log_eig1.append(eigvals[0].item())
        log_eig2.append(eigvals[1].item())
        log_align_l.append(align_l.item())
        log_align_s.append(align_s.item())
    
    print(f"Step {t+1}/{steps} - Loss: {loss.item():.4f} - Eig1: {eigvals[0].item():.4f} - Eig2: {eigvals[1].item():.4f} - Align_l: {align_l.item():.4f} - Align_s: {align_s.item():.4f}", end='\r')

# --- 6. Visualization (Replicating Figure 1) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Loss Evolution
axes[0].plot(range(steps), log_loss, color='green', label='Loss')
axes[0].set_title('Loss Evolution')
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Loss')
axes[0].grid(True, linestyle='--', alpha=0.6)

# Plot 2: Eigenvalues of Covariance Matrix
axes[1].plot(range(steps), log_eig1, color='blue', label=r'$\lambda_1$')
axes[1].plot(range(steps), log_eig2, color='red', label=r'$\lambda_2$')
axes[1].set_title('Eigenvalues of Covariance Matrix')
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Eigenvalue')
axes[1].set_ylim(-0.1, 1.2) # Should go from 0 to 1
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.6)

# Plot 3: Feature Alignment
axes[2].plot(range(steps), log_align_l, color='blue', label=r'$||We_l||^2$ (Background)')
axes[2].plot(range(steps), log_align_s, color='red', label=r'$||We_s||^2$ (Object)')
axes[2].set_title('Feature Alignment')
axes[2].set_xlabel('Step')
axes[2].set_ylabel(r'$||We||^2$')
axes[2].legend()
axes[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()