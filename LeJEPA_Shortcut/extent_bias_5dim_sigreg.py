import lejepa
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Settings from Section 4.1 & D.1 ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Dimensions & Parameters
n = 1000        # Dataset size
m = 31          # Total feature dimension
m1 = 16          # Size of larger part (background-like)
m2 = 8
m3 = 4
m4 = 2
m5 = 1          # Size of smaller part (object-like)
d = 5           # Output dimension

steps = 6000
noise_a = 0.01          # Augmentation noise parameter a
lr = 6e-4               # Learning rate (Section D.1)
bt_lambda = 0.1        # Scaling factor / Off-diagonal weight (Section D.1)
init_std = 0.01         # Small initialization to start from 0 eigenvalues

# --- 2. Data Generation (Section 4.1) ---
class ExtentBiasDataset:
    def __init__(self, n, m1, m2, m3, m4, m5, noise_a):
        self.n = n
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.m4 = m4
        self.m5 = m5
        self.noise_a = noise_a
        
        # Generate x_base: bl, bs ~ Bernoulli(0.5) -> {-1, 1}
        # Shape: (n, 1)
        b1 = torch.randint(0, 2, (n, 1)).float() * 2 - 1
        b2 = torch.randint(0, 2, (n, 1)).float() * 2 - 1
        b3 = torch.randint(0, 2, (n, 1)).float() * 2 - 1
        b4 = torch.randint(0, 2, (n, 1)).float() * 2 - 1
        b5 = torch.randint(0, 2, (n, 1)).float() * 2 - 1
        
        # Construct base vectors: [bl * 1_ml, bs * 1_ms]
        part_1 = b1 @ torch.ones(1, m1)
        part_2 = b2 @ torch.ones(1, m2)
        part_3 = b3 @ torch.ones(1, m3)
        part_4 = b4 @ torch.ones(1, m4)
        part_5 = b5 @ torch.ones(1, m5)
        
        # x_base: (n, m)
        self.x_base = torch.cat([part_1, part_2, part_3, part_4, part_5], dim=1)

    def get_batch(self):
        # Augmentation: x = x_base + epsilon
        noise1 = torch.randn_like(self.x_base) * self.noise_a
        noise2 = torch.randn_like(self.x_base) * self.noise_a
        
        x1 = self.x_base + noise1
        x2 = self.x_base + noise2
        return x1, x2

# --- 3. Feature Definitions for Alignment ---
# Unit vectors for alignment measurement
v_1 = torch.cat([torch.ones(m1), torch.zeros(m2 + m3 + m4 + m5)]).float()
v_2 = torch.cat([torch.zeros(m1), torch.ones(m2), torch.zeros(m3 + m4 + m5)]).float()
v_3 = torch.cat([torch.zeros(m1 + m2), torch.ones(m3), torch.zeros(m4 + m5)]).float()
v_4 = torch.cat([torch.zeros(m1 + m2 + m3), torch.ones(m4), torch.zeros(m5)]).float()
v_5 = torch.cat([torch.zeros(m1 + m2 + m3 + m4), torch.ones(m5)]).float()
# e_l = v_l / torch.norm(v_l)
# e_s = v_s / torch.norm(v_s)
e_1 = v_1
e_2 = v_2
e_3 = v_3
e_4 = v_4
e_5 = v_5
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

    univariate_test = lejepa.univariate.EppsPulley(n_points=17)
    loss_fn = lejepa.multivariate.SlicingUnivariateTest(
        univariate_test=univariate_test,
        num_slices=128
    )
    loss_sigreg = (loss_fn(z1) + loss_fn(z2)) / 2
    
    loss = (1 - lambd) * on_diag + lambd * loss_sigreg
    return loss, c

# --- 5. Training Loop ---
dataset = ExtentBiasDataset(n, m1, m2, m3, m4, m5, noise_a)

log_loss = []
log_eig1 = []
log_eig2 = []
log_eig3 = []
log_eig4 = []
log_eig5 = []
log_align_1 = []
log_align_2 = []
log_align_3 = []
log_align_4 = []
log_align_5 = []


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
        align_1 = torch.norm(W @ e_1).pow(2)
        align_2 = torch.norm(W @ e_2).pow(2)
        align_3 = torch.norm(W @ e_3).pow(2)
        align_4 = torch.norm(W @ e_4).pow(2)
        align_5 = torch.norm(W @ e_5).pow(2)
        
        log_loss.append(loss.item())
        log_eig1.append(eigvals[0].item())
        log_eig2.append(eigvals[1].item())
        log_eig3.append(eigvals[2].item())
        log_eig4.append(eigvals[3].item())
        log_eig5.append(eigvals[4].item())
        
        log_align_1.append(align_1.item())
        log_align_2.append(align_2.item())
        log_align_3.append(align_3.item())
        log_align_4.append(align_4.item())
        log_align_5.append(align_5.item())

    print(f"Step {t+1}/{steps} - Loss: {loss.item():.4f}", end='\r')


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
axes[1].plot(range(steps), log_eig3, color='green', label=r'$\lambda_3$')
axes[1].plot(range(steps), log_eig4, color='orange', label=r'$\lambda_4$')
axes[1].plot(range(steps), log_eig5, color='purple', label=r'$\lambda_5$')
axes[1].set_title('Eigenvalues of Covariance Matrix')
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Eigenvalue')
axes[1].set_ylim(-0.1, 1.2) # Should go from 0 to 1
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.6)

# Plot 3: Feature Alignment
axes[2].plot(range(steps), log_align_1, color='blue', label=r'$||We_1||^2$ (Background1)')
axes[2].plot(range(steps), log_align_2, color='red', label=r'$||We_2||^2$ (Background2)')
axes[2].plot(range(steps), log_align_3, color='green', label=r'$||We_3||^2$ (Normal)')
axes[2].plot(range(steps), log_align_4, color='orange', label=r'$||We_4||^2$ (Object1)')
axes[2].plot(range(steps), log_align_5, color='purple', label=r'$||We_5||^2$ (Object2)')
axes[2].set_title('Feature Alignment')
axes[2].set_xlabel('Step')
axes[2].set_ylabel(r'$||We||^2$')
axes[2].legend()
axes[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('graph_5dim_sigreg.png')
plt.show()

