import torch
import torch.nn as nn
import os
import tqdm
from torch.amp import autocast

# -------------------------------------------------------------------------
# main.py의 클래스들 (가정)
# -------------------------------------------------------------------------
from lejepa_difflambda import ViTEncoder, HFDataset, OnlineCovariance 

def extract_and_save_basis():
    # 설정
    ckpt_path = "70, 0.08.pth"      # 체크포인트 경로
    save_path = "final_basis_0.08.pth"     
    device = "cuda"
    batch_size = 256
    proj_dim = 16  # [주의] 체크포인트 설정에 맞춤 (128 -> 16)
    
    print(f"Loading Checkpoint from {ckpt_path}...")
    
    # 1. 모델 로드
    model = ViTEncoder(proj_dim=proj_dim).to(device)
    
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        if 'net_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['net_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    model.eval()

    # ---------------------------------------------------------------------
    # 2. Colored Dataset (Shortcut Basis) 처리
    # ---------------------------------------------------------------------
    print("\n[1/2] Processing Colored Dataset (Shortcut)...")
    ds_colored = HFDataset("train", V=1, mode='colored') 
    loader_colored = torch.utils.data.DataLoader(
        ds_colored, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    cov_emb_colored = OnlineCovariance(device)
    cov_proj_colored = OnlineCovariance(device)

    with torch.inference_mode():
        for vs, _ in tqdm.tqdm(loader_colored, desc="Scanning Colored"):
            vs = vs.to(device, non_blocking=True)
            with autocast("cuda", dtype=torch.bfloat16):
                emb, proj = model(vs)
                
                # [수정] 차원 불일치 해결: proj는 [View, Batch, Dim] -> [B*V, D]
                cov_emb_colored.update(emb)
                cov_proj_colored.update(proj.flatten(0, 1))

    # ---------------------------------------------------------------------
    # 3. Clean Dataset (Core Feature Basis) 처리
    # ---------------------------------------------------------------------
    print("\n[2/2] Processing Clean Dataset (Real Features)...")
    # 기저 추출이므로 Clean 데이터도 'train' split을 사용하여 충분한 통계량 확보
    ds_clean = HFDataset("train", V=1, mode='clean') 
    loader_clean = torch.utils.data.DataLoader(
        ds_clean, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    cov_emb_clean = OnlineCovariance(device)
    cov_proj_clean = OnlineCovariance(device)

    with torch.inference_mode():
        for vs, _ in tqdm.tqdm(loader_clean, desc="Scanning Clean"):
            vs = vs.to(device, non_blocking=True)
            with autocast("cuda", dtype=torch.bfloat16):
                emb, proj = model(vs)
                
                # [수정] 차원 불일치 해결
                cov_emb_clean.update(emb)
                cov_proj_clean.update(proj.flatten(0, 1))

    # ---------------------------------------------------------------------
    # 4. Eigen Decomposition & Save
    # ---------------------------------------------------------------------
    print("\nComputing Spectra...")
    
    # Colored
    c_emb_vals, c_emb_vecs = cov_emb_colored.compute_spectrum(return_tensors=True)
    c_proj_vals, c_proj_vecs = cov_proj_colored.compute_spectrum(return_tensors=True)
    
    # Clean
    cl_emb_vals, cl_emb_vecs = cov_emb_clean.compute_spectrum(return_tensors=True)
    cl_proj_vals, cl_proj_vecs = cov_proj_clean.compute_spectrum(return_tensors=True)

    basis_data = {
        # --- Colored Basis (Shortcut이 포함됨) ---
        "colored_emb_eigvals": c_emb_vals.cpu(),
        "colored_proj_eigvals": c_proj_vals.cpu(),
        "colored_emb_eigvecs": c_emb_vecs.cpu(),
        "colored_proj_eigvecs": c_proj_vecs.cpu(),
        
        # --- Clean Basis (Core Feature 위주) ---
        "clean_emb_eigvals": cl_emb_vals.cpu(),
        "clean_proj_eigvals": cl_proj_vals.cpu(),
        "clean_emb_eigvecs": cl_emb_vecs.cpu(),
        "clean_proj_eigvecs": cl_proj_vecs.cpu(),
        
        "source_checkpoint": ckpt_path
    }

    torch.save(basis_data, save_path)
    print(f"Successfully saved combined basis to {save_path}")

if __name__ == "__main__":
    extract_and_save_basis()