# -*- coding: utf-8 -*-
"""
[Step 8] Zero-Shot Domain Adaptation using MMD Loss 
- DI Scale 0~1 
- RESIDUAL AMPLIFICATION (x10) to boost generated damage size.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from scipy.stats import kurtosis
from scipy.signal import welch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# =========================================================
# 1. Configuration
# =========================================================
class Config:
    DIR_A = r"E:\Benchmark Code\benchmarktu1402-master\f_accerlerations\ds1"
    DIR_B = r"E:\2ndstructuredata\raw data"
    FILE_B = "healthyclean.txt"
    SAVE_DIR = r"E:\2ndstructuredata\Code_5_MMD"
    
    WINDOW_SIZE = 128
    LATENT_DIM = 8
    SELECTED_NODES = [3, 21, 39, 57, 63, 81, 99, 117]
    
    DAMAGE_CASES_A = list(range(1, 11))
    DAMAGE_CASES_B = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48] 
    
    AE_EPOCHS = 500
    DON_EPOCHS = 500
    BATCH_SIZE = 128
    LR = 0.001
    
    LAMBDA_MMD = 0.3
    
    # [핵심 수정] 잔차 증폭 스케일 추가 (모델이 더 큰 잔차를 학습하도록 유도)
    RESIDUAL_SCALE = 0.9 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()
if not os.path.exists(cfg.SAVE_DIR): os.makedirs(cfg.SAVE_DIR)

# =========================================================
# 2. Models & MMD
# =========================================================
def rbf_kernel(x, y, gamma=1.0):
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    return torch.exp(-gamma * torch.sum((x - y) ** 2, dim=2))

def mmd_loss(x, y, gamma=1.0):
    xx = rbf_kernel(x, x, gamma)
    yy = rbf_kernel(y, y, gamma)
    xy = rbf_kernel(x, y, gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()

class Autoencoder(nn.Module):
    def __init__(self, input_dim=128*8, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.SiLU(),
            nn.Linear(512, 256), nn.SiLU(),
            nn.Linear(256, 64), nn.SiLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.SiLU(),
            nn.Linear(64, 256), nn.SiLU(),
            nn.Linear(256, 512), nn.SiLU(),
            nn.Linear(512, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

class FourierFeature(nn.Module):
    def __init__(self, input_dim, mapping_size=128, scale=10):
        super().__init__()
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)
    def forward(self, x):
        projected = 2 * np.pi * (x @ self.B)
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)

class DeepONet(nn.Module):
    def __init__(self, branch_dim=9, trunk_dim=2, hidden_dim=128):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(branch_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.fourier = FourierFeature(trunk_dim)
        self.trunk = nn.Sequential(
            nn.Linear(256, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, branch_in, trunk_in):
        B = self.branch(branch_in)
        T = self.trunk(self.fourier(trunk_in))
        return torch.matmul(B, T.T) + self.bias

# =========================================================
# 3. Utility Functions
# =========================================================
def load_data(path, is_A=False):
    try:
        if is_A:
            data = np.loadtxt(path)
            data = data[:, cfg.SELECTED_NODES].T if data.shape[0] > data.shape[1] else data[cfg.SELECTED_NODES, :]
        else:
            raw = [float(line.split()[1]) for line in open(path, 'r') if len(line.split()) >= 2]
            data = np.array(raw, dtype=np.float32).reshape(8, -1)
        n_samples = data.shape[1] // cfg.WINDOW_SIZE
        return data[:, :n_samples * cfg.WINDOW_SIZE].astype(np.float32)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def calc_di(healthy, damaged):
    window = 2000
    n_wins = min(healthy.shape[1], damaged.shape[1]) // window
    if n_wins < 1: n_wins = 1; window = min(healthy.shape[1], damaged.shape[1])
    di_list = [abs(np.percentile(kurtosis(healthy[i, :n_wins*window].reshape(n_wins, window), axis=1, fisher=False), 95) - 
                   np.percentile(kurtosis(damaged[i, :n_wins*window].reshape(n_wins, window), axis=1, fisher=False), 95)) 
               for i in range(8)]
    return np.mean(di_list)

def reshape_for_scaler(data):
    ns = data.shape[1] // cfg.WINDOW_SIZE
    return data.reshape(8, ns, cfg.WINDOW_SIZE).transpose(1, 0, 2).reshape(ns, -1), ns

# =========================================================
# 4. Pipeline Execution
# =========================================================
def main():
    print("=== Phase 1: Healthy-based Normalization ===")
    raw_h_A = load_data(os.path.join(cfg.DIR_A, "fh_accelerations.dat"), is_A=True)
    raw_h_B = load_data(os.path.join(cfg.DIR_B, cfg.FILE_B), is_A=False)
    
    reshaped_h_A, ns_A = reshape_for_scaler(raw_h_A)
    reshaped_h_B, ns_B = reshape_for_scaler(raw_h_B)
    
    scaler_A = MinMaxScaler(feature_range=(0, 1)).fit(reshaped_h_A)
    scaler_B = MinMaxScaler(feature_range=(0, 1)).fit(reshaped_h_B)
    
    norm_h_A = scaler_A.transform(reshaped_h_A)
    norm_h_B = scaler_B.transform(reshaped_h_B)
    
    print("\n=== Phase 2: Autoencoder Training (with MMD) ===")
    ae = Autoencoder().to(cfg.device)
    optimizer_ae = optim.Adam(ae.parameters(), lr=cfg.LR)
    criterion_recon = nn.MSELoss()
    
    loader_A = DataLoader(TensorDataset(torch.FloatTensor(norm_h_A)), batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)
    loader_B = DataLoader(TensorDataset(torch.FloatTensor(norm_h_B)), batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)
    
    for epoch in range(cfg.AE_EPOCHS):
        ae.train()
        total_recon, total_mmd = 0, 0
        for (batch_A,), (batch_B,) in zip(loader_A, loader_B):
            b_A, b_B = batch_A.to(cfg.device), batch_B.to(cfg.device)
            optimizer_ae.zero_grad()
            
            recon_A, z_A_batch = ae(b_A)
            recon_B, z_B_batch = ae(b_B)
            
            loss_r = criterion_recon(recon_A, b_A) + criterion_recon(recon_B, b_B)
            loss_m = mmd_loss(z_A_batch, z_B_batch)
            
            loss = loss_r + (cfg.LAMBDA_MMD * loss_m)
            loss.backward()
            optimizer_ae.step()
            
            total_recon += loss_r.item()
            total_mmd += loss_m.item()
            
        if (epoch+1) % 50 == 0: 
            print(f"   AE Epoch {epoch+1}/{cfg.AE_EPOCHS} | Recon Loss: {total_recon/len(loader_A):.6f} | MMD Loss: {total_mmd/len(loader_A):.6f}")
    
    ae.eval()
    with torch.no_grad():
        z_A = ae.encoder(torch.FloatTensor(norm_h_A).to(cfg.device)).cpu().numpy()
        z_B = ae.encoder(torch.FloatTensor(norm_h_B).to(cfg.device)).cpu().numpy()

    print("\n=== Phase 3: DeepONet Residual Training (Structure A) ===")
    dis_A_raw = [calc_di(raw_h_A, load_data(os.path.join(cfg.DIR_A, f"f{c}_accelerations.dat"), is_A=True)) for c in cfg.DAMAGE_CASES_A]
    min_di_A, max_di_A = min(dis_A_raw), max(dis_A_raw)
    
    X_branch, Y_target = [], []
    for i, c in enumerate(cfg.DAMAGE_CASES_A):
        raw_d = load_data(os.path.join(cfg.DIR_A, f"f{c}_accelerations.dat"), is_A=True)
        reshaped_d, _ = reshape_for_scaler(raw_d)
        norm_d = scaler_A.transform(reshaped_d)
        
        # [핵심 수정] 타겟 잔차를 10배 증폭하여 모델이 더 확실하게 파형 크기 변화를 인지하도록 함
        target_res = (norm_d - norm_h_A) * cfg.RESIDUAL_SCALE 
        di_norm = (dis_A_raw[i] - min_di_A) / (max_di_A - min_di_A)
        
        b_in = np.hstack([z_A, np.full((ns_A, 1), di_norm)])
        X_branch.append(b_in)
        Y_target.append(target_res)
        
    X_branch = np.vstack(X_branch)
    Y_target = np.vstack(Y_target)
    
    don = DeepONet(branch_dim=9, trunk_dim=2).to(cfg.device)
    optimizer_don = optim.Adam(don.parameters(), lr=cfg.LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer_don, step_size=300, gamma=0.5) 
    
    T_grid, X_grid = np.meshgrid(np.linspace(0, 1, cfg.WINDOW_SIZE), np.linspace(0, 1, 8))
    trunk_in = torch.FloatTensor(np.stack([T_grid.flatten(), X_grid.flatten()], axis=1)).to(cfg.device)
    loader_don = DataLoader(TensorDataset(torch.FloatTensor(X_branch), torch.FloatTensor(Y_target)), batch_size=cfg.BATCH_SIZE, shuffle=True)
    
    for epoch in range(cfg.DON_EPOCHS):
        don.train()
        total_loss = 0
        for b_in, target in loader_don:
            optimizer_don.zero_grad()
            preds = don(b_in.to(cfg.device), trunk_in)
            loss = nn.MSELoss()(preds, target.to(cfg.device))
            loss.backward()
            optimizer_don.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch+1) % 100 == 0: 
            print(f"   DeepONet Epoch {epoch+1}/{cfg.DON_EPOCHS} | Loss: {total_loss/len(loader_don):.6f}")

    print("\n=== Phase 4: Validation & Visualization ===")
    
    def validate_structure(name, raw_h, norm_h, scaler, z_actual, damage_cases, min_di, max_di, is_A=False):
        print(f"\n[{name} Logs] {'Case':<4} | {'Real DI':<10} | {'Input(0~1)':<10} | {'Gen DI':<10}")
        real_dis, gen_dis = [], []
        dis_raw = [calc_di(raw_h, load_data(os.path.join(cfg.DIR_A if is_A else cfg.DIR_B, f"f{c}_accelerations.dat" if is_A else f"D3_{c}_1.txt"), is_A=is_A)) for c in damage_cases]
        
        don.eval()
        for i, c in enumerate(damage_cases):
            if is_A:
                di_norm = (dis_raw[i] - min_di) / (max_di - min_di)
            else:
                di_norm = (i + 1) / len(damage_cases)
            
            b_in = torch.FloatTensor(np.hstack([z_actual, np.full((len(norm_h), 1), di_norm)])).to(cfg.device)
            
            with torch.no_grad():
                # [핵심 수정] 예측된 잔차를 다시 10으로 나누어 원본 스케일로 복구
                pred_res = don(b_in, trunk_in).cpu().numpy() / cfg.RESIDUAL_SCALE
                
            gen_phys = scaler.inverse_transform(norm_h + pred_res)
            gen_data = gen_phys.reshape(len(norm_h), 8, cfg.WINDOW_SIZE).transpose(1, 0, 2).reshape(8, -1)
            
            gen_di = calc_di(raw_h, gen_data)
            real_dis.append(dis_raw[i])
            gen_dis.append(gen_di)
            print(f"Case {c:<2} | {dis_raw[i]:.6f}   | {di_norm:.6f}   | {gen_di:.6f}")
        return real_dis, gen_dis

    # Validation
    real_A, gen_A = validate_structure("Structure A", raw_h_A, norm_h_A, scaler_A, z_A, cfg.DAMAGE_CASES_A, min_di_A, max_di_A, is_A=True)
    real_B, gen_B = validate_structure("Structure B", raw_h_B, norm_h_B, scaler_B, z_B, cfg.DAMAGE_CASES_B, min_di_A, max_di_A, is_A=False)

    # Plot Trends
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(cfg.DAMAGE_CASES_A, real_A, 'k--o', label='Real A DI')
    ax1.plot(cfg.DAMAGE_CASES_A, gen_A, 'b-s', label='Gen A DI')
    ax1.set_title("Structure A: Real vs Generated DI"); ax1.legend(); ax1.grid(True, alpha=0.3)
    
    ax2.plot(cfg.DAMAGE_CASES_B, real_B, 'k--o', label='Real B DI')
    ax2.plot(cfg.DAMAGE_CASES_B, gen_B, 'r-s', label='Gen B DI')
    ax2.set_title("Structure B: Real vs Generated DI (MMD + Scale)"); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.savefig(os.path.join(cfg.SAVE_DIR, "validation_trends_mmd.png"))
    plt.show()

    # =========================================================
    # 5. 5000 Samples Visualization (Time & PSD)
    # =========================================================
    def visualize_5000(name, raw_h, raw_d, scaler, z_actual, di_norm):
        print(f"\n   -> Generating 5000-sample plot for {name}...")
        reshaped_h, _ = reshape_for_scaler(raw_h)
        norm_h = scaler.transform(reshaped_h)
        
        b_in = torch.FloatTensor(np.hstack([z_actual, np.full((len(norm_h), 1), di_norm)])).to(cfg.device)
        with torch.no_grad():
            # 예측 시 증폭 비율 제거
            pred_res = don(b_in, trunk_in).cpu().numpy() / cfg.RESIDUAL_SCALE
            
        gen_phys = scaler.inverse_transform(norm_h + pred_res)
        gen_data = gen_phys.reshape(len(norm_h), 8, cfg.WINDOW_SIZE).transpose(1, 0, 2).reshape(8, -1)
        
        # 5000개 샘플 추출 (Sensor 1 기준)
        real_wave = raw_d[0, :1000]
        gen_wave = gen_data[0, :1000]
        
        f_real, p_real = welch(real_wave, fs=100, nperseg=256)
        f_gen, p_gen = welch(gen_wave, fs=100, nperseg=256)
        
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        axs[0].plot(real_wave, 'k-', label='Real Input', alpha=0.6, linewidth=1)
        axs[0].plot(gen_wave, 'r--', label='Generated', alpha=0.8, linewidth=1)
        axs[0].set_title(f"{name} - Time Domain (First 5000 pts)")
        axs[0].legend(loc='upper right'); axs[0].grid(True, alpha=0.3)
        
        axs[1].semilogy(f_real, p_real, 'k-', label='Real PSD', alpha=0.6)
        axs[1].semilogy(f_gen, p_gen, 'r--', label='Generated PSD', alpha=0.8)
        axs[1].set_title(f"{name} - Frequency Domain (PSD)")
        axs[1].set_xlabel("Frequency (Hz)"); axs[1].set_ylabel("PSD")
        axs[1].legend(loc='upper right'); axs[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.SAVE_DIR, f"vis_5000_{name}.png"))
        plt.show()

    # A 구조물 50% 손상 시각화 (Case f5)
    raw_d_A50 = load_data(os.path.join(cfg.DIR_A, "f5_accelerations.dat"), is_A=True)
    visualize_5000("Structure_A_50", raw_h_A, raw_d_A50, scaler_A, z_A, (dis_A_raw[4]-min_di_A)/(max_di_A-min_di_A))
    
    # B 구조물 48% 손상 시각화 (Case 48) -> B의 최대 손상(1.0) 모방
    raw_d_B48 = load_data(os.path.join(cfg.DIR_B, "D3_48_1.txt"), is_A=False)
    visualize_5000("Structure_B3_48_1", raw_h_B, raw_d_B48, scaler_B, z_B, 1.0)

if __name__ == "__main__":
    main()