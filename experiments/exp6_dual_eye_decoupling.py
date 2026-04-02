# experiments/exp6_dual_eye_decoupling.py

import os
import zarr
import numpy as np
import cv2
import cupy as cp
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
import concurrent.futures
from functools import partial
import traceback

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.config_loader import IDUQConfig
from core.perception import PhysicsAwarePerception
from core.data_loader import safe_open_zarr, get_episode_data

def evaluate_dual_eyes(ep_id, config_path):
    cv2.setNumThreads(1) 
    try:
        cfg = IDUQConfig.from_yaml(config_path)
        root = safe_open_zarr(cfg.io['data_path'])
        perception = PhysicsAwarePerception(cfg)
        
        images, poses = get_episode_data(root, ep_id)
        if len(images) < 100: return None
            
        step = cfg.perception.get('step', 1)
        trim = cfg.perception.get('trim_edge', 20)
        
        xi_trim, s_dot_trim = perception.process_episode(images, poses)
        min_len = min(len(xi_trim), len(s_dot_trim))
        
        # --- 提取几何之眼 (Geometric Observability) ---
        # 我们用 ROI 区域内的平均空间梯度（纹理丰富度）来量化几何可观测性
        S_geo_list = []
        contact_ratios = []
        
        for k in range(min_len):
            curr_idx = step + trim + 1 + k
            if curr_idx >= len(images): break
            img_curr = images[curr_idx]
            gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_RGB2GRAY) if img_curr.ndim == 3 else img_curr
            
            # 模糊去噪
            gray_blur = cv2.GaussianBlur(gray_curr, (7, 7), 0)
            gray_cp = cp.array(gray_blur, dtype=cp.float64)
            W_cp = perception.get_confidence_mask_gpu(gray_cp)
            
            # 计算声学窗口比例
            mask_roi = W_cp > 0.5
            contact_ratios.append(float(cp.mean(mask_roi)))
            
            # 计算几何观测梯度 (Sobel 能量)
            grad_x = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = cp.array(np.sqrt(grad_x**2 + grad_y**2))
            
            if cp.any(mask_roi):
                s_geo_val = float(cp.mean(grad_mag[mask_roi]))
            else:
                s_geo_val = 0.0
            S_geo_list.append(s_geo_val)
            
        N_valid = len(S_geo_list)
        contact_arr = np.array(contact_ratios)
        S_geo_arr = np.array(S_geo_list)
        
        max_contact = np.max(contact_arr)
        if max_contact < 0.1: return None

        calib_threshold = max_contact * 0.85
        calib_idx = np.where(contact_arr > calib_threshold)[0][:150]
        if len(calib_idx) < 30: return None
            
        # --- 提取物理之眼 (Physical Risk Residual) ---
        X_norm = StandardScaler().fit_transform(xi_trim[:N_valid, 2].reshape(-1, 1))
        Y_norm = StandardScaler().fit_transform(s_dot_trim[:N_valid, 2].reshape(-1, 1))
        
        model = Ridge(alpha=1.0).fit(X_norm[calib_idx], Y_norm[calib_idx])
        R_phys_raw = np.abs(Y_norm - model.predict(X_norm)).flatten()
        sigma_calib = np.std(R_phys_raw[calib_idx]) + 1e-6
        R_phys_sigma = R_phys_raw / sigma_calib 
        
        return {
            "R_phys": R_phys_sigma,
            "S_geo": S_geo_arr,
            "contact": contact_arr
        }
    except Exception:
        return None

def run_decoupling_analysis():
    print("🚀 开始抽取数据验证双轨正交解耦 (Dual-Eye Orthogonal Decoupling)...")
    config_path = "configs/default_config.yaml"
    cfg = IDUQConfig.from_yaml(config_path)
    out_dir = os.path.join(cfg.io.get('output_dir', 'Results'), 'EXP6_Dual_Eye')
    os.makedirs(out_dir, exist_ok=True)
    
    root = safe_open_zarr(cfg.io['data_path'])
    # 我们不需要跑全部，抽取 20 个复杂的 Episode 就足以绘制致密的散点图
    episodes = sorted(list(root.group_keys()))[:20] 
    
    all_R = []
    all_S = []
    
    process_func = partial(evaluate_dual_eyes, config_path=config_path)
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        for res in tqdm(executor.map(process_func, episodes), total=len(episodes), desc="Extracting Eyes"):
            if res is not None:
                all_R.extend(res['R_phys'])
                all_S.extend(res['S_geo'])
                
    if len(all_R) == 0:
        print("💥 提取失败。")
        return
        
    R_arr = np.array(all_R)
    S_arr = np.array(all_S)
    
    # 归一化几何之眼到 [0, 1] 区间以便展示
    S_arr = MinMaxScaler().fit_transform(S_arr.reshape(-1, 1)).flatten()
    
    # ==========================================
    # 顶级学术绘图：四象限工作包络图 (Phase Plot)
    # ==========================================
    print("📊 正在绘制双轨正交解耦四象限图...")
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 9))
    
    # 使用 KDE (核密度估计) 绘制底层热力图，展示数据分布中心
    sns.kdeplot(x=S_arr, y=R_arr, fill=True, cmap="Blues", alpha=0.6, thresh=0.05, ax=ax)
    # 叠加散点图展示长尾和离群点 (异常)
    ax.scatter(S_arr, R_arr, color='navy', alpha=0.1, s=10)
    
    # 定义十字切割线 (Thresholds)
    # 物理风险大于 4.0 Sigma 视为异常
    phys_thresh = 4.0 
    # 几何观测度低于 0.4 视为阴影/特征缺失
    geo_thresh = 0.4
    
    ax.axhline(y=phys_thresh, color='crimson', linestyle='--', linewidth=2)
    ax.axvline(x=geo_thresh, color='darkorange', linestyle='--', linewidth=2)
    
    # 填充四象限背景色
    ax.axhspan(phys_thresh, max(10, np.max(R_arr)), xmin=geo_thresh, xmax=1, color='salmon', alpha=0.15) # Q1
    ax.axhspan(phys_thresh, max(10, np.max(R_arr)), xmin=0, xmax=geo_thresh, color='gray', alpha=0.15)    # Q2
    ax.axhspan(0, phys_thresh, xmin=0, xmax=geo_thresh, color='gold', alpha=0.15)                         # Q3
    ax.axhspan(0, phys_thresh, xmin=geo_thresh, xmax=1, color='lightgreen', alpha=0.15)                   # Q4
    
    # 标注四大物理状态
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.75, phys_thresh + 1.5, "Quadrant I\nPure Sliding / Kinematic Mismatch\n(High Risk, Clear Image)", 
            fontsize=11, fontweight='bold', ha='center', va='center', bbox=props, color='crimson')
            
    ax.text(0.20, phys_thresh + 1.5, "Quadrant II\nSevere Decoupling\n(High Risk, Air Gap)", 
            fontsize=11, fontweight='bold', ha='center', va='center', bbox=props, color='dimgray')
            
    ax.text(0.20, phys_thresh / 2, "Quadrant III\nAcoustic Shadowing (Ribs/Gas)\n(Safe Press, Poor Visibility)", 
            fontsize=11, fontweight='bold', ha='center', va='center', bbox=props, color='darkgoldenrod')
            
    ax.text(0.75, phys_thresh / 2, "Quadrant IV\nIdeal Servoing Envelope\n(Safe Press, Clear Tissue)", 
            fontsize=11, fontweight='bold', ha='center', va='center', bbox=props, color='forestgreen')
    
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, max(8, np.percentile(R_arr, 99.5))) # 限制Y轴高度防止被极端值拉坏
    
    ax.set_xlabel('Geometric Eye $\mathcal{S}_{geo}$ (Normalized Acoustic Observability)', fontsize=14)
    ax.set_ylabel('Physical Eye $\mathcal{R}_{phys}$ (Kinematic-Affine Residual Z-Score)', fontsize=14)
    ax.set_title('Dual-Track Decoupling: Orthogonal Uncertainty Phase Space', fontsize=16, fontweight='bold', pad=15)
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, 'Orthogonal_Decoupling_Phase_Plot.png')
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"🎉 成功！四象限工作包络图已保存至: {save_path}")

if __name__ == "__main__":
    run_decoupling_analysis()