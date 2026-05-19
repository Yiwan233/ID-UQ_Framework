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
        
        S_geo_list = []
        contact_ratios = []
        
        for k in range(min_len):
            curr_idx = step + trim + 1 + k
            if curr_idx >= len(images): break
            img_curr = images[curr_idx]
            gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_RGB2GRAY) if img_curr.ndim == 3 else img_curr
            
            gray_blur = cv2.GaussianBlur(gray_curr, (7, 7), 0)
            gray_cp = cp.array(gray_blur, dtype=cp.float64)
            W_cp = perception.get_confidence_mask_gpu(gray_cp)
            
            mask_roi = W_cp > 0.5
            contact_ratios.append(float(cp.mean(mask_roi)))
            
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
            
        X_norm = StandardScaler().fit_transform(xi_trim[:N_valid, 2].reshape(-1, 1))
        Y_norm = StandardScaler().fit_transform(s_dot_trim[:N_valid, 2].reshape(-1, 1))
        
        model = Ridge(alpha=1.0).fit(X_norm[calib_idx], Y_norm[calib_idx])
        
        # 🚀 致命 Bug 修复：强制将两者拉平成 1D，彻底切断 Numpy 生成 NxN 广播矩阵的可能！
        Y_flat = Y_norm.flatten()
        pred_flat = model.predict(X_norm).flatten()
        
        R_phys_raw = np.abs(Y_flat - pred_flat)
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
    episodes = sorted(list(root.group_keys()))[:25] # 抽取 25 个
    
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
    
    # 🛡️ 绘图前的终极防御：剔除 NaN 和 Inf 数据，防止 Seaborn 内核崩溃
    valid_mask = np.isfinite(R_arr) & np.isfinite(S_arr)
    R_arr = R_arr[valid_mask]
    S_arr = S_arr[valid_mask]
    
    # 归一化几何之眼到 [0, 1] 区间
    S_arr = MinMaxScaler().fit_transform(S_arr.reshape(-1, 1)).flatten()
    
    print(f"📊 正在绘制双轨正交解耦四象限图 (共 {len(R_arr)} 帧数据)...")
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 9))
    
    sns.kdeplot(x=S_arr, y=R_arr, fill=True, cmap="Blues", alpha=0.6, thresh=0.05, ax=ax)
    ax.scatter(S_arr, R_arr, color='navy', alpha=0.1, s=10)
    
    phys_thresh = 4.0 
    geo_thresh = 0.4
    
    ax.axhline(y=phys_thresh, color='crimson', linestyle='--', linewidth=2)
    ax.axvline(x=geo_thresh, color='darkorange', linestyle='--', linewidth=2)
    
    # 获取 R_arr 的 99% 分位数作为顶部高度限制
    y_max_plot = max(10, np.percentile(R_arr, 99.5))
    
    ax.axhspan(phys_thresh, y_max_plot * 1.5, xmin=geo_thresh, xmax=1, color='salmon', alpha=0.15) # Q1
    ax.axhspan(phys_thresh, y_max_plot * 1.5, xmin=0, xmax=geo_thresh, color='gray', alpha=0.15)   # Q2
    ax.axhspan(0, phys_thresh, xmin=0, xmax=geo_thresh, color='gold', alpha=0.15)                # Q3
    ax.axhspan(0, phys_thresh, xmin=geo_thresh, xmax=1, color='lightgreen', alpha=0.15)          # Q4
    
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
    ax.set_ylim(0, y_max_plot)
    
    ax.set_xlabel('Geometric Eye $\\mathcal{S}_{geo}$ (Normalized Acoustic Observability)', fontsize=14)
    ax.set_ylabel('Physical Eye $\\mathcal{R}_{phys}$ (Kinematic-Affine Residual Z-Score)', fontsize=14)
    ax.set_title('Dual-Track Decoupling: Orthogonal Uncertainty Phase Space', fontsize=16, fontweight='bold', pad=15)
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, 'Orthogonal_Decoupling_Phase_Plot.png')
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"🎉 成功！四象限工作包络图已保存至: {save_path}")

if __name__ == "__main__":
    run_decoupling_analysis()
