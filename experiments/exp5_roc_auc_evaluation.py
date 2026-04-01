# experiments/exp5_roc_auc_evaluation.py

import os
import zarr
import numpy as np
import cv2
import cupy as cp
import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import concurrent.futures
from functools import partial
import traceback

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.config_loader import IDUQConfig
from core.perception import PhysicsAwarePerception
from core.data_loader import safe_open_zarr, get_episode_data

def evaluate_single_episode(ep_id, config_path):
    cv2.setNumThreads(1) 
    try:
        cfg = IDUQConfig.from_yaml(config_path)
        root = safe_open_zarr(cfg.io['data_path'])
        perception = PhysicsAwarePerception(cfg)
        
        images, poses = get_episode_data(root, ep_id)
        if len(images) < 100: 
            return {"Episode": ep_id, "Error": "Sequence too short"}
            
        step = cfg.perception.get('step', 1)
        trim = cfg.perception.get('trim_edge', 20)
        
        # 1. 核心 GPU 物理特征提取 (散度 D 与 机器人扭矩)
        xi_trim, s_dot_trim = perception.process_episode(images, poses)
        min_len = min(len(xi_trim), len(s_dot_trim))
        xi_trim, s_dot_trim = xi_trim[:min_len], s_dot_trim[:min_len]
        
        ssim_list, contact_ratios = [], []
        
        for k in range(min_len):
            curr_idx = step + trim + 1 + k
            prev_idx = curr_idx - step
            if curr_idx >= len(images): break
                
            img_curr, img_prev = images[curr_idx], images[prev_idx]
            gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_RGB2GRAY) if img_curr.ndim == 3 else img_curr
            gray_prev = cv2.cvtColor(img_prev, cv2.COLOR_RGB2GRAY) if img_prev.ndim == 3 else img_prev
            
            ssim_list.append(ssim(gray_prev, gray_curr, data_range=255))
            
            # 使用 GPU 掩膜计算接触质量
            gray_blur = cv2.GaussianBlur(gray_curr, (7, 7), 0)
            W_cp = perception.get_confidence_mask_gpu(cp.array(gray_blur, dtype=cp.float64))
            contact_ratios.append(float(cp.mean(W_cp > 0.5)))
            
        N_valid = len(ssim_list)
        ssim_arr, contact_arr = np.array(ssim_list), np.array(contact_ratios)
        
        # 2. 动态相对阈值逻辑
        max_contact = np.max(contact_arr)
        if max_contact < 0.1:
            return {"Episode": ep_id, "Error": "Invalid scan"}

        # 标定区：最高质量的 85% 以上
        calib_threshold = max_contact * 0.85
        calib_idx = np.where(contact_arr > calib_threshold)[0][:150]
        
        if len(calib_idx) < 30: 
            return {"Episode": ep_id, "Error": "Unstable calibration"}
            
        # 3. 训练 Digital Twin 并计算 Sigma 残差
        X_norm = StandardScaler().fit_transform(xi_trim[:N_valid, 2].reshape(-1, 1))
        Y_norm = StandardScaler().fit_transform(s_dot_trim[:N_valid, 2].reshape(-1, 1))
        
        model = Ridge(alpha=1.0).fit(X_norm[calib_idx], Y_norm[calib_idx])
        R_phys_raw = np.abs(Y_norm - model.predict(X_norm)).flatten()
        sigma_calib = np.std(R_phys_raw[calib_idx]) + 1e-6
        R_phys_sigma = R_phys_raw / sigma_calib 
        
        # 🚀 提升敏感度：接触面积跌破最高值的 75% 即视为异常 (25% 的流失)
        slip_threshold = max_contact * 0.75 
        y_true = (contact_arr < slip_threshold).astype(int)
        
        return {"Episode": ep_id, "R_phys": R_phys_sigma, "SSIM": ssim_arr, "y_true": y_true}
        
    except Exception as e:
        return {"Episode": ep_id, "Error": str(e)}

def run_roc_analysis():
    print("🚀 开始 300+ 序列全量评估 (敏感度增强版)...")
    config_path = "configs/default_config.yaml"
    cfg = IDUQConfig.from_yaml(config_path)
    out_dir = os.path.join(cfg.io.get('output_dir', 'Results'), 'EXP5_ROC')
    os.makedirs(out_dir, exist_ok=True)
    
    root = safe_open_zarr(cfg.io['data_path'])
    episodes = sorted(list(root.group_keys()))
    
    all_R_phys, all_ssim, all_y_true = [], [], []
    valid_count = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        process_func = partial(evaluate_single_episode, config_path=config_path)
        for res in tqdm(executor.map(process_func, episodes), total=len(episodes), desc="Evaluating"):
            if "Error" not in res:
                all_R_phys.extend(res['R_phys'])
                all_ssim.extend(res['SSIM'])
                all_y_true.extend(res['y_true'])
                valid_count += 1
            else:
                if "too short" not in res['Error'].lower():
                    tqdm.write(f"⚠️ {res['Episode']} Skipped: {res['Error']}")

    y_true_global = np.array(all_y_true)
    R_phys_global = np.array(all_R_phys)
    ssim_global = np.array(all_ssim)

    # 📊 诊断打印
    n_pos = np.sum(y_true_global)
    n_neg = len(y_true_global) - n_pos
    print(f"\n✅ 数据提取完毕：成功处理 {valid_count} 个 Episode")
    print(f"📈 标签分布: [正常帧: {n_neg}] | [异常帧: {n_pos}]")

    if n_pos == 0 or n_neg == 0:
        print("💥 错误：依然没有检测到异常样本。请尝试进一步调高阈值（例如将 0.75 改为 0.85）。")
        return

    # ROC 计算
    fpr_base, tpr_base, _ = roc_curve(y_true_global, -ssim_global)
    fpr_ours, tpr_ours, thresholds = roc_curve(y_true_global, R_phys_global)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_base, tpr_base, color='gray', linestyle='--', label=f'Baseline (SSIM) AUC={auc(fpr_base, tpr_base):.3f}')
    plt.plot(fpr_ours, tpr_ours, color='crimson', lw=3, label=f'Ours (R_phys) AUC={auc(fpr_ours, tpr_ours):.3f}')
    plt.plot([0, 1], [0, 1], color='navy', linestyle=':')
    plt.title(f'Global ROC Evaluation ({valid_count} episodes)', fontsize=15, fontweight='bold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    
    save_path = os.path.join(out_dir, 'Global_ROC_AUC_Sensitivity_Enhanced.png')
    plt.savefig(save_path, dpi=300)
    print(f"🎉 评估完成！终极图表已保存至: {save_path}")

if __name__ == "__main__":
    run_roc_analysis()