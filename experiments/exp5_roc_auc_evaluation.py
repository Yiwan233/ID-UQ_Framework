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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
    """
    独立子进程：处理单个 Episode，自动提取标定区和 Ground Truth
    """
    cv2.setNumThreads(1) # 防止 OpenCV 线程灾难
    
    try:
        cfg = IDUQConfig.from_yaml(config_path)
        root = safe_open_zarr(cfg.io['data_path'])
        perception = PhysicsAwarePerception(cfg)
        
        images, poses = get_episode_data(root, ep_id)
        if len(images) < 100: 
            return {"Episode": ep_id, "Error": "Sequence too short"}
            
        step = cfg.perception.get('step', 1)
        trim = cfg.perception.get('trim_edge', 20)
        
        # 1. 核心 GPU 物理特征提取
        xi_trim, s_dot_trim = perception.process_episode(images, poses)
        N_valid = len(xi_trim)
        
        # 2. 时序严格对齐的 SSIM 与掩膜质量提取
        ssim_list = []
        contact_ratios = []
        
        for k in range(N_valid):
            curr_idx = step + trim + k
            prev_idx = curr_idx - step
            
            img_curr = images[curr_idx]
            img_prev = images[prev_idx]
            
            gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_RGB2GRAY) if img_curr.ndim == 3 else img_curr
            gray_prev = cv2.cvtColor(img_prev, cv2.COLOR_RGB2GRAY) if img_prev.ndim == 3 else img_prev
            
            # 记录 Baseline: 纯图像 SSIM
            ssim_list.append(ssim(gray_prev, gray_curr, data_range=255))
            
            # 计算掩膜接触质量 (用于自动打标)
            gray_cp = cp.array(gray_curr, dtype=cp.float64)
            W_cp = perception.get_confidence_mask_gpu(gray_cp)
            contact_ratios.append(float(cp.mean(W_cp > 0.5)))
            
        ssim_arr = np.array(ssim_list)
        contact_arr = np.array(contact_ratios)
        
        # 3. 物理弱监督：自动寻找 Calibration Zone
        # 选取接触质量好的帧作为标定基础
        good_contact_idx = np.where(contact_arr > 0.6)[0]
        if len(good_contact_idx) < 50:
            return {"Episode": ep_id, "Error": "Not enough nominal contact frames for calibration."}
            
        # 使用前 150 个优质接触帧作为标定
        calib_idx = good_contact_idx[:150]
        
        vz = xi_trim[:, 2].reshape(-1, 1)
        div = s_dot_trim[:, 2].reshape(-1, 1)
        
        X_norm = StandardScaler().fit_transform(vz)
        Y_norm = StandardScaler().fit_transform(div)
        
        # 训练标定模型
        model = Ridge(alpha=1.0)
        model.fit(X_norm[calib_idx], Y_norm[calib_idx])
        
        # 4. 计算交互风险残差 R_phys
        R_phys = np.abs(Y_norm - model.predict(X_norm)).flatten()
        # 归一化残差以便在全局数据集上统一阈值
        R_phys = MinMaxScaler().fit_transform(R_phys.reshape(-1, 1)).flatten()
        
        # 5. 自动生成 Ground Truth: 当接触掩膜面积跌破 30% 视为发生严重滑脱异常
        y_true = (contact_arr < 0.3).astype(int)
        
        return {
            "Episode": ep_id,
            "R_phys": R_phys,
            "SSIM": ssim_arr,
            "y_true": y_true
        }
        
    except Exception as e:
        return {"Episode": ep_id, "Error": f"{str(e)}\n{traceback.format_exc()}"}

def run_roc_analysis():
    print("🚀 开始 300+ 序列全量评估 (Physical Weak Supervision Auto-Labeling)...")
    config_path = "configs/default_config.yaml"
    cfg = IDUQConfig.from_yaml(config_path)
    out_dir = cfg.io.get('output_dir_exp5', 'Results_EXP5_ROC_AUC')
    os.makedirs(out_dir, exist_ok=True)
    
    root = safe_open_zarr(cfg.io['data_path'])
    episodes = sorted(list(root.group_keys()))
    
    # 控制多进程数量，防止显存与 PCIe 爆炸
    max_workers = 6 
    process_func = partial(evaluate_single_episode, config_path=config_path)
    
    all_R_phys, all_ssim, all_y_true = [], [], []
    valid_episodes = 0
    errors = []

    print(f"⚡ 启动并行计算池 (Workers: {max_workers})...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in tqdm(executor.map(process_func, episodes), total=len(episodes), desc="Evaluating"):
            if "Error" not in res:
                all_R_phys.extend(res['R_phys'])
                all_ssim.extend(res['SSIM'])
                all_y_true.extend(res['y_true'])
                valid_episodes += 1
            else:
                errors.append(res['Episode'])
                # tqdm.write(f"⚠️ 跳过 {res['Episode']}: {res['Error'].splitlines()[0]}")

    if len(all_y_true) == 0:
        print("\n💥 灾难性错误：所有数据提取失败。")
        return
        
    print(f"\n✅ 数据提取完毕！成功处理了 {valid_episodes} 个 Episodes，共获取了 {len(all_y_true)} 帧数据样本。")
    if errors:
        print(f"⚠️ 有 {len(errors)} 个序列由于数据过短或无合法接触区间被跳过。")

    # ==========================================
    # 计算全局 ROC 与 AUC
    # ==========================================
    print("📊 正在计算全局 ROC 曲线...")
    y_true_global = np.array(all_y_true)
    R_phys_global = np.array(all_R_phys)
    ssim_global = np.array(all_ssim)
    
    # 过滤掉全是 0 或全是 1 的极端情况防报错
    if len(np.unique(y_true_global)) < 2:
        print("💥 数据集中没有检测到异常标签 (或全是异常)，无法绘制 ROC。请检查接触掩膜的阈值。")
        return

    # Baseline: SSIM (负相关，SSIM 越低越可能是异常)
    fpr_base, tpr_base, _ = roc_curve(y_true_global, -ssim_global)
    auc_base = auc(fpr_base, tpr_base)
    
    # Ours: R_phys (正相关，残差越大越异常)
    fpr_ours, tpr_ours, thresholds = roc_curve(y_true_global, R_phys_global)
    auc_ours = auc(fpr_ours, tpr_ours)
    optimal_idx = np.argmax(tpr_ours - fpr_ours)
    
    # ==========================================
    # 顶级学术绘图
    # ==========================================
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(10, 8))
    
    plt.plot(fpr_base, tpr_base, color='gray', linestyle='--', lw=2.5, 
             label=f'Baseline (Pure Image SSIM) AUC = {auc_base:.3f}')
             
    plt.plot(fpr_ours, tpr_ours, color='crimson', lw=3, 
             label=fr'Proposed ($\mathcal{{R}}_{{phys}}$) AUC = {auc_ours:.3f}')
             
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle=':')
    plt.fill_between(fpr_ours, tpr_ours, alpha=0.15, color='crimson')
    
    # 标记最佳操作阈值点
    plt.scatter(fpr_ours[optimal_idx], tpr_ours[optimal_idx], marker='*', color='gold', s=250, edgecolor='black', zorder=5, 
                label=f'Optimal Trigger Point ($R_{{phys}} \geq {thresholds[optimal_idx]:.2f}$)')
    
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR) - [False Alarms]', fontsize=14)
    plt.ylabel('True Positive Rate (TPR) - [Successful Detections]', fontsize=14)
    plt.title(f'Global ROC Evaluation across {valid_episodes} Unannotated Sequences', fontsize=16, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=13)
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, 'Global_ROC_AUC_Evaluation.png')
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    
    print(f"\n🎉 惊人的结果！")
    print(f"👉 Baseline (纯视觉 SSIM) AUC: {auc_base:.3f}")
    print(f"👉 Ours (物理驱动 R_phys) AUC: {auc_ours:.3f}")
    print(f"🚀 图表已保存至: {save_path}")

if __name__ == "__main__":
    run_roc_analysis()