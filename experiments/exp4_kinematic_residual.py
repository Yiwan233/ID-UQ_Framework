# experiments/exp4_kinematic_residual.py

import os
import sys
import numpy as np
import scipy.stats as stats
import traceback
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# 引入核心库
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.config_loader import IDUQConfig
from core.perception import PhysicsAwarePerception
from core.data_loader import safe_open_zarr, get_episode_data

def run_batch_residual_analysis_gpu():
    print(f"🚀 开始运动学-视觉残差分析 (GPU 加速 + 3σ 置信管可视化)...")
    cfg = IDUQConfig.from_yaml("configs/default_config.yaml")
    out_dir = os.path.join(cfg.io['output_dir'], 'EXP4_Residual_GPU')
    os.makedirs(out_dir, exist_ok=True)
    
    root = safe_open_zarr(cfg.io['data_path'])
    perception = PhysicsAwarePerception(cfg) # 实例化 GPU 感知器
    
    train_ratio = 0.4 
    episodes = sorted(list(root.group_keys()))
    
    for ep in tqdm(episodes, desc="Analyzing Residuals"):
        try:
            images, poses = get_episode_data(root, ep)
            if len(images) < 50: continue
            
            # 🚀 一行代码，调用 GPU 加速的核心特征提取 (内置了 NLM + 轻微平滑)
            xi_tool, s_dot = perception.process_episode(images, poses)
            
            # 我们只看 Z 轴和散度
            X_input = xi_tool[:, 2].reshape(-1, 1) 
            Y_target = s_dot[:, 2].reshape(-1, 1) 
            
            # --- 数据建模与残差计算 (与你原来完全一致) ---
            scaler_X, scaler_Y = StandardScaler(), StandardScaler()
            X_norm = scaler_X.fit_transform(X_input)
            Y_norm = scaler_Y.fit_transform(Y_target)
            
            split_idx = int(len(X_norm) * train_ratio)
            model = Ridge(alpha=1.0)
            model.fit(X_norm[:split_idx], Y_norm[:split_idx])
            
            Y_pred = model.predict(X_norm).flatten()
            Y_obs = Y_norm.flatten()
            
            # --- 3σ 置信界限 (与你原来完全一致) ---
            error_calib = Y_obs[:split_idx] - Y_pred[:split_idx]
            std_calib = np.std(error_calib)  
            confidence_bound = 1.96 * std_calib # 95% 置信区间
            
            R_phys = np.abs(Y_obs - Y_pred)

            # --- 顶级学术绘图 (与你原来完全一致) ---
            sns.set_theme(style="whitegrid")
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1])
            fig.suptitle(f'Interaction State Estimation & Anomaly Detection - {ep}', fontsize=18, fontweight='bold', y=0.96)
            
            # 图 1: 置信管
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(Y_pred, label='Predicted Nominal State ($L_{emp}\\cdot\\xi$)', color='darkorange', linestyle='--', linewidth=2)
            ax1.fill_between(range(len(Y_pred)), Y_pred - confidence_bound, Y_pred + confidence_bound, color='darkorange', alpha=0.2, label='$\pm 2\sigma_{calib}$ Tube')
            ax1.plot(Y_obs, label='Observed Affine State ($\dot{s}_{obs}$)', color='royalblue', alpha=0.85, linewidth=1.5)
            ax1.axvline(x=split_idx, color='gray', linestyle=':', label='Calibration/Test Split')
            ax1.set_title('Forward Model Tracking with Statistical Confidence Bounds', fontsize=14)
            ax1.legend(loc='upper right')
            
            # 图 2: 残差
            ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
            ax2.plot(R_phys, label='Kinematic-Visual Residual $\mathcal{R}_{phys}$', color='crimson', linewidth=1.5)
            ax2.axhline(y=confidence_bound, color='purple', linestyle='--', label='Anomaly Trigger Threshold')
            ax2.fill_between(range(len(R_phys)), R_phys, color='crimson', alpha=0.2)
            ax2.set_title('Interaction Uncertainty Metric ($\mathcal{R}_{phys}$)', fontsize=14)
            ax2.set_xlabel('Aligned Frames (Time)')
            ax2.legend(loc='upper right')
            
            # 图 3: 概率分布
            ax3 = fig.add_subplot(gs[:, 1])
            error_test = Y_obs[split_idx:] - Y_pred[split_idx:]
            sns.histplot(error_calib, kde=True, stat="density", color='lightgreen', label='Calib ($H_0$: In-Dist)', ax=ax3, alpha=0.5)
            sns.histplot(error_test, kde=True, stat="density", color='salmon', label='Test ($H_1$: OOD)', ax=ax3, alpha=0.5)
            x_g = np.linspace(*ax3.get_xlim(), 100)
            ax3.plot(x_g, stats.norm.pdf(x_g, *stats.norm.fit(error_calib)), 'k', linewidth=2, label='Fitted Normal Prior')
            ax3.set_title('Error Probability Density', fontsize=14)
            ax3.legend(loc='upper right')

            plt.tight_layout(rect=[0, 0.03, 1, 0.93])
            plt.savefig(os.path.join(out_dir, f'residual_analysis_confidence_{ep}.png'), dpi=300)
            plt.close(fig)
        except Exception as e:
            print(f"❌ 处理 {ep} 时发生错误:")
            traceback.print_exc()
        
    print(f"✅ 置信区间绘图完毕！请查看 {out_dir}")

if __name__ == "__main__":
    run_batch_residual_analysis_gpu()