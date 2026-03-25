# experiments/exp4_kinematic_residual.py

import os
import zarr
import numpy as np
import cv2
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter

from core.config_loader import IDUQConfig
from core.perception import PhysicsAwarePerception

def run_residual_analysis():
    print(f"🚀 开始运动学-视觉残差分析 (已启用 3σ 置信管可视化)...")
    cfg = IDUQConfig.from_yaml("configs/default_config.yaml")
    out_dir = cfg.io.get('output_dir_exp4', 'Results_EXP4_Residual')
    os.makedirs(out_dir, exist_ok=True)
    
    root = zarr.open(cfg.io['data_path'], mode='r')
    perception = PhysicsAwarePerception(cfg)
    trim = cfg.perception['trim_edge']
    train_ratio = 0.4 
    blur_kernel = (5, 5)

    for ep in root.group_keys():
        print(f"  正在拟合与评估 {ep} ...")
        images, poses = root[ep]['images'][:], root[ep]['poses'][:]
        
        # 提取速度
        xi = perception.extract_kinematics(poses)
        
        # ⚠️ 物理机制保留：采用高斯模糊以保持对高频动作的敏锐响应
        s_dot = []
        prev_blur = None
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
            curr_blur = cv2.GaussianBlur(gray, blur_kernel, 0)
            
            W = perception.get_confidence_mask(curr_blur)
            if prev_blur is not None:
                s_dot.append(perception.calculate_affine_flow(prev_blur, curr_blur, W))
            else:
                s_dot.append(np.array([0,0,0,0]))
            prev_blur = curr_blur
            
        s_dot = savgol_filter(np.array(s_dot), window_length=15, polyorder=2, axis=0)
        
        X_input = xi[trim:-trim, 2].reshape(-1, 1)  # Vz
        Y_target = s_dot[trim:-trim, 2].reshape(-1, 1) # Divergence
        
        # 数据建模与残差计算
        scaler_X, scaler_Y = StandardScaler(), StandardScaler()
        X_norm = scaler_X.fit_transform(X_input)
        Y_norm = scaler_Y.fit_transform(Y_target)
        
        split_idx = int(len(X_norm) * train_ratio)
        model = Ridge(alpha=1.0)
        model.fit(X_norm[:split_idx], Y_norm[:split_idx])
        
        Y_pred = model.predict(X_norm).flatten()
        Y_obs = Y_norm.flatten()
        R_phys = np.abs(Y_obs - Y_pred)
        
        # 3σ 置信界限
        std_calib = np.std(Y_obs[:split_idx] - Y_pred[:split_idx])
        conf_bound = 1.96 * std_calib 
        
        # ==================== 顶级学术绘图 ====================
        sns.set_theme(style="whitegrid")
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1])
        fig.suptitle(f'Interaction State Estimation & Anomaly Detection - {ep}', fontsize=18, fontweight='bold', y=0.96)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(Y_pred, label='Predicted Nominal State ($L_{emp}\cdot\\xi$)', color='darkorange', linestyle='--', linewidth=2)
        ax1.fill_between(range(len(Y_pred)), Y_pred - conf_bound, Y_pred + conf_bound, color='darkorange', alpha=0.2, label='$\pm 3\sigma_{calib}$ Tube')
        ax1.plot(Y_obs, label='Observed Affine State ($\dot{s}_{obs}$)', color='royalblue', alpha=0.85, linewidth=1.5)
        ax1.axvline(x=split_idx, color='gray', linestyle=':', label='Calibration/Test Split')
        ax1.set_title('Forward Model Tracking with Statistical Confidence Bounds', fontsize=14)
        ax1.legend(loc='upper right')
        ax1.set_xlim(0, len(Y_obs))
        
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(R_phys, label='Kinematic-Visual Residual $\mathcal{R}_{phys}$', color='crimson', linewidth=1.5)
        ax2.axhline(y=conf_bound, color='purple', linestyle='--', label='Anomaly Trigger Threshold')
        ax2.fill_between(range(len(R_phys)), R_phys, color='crimson', alpha=0.2)
        ax2.set_title('Interaction Uncertainty Metric ($\mathcal{R}_{phys}$)', fontsize=14)
        ax2.set_xlabel('Aligned Frames (Time)')
        ax2.legend(loc='upper right')
        ax2.set_xlim(0, len(R_phys))
        
        ax3 = fig.add_subplot(gs[:, 1])
        error_calib = Y_obs[:split_idx] - Y_pred[:split_idx]
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
        
    print(f"✅ 置信区间绘图完毕！请查看 {out_dir}")

if __name__ == "__main__":
    run_residual_analysis()