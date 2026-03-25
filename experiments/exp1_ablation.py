# experiments/exp1_ablation.py

import os
import zarr
import numpy as np
import cv2
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 引入我们重构好的核心引擎
from core.config_loader import IDUQConfig
from core.perception import PhysicsAwarePerception

def compute_best_correlation(sig1: np.ndarray, sig2: np.ndarray, max_lag: int = 25):
    """(复用你原有的滑动相关性计算逻辑)"""
    scaler = StandardScaler()
    s1_norm = scaler.fit_transform(sig1.reshape(-1, 1)).flatten()
    s2_norm = scaler.fit_transform(sig2.reshape(-1, 1)).flatten()

    best_offset, max_corr = 0, -1.0
    for offset in range(-max_lag, max_lag + 1):
        if offset > 0:
            x_test, y_test = s1_norm[offset:], s2_norm[:-offset]
        elif offset < 0:
            x_test, y_test = s1_norm[:offset], s2_norm[-offset:]
        else:
            x_test, y_test = s1_norm, s2_norm
            
        corr = abs(np.corrcoef(x_test, y_test)[0, 1])
        if corr > max_corr:
            max_corr, best_offset = corr, offset
            
    # 返回对齐后的序列
    if best_offset > 0:
        return s1_norm[best_offset:], s2_norm[:-best_offset], max_corr
    elif best_offset < 0:
        return s1_norm[:best_offset], s2_norm[-best_offset:], max_corr
    return s1_norm, s2_norm, max_corr

def run_ablation_experiment():
    print("🚀 开始运行实验一：感知层消融与 Pearson 相关性分析...")
    cfg = IDUQConfig.from_yaml("configs/default_config.yaml")
    root = zarr.open(cfg.io['data_path'], mode='r')
    
    # 获取指定的 episode
    ep = list(root.group_keys())[4] 
    images = root[ep]['images'][:]
    poses = root[ep]['poses'][:]
    
    # ==========================================
    # 1. 核心数据提取 (直接调用生产级 Core 模块)
    # ==========================================
    perception = PhysicsAwarePerception(cfg)
    # 提取 Adjoint 工具系速度和仿射流场特征
    xi_tool, s_dot = perception.process_episode(images, poses)
    robot_z_tool = xi_tool[:, 2]
    div_feat = s_dot[:, 2]
    
    # [Baseline] 提取 Base 系 Z 速度 (模拟传统的未经过 Adjoint 变换的速度)
    dt = cfg.kinematics['dt']
    pos_z_smooth = savgol_filter(poses[:, 2], window_length=31, polyorder=3)
    robot_z_base = (np.diff(pos_z_smooth) / dt)[cfg.perception['trim_edge']:-cfg.perception['trim_edge']]

    # [Baseline] 提取传统的面积特征 (Area Rate)
    area_feat = []
    for img in images[1:]:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        _, bin_img = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
        area_feat.append(-np.sum(bin_img > 0))
    area_dot = savgol_filter(np.diff(area_feat) / dt, window_length=15, polyorder=2)
    area_dot = np.append(area_dot, area_dot[-1])[cfg.perception['trim_edge']:-cfg.perception['trim_edge']]

    # ==========================================
    # 2. 相关性计算
    # ==========================================
    print("\n📊 计算消融组合相关性:")
    x1, y1, corr_1 = compute_best_correlation(robot_z_base, area_dot)
    print(f"  [Baseline 1] Base_Vel vs Area_Dot : Corr = {corr_1:.4f}")
    
    x2, y2, corr_2 = compute_best_correlation(robot_z_tool, area_dot)
    print(f"  [Baseline 2] Tool_Vel vs Area_Dot : Corr = {corr_2:.4f}")
    
    x3, y3, corr_3 = compute_best_correlation(robot_z_tool, div_feat)
    print(f"  [Ours Full]  Tool_Vel vs Div_Flow : Corr = {corr_3:.4f}")

    # ==========================================
    # 3. 顶级学术绘图
    # ==========================================
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Ablation Study: Kinematic Consistency and Feature Observability', fontsize=16, fontweight='bold', y=0.96)
    
    # Plot 1
    axes[0].plot(x1, label='Robot Z Velocity (Base)', color='dimgray', linewidth=1.5)
    axes[0].plot(y1, label='Image Area Rate', color='salmon', linestyle='--', linewidth=1.5)
    axes[0].set_title(f'Baseline 1: Base Kinematics + Geometric Feature (Pearson $r = {corr_1:.2f}$)', fontsize=13)
    axes[0].legend(loc='upper right')
    
    # Plot 2
    axes[1].plot(x2, label='Robot Z Velocity (Tool)', color='teal', linewidth=1.5)
    axes[1].plot(y2, label='Image Area Rate', color='salmon', linestyle='--', linewidth=1.5)
    axes[1].set_title(f'Baseline 2: Adjoint Kinematics + Geometric Feature (Pearson $r = {corr_2:.2f}$)', fontsize=13)
    axes[1].legend(loc='upper right')
    
    # Plot 3 (Ours)
    axes[2].plot(x3, label='Robot Z Velocity (Tool)', color='teal', linewidth=2)
    axes[2].plot(y3, label='Continuum Divergence Flow (D)', color='crimson', linestyle='-.', linewidth=2)
    axes[2].set_title(f'Ours: Physics-Aware Adjoint Kinematics + Affine Flow (Pearson $r = {corr_3:.2f}$)', fontsize=13, fontweight='bold')
    axes[2].set_facecolor('#fdf6e3') 
    axes[2].legend(loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(cfg.io['output_dir'], exist_ok=True)
    save_path = os.path.join(cfg.io['output_dir'], 'ablation_pearson_correlation.png')
    plt.savefig(save_path, dpi=300)
    print(f"✅ 可视化图表已保存为: {save_path}")

if __name__ == "__main__":
    run_ablation_experiment()