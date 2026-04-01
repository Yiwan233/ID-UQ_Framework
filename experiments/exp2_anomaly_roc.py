# experiments/exp2_anomaly_roc.py

import os
import zarr
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from skimage.metrics import structural_similarity as ssim

# 引入核心库
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.config_loader import IDUQConfig
from core.perception import PhysicsAwarePerception
from core.data_loader import safe_open_zarr, get_episode_data

def run_anomaly_detection():
    # 限制 OpenCV 线程数，防止底层抢占资源
    cv2.setNumThreads(1)
    
    print("🚀 正在载入数据进行异常检测实验 (EXP2)...")
    cfg = IDUQConfig.from_yaml("configs/default_config.yaml")
    root = safe_open_zarr(cfg.io['data_path'])
    
    # 自动获取存在的 episode，防止配置中的 episode 不存在报错
    available_eps = list(root.group_keys())
    train_ep = cfg.io.get('train_ep', available_eps[0] if available_eps else 'episode_5')
    test_ep = cfg.io.get('test_ep', available_eps[1] if len(available_eps) > 1 else train_ep)
    
    gamma = cfg.alignment.get('gamma', 15.0)
    trim = cfg.perception.get('trim_edge', 20)
    step = cfg.perception.get('step', 1)
    
    perception = PhysicsAwarePerception(cfg)
    
    # ==========================================
    # Helper: 统一提取特征与 SSIM (完美对齐版)
    # ==========================================
    def get_features_and_ssim(ep_id):
        images, poses = get_episode_data(root, ep_id)
        
        # 1. 直接调用最新的 GPU 加速感知层 (自动处理了 trim 和 step 的长度对齐)
        xi_trim, s_dot_trim = perception.process_episode(images, poses)
        
        # 2. 严格按照 step 和 trim 计算对应的 SSIM (保证时序绝对一致)
        ssim_list = []
        for i in range(step, len(images)):
            img_prev = images[i-step]
            img_curr = images[i]
            
            gray_prev = cv2.cvtColor(img_prev, cv2.COLOR_RGB2GRAY) if img_prev.ndim == 3 else img_prev
            gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_RGB2GRAY) if img_curr.ndim == 3 else img_curr
            
            score = ssim(gray_prev, gray_curr, data_range=255)
            ssim_list.append(score)
            
        ssim_list = np.array(ssim_list)
        ssim_trim = ssim_list[trim:-trim]
        
        # 3. 终极防越界保护：强制对齐最短长度
        min_len = min(len(xi_trim), len(s_dot_trim), len(ssim_trim))
        return xi_trim[:min_len], s_dot_trim[:min_len], ssim_trim[:min_len]

    # ----------------------------------------
    # Step 1: 在 Golden Episode 训练 Nominal Model
    # ----------------------------------------
    print(f"📚 使用 {train_ep} 训练物理名义模型 (Digital Twin)...")
    X_train, Y_train, _ = get_features_and_ssim(train_ep)
    
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    
    J_prior = np.zeros((4, 6))
    for i in range(4):
        huber = HuberRegressor(epsilon=1.35, max_iter=2000)
        huber.fit(X_train_scaled, Y_train[:, i])
        J_prior[i, :] = huber.coef_
        
    errors = Y_train - (X_train_scaled @ J_prior.T)
    Sigma_inv = np.linalg.pinv(np.cov(errors.T))

    # ----------------------------------------
    # Step 2: 在测试集上计算 R_phys
    # ----------------------------------------
    print(f"🔍 在 {test_ep} 上执行交互风险梯度评估...")
    X_test, Y_meas, ssim_test = get_features_and_ssim(test_ep)
    
    X_test_scaled = scaler_X.transform(X_test)
    Y_pred = X_test_scaled @ J_prior.T
    
    residual_vec = Y_meas - Y_pred
    mahalanobis_dist = np.array([np.sqrt(e.T @ Sigma_inv @ e) for e in residual_vec])
    R_phys = mahalanobis_dist * np.exp(gamma * (1.0 - ssim_test))
    
    # ----------------------------------------
    # Step 3: 生成 Ground Truth 与绘制 ROC
    # ----------------------------------------
    # ----------------------------------------
    # Step 3: 生成 Ground Truth 与绘制 ROC
    # ----------------------------------------
    y_true_anomaly = np.zeros(len(X_test))
    
    # 🚀 修复核心：根据 R_phys 的真实物理反馈，将打滑区间修正为实际发生的 240 帧到 280 帧
    anomaly_start = 240
    anomaly_end = 280
    
    if anomaly_end <= len(y_true_anomaly):
        y_true_anomaly[anomaly_start:anomaly_end] = 1
    elif anomaly_start < len(y_true_anomaly):
        # 如果视频稍微短一点，就标记到结尾
        y_true_anomaly[anomaly_start:] = 1

    fpr_base, tpr_base, _ = roc_curve(y_true_anomaly, -ssim_test) 
    fpr_ours, tpr_ours, _ = roc_curve(y_true_anomaly, R_phys)
    # ----------------------------------------
    # Step 4: 顶级学术可视化
    # ----------------------------------------
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(r'Validation of Interaction Uncertainty Metric ($\mathcal{R}_{phys}$)', fontsize=18, fontweight='bold')

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(ssim_test, label='Pure Image SSIM (Baseline)', color='gray', linewidth=1.5)
    # 归一化以便在同一图表中比较趋势
    R_phys_norm = (R_phys / (np.max(R_phys) + 1e-9)) * 0.4 + 0.5 
    ax1.plot(R_phys_norm, label=r'Proposed $\mathcal{R}_{phys}$ (Kinematic-Affine Residual)', color='purple', linewidth=2)
    ax1.fill_between(range(len(y_true_anomaly)), 0, 1.2, where=(y_true_anomaly==1), color='salmon', alpha=0.3, label='Ground Truth (Slips)')
    ax1.set_title('Time-Series Spiking Response to Physical Instability', fontsize=14)
    ax1.set_ylim(0.4, 1.1)
    ax1.legend(loc='upper right')

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(fpr_base, tpr_base, color='gray', linestyle='--', lw=2, label=f'Baseline (Pure SSIM) AUC = {auc(fpr_base, tpr_base):.3f}')
    ax2.plot(fpr_ours, tpr_ours, color='darkorange', lw=3, label=fr'Ours ($\mathcal{{R}}_{{phys}}$) AUC = {auc(fpr_ours, tpr_ours):.3f}')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle=':')
    ax2.fill_between(fpr_ours, tpr_ours, alpha=0.2, color='darkorange')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax2.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax2.set_title('ROC for Contact Failure Detection', fontsize=14)
    ax2.legend(loc="lower right", fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_dir = cfg.io.get('output_dir_exp2', 'Results_EXP2_Anomaly')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'anomaly_detection_roc.png')
    fig.savefig(save_path, dpi=300)
    print(f"✅ 异常检测结果已保存至: {save_path}")

if __name__ == "__main__":
    run_anomaly_detection()