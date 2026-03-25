# experiments/exp2_anomaly_roc.py

import os
import zarr
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from skimage.metrics import structural_similarity as ssim

from core.config_loader import IDUQConfig
from core.perception import PhysicsAwarePerception

def run_anomaly_detection():
    print("🚀 正在载入数据进行异常检测实验 (EXP2)...")
    cfg = IDUQConfig.from_yaml("configs/default_config.yaml")
    root = zarr.open(cfg.io['data_path'], mode='r')
    
    # 获取特定配置
    train_ep = cfg.io.get('train_ep', 'episode_5')
    test_ep = cfg.io.get('test_ep', 'episode_7')
    gamma = cfg.alignment.get('gamma', 15.0)
    trim = cfg.perception['trim_edge']
    
    perception = PhysicsAwarePerception(cfg)
    
    # ==========================================
    # Helper: 提取特征与 SSIM
    # ==========================================
    def get_features_and_ssim(ep_id):
        images, poses = root[ep_id]['images'][:], root[ep_id]['poses'][:]
        xi = perception.extract_kinematics(poses)
        
        s_dot, ssim_list = [], []
        prev_gray = None
        for img in images:
            curr_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
            if prev_gray is not None:
                W = perception.get_confidence_mask(curr_gray)
                s_dot.append(perception.calculate_affine_flow(prev_gray, curr_gray, W))
                ssim_list.append(ssim(prev_gray, curr_gray, data_range=255))
            prev_gray = curr_gray
            
        s_dot = np.array(s_dot)
        ssim_list = np.array(ssim_list)
        return xi[trim:-trim], s_dot[trim:-trim], ssim_list[trim:-trim]

    # ----------------------------------------
    # Step 1: 在 Golden Episode 训练 Nominal Model
    # ----------------------------------------
    print(f"📚 使用 {train_ep} 训练物理名义模型...")
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
    y_true_anomaly = np.zeros(len(X_test))
    anomaly_start, anomaly_end = 95, 105
    if anomaly_end < len(y_true_anomaly):
        y_true_anomaly[anomaly_start:anomaly_end] = 1
    else:
        y_true_anomaly[-10:] = 1 

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
    R_phys_norm = (R_phys / np.max(R_phys)) * 0.4 + 0.5 
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
    fig.savefig(os.path.join(out_dir, 'anomaly_detection_roc.png'), dpi=300)
    print("✅ 异常检测结果已保存！")

if __name__ == "__main__":
    run_anomaly_detection()