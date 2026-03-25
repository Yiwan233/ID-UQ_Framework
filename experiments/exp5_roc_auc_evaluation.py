# experiments/exp5_roc_auc_evaluation.py

import os
import zarr
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, auc
from scipy.signal import savgol_filter

from core.config_loader import IDUQConfig
from core.perception import PhysicsAwarePerception

CALIB_ZONES = {
    'episode_2': (220, 450),
    'episode_6': (50, 300),
    'episode_9': (400, 600)
}

GT_INTERVALS = {
    'episode_2': [(50, 180), (720, 780)], 
    'episode_6': [(400, 500)],            
    'episode_9': [(100, 300)]             
}

def run_roc_analysis():
    print("🚀 开始使用纯净标定区间计算全局 ROC 曲线...")
    cfg = IDUQConfig.from_yaml("configs/default_config.yaml")
    out_dir = cfg.io.get('output_dir_exp5', 'Results_EXP5_ROC_AUC')
    os.makedirs(out_dir, exist_ok=True)
    
    root = zarr.open(cfg.io['data_path'], mode='r')
    perception = PhysicsAwarePerception(cfg)
    trim = cfg.perception['trim_edge']
    
    episodes = [ep for ep in root.group_keys() if ep in GT_INTERVALS and ep in CALIB_ZONES] 
    all_R_phys, all_y_true = [], []
    
    for ep in episodes:
        print(f"  正在标定并评估序列 {ep} ...")
        images, poses = root[ep]['images'][:], root[ep]['poses'][:]
        
        xi = perception.extract_kinematics(poses)
        
        # 使用高频响应滤波提取散度
        s_dot = []
        prev_blur = None
        for img in images:
            curr_blur = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img, (5, 5), 0)
            W = perception.get_confidence_mask(curr_blur)
            if prev_blur is not None:
                s_dot.append(perception.calculate_affine_flow(prev_blur, curr_blur, W)[2]) # 仅取 Divergence
            else:
                s_dot.append(0.0)
            prev_blur = curr_blur
            
        div_flow = savgol_filter(s_dot, window_length=15, polyorder=2)[trim:-trim]
        vz = xi[trim:-trim, 2]
        
        # 标准化与模型拟合
        X_norm = StandardScaler().fit_transform(vz.reshape(-1, 1))
        Y_norm = StandardScaler().fit_transform(div_flow.reshape(-1, 1))
        
        c_start, c_end = max(0, CALIB_ZONES[ep][0] - trim), min(len(X_norm), CALIB_ZONES[ep][1] - trim)
        model = Ridge(alpha=1.0)
        model.fit(X_norm[c_start:c_end], Y_norm[c_start:c_end])
        
        R_phys = np.abs(Y_norm - model.predict(X_norm)).flatten()
        R_phys = MinMaxScaler().fit_transform(R_phys.reshape(-1, 1)).flatten()
        
        y_true = np.zeros(len(R_phys), dtype=int)
        for start, end in GT_INTERVALS[ep]:
            y_true[max(0, start - trim) : min(len(R_phys), end - trim)] = 1 
            
        all_R_phys.extend(R_phys)
        all_y_true.extend(y_true)
        
    # 计算 ROC
    fpr, tpr, thresholds = roc_curve(all_y_true, all_R_phys)
    roc_auc = auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    
    # 绘图
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(9, 8))
    plt.plot(fpr, tpr, color='crimson', lw=2.5, label=f'Physics-Aware Div Flow (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='*', color='gold', s=200, edgecolor='black', zorder=5, 
                label=f'Optimal Threshold ($R_{{phys}} \geq {thresholds[optimal_idx]:.2f}$)')
    
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=14)
    plt.ylabel('True Positive Rate (TPR)', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) for Contact Failure Detection', fontsize=16, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'Global_ROC_Curve_Clean.png'), dpi=300)
    plt.close(fig)
    print(f"\n🎉 全局 ROC 评估完成！终极 AUC 分数: {roc_auc:.3f}")

if __name__ == "__main__":
    run_roc_analysis()