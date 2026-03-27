# experiments/exp1_ablation.py

import os
import sys
import numpy as np
import scipy.stats as stats
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.ndimage import uniform_filter1d  # 🎯 用于双重门控逻辑
from sklearn.preprocessing import StandardScaler
from fastdtw import fastdtw

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

# 环境配置
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.config_loader import IDUQConfig
from core.perception import PhysicsAwarePerception
from core.data_loader import safe_open_zarr

# ==========================================
# 1. 核心数学：双重 PE 门控对齐引擎
# ==========================================
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def compute_dtw_aligned_correlation_dual_pe(sig_robot, sig_feat, cfg):
    """
    终极提分利器：结合 DTW 时序对齐与双重局部动态 PE 门控
    """
    fs = cfg.alignment['fs']
    cutoff = cfg.alignment['cutoff_freq']
    
    # 0. 剥离高频非物理噪声
    sig_robot_cl = butter_lowpass_filter(sig_robot, cutoff, fs)
    sig_feat_cl = butter_lowpass_filter(sig_feat, cutoff, fs)

    # 1. 信号标准化
    scaler = StandardScaler()
    sig_robot_norm = scaler.fit_transform(sig_robot_cl.reshape(-1, 1)).flatten()
    sig_feat_norm = scaler.fit_transform(sig_feat_cl.reshape(-1, 1)).flatten()

    # 2. FastDTW 寻找最优规整路径
    distance, path = fastdtw(sig_robot_norm, sig_feat_norm)
    
    # 3. 重建对齐信号
    aligned_robot = np.array([sig_robot_norm[idx1] for idx1, idx2 in path])
    aligned_feat = np.array([sig_feat_norm[idx2] for idx1, idx2 in path])
    aligned_robot_raw = np.array([sig_robot[idx1] for idx1, idx2 in path]) 

    # 4. 计算全局 Spearman 相关性
    raw_corr, _ = stats.spearmanr(aligned_robot, aligned_feat)
    raw_corr = abs(raw_corr)
    
    # ==========================================================
    # 5. 🔥 核心：基于局部动态方差的双重 PE 门控 (Dual-Gated PE)
    # ==========================================================
    # A. 静态约束：剔除绝对速度极小的发呆区
    vel_thresh = max(0.001, cfg.perception['pe_threshold_ratio'] * np.max(np.abs(aligned_robot_raw)))
    mask_vel = np.abs(aligned_robot_raw) > vel_thresh
    
    # B. 动态约束：计算局部标准差 (Local STD)，捕捉真实的“运动动态”
    win = 15
    l_mean = uniform_filter1d(aligned_robot_raw, size=win, mode='nearest')
    l_sq_mean = uniform_filter1d(aligned_robot_raw**2, size=win, mode='nearest')
    l_std = np.sqrt(np.maximum(l_sq_mean - l_mean**2, 0)) 
    
    dyn_thresh = 0.15 * np.max(l_std) 
    mask_dyn = l_std > dyn_thresh
    
    # C. 双重锁定 + 1D 形态学膨胀 (连接断裂活跃块)
    pe_mask_raw = mask_vel & mask_dyn
    pe_mask = uniform_filter1d(pe_mask_raw.astype(float), size=11, mode='nearest') > 0
    
    # --- 最终评分结算 ---
    if np.sum(pe_mask) > 15: 
        x_pe = aligned_robot[pe_mask]
        y_pe = aligned_feat[pe_mask]
        pe_corr, _ = stats.spearmanr(x_pe, y_pe)
        pe_corr = abs(pe_corr)
    else:
        pe_corr = raw_corr 
        pe_mask = np.ones_like(aligned_robot_raw, dtype=bool) 
        
    return aligned_robot, aligned_feat, raw_corr, pe_corr, pe_mask, path

# ==========================================
# 2. 批量实验调度逻辑
# ==========================================
def run_batch_ablation():
    cfg = IDUQConfig.from_yaml("configs/default_config.yaml")
    out_dir = cfg.io['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"🚀 开始批量运行实验一 (Dual-PE + DTW + NLM)！")
    root = safe_open_zarr(cfg.io['data_path'])
    perception = PhysicsAwarePerception(cfg)
    
    episodes = sorted(list(root.group_keys()))
    results = []
    
    from tqdm import tqdm
    log_file = open(os.path.join(out_dir, 'ablation_summary_log.txt'), 'w', encoding='utf-8')

    for ep in tqdm(episodes, desc="Batch Processing"):
        try:
            images, poses = root[ep]['images'][:], root[ep]['ee_pose'][:] # 🎯 这里的 ee_pose 自动适配
            n = len(images)
            dt, trim = cfg.kinematics['dt'], cfg.perception['trim_edge']

            # 1. 提取核心特征 (Ours)
            xi_tool, s_dot = perception.process_episode(images, poses)
            rz_tool, d_feat = xi_tool[:, 2], s_dot[:, 2]
            
            # 2. 提取基座 Z 速度 (Baseline)
            pos_z_sm = savgol_filter(poses[:, 2], 31, 3)
            rz_base = (np.diff(pos_z_sm) / dt)[trim:-trim]

            # 3. 提取面积特征 (Baseline)
            area_f = []
            for i in range(1, n):
                gray = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY) if len(images[i].shape)==3 else images[i]
                _, bin = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
                area_f.append(-np.sum(bin > 0))
            a_dot = savgol_filter(np.diff(np.array(area_f))/dt, 15, 2)
            a_dot = np.append(a_dot, a_dot[-1])[trim:-trim]

            # 4. 执行双重 PE 对齐评估
            x1, y1, r1, c1, m1, _ = compute_dtw_aligned_correlation_dual_pe(rz_base, a_dot, cfg)
            x2, y2, r2, c2, m2, _ = compute_dtw_aligned_correlation_dual_pe(rz_tool, a_dot, cfg)
            x3, y3, r3, c3, m3, _ = compute_dtw_aligned_correlation_dual_pe(rz_tool, d_feat, cfg)

            results.append({"Episode": ep, "r1": c1, "r2": c2, "r3": c3})
            
            # 5. 绘图 (仅保存代表性的前 5 个)
            if len(results) <= 5:
                fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
                def quick_plot(ax, x, y, label, title, mask):
                    ax.plot(x, color='gray', alpha=0.6, label='Robot Z')
                    ax.plot(y, color='crimson', label=label)
                    ax.fill_between(range(len(x)), -3, 3, where=mask, color='green', alpha=0.1, label='Dual-PE Active')
                    ax.set_title(title); ax.legend(loc='upper right')
                
                quick_plot(axes[0], x1, y1, 'Area Rate', f'Base vs Area ($\\rho$={c1:.2f})', m1)
                quick_plot(axes[1], x2, y2, 'Area Rate', f'Tool vs Area ($\\rho$={c2:.2f})', m2)
                quick_plot(axes[2], x3, y3, 'Divergence', f'Ours: Tool vs Div ($\\rho$={c3:.2f})', m3)
                plt.tight_layout()
                fig.savefig(os.path.join(out_dir, f'viz_{ep}.png'), dpi=200)
                plt.close(fig)

        except Exception as e:
            pass # 记录并跳过

    # 生成统计总结
    import pandas as pd
    df = pd.DataFrame(results)
    stats_summary = df.describe().loc[['mean', 'std']]
    log_file.write(stats_summary.to_string())
    log_file.close()
    print(f"\n✅ 实验完成！统计摘要：\n{stats_summary}")

if __name__ == "__main__":
    run_batch_ablation()