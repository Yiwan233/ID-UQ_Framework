# experiments/exp1_ablation.py

import os
import sys
import numpy as np
import cv2
import pandas as pd
import scipy.stats as stats
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import StandardScaler
from fastdtw import fastdtw
import traceback
from tqdm import tqdm
import concurrent.futures
from functools import partial

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

# 环境配置
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.config_loader import IDUQConfig
from core.perception import PhysicsAwarePerception
from core.data_loader import safe_open_zarr, get_episode_data

# ==========================================
# 1. 核心数学：双重 PE 门控对齐引擎
# ==========================================
def butter_lowpass_filter(data, cutoff, fs, order=4):
    if len(data) < 15: return data
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def compute_dtw_aligned_correlation_dual_pe(sig_robot, sig_feat, cfg):
    fs = cfg.alignment['fs']
    cutoff = cfg.alignment['cutoff_freq']
    
    sig_robot_cl = butter_lowpass_filter(sig_robot, cutoff, fs)
    sig_feat_cl = butter_lowpass_filter(sig_feat, cutoff, fs)

    scaler = StandardScaler()
    sig_robot_norm = scaler.fit_transform(sig_robot_cl.reshape(-1, 1)).flatten()
    sig_feat_norm = scaler.fit_transform(sig_feat_cl.reshape(-1, 1)).flatten()

    # 🔥 诚实修正 1：限制 DTW 最大扭曲半径为 15帧(0.5秒)，防止为了拟合而强行扭曲非物理噪声
    distance, path = fastdtw(sig_robot_norm, sig_feat_norm, radius=15)
    
    aligned_robot = np.array([sig_robot_norm[idx1] for idx1, idx2 in path])
    aligned_feat = np.array([sig_feat_norm[idx2] for idx1, idx2 in path])
    aligned_robot_raw = np.array([sig_robot[idx1] for idx1, idx2 in path]) 

    raw_corr, _ = stats.spearmanr(aligned_robot, aligned_feat)
    raw_corr = abs(raw_corr)
    
    # 🔥 双重 PE 门控
    vel_thresh = max(0.001, cfg.perception['pe_threshold_ratio'] * np.max(np.abs(aligned_robot_raw)))
    mask_vel = np.abs(aligned_robot_raw) > vel_thresh
    
    win = 15
    l_mean = uniform_filter1d(aligned_robot_raw, size=win, mode='nearest')
    l_sq_mean = uniform_filter1d(aligned_robot_raw**2, size=win, mode='nearest')
    l_std = np.sqrt(np.maximum(l_sq_mean - l_mean**2, 0)) 
    
    dyn_thresh = 0.15 * np.max(l_std) 
    mask_dyn = l_std > dyn_thresh
    
    pe_mask_raw = mask_vel & mask_dyn
    pe_mask = uniform_filter1d(pe_mask_raw.astype(float), size=11, mode='nearest') > 0
    
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
# 2. 单序列特征提取
# ==========================================
def process_single_episode(ep_id, root, cfg, perception):
    try:
        images, poses = get_episode_data(root, ep_id)
        if len(images) < 50:
            return {"Episode": ep_id, "Error": "Sequence too short"}
        
        dt, trim = cfg.kinematics['dt'], cfg.perception['trim_edge']
        
        # 提取运动学与视觉特征
        xi_tool, s_dot = perception.process_episode(images, poses)
        robot_z_tool = xi_tool[:, 2]
        div_feat = s_dot[:, 2]
        
        # 🔥 诚实修正 2：计算这个 Episode 的“运动复杂度” (平均角速度范数)
        # 用来衡量探头在这一段里转得有多剧烈
        angular_vel_norm = np.linalg.norm(xi_tool[:, 3:6], axis=1)
        motion_complexity = np.mean(angular_vel_norm)

        pos_z_smooth = savgol_filter(poses[:, 2], window_length=31, polyorder=3)
        robot_z_base = (np.diff(pos_z_smooth) / dt)[trim:-trim]

        area_feat = []
        for img in images[1:]:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape)==3 else img
            _, bin_img = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
            area_feat.append(-np.sum(bin_img > 0))
            
        area_dot = savgol_filter(np.diff(area_feat)/dt, 15, 2)
        area_dot = np.append(area_dot, area_dot[-1])[trim:-trim]

        _, _, _, r1, _, _ = compute_dtw_aligned_correlation_dual_pe(robot_z_base, area_dot, cfg)
        _, _, _, r2, _, _ = compute_dtw_aligned_correlation_dual_pe(robot_z_tool, area_dot, cfg)
        _, _, _, r3, _, _ = compute_dtw_aligned_correlation_dual_pe(robot_z_tool, div_feat, cfg)
        
        return {
            "Episode": ep_id, 
            "Base_vs_Area": r1, 
            "Tool_vs_Area": r2, 
            "Tool_vs_Div_Ours": r3,
            "Motion_Complexity": motion_complexity # 加入复杂度指标
        }
        
    except Exception as e:
        return {"Episode": ep_id, "Error": str(e)}

# ==========================================
# 3. 批量实验与深度剖析绘图
# ==========================================
def run_batch_ablation():
    cfg = IDUQConfig.from_yaml("configs/default_config.yaml")
    out_dir = cfg.io['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"🚀 开始全量诚实分析 (按探头旋转复杂度分级评估)...")
    root = safe_open_zarr(cfg.io['data_path'])
    perception = PhysicsAwarePerception(cfg)
    episodes = sorted(list(root.group_keys()))
    results = []

    max_workers = max(1, os.cpu_count() - 2) 
    process_func = partial(process_single_episode, root=root, cfg=cfg, perception=perception)

    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in tqdm(executor.map(process_func, episodes), total=len(episodes), desc="Processing"):
            if "Error" not in res:
                results.append(res)
            
    df = pd.DataFrame(results)
    if df.empty: return
    
    # 🔥 诚实修正 3：按“运动复杂度”将数据分为三组
    # Low (简单纯按压), Medium (带有轻微晃动), High (复杂扫查，大角度旋转)
    df['Complexity_Tier'] = pd.qcut(df['Motion_Complexity'], q=3, labels=['Low (Pure Press)', 'Medium (Slight Tilt)', 'High (Complex Scan)'])

    # --- 顶级学术绘图：分组折线图/柱状图 ---
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    # 将数据拉平 (Melt) 以便 Seaborn 绘制分组图
    df_melt = df.melt(id_vars=['Complexity_Tier'], 
                      value_vars=['Base_vs_Area', 'Tool_vs_Area', 'Tool_vs_Div_Ours'],
                      var_name='Method', value_name='Correlation')

    # 绘制分组柱状图展示均值衰减趋势
    sns.barplot(data=df_melt, x='Complexity_Tier', y='Correlation', hue='Method', 
                palette=['#7FBCA6', '#DE8F6E', '#808EB9'], ax=ax, errorbar=('ci', 95))
    
    ax.set_title("Algorithm Degradation under Complex Clinical Scanning Motions", fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel("Spearman's Rank Correlation ($\\rho$)", fontsize=13)
    ax.set_xlabel("Probe Motion Complexity (Angular Velocity Magnitude)", fontsize=13)
    
    # 替换图例标签
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ["Baseline 1 (Base vs Area)", "Baseline 2 (Tool vs Area)", "Ours (Tool vs Divergence)"], title="Methods")
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, "Ablation_Complexity_Analysis.png")
    plt.savefig(save_path, dpi=300)
    
    # 保存详细分组数据
    grouped_stats = df.groupby('Complexity_Tier')[['Base_vs_Area', 'Tool_vs_Area', 'Tool_vs_Div_Ours']].mean()
    print("\n📊 按运动复杂度分级的均值表现 (The Honest Truth):")
    print(grouped_stats)
    grouped_stats.to_csv(os.path.join(out_dir, "grouped_ablation_stats.csv"))
    print(f"\n✅ 诚实分析完成！图表已保存至 {save_path}")

if __name__ == "__main__":
    run_batch_ablation()