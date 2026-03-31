# experiments/exp1_diagnostic_tails.py

import os
import sys
import numpy as np
import cv2
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import concurrent.futures
from functools import partial

# 引入核心库
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.config_loader import IDUQConfig
from core.perception import PhysicsAwarePerception
from core.data_loader import safe_open_zarr, get_episode_data

# (复用之前的带通滤波和 DTW 核心代码...)
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from fastdtw import fastdtw
from scipy.ndimage import uniform_filter1d

def butter_lowpass_filter(data, cutoff, fs, order=4):
    if len(data) < 15: return data
    b, a = butter(order, cutoff / (0.5 * fs), btype='low', analog=False)
    return filtfilt(b, a, data)

def compute_correlation(sig_robot, sig_feat, cfg):
    # 简化的对齐计算，专注于返回相关性
    fs, cutoff = cfg.alignment['fs'], cfg.alignment['cutoff_freq']
    s1 = StandardScaler().fit_transform(butter_lowpass_filter(sig_robot, cutoff, fs).reshape(-1, 1)).flatten()
    s2 = StandardScaler().fit_transform(butter_lowpass_filter(sig_feat, cutoff, fs).reshape(-1, 1)).flatten()
    _, path = fastdtw(s1, s2, radius=15)
    s1_al = np.array([s1[i] for i, j in path])
    s2_al = np.array([s2[j] for i, j in path])
    corr, _ = stats.spearmanr(s1_al, s2_al)
    return abs(corr)

def extract_meta_features(ep_id, root, cfg, perception):
    """
    🔥 核心：不仅仅计算相关性，还要提取当前 Episode 的宏观物理属性
    """
    try:
        images, poses = get_episode_data(root, ep_id)
        if len(images) < 50: return {"Episode": ep_id, "Error": "Too short"}
        
        # 1. 提取运动学与视觉流
        xi_tool, s_dot = perception.process_episode(images, poses)
        
        # ==========================================
        # 💎 提取物理元特征 (Meta-features)
        # ==========================================
        # A. 法向激励能量 (Z-axis Excitation Energy): 这个序列里，下压动作到底有多剧烈？
        z_energy = np.var(xi_tool[:, 2])
        
        # B. 侧滑/翻滚干扰 (Lateral & Rotational Interference): 除了下压，它是不是在乱晃？
        lateral_energy = np.var(xi_tool[:, 0]) + np.var(xi_tool[:, 1])
        rotational_energy = np.linalg.norm(np.var(xi_tool[:, 3:6], axis=0))
        
        # C. 图像接触质量 (Acoustic Contact Quality): 图像有没有黑掉？(置信度掩膜的平均占比)
        contact_ratios = []
        for img in images[1::5]: # 抽样检查以加快速度
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape)==3 else img
            W = perception.get_confidence_mask(gray)
            contact_ratios.append(np.mean(W > 0.5))
        contact_quality = np.mean(contact_ratios)

        # 2. 计算相关性
        corr_ours = compute_correlation(xi_tool[:, 2], s_dot[:, 2], cfg)
        
        return {
            "Episode": ep_id,
            "Correlation": corr_ours,
            "Z_Energy": z_energy,
            "Lateral_Energy": lateral_energy,
            "Rotational_Energy": rotational_energy,
            "Contact_Quality": contact_quality
        }
    except Exception as e:
        return {"Episode": ep_id, "Error": str(e)}

def run_diagnostic():
    print("🚀 开始拖尾诊断分析 (Tail Diagnostics)...")
    cfg = IDUQConfig.from_yaml("configs/default_config.yaml")
    out_dir = os.path.join(cfg.io['output_dir'], "Diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    
    root = safe_open_zarr(cfg.io['data_path'])
    perception = PhysicsAwarePerception(cfg)
    episodes = sorted(list(root.group_keys()))
    
    results = []
    max_workers = max(1, os.cpu_count() - 2)
    process_func = partial(extract_meta_features, root=root, cfg=cfg, perception=perception)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in tqdm(executor.map(process_func, episodes), total=len(episodes)):
            if "Error" not in res: results.append(res)
            
    df = pd.DataFrame(results)
    if df.empty: return

    # ==========================================
    # 💎 绘制诊断散点图：寻找拖尾的真相
    # ==========================================
    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Diagnostic Analysis of Correlation Tails (Failure Modes)", fontsize=16, fontweight='bold', y=1.05)

    # 散点图 1：相关性 vs. 法向下压能量
    sns.scatterplot(data=df, x='Z_Energy', y='Correlation', hue='Contact_Quality', palette='coolwarm', ax=axes[0], alpha=0.7)
    axes[0].set_xscale('log') # 能量通常跨度很大，用对数轴
    axes[0].set_title("Correlation vs. Z-axis Excitation\n(Proof of 'No Press, No Correlation')")
    axes[0].set_xlabel("Z-axis Variance (Log Scale)")
    axes[0].axhline(0.4, color='red', linestyle='--', alpha=0.5) # 拖尾警戒线

    # 散点图 2：相关性 vs. 图像接触质量
    sns.scatterplot(data=df, x='Contact_Quality', y='Correlation', color='purple', ax=axes[1], alpha=0.6)
    axes[1].set_title("Correlation vs. Acoustic Contact Quality\n(Proof of Coupling Loss)")
    axes[1].set_xlabel("Valid ROI Ratio (%)")
    axes[1].axhline(0.4, color='red', linestyle='--', alpha=0.5)

    # 散点图 3：相关性 vs. 干扰能量 (侧滑+旋转)
    df['Interference'] = df['Lateral_Energy'] + df['Rotational_Energy']
    sns.scatterplot(data=df, x='Interference', y='Correlation', color='darkorange', ax=axes[2], alpha=0.6)
    axes[2].set_xscale('log')
    axes[2].set_title("Correlation vs. Motion Interference\n(Robustness to Complex Sweeps)")
    axes[2].set_xlabel("Lateral & Rotational Variance (Log Scale)")
    axes[2].axhline(0.4, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Diagnostic_Scatter_Plots.png"), dpi=300, bbox_inches='tight')
    
    # 提取最烂的 10 个 Episode 输出名单，让你去肉眼排查
    worst_tails = df[df['Correlation'] < 0.4].sort_values(by='Correlation')
    worst_tails.to_csv(os.path.join(out_dir, "Worst_Tails_Report.csv"), index=False)
    
    print(f"\n✅ 诊断完成！共发现 {len(worst_tails)} 个拖尾样本 (rho < 0.4)。")
    print(f"📊 图表和坏点报告已保存至: {out_dir}")

if __name__ == "__main__":
    run_diagnostic()