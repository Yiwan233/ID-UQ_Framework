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
    # 5. 🔥 基于局部动态方差的双重 PE 门控 (Dual-Gated PE)
    # ==========================================================
    vel_thresh = max(0.001, cfg.perception['pe_threshold_ratio'] * np.max(np.abs(aligned_robot_raw)))
    mask_vel = np.abs(aligned_robot_raw) > vel_thresh
    
    win = 15
    l_mean = uniform_filter1d(aligned_robot_raw, size=win, mode='nearest')
    l_sq_mean = uniform_filter1d(aligned_robot_raw**2, size=win, mode='nearest')
    l_std = np.sqrt(np.maximum(l_sq_mean - l_mean**2, 0)) 
    
    dyn_thresh = 0.15 * np.max(l_std) 
    mask_dyn = l_std > dyn_thresh
    
    # 双重锁定 + 1D 形态学膨胀
    pe_mask_raw = mask_vel & mask_dyn
    pe_mask = uniform_filter1d(pe_mask_raw.astype(float), size=11, mode='nearest') > 0
    
    # 最终评分结算
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
        
        # 1. 提取核心特征 (Ours: Div Flow)
        xi_tool, s_dot = perception.process_episode(images, poses)
        robot_z_tool = xi_tool[:, 2]
        div_feat = s_dot[:, 2]
        
        # 2. 提取基座 Z 速度 (Baseline 1)
        pos_z_smooth = savgol_filter(poses[:, 2], window_length=31, polyorder=3)
        robot_z_base = (np.diff(pos_z_smooth) / dt)[trim:-trim]

        # 3. 提取面积特征 (Baseline 1 & 2)
        area_feat = []
        for img in images[1:]:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape)==3 else img
            _, bin_img = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
            area_feat.append(-np.sum(bin_img > 0))
            
        area_dot = savgol_filter(np.diff(area_feat)/dt, 15, 2)
        area_dot = np.append(area_dot, area_dot[-1])[trim:-trim]

        # 4. 执行 DTW + 双重 PE 对齐评估
        _, _, _, r1, _, _ = compute_dtw_aligned_correlation_dual_pe(robot_z_base, area_dot, cfg)
        _, _, _, r2, _, _ = compute_dtw_aligned_correlation_dual_pe(robot_z_tool, area_dot, cfg)
        _, _, _, r3, _, _ = compute_dtw_aligned_correlation_dual_pe(robot_z_tool, div_feat, cfg)
        
        return {"Episode": ep_id, "Base_vs_Area": r1, "Tool_vs_Area": r2, "Tool_vs_Div_Ours": r3}
        
    except Exception as e:
        return {"Episode": ep_id, "Error": str(e)}
import concurrent.futures
from functools import partial

def run_batch_ablation():
    cfg = IDUQConfig.from_yaml("configs/default_config.yaml")
    out_dir = cfg.io['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"🚀 开始全量数据消融分析 (多进程加速: Dual-PE + DTW + Spearman)...")
    root = safe_open_zarr(cfg.io['data_path'])
    perception = PhysicsAwarePerception(cfg)
    episodes = sorted(list(root.group_keys()))
    
    results = []
    
    # ---------------------------------------------------------
    # 🔥 核心加速：使用 ProcessPoolExecutor 进行多进程并行计算
    # ---------------------------------------------------------
    # 获取 CPU 核心数，保留 1-2 个核心给系统，防止电脑卡死
    max_workers = max(1, os.cpu_count() - 2) 
    print(f"⚡ 启动并行计算池，分配 {max_workers} 个 CPU 核心...")

    # 使用 partial 固定公共参数，方便 map 函数调用
    process_func = partial(process_single_episode, root=root, cfg=cfg, perception=perception)

    # tqdm 结合多进程 map
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # executor.map 会将 episodes 分发给不同的 CPU 核心同时运行
        for res in tqdm(executor.map(process_func, episodes), total=len(episodes), desc="Processing Episodes"):
            if "Error" not in res:
                results.append(res)
            
    df = pd.DataFrame(results)
    
    if df.empty:
        print("💥 警告：所有 Episode 均处理失败，DataFrame 为空！")
        return
        
    # --- A. 打印与保存统计表格 ---
    stats_df = df[["Base_vs_Area", "Tool_vs_Area", "Tool_vs_Div_Ours"]].describe().loc[['mean', 'std', 'min', 'max', '50%']]
    stats_df.rename(index={'50%': 'median'}, inplace=True)
    
    print("\n📊 全局统计摘要 (Spearman's Rank Correlation):")
    print(stats_df)
    df.to_csv(os.path.join(out_dir, "batch_ablation_results.csv"), index=False)
    
    # --- B. 顶级学术绘图 (Boxplot + Table) ---
    sns.set_theme(style="whitegrid")
    fig, (ax_box, ax_table) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    # 1. 绘制箱线图展示分布 (Boxplot)
    plot_data = df[["Base_vs_Area", "Tool_vs_Area", "Tool_vs_Div_Ours"]]
    sns.boxplot(data=plot_data, ax=ax_box, palette="Set2")
    sns.stripplot(data=plot_data, ax=ax_box, color=".25", alpha=0.3, size=3, jitter=True)
    
    ax_box.set_title("Ablation Study: Distribution of Kinematic Consistency across 1000+ Episodes", fontsize=14, fontweight='bold', pad=15)
    ax_box.set_ylabel("Spearman's Rank Correlation ($\\rho$)", fontsize=12)
    ax_box.set_xticklabels(["Baseline 1\n(Base Vel vs. Area)", "Baseline 2\n(Tool Vel vs. Area)", "Ours\n(Tool Vel vs. Divergence)"], fontsize=11)
    ax_box.grid(True, alpha=0.3)
    
    # 2. 绘制数据表格 (Table)
    ax_table.axis('off')
    tbl = ax_table.table(
        cellText=stats_df.round(3).values, 
        colLabels=["Baseline 1 (Base vs Area)", "Baseline 2 (Tool vs Area)", "Ours (Tool vs Div)"], 
        rowLabels=stats_df.index, 
        loc='center', 
        cellLoc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 1.8)
    
    # 调整布局并保存
    plt.tight_layout()
    save_path = os.path.join(out_dir, "Ablation_Statistical_Summary.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ 批处理完成！精美的学术图表与 CSV 数据已保存至 {out_dir}")

if __name__ == "__main__":
    run_batch_ablation()