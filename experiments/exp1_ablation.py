# experiments/exp1_ablation.py

import os
import sys
import traceback
import numpy as np
import cv2
import pandas as pd
from scipy.signal import savgol_filter, butter, filtfilt
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# 环境与路径配置
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.config_loader import IDUQConfig
from core.perception import PhysicsAwarePerception
from core.data_loader import safe_open_zarr, get_episode_data

# ==========================================
# 1. 信号处理工具
# ==========================================
def bandpass_filter(data, lowcut=0.1, highcut=5.0, fs=30.0, order=2):
    """
    零相移带通滤波：
    1. 滤除极低频基线漂移 (Drift)
    2. 滤除极高频硬件噪声 (Electrical Noise)
    """
    if len(data) < 15: # 信号太短直接返回
        return data
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def compute_robust_correlation(sig1: np.ndarray, sig2: np.ndarray, max_lag: int = 25):
    """
    使用 Spearman 秩相关替代 Pearson，无需正态性假设，
    并结合滑动窗口解决整体的粘弹性相位延迟。
    """
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
            
        # 🚀 核心升级：使用 Spearman 秩相关系数
        corr, _ = stats.spearmanr(x_test, y_test)
        corr = abs(corr)
        
        if corr > max_corr:
            max_corr, best_offset = corr, offset
            
    return max_corr

# ==========================================
# 2. 单序列处理逻辑
# ==========================================
def process_single_episode(ep_id, root, cfg, perception):
    try:
        images, poses = get_episode_data(root, ep_id)
        if len(images) < 50:
            raise ValueError("Episode 长度过短，无法进行统计分析")
        
        dt = cfg.kinematics['dt']
        fs = 1.0 / dt
        trim = cfg.perception['trim_edge']
        
        # -----------------------------------
        # A. 提取核心特征 (Ours: Div_Flow)
        # -----------------------------------
        xi_tool, s_dot = perception.process_episode(images, poses)
        
        # 物理机制对齐：应用带通滤波统一频率基准
        robot_z_tool = bandpass_filter(xi_tool[:, 2], fs=fs)
        div_feat = bandpass_filter(s_dot[:, 2], fs=fs)
        
        # -----------------------------------
        # B. 提取基准特征 1 (Baseline: Area Rate)
        # -----------------------------------
        area_feat = []
        for img in images[1:]:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape)==3 else img
            _, bin_img = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
            area_feat.append(-np.sum(bin_img > 0))
            
        area_dot_raw = savgol_filter(np.diff(area_feat)/dt, 15, 2)
        area_dot_raw = np.append(area_dot_raw, area_dot_raw[-1])[trim:-trim]
        area_dot = bandpass_filter(area_dot_raw, fs=fs)
        
        # -----------------------------------
        # C. 提取基准特征 2 (Baseline: Base Vel)
        # -----------------------------------
        pos_z_smooth = savgol_filter(poses[:, 2], 31, 3)
        robot_z_base_raw = (np.diff(pos_z_smooth)/dt)[trim:-trim]
        robot_z_base = bandpass_filter(robot_z_base_raw, fs=fs)

        # -----------------------------------
        # D. 计算非参数统计相关性 (Spearman)
        # -----------------------------------
        r1 = compute_robust_correlation(robot_z_base, area_dot)
        r2 = compute_robust_correlation(robot_z_tool, area_dot)
        r3 = compute_robust_correlation(robot_z_tool, div_feat)
        
        return {"Episode": ep_id, "r1_Base_Area": r1, "r2_Tool_Area": r2, "r3_Ours_Full": r3}
        
    except Exception as e:
        print(f"\n❌ [Episode {ep_id} 崩溃] 真正的错误是:")
        traceback.print_exc()
        return {"Episode": ep_id, "Error": str(e)}

# ==========================================
# 3. 批量执行与评估
# ==========================================
def run_batch_ablation():
    print("🚀 开始全量数据消融分析 (Robust Non-parametric Mode)...")
    cfg = IDUQConfig.from_yaml("configs/default_config.yaml")
    root = safe_open_zarr(cfg.io['data_path'])
    perception = PhysicsAwarePerception(cfg)
    
    episodes = sorted(list(root.group_keys()))
    results = []
    
    # 为了测试，先取前 10 个跑跑看。如果没问题，把 [:10] 删掉跑全量
    for ep in tqdm(episodes[:10], desc="Processing Episodes"):
        res = process_single_episode(ep, root, cfg, perception)
        if "Error" not in res:
            results.append(res)
            
    df = pd.DataFrame(results)
    
    if df.empty:
        print("💥 警告：所有 Episode 均处理失败，DataFrame 为空！")
        return
        
    # 计算统计摘要
    stats_df = df.describe().loc[['mean', 'std', 'min', 'max']]
    print("\n📊 统计摘要 (Spearman's Rank Correlation):")
    print(stats_df)
    
    # -----------------------------------
    # 绘图：生成箱型图与数据表
    # -----------------------------------
    out_dir = cfg.io['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "batch_ablation_results.csv"), index=False)
    
    fig, (ax_box, ax_table) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    # 绘制箱线图展示分布
    plot_data = df[["r1_Base_Area", "r2_Tool_Area", "r3_Ours_Full"]]
    sns.boxplot(data=plot_data, ax=ax_box, palette="Set2")
    sns.stripplot(data=plot_data, ax=ax_box, color=".25", alpha=0.5, size=4)
    ax_box.set_title("Distribution of Non-parametric Correlation (Spearman's $\\rho$)", fontsize=14, fontweight='bold')
    ax_box.set_ylabel("Spearman Rank Correlation")
    ax_box.set_xticklabels(["Base Vel vs Area", "Tool Vel vs Area", "Tool Vel vs Div (Ours)"])
    ax_box.grid(True, alpha=0.3)
    
    # 绘制表格
    ax_table.axis('off')
    tbl = ax_table.table(cellText=stats_df.round(4).values, 
                         colLabels=["Base vs Area", "Tool vs Area", "Tool vs Div (Ours)"], 
                         rowLabels=stats_df.index, 
                         loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 1.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "statistical_ablation_summary.png"), dpi=300)
    print(f"\n✅ 批处理完成！图表与数据已保存至 {out_dir}")

if __name__ == "__main__":
    run_batch_ablation()