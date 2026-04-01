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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.config_loader import IDUQConfig
from core.perception import PhysicsAwarePerception
from core.data_loader import safe_open_zarr, get_episode_data

def butter_lowpass_filter(data, cutoff, fs, order=4):
    if len(data) < 15: return data
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def compute_dtw_aligned_correlation_dual_pe(sig_robot, sig_feat, cfg):
    if len(sig_robot) == 0 or len(sig_feat) == 0:
        return np.array([]), np.array([]), 0.0, 0.0, np.array([]), []
        
    fs = cfg.alignment['fs']
    cutoff = cfg.alignment['cutoff_freq']
    
    sig_robot_cl = butter_lowpass_filter(sig_robot, cutoff, fs)
    sig_feat_cl = butter_lowpass_filter(sig_feat, cutoff, fs)

    scaler = StandardScaler()
    sig_robot_norm = scaler.fit_transform(sig_robot_cl.reshape(-1, 1)).flatten()
    sig_feat_norm = scaler.fit_transform(sig_feat_cl.reshape(-1, 1)).flatten()

    distance, path = fastdtw(sig_robot_norm, sig_feat_norm, radius=15)
    
    aligned_robot = np.array([sig_robot_norm[idx1] for idx1, idx2 in path])
    aligned_feat = np.array([sig_feat_norm[idx2] for idx1, idx2 in path])
    aligned_robot_raw = np.array([sig_robot[idx1] for idx1, idx2 in path]) 

    raw_corr, _ = stats.spearmanr(aligned_robot, aligned_feat)
    raw_corr = 0.0 if np.isnan(raw_corr) else abs(raw_corr)
    
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
        pe_corr = 0.0 if np.isnan(pe_corr) else abs(pe_corr)
    else:
        pe_corr = raw_corr 
        pe_mask = np.ones_like(aligned_robot_raw, dtype=bool) 
        
    return aligned_robot, aligned_feat, raw_corr, pe_corr, pe_mask, path

# 🔥 完全隔离子进程状态，杜绝内存/类型泄漏
def process_single_episode(ep_id, config_path):
    try:
        cfg = IDUQConfig.from_yaml(config_path)
        root = safe_open_zarr(cfg.io['data_path'])
        perception = PhysicsAwarePerception(cfg)
        
        images, poses = get_episode_data(root, ep_id)
        if len(images) < 50:
            return {"Episode": ep_id, "Error": "Sequence too short"}
        
        dt, trim = cfg.kinematics['dt'], cfg.perception['trim_edge']
        step = cfg.perception.get('step', 1)
        
        xi_tool, s_dot = perception.process_episode(images, poses)
        robot_z_tool = xi_tool[:, 2]
        div_feat = s_dot[:, 2]
        
        angular_vel_norm = np.linalg.norm(xi_tool[:, 3:6], axis=1)
        motion_complexity = np.mean(angular_vel_norm)
        
        pos_z_smooth = savgol_filter(poses[:, 2], window_length=31, polyorder=3)
        robot_z_base = (np.diff(pos_z_smooth) / dt)[trim:-trim]

        area_feat = []
        for img in images[1:]:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape)==3 else img
            _, bin_img = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
            area_feat.append(-np.sum(bin_img > 0))
            
        # 🔥 核心修复：将 list 转换为 numpy array，否则无法相减
        area_feat = np.array(area_feat)
        area_dot = (area_feat[step:] - area_feat[:-step]) / (dt * step)
        area_dot = savgol_filter(area_dot, 15, 2)[trim:-trim]

        _, _, _, r1, _, _ = compute_dtw_aligned_correlation_dual_pe(robot_z_base, area_dot, cfg)
        _, _, _, r2, _, _ = compute_dtw_aligned_correlation_dual_pe(robot_z_tool, area_dot, cfg)
        _, _, _, r3, _, _ = compute_dtw_aligned_correlation_dual_pe(robot_z_tool, div_feat, cfg)
        
        return {
            "Episode": ep_id, 
            "Base_vs_Area": r1, 
            "Tool_vs_Area": r2, 
            "Tool_vs_Div_Ours": r3,
            "Motion_Complexity": motion_complexity
        }
    except Exception as e:
        err_trace = traceback.format_exc()
        return {"Episode": ep_id, "Error": f"{str(e)}\n{err_trace}"}

def run_batch_ablation():
    config_path = "configs/default_config.yaml"
    cfg = IDUQConfig.from_yaml(config_path)
    out_dir = cfg.io['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"🚀 开始全量诚实分析 (按探头旋转复杂度分级评估)...")
    root = safe_open_zarr(cfg.io['data_path'])
    episodes = sorted(list(root.group_keys()))

    max_workers = max(1, os.cpu_count() - 2) 
    process_func = partial(process_single_episode, config_path=config_path)

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in tqdm(executor.map(process_func, episodes), total=len(episodes), desc="Processing"):
            if "Error" not in res:
                results.append(res)
            else:
                # 使用 tqdm.write 防止日志被进度条覆盖
                tqdm.write(f"❌ [Error in {res.get('Episode', 'Unknown')}]:\n{res['Error']}")
            
    df = pd.DataFrame(results)
    if df.empty: 
        print("\n💥 严重错误: 处理结果为空，请查看上方的 ❌ Error 日志。")
        return
    
    df['Complexity_Tier'] = pd.qcut(df['Motion_Complexity'], q=3, labels=['Low (Pure Press)', 'Medium (Slight Tilt)', 'High (Complex Scan)'])

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    df_melt = df.melt(id_vars=['Complexity_Tier'], 
                      value_vars=['Base_vs_Area', 'Tool_vs_Area', 'Tool_vs_Div_Ours'],
                      var_name='Method', value_name='Correlation')

    sns.barplot(data=df_melt, x='Complexity_Tier', y='Correlation', hue='Method', 
                palette=['#7FBCA6', '#DE8F6E', '#808EB9'], ax=ax, errorbar=('ci', 95))
    
    ax.set_title("Algorithm Degradation under Complex Clinical Scanning Motions", fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel("Spearman's Rank Correlation ($\\rho$)", fontsize=13)
    ax.set_xlabel("Probe Motion Complexity (Angular Velocity Magnitude)", fontsize=13)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ["Baseline 1 (Base vs Area)", "Baseline 2 (Tool vs Area)", "Ours (Tool vs Divergence)"], title="Methods")
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, "Ablation_Complexity_Analysis.png")
    plt.savefig(save_path, dpi=300)
    
    grouped_stats = df.groupby('Complexity_Tier')[['Base_vs_Area', 'Tool_vs_Area', 'Tool_vs_Div_Ours']].mean()
    print("\n📊 按运动复杂度分级的均值表现 (The Honest Truth):")
    print(grouped_stats)
    grouped_stats.to_csv(os.path.join(out_dir, "grouped_ablation_stats.csv"))
    print(f"\n✅ 诚实分析完成！图表已保存至 {save_path}")

if __name__ == "__main__":
    run_batch_ablation()