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
import traceback

# 引入核心库
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.config_loader import IDUQConfig
from core.perception import PhysicsAwarePerception
from core.data_loader import safe_open_zarr, get_episode_data

# (复用之前的带通滤波和 DTW 核心代码...)
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from fastdtw import fastdtw

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
# 注意：确保在 exp1_diagnostic_tails.py 文件顶部导入 cupy
import cupy as cp
import traceback

def extract_meta_features(ep_id, config_path):
    """
    🔥 核心修复：子进程完全独立运行，适配 GPU 加速版的 Perception
    """
    try:
        # 1. 子进程独立初始化 (这完美避开了多进程 CUDA Context 冲突的坑)
        cfg = IDUQConfig.from_yaml(config_path)
        root = safe_open_zarr(cfg.io['data_path'])
        perception = PhysicsAwarePerception(cfg)
        
        # 2. 读取数据
        images, poses = get_episode_data(root, ep_id)
        if len(images) < 50: 
            return {"Episode": ep_id, "Error": "Too short"}
        
        # 3. 提取运动学与视觉流 (内部已由 CuPy 和 einsum 加速)
        xi_tool, s_dot = perception.process_episode(images, poses)
        
        # ==========================================
        # 💎 提取物理元特征 (Meta-features)
        # ==========================================
        z_energy = np.var(xi_tool[:, 2])
        lateral_energy = np.var(xi_tool[:, 0]) + np.var(xi_tool[:, 1])
        rotational_energy = np.linalg.norm(np.var(xi_tool[:, 3:6], axis=0))
        
        contact_ratios = []
        for img in images[1::5]: 
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape)==3 else img
            
            # 🚀 适配 GPU 架构：将 CPU 图像转为 CuPy 数组推入显存
            gray_cp = cp.array(gray, dtype=cp.float64)
            
            # 调用新的 GPU 版掩膜函数
            W_cp = perception.get_confidence_mask_gpu(gray_cp)
            
            # 在 GPU 上计算掩膜有效率，然后拉回 CPU 存入列表
            contact_ratios.append(float(cp.mean(W_cp > 0.5)))
            
        contact_quality = np.mean(contact_ratios)

        # 4. 计算相关性
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
        err_msg = traceback.format_exc()
        return {"Episode": ep_id, "Error": str(e), "Traceback": err_msg}

def run_diagnostic():
    print("🚀 开始拖尾诊断分析 (Tail Diagnostics)...")
    config_path = "configs/default_config.yaml"
    cfg = IDUQConfig.from_yaml(config_path)
    out_dir = os.path.join(cfg.io['output_dir'], "Diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    
    # 主进程只负责获取 episode 列表
    root = safe_open_zarr(cfg.io['data_path'])
    episodes = sorted(list(root.group_keys()))
    
    results = []
    errors = [] # 记录报错的 Episode
    
    max_workers = max(1, os.cpu_count() - 2)
    # 只把配置路径传给子进程，避开 Zarr 对象的跨进程传输
    process_func = partial(extract_meta_features, config_path=config_path)

    print(f"⚡ 启动多进程池 (Workers: {max_workers})...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in tqdm(executor.map(process_func, episodes), total=len(episodes)):
            if "Error" not in res: 
                results.append(res)
            else:
                errors.append(res)
                
    # 🚨 防吞错机制：如果全军覆没，大声报错！
    df = pd.DataFrame(results)
    if df.empty: 
        print("\n💥 灾难性警告：所有 Episode 均处理失败！DataFrame 为空！")
        print("👇 截取第一个失败的详细报错信息：")
        print(errors[0].get("Traceback", errors[0]["Error"]))
        return

    # 如果有部分失败，打印个警告
    if len(errors) > 0:
        print(f"\n⚠️ 警告：有 {len(errors)} 个 Episode 处理失败被跳过。")

    # ==========================================
    # 💎 绘制诊断散点图：寻找拖尾的真相
    # ==========================================
    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Diagnostic Analysis of Correlation Tails (Failure Modes)", fontsize=16, fontweight='bold', y=1.05)

    # 散点图 1：相关性 vs. 法向下压能量
    sns.scatterplot(data=df, x='Z_Energy', y='Correlation', hue='Contact_Quality', palette='coolwarm', ax=axes[0], alpha=0.7)
    axes[0].set_xscale('log') 
    axes[0].set_title("Correlation vs. Z-axis Excitation\n(Proof of 'No Press, No Correlation')")
    axes[0].set_xlabel("Z-axis Variance (Log Scale)")
    axes[0].axhline(0.4, color='red', linestyle='--', alpha=0.5) 

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
    # ==========================================
    # 🏆 核心升级：量化物理病理对算法的打击 (Quantitative Pathology Analysis)
    # ==========================================
    print("\n" + "="*60)
    print("🏆 真实工作边界与成功率分析 (Operating Envelope Analysis)")
    print("="*60)
    
    # 成功标准：相关性 > 0.6 视为有效跟踪
    success_threshold = 0.6
    
    # 1. 原始的全局成功率
    raw_success_rate = np.mean(df['Correlation'] >= success_threshold) * 100
    print(f"🌍 全局原始成功率 (All 1000+ Episodes): {raw_success_rate:.2f}%")
    
    # 2. 定义两大病理区 (Pathology Zones)
    # 病理 A：发呆/无激励 (Z轴方差 < 1e-6)
    mask_pathology_A = df['Z_Energy'] < 1e-6
    # 病理 B：极端翻滚干扰 (侧滑+旋转方差 > 1.0)
    mask_pathology_B = df['Interference'] > 1.0
    
    # 3. 计算两大病理区内的“崩溃率” (证明病理假设成立)
    df_pathology_A = df[mask_pathology_A]
    df_pathology_B = df[mask_pathology_B]
    
    if len(df_pathology_A) > 0:
        fail_rate_A = np.mean(df_pathology_A['Correlation'] < success_threshold) * 100
        print(f"🚨 [病理 A] 无激励区 (Z < 1e-6): 共有 {len(df_pathology_A)} 个样本，崩溃率为 {fail_rate_A:.2f}% (远高于全局平均！)")
        
    if len(df_pathology_B) > 0:
        fail_rate_B = np.mean(df_pathology_B['Correlation'] < success_threshold) * 100
        print(f"🚨 [病理 B] 极端干扰区 (干扰 > 1.0): 共有 {len(df_pathology_B)} 个样本，崩溃率为 {fail_rate_B:.2f}% (远高于全局平均！)")
    
    # 4. 物理边界过滤 (Define Operating Envelope) -> 剔除病理
    envelope_mask = ~(mask_pathology_A | mask_pathology_B)
    df_envelope = df[envelope_mask]
    
    envelope_ratio = np.mean(envelope_mask) * 100
    print(f"\n🛡️ 剔除病理后的工作边界样本占比: {envelope_ratio:.2f}% ({len(df_envelope)}/{len(df)} episodes)")
    
    # 5. 计算真实成功率
    if len(df_envelope) > 0:
        true_success_rate = np.mean(df_envelope['Correlation'] >= success_threshold) * 100
        print(f"🎯 物理边界内的【真实成功率】 (True Success Rate): {true_success_rate:.2f}%")
        
        true_mean_corr = df_envelope['Correlation'].mean()
        print(f"📈 物理边界内的平均相关性 (True Mean Correlation): {true_mean_corr:.4f}")
    
    print("="*60)
    
    # --- 绘制一张“剔除病理数据后的提升对比图” ---
    fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
    categories = ['Raw Dataset\n(Contains Pathologies)', 'Operating Envelope\n(Valid Mechanics Only)']
    rates = [raw_success_rate, true_success_rate]
    
    bars = ax_bar.bar(categories, rates, color=['#B0BEC5', '#D62728'], width=0.5)
    ax_bar.set_ylim(0, 105)
    ax_bar.set_ylabel("Tracking Success Rate (%)", fontsize=12)
    ax_bar.set_title("Success Rate Boost by Excluding Physical Pathologies", fontsize=14, fontweight='bold', pad=15)
    
    # 在柱子上加数字
    for bar in bars:
        yval = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "True_Success_Rate_Pathology.png"), dpi=300)

if __name__ == "__main__":
    run_diagnostic()