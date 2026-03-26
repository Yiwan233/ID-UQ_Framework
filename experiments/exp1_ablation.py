# experiments/exp1_ablation.py

import os
import sys
import traceback
import numpy as np
import cv2
import pandas as pd  # 用于生成统计表格
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm # 用于显示进度条

# 环境与路径配置
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.config_loader import IDUQConfig
from core.perception import PhysicsAwarePerception
from core.data_loader import safe_open_zarr, get_episode_data

def compute_best_correlation(sig1: np.ndarray, sig2: np.ndarray, max_lag: int = 25):
    """保持原有的滑动相关性计算逻辑"""
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
        corr = abs(np.corrcoef(x_test, y_test)[0, 1])
        if corr > max_corr:
            max_corr, best_offset = corr, offset
            
    return max_corr

def process_single_episode(ep_id, root, cfg, perception):
    """处理单个 Episode 并返回三个相关性系数"""
    try:
        images, poses = get_episode_data(root, ep_id)
        
        # 1. 提取核心特征 (Ours)
        xi_tool, s_dot = perception.process_episode(images, poses)
        robot_z_tool = xi_tool[:, 2]
        div_feat = s_dot[:, 2]
        
        # 2. 提取基准特征 (Baselines)
        dt = cfg.kinematics['dt']
        trim = cfg.perception['trim_edge']
        
        # Baseline 1 & 2: 面积提取
        area_feat = []
        for img in images[1:]:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape)==3 else img
            _, bin_img = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
            area_feat.append(-np.sum(bin_img > 0))
        area_dot = savgol_filter(np.diff(area_feat)/dt, 15, 2)
        area_dot = np.append(area_dot, area_dot[-1])[trim:-trim]
        
        # Baseline 1: Base 系速度
        pos_z_smooth = savgol_filter(poses[:, 2], 31, 3)
        robot_z_base = (np.diff(pos_z_smooth)/dt)[trim:-trim]

        # 计算三个相关性系数
        r1 = compute_best_correlation(robot_z_base, area_dot)
        r2 = compute_best_correlation(robot_z_tool, area_dot)
        r3 = compute_best_correlation(robot_z_tool, div_feat)
        
        return {"Episode": ep_id, "r1_Base_Area": r1, "r2_Tool_Area": r2, "r3_Ours_Full": r3}
    except Exception as e:
        # 🎯 新增：打印红色的详细错误栈，方便我们定位问题！
        print(f"\n❌ [Episode {ep_id} 崩溃] 真正的错误是:")
        traceback.print_exc()
    
    # 依然返回 Error 字典，保证程序不闪退
    return {"Episode": ep_id, "Error": str(e)}

def run_batch_ablation():
    print("🚀 开始全量数据消融分析 (Batch Mode)...")
    cfg = IDUQConfig.from_yaml("configs/default_config.yaml")
    root = safe_open_zarr(cfg.io['data_path'])
    perception = PhysicsAwarePerception(cfg)
    
    episodes = sorted(list(root.group_keys()))
    results = []
    
    # 使用 tqdm 进度条，对于 1000+ 数据很有用
    for ep in tqdm(episodes, desc="Processing Episodes"):
        res = process_single_episode(ep, root, cfg, perception)
        if "Error" not in res:
            results.append(res)
            
    # 生成 Pandas 表格
    df = pd.DataFrame(results)
    
    # 计算统计摘要
    stats_df = df.describe().loc[['mean', 'std', 'min', 'max']]
    print("\n📊 统计摘要 (Correlation Statistics):")
    print(stats_df)
    
    # 保存结果
    out_dir = cfg.io['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "batch_ablation_results.csv"), index=False)
    stats_df.to_csv(os.path.join(out_dir, "statistical_summary.csv"))
    
    # 生成学术级表格图
    plt.figure(figsize=(10, 4))
    plt.axis('off')
    tbl = plt.table(cellText=stats_df.round(4).values, 
                   colLabels=stats_df.columns, 
                   rowLabels=stats_df.index, 
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 1.8)
    plt.title("Statistical Correlation Table (Mean/Std across all episodes)", pad=20)
    plt.savefig(os.path.join(out_dir, "statistical_table.png"), dpi=300)
    
    print(f"\n✅ 批处理完成！表格已保存至 {out_dir}")

if __name__ == "__main__":
    run_batch_ablation()