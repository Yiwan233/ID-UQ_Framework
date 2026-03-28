import os
import sys
import numpy as np
import scipy.stats as stats
from scipy.signal import butter, filtfilt
import cv2
import cupy as cp  # 🚀 GPU 加速核心
from cupyx.scipy.ndimage import uniform_filter1d as cupy_uniform_filter1d
from sklearn.preprocessing import StandardScaler
from fastdtw import fastdtw
import matplotlib.pyplot as plt
from tqdm import tqdm

# 环境配置与原代码一致
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.config_loader import IDUQConfig
from core.perception import PhysicsAwarePerception
from core.data_loader import safe_open_zarr

# ==========================================
# 1. GPU 加速的数学引擎
# ==========================================
def butter_lowpass_filter(data, cutoff, fs, order=4):
    # 滤波器系数计算保留在 CPU，过滤过程如果是大批量可以移至 GPU
    # 但由于 filtfilt 的递归特性，对于单条短序列，CPU 依然很快
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def compute_dtw_aligned_correlation_dual_pe_gpu(sig_robot, sig_feat, cfg):
    """
    使用 CuPy 加速逻辑门控计算
    注意：FastDTW 目前主要在 CPU 运行，但我们通过加速前后置处理来提升效率
    """
    fs = cfg.alignment['fs']
    cutoff = cfg.alignment['cutoff_freq']
    
    # 0. 滤波 (CPU)
    sig_robot_cl = butter_lowpass_filter(sig_robot, cutoff, fs)
    sig_feat_cl = butter_lowpass_filter(sig_feat, cutoff, fs)

    # 1. 信号标准化 (CPU for DTW input)
    scaler = StandardScaler()
    sig_robot_norm = scaler.fit_transform(sig_robot_cl.reshape(-1, 1)).flatten()
    sig_feat_norm = scaler.fit_transform(sig_feat_cl.reshape(-1, 1)).flatten()

    # 2. FastDTW (核心瓶颈，目前在 CPU 运行)
    # 如果需要纯 GPU DTW，建议更换为 tslearn.metrics.soft_dtw_gpu
    distance, path = fastdtw(sig_robot_norm, sig_feat_norm)
    
    # 3. 重建对齐信号并上传至 GPU
    aligned_robot = cp.array([sig_robot_norm[idx1] for idx1, idx2 in path])
    aligned_feat = cp.array([sig_feat_norm[idx2] for idx1, idx2 in path])
    aligned_robot_raw = cp.array([sig_robot[idx1] for idx1, idx2 in path]) 

    # 4. 全局相关性 (Spearman 需要转回 CPU，或使用 cupy 实现的秩相关)
    raw_corr = abs(float(cp.corrcoef(aligned_robot, aligned_feat)[0, 1])) # 近似线性相关
    
    # ==========================================================
    # 5. 🔥 GPU 加速：双重 PE 门控逻辑
    # ==========================================================
    # A. 静态约束
    vel_thresh = max(0.001, cfg.perception['pe_threshold_ratio'] * cp.max(cp.abs(aligned_robot_raw)))
    mask_vel = cp.abs(aligned_robot_raw) > vel_thresh
    
    # B. 动态约束：计算局部标准差 (Local STD)
    win = 15
    l_mean = cupy_uniform_filter1d(aligned_robot_raw, size=win, mode='nearest')
    l_sq_mean = cupy_uniform_filter1d(aligned_robot_raw**2, size=win, mode='nearest')
    l_std = cp.sqrt(cp.maximum(l_sq_mean - l_mean**2, 0)) 
    
    dyn_thresh = 0.15 * cp.max(l_std) 
    mask_dyn = l_std > dyn_thresh
    
    # C. 双重锁定
    pe_mask_raw = mask_vel & mask_dyn
    pe_mask = cupy_uniform_filter1d(pe_mask_raw.astype(cp.float32), size=11, mode='nearest') > 0
    
    # --- 最终评分结算 ---
    active_count = cp.sum(pe_mask)
    if active_count > 15: 
        # 转回 CPU 计算最终的 Spearman (因为 stats.spearmanr 不支持 GPU)
        x_pe = cp.asnumpy(aligned_robot[pe_mask])
        y_pe = cp.asnumpy(aligned_feat[pe_mask])
        pe_corr = abs(stats.spearmanr(x_pe, y_pe)[0])
    else:
        pe_corr = raw_corr 
        pe_mask = cp.ones_like(aligned_robot_raw, dtype=cp.bool_)
        
    return (cp.asnumpy(aligned_robot), cp.asnumpy(aligned_feat), 
            raw_corr, pe_corr, cp.asnumpy(pe_mask), path)

# ==========================================
# 2. 批量实验调度逻辑 (含预实验功能)
# ==========================================
def run_batch_ablation(test_mode=False):
    cfg = IDUQConfig.from_yaml("configs/default_config.yaml")
    out_dir = cfg.io['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    root = safe_open_zarr(cfg.io['data_path'])
    perception = PhysicsAwarePerception(cfg)
    
    episodes = sorted(list(root.group_keys()))
    
    # 🎯 预实验逻辑：仅取前 3 个样本
    if test_mode:
        print("🧪 [TEST MODE] 正在进行预实验，仅处理前 3 个样本...")
        episodes = episodes[:3]
    else:
        print(f"🚀 开始全量批量运行实验！总数: {len(episodes)}")

    results = []
    log_file = open(os.path.join(out_dir, 'ablation_summary_log.txt'), 'w', encoding='utf-8')

    for ep in tqdm(episodes, desc="Processing"):
        try:
            images = root[ep]['images'][:] 
            poses = root[ep]['ee_pose'][:]
            n = len(images)
            dt, trim = cfg.kinematics['dt'], cfg.perception['trim_edge']

            # 1. 提取核心特征
            xi_tool, s_dot = perception.process_episode(images, poses)
            rz_tool, d_feat = xi_tool[:, 2], s_dot[:, 2]
            
            # 2. 提取基座 Z 速度 (Baseline)
            from scipy.signal import savgol_filter
            pos_z_sm = savgol_filter(poses[:, 2], 31, 3)
            rz_base = (np.diff(pos_z_sm) / dt)[trim:-trim]

            # 3. 提取面积特征 (这一步如果很慢，可以考虑 GPU 化的 OpenCV)
            area_f = []
            for i in range(1, n):
                gray = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY) if len(images[i].shape)==3 else images[i]
                _, bin_img = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
                area_f.append(-np.sum(bin_img > 0))
            
            a_dot = savgol_filter(np.diff(np.array(area_f))/dt, 15, 2)
            a_dot = np.append(a_dot, a_dot[-1])[trim:-trim]

            # 4. 执行 GPU 加速的对齐评估
            x3, y3, r3, c3, m3, _ = compute_dtw_aligned_correlation_dual_pe_gpu(rz_tool, d_feat, cfg)

            results.append({"Episode": ep, "r3": c3})
            
            if test_mode:
                print(f"   - {ep} 结果: Spearman Corr = {c3:.4f}")

        except Exception as e:
            print(f"❌ 错误 {ep}: {e}")

    # 统计总结
    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        stats_summary = df.describe().loc[['mean', 'std']]
        print(f"\n✅ {'预实验' if test_mode else '全量实验'}完成！\n{stats_summary}")
        if not test_mode:
            log_file.write(stats_summary.to_string())
    
    log_file.close()

if __name__ == "__main__":
    # 1. 先跑预实验
    run_batch_ablation(test_mode=True)
    
    # 2. 如果预实验没问题，询问用户是否继续全量跑
    # (或者直接注释掉下面这行，在确认预实验 OK 后再手动运行)
    # run_batch_ablation(test_mode=False)