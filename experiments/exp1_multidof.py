# experiments/exp1_multidof.py

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from core.config_loader import IDUQConfig
from core.perception import PhysicsAwarePerception
from core.data_loader import safe_open_zarr, get_episode_data

def compute_best_correlation_pe_gated(sig_robot, sig_feat, pe_ratio=0.15, max_lag=25):
    """Cross-correlation alignment with persistent excitation (PE) gating."""
    n = len(sig_robot)
    if n < 20:
        return sig_robot, sig_feat, 0.0, 0.0, np.ones(n, dtype=bool)

    scaler = StandardScaler()
    sig_robot_norm = scaler.fit_transform(sig_robot.reshape(-1, 1)).flatten()
    sig_feat_norm = scaler.fit_transform(sig_feat.reshape(-1, 1)).flatten()

    best_offset, best_corr = 0, -1.0
    for offset in range(-max_lag, max_lag + 1):
        x_shifted = np.roll(sig_robot_norm, offset)
        y_ref = sig_feat_norm
        if offset > 0:
            mask = np.arange(n) >= offset
        elif offset < 0:
            mask = np.arange(n) < n + offset
        else:
            mask = np.ones(n, dtype=bool)
        if np.sum(mask) < 15:
            continue
        corr_val = abs(np.corrcoef(x_shifted[mask], y_ref[mask])[0, 1])
        if corr_val > best_corr:
            best_corr = corr_val
            best_offset = offset

    x_aligned = np.roll(sig_robot_norm, best_offset)
    if best_offset > 0:
        x_aligned[:best_offset] = x_aligned[best_offset]
    elif best_offset < 0:
        x_aligned[best_offset:] = x_aligned[best_offset - 1]

    x_al_raw = np.roll(sig_robot, best_offset)
    pe_threshold = max(0.001, pe_ratio * np.max(np.abs(x_al_raw)))
    pe_mask = np.abs(x_al_raw) > pe_threshold

    pe_corr = best_corr
    if np.sum(pe_mask) > 15:
        pe_corr = abs(np.corrcoef(x_aligned[pe_mask], sig_feat_norm[pe_mask])[0, 1])

    return x_aligned, sig_feat_norm, best_corr, pe_corr, pe_mask

def run_batch_multi_dof_experiment():
    cfg = IDUQConfig.from_yaml("configs/default_config.yaml")
    output_dir = os.path.join(cfg.io['output_dir'], 'EXP1_MultiDOF')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🚀 开始全自由度遍历！结果将保存在 '{output_dir}'")
    root = safe_open_zarr(cfg.io['data_path'])
    episodes = list(root.group_keys())

    perception = PhysicsAwarePerception(cfg)

    log_file_path = os.path.join(output_dir, 'multi_dof_summary_log.txt')
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        def log_print(msg):
            print(msg)
            log_file.write(msg + '\n')

        log_print("="*70)
        log_print("全自由度解耦与可观测性报告 (Multi-DOF Decoupling Report)")
        log_print("="*70)

        for ep in episodes:
            log_print(f"\n>>> 开始处理 {ep} ...")
            images, poses = get_episode_data(root, ep)
            
            # 🔥 1. 一键提取特征
            xi_tool, s_dot = perception.process_episode(images, poses)
            
            mappings = [
                (0, 0, "Trans X ($v_x$) vs Flow X ($t_x$)", "royalblue"),
                (1, 1, "Trans Y ($v_y$) vs Flow Y ($t_y$)", "darkorange"),
                (2, 2, "Trans Z ($v_z$) vs Divergence ($D$)", "crimson"),
                (5, 3, "Roll Rot ($\omega_z$) vs Curl ($R$)", "purple")
            ]
            
            sns.set_theme(style="whitegrid")
            fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
            fig.suptitle(f'Multi-DOF Kinematic-Affine Tracking ({ep})', fontsize=18, fontweight='bold', y=0.97)
            
            for idx, (robot_dim, img_dim, title, color) in enumerate(mappings):
                robot_sig = xi_tool[:, robot_dim]
                img_sig = s_dot[:, img_dim]
                
                # 计算门控相关性
                x_al, y_al, raw_corr, pe_corr, pe_mask = compute_best_correlation_pe_gated(robot_sig, img_sig)
                log_print(f"  [{title.split(' ')[0]}] 全局 r={raw_corr:.4f} -> PE有效 r={pe_corr:.4f}")
                
                ax = axes[idx]
                ax.plot(x_al, label='Robot Command (Tool)', color='dimgray', linewidth=1.5)
                ax.plot(y_al, label='Affine Flow Obs', color=color, linestyle='--', linewidth=2)
                ax.fill_between(range(len(x_al)), -4, 4, where=pe_mask, color='lightgreen', alpha=0.25, label='PE Active Region')
                
                ax.set_title(f'{title} | PE Gated $r = {pe_corr:.2f}$', fontsize=13)
                ax.set_ylim(-3.5, 3.5)
                ax.legend(loc='upper right')
                if idx in [2, 3]: ax.set_facecolor('#fdf6e3') 

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_path = os.path.join(output_dir, f'multi_dof_tracking_{ep}.png')
            fig.savefig(save_path, dpi=300)
            plt.close(fig) 
            log_print(f"  > 图表已保存至: {save_path}")

if __name__ == "__main__":
    run_batch_multi_dof_experiment()