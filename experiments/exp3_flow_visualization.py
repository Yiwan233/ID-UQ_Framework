# experiments/exp3_flow_visualization.py

import os
import sys
import zarr
import numpy as np
import cv2
import cupy as cp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 引入核心库
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.config_loader import IDUQConfig
from core.perception import PhysicsAwarePerception

def run_evolution_analysis_gpu():
    cfg = IDUQConfig.from_yaml("configs/default_config.yaml")
    out_dir = os.path.join(cfg.io['output_dir'], 'EXP3_Flow_Viz_GPU')
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"🚀 开始全量时空演化分析 (GPU 加速版)...")
    root = zarr.open(cfg.io['data_path'], mode='r')
    perception = PhysicsAwarePerception(cfg)
    
    observe_frames = [50, 100, 150, 200]
    observe_steps = [1, 5]

    for ep in list(root.group_keys())[:5]: # 为了演示，只跑前5个
        ep_dir = os.path.join(out_dir, ep)
        os.makedirs(ep_dir, exist_ok=True)
        images = root[ep]['images'][:]
        
        for step in observe_steps:
            for idx in observe_frames:
                if idx + step >= len(images): continue
                
                # --- A. CPU 端 NLM 去噪 ---
                prev_clean = cv2.fastNlMeansDenoising(images[idx], None, **cfg.perception['nlm'])
                curr_clean = cv2.fastNlMeansDenoising(images[idx+step], None, **cfg.perception['nlm'])
                
                # --- B. GPU 端计算 Mask 和光流 ---
                W_cp = perception.get_confidence_mask_gpu(cp.array(curr_clean, dtype=cp.float64))
                flow = cv2.calcOpticalFlowFarneback(prev_clean, curr_clean, None, **cfg.perception['optical_flow'])
                
                # --- C. GPU 端生成 HSV 和散度热力图 ---
                flow_cp = cp.array(flow)
                mag, ang = cp.angle(flow_cp[..., 0] + 1j * flow_cp[..., 1]), cp.abs(flow_cp[..., 0] + 1j * flow_cp[..., 1])
                hsv_cp = cp.zeros((flow_cp.shape[0], flow_cp.shape[1], 3), dtype=cp.uint8)
                hsv_cp[..., 0] = ang * 180 / cp.pi / 2
                hsv_cp[..., 1] = 255
                normalized_mag = cp.asnumpy(cv2.normalize(cp.asnumpy(mag), None, 0, 255, cv2.NORM_MINMAX))
                hsv_cp[..., 2] = cp.array(normalized_mag)
                
                flow_rgb = cv2.cvtColor(cp.asnumpy(hsv_cp), cv2.COLOR_HSV2RGB)
                flow_rgb[cp.asnumpy(W_cp <= 0.5)] = 0
                
                div_map_cp = perception.calculate_affine_flow_gpu(prev_clean, curr_clean, W_cp)[2]
                div_map = cp.asnumpy(div_map_cp)
                div_map[cp.asnumpy(W_cp <= 0.5)] = np.nan
                
                # 绘图 1x3
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                fig.suptitle(f'{ep} | Frame {idx} | Step {step}', fontsize=15, fontweight='bold')
                
                axes[0].imshow(curr_clean, cmap='gray')
                axes[0].contour(valid_mask, levels=[0.5], colors='yellow', linewidths=1)
                axes[0].set_title('1: NLM Denoised B-Mode & ROI')
                axes[0].axis('off')
                
                axes[1].imshow(flow_rgb)
                axes[1].set_title('2: Edge-Preserved Flow (HSV)')
                axes[1].axis('off')
                
                vmax = np.nanpercentile(np.abs(div_map), 95) if not np.isnan(np.nanpercentile(np.abs(div_map), 95)) else 0.1
                im = axes[2].imshow(div_map, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                axes[2].set_title('3: Divergence ($D$)')
                axes[2].axis('off')
                
                divider = make_axes_locatable(axes[2])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)

                plt.tight_layout()
                plt.savefig(os.path.join(ep_dir, f'evolution_f{idx}_s{step}.png'), dpi=150)
                plt.close(fig)

    print(f"✅ 处理完毕！图表保存在 {out_dir}")

if __name__ == "__main__":
    run_evolution_analysis_gpu()