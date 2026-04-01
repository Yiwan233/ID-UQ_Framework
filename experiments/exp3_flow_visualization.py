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
from core.data_loader import safe_open_zarr

def run_evolution_analysis_gpu():
    cfg = IDUQConfig.from_yaml("configs/default_config.yaml")
    out_dir = os.path.join(cfg.io['output_dir'], 'EXP3_Flow_Viz_GPU')
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"🚀 开始全量时空演化分析 (GPU 加速版)...")
    root = safe_open_zarr(cfg.io['data_path'])
    perception = PhysicsAwarePerception(cfg)
    
    observe_frames = [50, 100, 150, 200]
    observe_steps = [1, 5]

    # --- 🎯 提取 NLM 参数 ---
    nlm = cfg.perception['nlm']
    h, tw, sw = nlm['h'], nlm['template_window'], nlm['search_window']
    
    # --- 🎯 提取光流参数 ---
    f_cfg = cfg.perception['optical_flow']

    for ep in list(root.group_keys())[:5]: # 遍历前 5 个 Episode 
        ep_dir = os.path.join(out_dir, ep)
        os.makedirs(ep_dir, exist_ok=True)
        images = root[ep]['images'][:]
        
        for step in observe_steps:
            
            for idx in observe_frames:
                if idx + step >= len(images): continue
                
                # --- A. 极其鲁棒的图像提取 ---
                img_p = images[idx]
                img_c = images[idx+step]

                # 🎯 强行降为 2D NumPy 数组
                def force_2d_gray(img):
                    img = np.squeeze(img) # 去掉 (1, H, W) 或 (H, W, 1)
                    if img.ndim == 3:
                        # 如果是 RGB (H, W, 3)
                        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    return img

                prev_gray = force_2d_gray(img_p)
                curr_gray = force_2d_gray(img_c)

                # --- B. NLM 去噪 ---
                prev_clean = cv2.fastNlMeansDenoising(prev_gray, None, h, tw, sw)
                curr_clean = cv2.fastNlMeansDenoising(curr_gray, None, h, tw, sw)
                # --- B. GPU 端计算 Mask ---
                curr_clean_cp = cp.array(curr_clean, dtype=cp.float64)
                W_cp = perception.get_confidence_mask_gpu(curr_clean_cp)
                valid_mask_np = cp.asnumpy(W_cp > 0.5) # 转回 NumPy 用于绘图
                
                # --- C. 计算稠密光流 (Farneback) ---
                flow = cv2.calcOpticalFlowFarneback(
                    prev_clean, curr_clean, None,
                    f_cfg['pyr_scale'], f_cfg['levels'], f_cfg['winsize'],
                    f_cfg['iterations'], f_cfg['poly_n'], f_cfg['poly_sigma'], f_cfg['flags']
                )
                
                # --- D. GPU 加速特征场生成 (HSV & Divergence Map) ---
                flow_cp = cp.array(flow)
                vx, vy = flow_cp[..., 0], flow_cp[..., 1]
                
                # 1. 生成 HSV 颜色图
                mag_cp = cp.sqrt(vx**2 + vy**2)
                ang_cp = (cp.arctan2(vy, vx) + cp.pi) * (180 / cp.pi / 2) # 映射到 [0, 180]
                
                hsv_cp = cp.zeros((flow_cp.shape[0], flow_cp.shape[1], 3), dtype=cp.uint8)
                hsv_cp[..., 0] = ang_cp.astype(cp.uint8)
                hsv_cp[..., 1] = 255
                # 归一化强度
                mag_max = cp.max(mag_cp) if cp.max(mag_cp) > 0 else 1
                hsv_cp[..., 2] = (mag_cp * 255 / mag_max).astype(cp.uint8)
                
                flow_rgb = cv2.cvtColor(cp.asnumpy(hsv_cp), cv2.COLOR_HSV2RGB)
                flow_rgb[~valid_mask_np] = 0 # 屏蔽背景
                
                # 2. 生成散度场热力图 (使用 Perception 模块的求导逻辑)
                from cupyx.scipy.ndimage import sobel
                dvx_dx = sobel(vx, axis=1)
                dvy_dy = sobel(vy, axis=0)
                div_map_cp = dvx_dx + dvy_dy
                div_map = cp.asnumpy(div_map_cp)
                div_map[~valid_mask_np] = np.nan # 非 ROI 设为 NaN 方便热力图显示

                # --- E. 顶级学术绘图 (1x3 布局) ---
                fig, axes = plt.subplots(1, 3, figsize=(20, 6))
                fig.suptitle(f'{ep} | Frame {idx} | Step {step} | NLM-Accelerated Flow', fontsize=16, fontweight='bold')
                
                # 子图 1: 去噪后的 B-Mode
                axes[0].imshow(curr_clean, cmap='gray')
                axes[0].contour(valid_mask_np, levels=[0.5], colors='yellow', linewidths=1, alpha=0.5)
                axes[0].set_title('1: NLM Denoised B-Mode & ROI')
                axes[0].axis('off')
                
                # 子图 2: HSV 致密光流
                axes[1].imshow(flow_rgb)
                axes[1].set_title('2: Edge-Preserved Dense Flow (HSV)')
                axes[1].axis('off')
                
                # 子图 3: 散度场热力图 (D)
                # 计算 95% 分位数用于颜色映射，避免噪声噪点干扰色阶
                vlim = np.nanpercentile(np.abs(div_map), 95) if not np.all(np.isnan(div_map)) else 0.1
                im = axes[2].imshow(div_map, cmap='RdBu_r', vmin=-vlim, vmax=vlim)
                axes[2].set_title('3: Volumetric Strain Map (Divergence $D$)')
                axes[2].axis('off')
                
                divider = make_axes_locatable(axes[2])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)

                plt.tight_layout()
                save_path = os.path.join(ep_dir, f'evolution_f{idx}_s{step}.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

    print(f"✅ 处理完毕！所有图表已按 Episode 分类保存在 {out_dir}")

if __name__ == "__main__":
    run_evolution_analysis_gpu()