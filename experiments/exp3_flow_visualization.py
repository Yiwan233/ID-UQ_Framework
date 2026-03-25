# experiments/exp3_flow_visualization.py

import os
import zarr
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from core.config_loader import IDUQConfig
from core.perception import PhysicsAwarePerception

def run_evolution_analysis():
    cfg = IDUQConfig.from_yaml("configs/default_config.yaml")
    out_dir = cfg.io.get('output_dir_exp3', 'Results_EXP3_Evolution_NLM')
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"🚀 开始全量时空演化分析 (已开启 Non-Local Means 高级超声去噪)...")
    root = zarr.open(cfg.io['data_path'], mode='r')
    perception = PhysicsAwarePerception(cfg)
    
    observe_frames = [50, 100, 150, 200]
    observe_steps = [1, 5]
    nlm_cfg = cfg.perception['nlm']

    for ep in root.group_keys():
        ep_dir = os.path.join(out_dir, ep)
        os.makedirs(ep_dir, exist_ok=True)
        images = root[ep]['images'][:]
        
        for step in observe_steps:
            for idx in observe_frames:
                if idx + step >= len(images): continue
                
                prev_gray = cv2.cvtColor(images[idx], cv2.COLOR_RGB2GRAY) if len(images[idx].shape)==3 else images[idx]
                curr_gray = cv2.cvtColor(images[idx+step], cv2.COLOR_RGB2GRAY) if len(images[idx+step].shape)==3 else images[idx+step]
                
                # NLM 去噪
                prev_clean = cv2.fastNlMeansDenoising(prev_gray, None, h=nlm_cfg['h'], 
                                                      templateWindowSize=nlm_cfg['template_window'], 
                                                      searchWindowSize=nlm_cfg['search_window'])
                curr_clean = cv2.fastNlMeansDenoising(curr_gray, None, h=nlm_cfg['h'], 
                                                      templateWindowSize=nlm_cfg['template_window'], 
                                                      searchWindowSize=nlm_cfg['search_window'])
                
                valid_mask = perception.get_confidence_mask(curr_clean) > 0.5
                
                # 光流场
                flow_args = cfg.perception['optical_flow']
                flow = cv2.calcOpticalFlowFarneback(
                    prev_clean, curr_clean, None,
                    flow_args['pyr_scale'], flow_args['levels'], flow_args['winsize'],
                    flow_args['iterations'], flow_args['poly_n'], flow_args['poly_sigma'], flow_args['flags']
                )
                vx, vy = flow[..., 0], flow[..., 1]
                
                # HSV 转换
                hsv = np.zeros((curr_gray.shape[0], curr_gray.shape[1], 3), dtype=np.uint8)
                hsv[..., 1] = 255
                mag, ang = cv2.cartToPolar(vx, vy)
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                flow_rgb[~valid_mask] = 0
                
                # 稠密散度场
                div_map = cv2.Sobel(vx, cv2.CV_64F, 1, 0, ksize=3) + cv2.Sobel(vy, cv2.CV_64F, 0, 1, ksize=3)
                div_map[~valid_mask] = np.nan
                
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
    run_evolution_analysis()