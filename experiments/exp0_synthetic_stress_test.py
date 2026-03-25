import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
from core.perception import PhysicsAwarePerception
from core.config_loader import IDUQConfig

logger = logging.getLogger(__name__)

class IDUQStressTester:
    def __init__(self, config_path: str):
        # 使用类方法 from_yaml 来加载字符串路径
        self.cfg = IDUQConfig.from_yaml(config_path) 
        self.perception = PhysicsAwarePerception(self.cfg)
        self.mask_size = (512, 512)

    def _generate_synthetic_texture(self) -> np.ndarray:
        """生成模拟超声散斑纹理的高斯随机场，用于基础图像源"""
        base = np.random.normal(128, 30, self.mask_size).astype(np.uint8)
        return cv2.GaussianBlur(base, (7, 7), 0)

    def test_pure_rotation(self, angle_range: float = 5.0, step: float = 1.0):
        """
        Test A: 验证纯原位旋转下 D 与 R 的解耦能力。
        物理期望：R 随角度线性增加，D 理论上保持为 0（体积不变量）。
        """
        logger.info("Starting Test A: Kinematic Singularity (Pure Rotation)...")
        img_source = self._generate_synthetic_texture()
        center = (self.mask_size[1] // 2, self.mask_size[0] // 2)
        
        angles = np.arange(-angle_range, angle_range + step, step)
        results = {"angle": [], "D": [], "R": []}

        # 预先生成 W 掩膜 (假设全域有效)
        W = np.ones(self.mask_size, dtype=np.float64)

        prev_img = img_source
        for ang in angles:
            # 执行纯刚体旋转变换
            rot_mat = cv2.getRotationMatrix2D(center, ang, 1.0)
            curr_img = cv2.warpAffine(img_source, rot_mat, self.mask_size, flags=cv2.INTER_LANCZOS4)
            
            # 提取仿射特征 [tx, ty, D, R]
            feats = self.perception.calculate_affine_flow(prev_img, curr_img, W)
            
            results["angle"].append(ang)
            results["D"].append(feats[2])
            results["R"].append(feats[3])
            
        return results

    def test_snr_breakdown(self, noise_max: float = 0.5, steps: int = 20):
        """
        Test B: 通过人为注入乘性斑点噪声，寻找散度提取的数值崩溃临界点。
        Speckle Noise Model: I_noisy = I + I * N(0, sigma^2)
        """
        logger.info("Starting Test B: SNR Breakdown (Multiplicative Speckle)...")
        
        # 生成一对带有真实形变（人为拉伸 2%）的图像作为 Ground Truth
        img1 = self._generate_synthetic_texture()
        # 构造微小拉伸矩阵 (散度 D 应约为 0.02)
        scale_val = 1.02
        pts1 = np.float32([[0,0], [511,0], [0,511]])
        pts2 = np.float32([[0,0], [511*scale_val,0], [0,511*scale_val]])
        M_scale = cv2.getAffineTransform(pts1, pts2)
        img2 = cv2.warpAffine(img1, M_scale, self.mask_size, flags=cv2.INTER_LANCZOS4)
        
        # 计算无噪声时的真值
        W = np.ones(self.mask_size, dtype=np.float64)
        gt_feats = self.perception.calculate_affine_flow(img1, img2, W)
        D_gt = gt_feats[2]
        
        variances = np.linspace(0.0, noise_max, steps)
        errors = []

        for sigma_sq in variances:
            # 注入乘性斑点噪声
            noise = np.random.normal(0, np.sqrt(sigma_sq), self.mask_size)
            noisy_img1 = np.clip(img1 + img1 * noise, 0, 255).astype(np.uint8)
            noisy_img2 = np.clip(img2 + img2 * noise, 0, 255).astype(np.uint8)
            
            # 在污染后的图像中提取特征
            noisy_feats = self.perception.calculate_affine_flow(noisy_img1, noisy_img2, W)
            D_noisy = noisy_feats[2]
            
            # 计算相对误差
            rel_error = abs(D_noisy - D_gt) / (abs(D_gt) + 1e-9)
            errors.append(rel_error)
        return variances, errors

    def run_all_tests(self):
        """运行完整测试套件并生成学术拼图"""
        res_a = self.test_pure_rotation()
        noise_vars, res_b = self.test_snr_breakdown()

        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(14, 6))
        plt.subplots_adjust(wspace=0.35)

        # --- Test A Plot ---
        ax1.set_title("Test A: Kinematic Decoupling (Pure Rotation)", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Rotation Angle [deg]")
        ax1.set_ylabel("Curl (R) [Response]", color='tab:blue')
        lns1 = ax1.plot(res_a["angle"], res_a["R"], 'o-', color='tab:blue', label="Curl (R)")
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Divergence (D) [Interference]", color='tab:red')
        lns2 = ax2.plot(res_a["angle"], res_a["D"], 's--', color='tab:red', label="Divergence (D)")
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.set_ylim(-0.01, 0.01) # 强行锁定 Y 轴以观察微小波动

        # 合并图例
        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper left')

        # --- Test B Plot ---
        ax3.set_title("Test B: SNR Breakdown Analysis", fontsize=12, fontweight='bold')
        ax3.set_xlabel("Speckle Noise Variance ($\sigma^2$)")
        ax3.set_ylabel("Relative Error in D extraction")
        ax3.plot(noise_vars, res_b, 'D-', color='purple', linewidth=2)
        
        # 标记 15% 崩溃阈值
        threshold = 0.15
        ax3.axhline(y=threshold, color='red', linestyle=':', label="15% Error Threshold")
        
        # 寻找临界点
        critical_idx = np.where(np.array(res_b) > threshold)[0]
        if len(critical_idx) > 0:
            crit_var = noise_vars[critical_idx[0]]
            ax3.annotate(f'Breakdown point\n($\sigma^2$={crit_var:.2f})', 
                        xy=(crit_var, threshold), xytext=(crit_var+0.05, threshold+0.1),
                        arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)

        ax3.set_ylim(0, 0.6)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = "stress_test_results.pdf"
        plt.savefig(save_path, dpi=300)
        logger.info(f"Stress test complete. Results saved to {save_path}")
        plt.show()

if __name__ == "__main__":
    # 自动定位到项目根目录下的 config
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "configs", "default_config.yaml")
    
    if not os.path.exists(config_path):
        print(f"Error: Cannot find config at {config_path}")
    else:
        tester = IDUQStressTester(config_path=config_path)
        tester.run_all_tests()