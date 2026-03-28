# core/perception.py

import cv2
import numpy as np
import cupy as cp
from typing import Tuple, List, Optional
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
from cupyx.scipy.ndimage import sobel, gaussian_filter, grey_closing
from core.config_loader import IDUQConfig

class PhysicsAwarePerception:
    """
    Handles physical state estimation from ultrasound images and robot kinematics.
    Accelerated via CuPy for dense field operations and Vectorization for Kinematics.
    """

    def __init__(self, config: IDUQConfig):
        self.cfg = config.perception
        self.kin_cfg = config.kinematics

    def extract_kinematics(self, poses: np.ndarray) -> np.ndarray:
        """
        [🚀 优化] 完全干掉 for 循环，使用 einsum 进行批量矩阵相乘
        """
        pos = poses[:, :3]
        quats = poses[:, 3:7]
        eulers = R.from_quat(quats).as_euler('xyz', degrees=False)
        poses_6d = np.hstack([pos, eulers])
        
        smooth_poses = np.zeros_like(poses_6d)
        for dim in range(6):
            smooth_poses[:, dim] = savgol_filter(
                poses_6d[:, dim], 
                window_length=self.kin_cfg['filter_win_pose'], 
                polyorder=self.kin_cfg['poly_order_pose']
            )
            
        xi_base = np.diff(smooth_poses, axis=0) / self.kin_cfg['dt']
        
        # 批量获取旋转矩阵 (N, 3, 3)
        rot_matrices = R.from_quat(poses[:-1, 3:7]).as_matrix() 
        v_b, w_b = xi_base[:, :3], xi_base[:, 3:6]
        
        # 批量转置矩阵并进行乘法运算 (N, 3, 3) @ (N, 3) -> (N, 3)
        rot_matrices_T = np.transpose(rot_matrices, (0, 2, 1))
        v_e = np.einsum('nij,nj->ni', rot_matrices_T, v_b)
        w_e = np.einsum('nij,nj->ni', rot_matrices_T, w_b)
        
        xi_tool = np.concatenate([v_e, w_e], axis=1)
        return xi_tool

    def get_confidence_mask_gpu(self, gray_img_cp: cp.ndarray) -> cp.ndarray:
        """
        [🚀 优化] 在 GPU 上执行形态学、平滑和梯度计算
        """
        # 1. 阈值分割 & 形态学闭运算
        fan_mask = (gray_img_cp > 5).astype(cp.float64)
        fan_mask = grey_closing(fan_mask, size=(15, 15))
        
        # 2. 高斯平滑
        smooth_gray = gaussian_filter(gray_img_cp, sigma=1.0) # sigma=1 近似 5x5 kernel
        
        # 3. Sobel 梯度计算
        grad_x = sobel(smooth_gray, axis=1)
        grad_y = sobel(smooth_gray, axis=0)
        grad_mag_sq = grad_x**2 + grad_y**2
        
        # 4. 掩码组合
        H_mask = (smooth_gray > self.cfg['masking']['noise_floor']).astype(cp.float64)
        sigma_sq = self.cfg['masking']['sigma_sq']
        
        W = cp.exp(-grad_mag_sq / (2 * sigma_sq)) * H_mask * fan_mask
        return W

    def calculate_affine_flow_gpu(self, prev_img: np.ndarray, curr_img: np.ndarray, W_mask_cp: cp.ndarray) -> cp.ndarray:
        """
        [🚀 优化] Farneback 留在 CPU，但散度和旋度的密集求导全部推给 GPU
        """
        if prev_img is None:
            return cp.array([0.0, 0.0, 0.0, 0.0])
            
        flow_args = self.cfg['optical_flow']
        # OpenCV 的 Farneback 默认使用 CPU (除非编译了 cv2.cuda)
        flow = cv2.calcOpticalFlowFarneback(
            prev_img, curr_img, None,
            flow_args['pyr_scale'], flow_args['levels'], flow_args['winsize'],
            flow_args['iterations'], flow_args['poly_n'], flow_args['poly_sigma'],
            flow_args['flags']
        )
        
        # 将计算好的光流场推送到 GPU
        flow_cp = cp.array(flow)
        vx, vy = flow_cp[..., 0], flow_cp[..., 1]
        valid_pixels = W_mask_cp > 0.5
        
        if not cp.any(valid_pixels):
            return cp.array([0.0, 0.0, 0.0, 0.0])
            
        tx = cp.mean(vx[valid_pixels])
        ty = cp.mean(vy[valid_pixels])
        
        # GPU 加速的偏导数计算
        dvx_dx = sobel(vx, axis=1)
        dvx_dy = sobel(vx, axis=0)
        dvy_dx = sobel(vy, axis=1)
        dvy_dy = sobel(vy, axis=0)
        
        div = cp.mean((dvx_dx + dvy_dy)[valid_pixels])
        curl = cp.mean((dvy_dx - dvx_dy)[valid_pixels])
        
        return cp.array([tx, ty, div, curl])

    def process_episode(self, images: np.ndarray, poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xi = self.extract_kinematics(poses)
        
        s_dot = []
        prev_clean = None
        nlm_cfg = self.cfg['nlm']
        
        for i in range(len(images)):
            curr_img = images[i]
            curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY) if len(curr_img.shape) == 3 else curr_img
            
            # NLM 降噪 (CPU)
            curr_clean = cv2.fastNlMeansDenoising(
                curr_gray, None, 
                h=nlm_cfg['h'], 
                templateWindowSize=nlm_cfg['template_window'], 
                searchWindowSize=nlm_cfg['search_window']
            )
            
            # 将清理后的图像转入 GPU
            curr_clean_cp = cp.array(curr_clean, dtype=cp.float64)
            
            # GPU 计算 Mask
            W_cp = self.get_confidence_mask_gpu(curr_clean_cp)
            
            if prev_clean is not None:
                # 混合计算光流与仿射特征
                affine_feat_cp = self.calculate_affine_flow_gpu(prev_clean, curr_clean, W_cp)
                s_dot.append(affine_feat_cp.get()) # 拉回 CPU 暂存
            else:
                s_dot.append(np.array([0.0, 0.0, 0.0, 0.0]))
                
            prev_clean = curr_clean
            
        s_dot = np.array(s_dot)[1:]  
        
        # 滤波保留在 CPU，因为 1D 数组 Savgol Filter CPU 速度极快
        for dim in range(4):
            s_dot[:, dim] = savgol_filter(
                s_dot[:, dim], 
                window_length=self.cfg['filter_win_feat'], 
                polyorder=self.cfg['poly_order_feat']
            )
            
        m = self.cfg['trim_edge']
        return xi[m:-m], s_dot[m:-m]