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
    def __init__(self, config: IDUQConfig):
        self.cfg = config.perception
        self.kin_cfg = config.kinematics

    def extract_kinematics(self, poses: np.ndarray) -> np.ndarray:
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
            
        step = self.cfg.get('step', 1)
        xi_base = (smooth_poses[step:] - smooth_poses[:-step]) / (self.kin_cfg['dt'] * step)
        
        # 修复 1: 确保旋转矩阵的切片长度与 xi_base 严格一致
        rot_matrices = R.from_quat(poses[:-step, 3:7]).as_matrix() 
        v_b, w_b = xi_base[:, :3], xi_base[:, 3:6]
        
        rot_matrices_T = np.transpose(rot_matrices, (0, 2, 1))
        v_e = np.einsum('nij,nj->ni', rot_matrices_T, v_b)
        w_e = np.einsum('nij,nj->ni', rot_matrices_T, w_b)
        
        xi_tool = np.concatenate([v_e, w_e], axis=1)
        return xi_tool

    def get_confidence_mask_gpu(self, gray_img_cp: cp.ndarray) -> cp.ndarray:
        gray_img_cp = gray_img_cp.squeeze()
        if gray_img_cp.ndim == 3:
            gray_img_cp = gray_img_cp[..., 0]
        if gray_img_cp.ndim != 2:
            raise ValueError(f"GPU感知器期望2D图像，但收到 {gray_img_cp.shape}")

        fan_mask = (gray_img_cp > 5).astype(cp.float64)
        fan_mask = grey_closing(fan_mask, size=15)
        smooth_gray = gaussian_filter(gray_img_cp, sigma=1.0) 
        
        grad_x = sobel(smooth_gray, axis=1)
        grad_y = sobel(smooth_gray, axis=0)
        grad_mag_sq = grad_x**2 + grad_y**2
        
        H_mask = (smooth_gray > self.cfg['masking']['noise_floor']).astype(cp.float64)
        sigma_sq = self.cfg['masking']['sigma_sq']
        
        W = cp.exp(-grad_mag_sq / (2 * sigma_sq)) * H_mask * fan_mask
        return W

    def calculate_affine_flow_gpu(self, prev_img: np.ndarray, curr_img: np.ndarray, W_mask_cp: cp.ndarray) -> cp.ndarray:
        if prev_img is None:
            return cp.array([0.0, 0.0, 0.0, 0.0])
            
        flow_args = self.cfg['optical_flow']
        flow = cv2.calcOpticalFlowFarneback(
            prev_img, curr_img, None,
            flow_args['pyr_scale'], flow_args['levels'], flow_args['winsize'],
            flow_args['iterations'], flow_args['poly_n'], flow_args['poly_sigma'], flow_args['flags']
        )
        
        flow_cp = cp.array(flow)
        vx, vy = flow_cp[..., 0], flow_cp[..., 1]
        valid_pixels = W_mask_cp > 0.5
        
        if not cp.any(valid_pixels):
            return cp.array([0.0, 0.0, 0.0, 0.0])
            
        tx = cp.mean(vx[valid_pixels])
        ty = cp.mean(vy[valid_pixels])
        
        dvx_dx = sobel(vx, axis=1)
        dvx_dy = sobel(vx, axis=0)
        dvy_dx = sobel(vy, axis=1)
        dvy_dy = sobel(vy, axis=0)
        
        div = cp.mean((dvx_dx + dvy_dy)[valid_pixels])
        curl = cp.mean((dvy_dx - dvx_dy)[valid_pixels])
        
        return cp.array([tx, ty, div, curl])

    def process_episode(self, images: np.ndarray, poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        step = self.cfg.get('step', 1)
        xi = self.extract_kinematics(poses)
        
        s_dot = []
        prev_clean = None
        
        nlm_cfg = self.cfg['nlm']
        h_val, tw_val, sw_val = nlm_cfg.get('h', 8), nlm_cfg.get('template_window', 5), nlm_cfg.get('search_window', 21)
        
        for i in range(step, len(images)):
            curr_img = images[i]
            curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY) if len(curr_img.shape) == 3 else curr_img
            curr_clean = cv2.fastNlMeansDenoising(curr_gray, None, h_val, tw_val, sw_val)
            
            curr_clean_cp = cp.array(curr_clean, dtype=cp.float64)
            W_cp = self.get_confidence_mask_gpu(curr_clean_cp)
            
            if prev_clean is not None:
                affine_feat_cp = self.calculate_affine_flow_gpu(prev_clean, curr_clean, W_cp)
                s_dot.append(affine_feat_cp.get()) 
            else:
                s_dot.append(np.array([0.0, 0.0, 0.0, 0.0]))
            prev_clean = curr_clean
            
        # 修复 2: s_dot 移除起始 dummy 帧后，xi 同步丢弃第一帧以保证维度完全对齐
        s_dot = np.array(s_dot)[1:]  
        xi = xi[1:] 
        
        for dim in range(4):
            s_dot[:, dim] = savgol_filter(
                s_dot[:, dim], 
                window_length=self.cfg['filter_win_feat'], 
                polyorder=self.cfg['poly_order_feat']
            )
            
        m = self.cfg['trim_edge']
        return xi[m:-m], s_dot[m:-m]