# core/perception.py

import cv2
import numpy as np
from typing import Tuple, List, Optional
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
from core.config_loader import IDUQConfig

class PhysicsAwarePerception:
    """
    Handles physical state estimation from ultrasound images and robot kinematics.
    Extracts affine optical flow features (Divergence, Curl) representing volumetric 
    continuum mechanics and transforms base velocities to the end-effector frame.
    """

    def __init__(self, config: IDUQConfig):
        self.cfg = config.perception
        self.kin_cfg = config.kinematics

    def extract_kinematics(self, poses: np.ndarray) -> np.ndarray:
        """
        Extracts and smooths the robot end-effector twist (velocities) from raw poses.
        
        Args:
            poses (np.ndarray): Array of shape (N, 7) containing [x, y, z, qx, qy, qz, qw].
            
        Returns:
            np.ndarray: Array of shape (N-1, 6) containing the tool-frame twist [vx, vy, vz, wx, wy, wz].
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
        
        xi_tool = []
        for i in range(len(xi_base)):
            q = poses[i, 3:7]
            rot_matrix = R.from_quat(q).as_matrix()
            v_b, w_b = xi_base[i, :3], xi_base[i, 3:6]
            # Adjoint transformation to tool frame
            v_e = rot_matrix.T @ v_b
            w_e = rot_matrix.T @ w_b
            xi_tool.append(np.concatenate([v_e, w_e]))
            
        return np.array(xi_tool)

    def get_confidence_mask(self, gray_img: np.ndarray) -> np.ndarray:
        """
        Generates a continuous spatial confidence mask W to isolate the acoustic window.
        Suppresses out-of-distribution acoustic shadows based on intensity and gradients.
        
        Args:
            gray_img (np.ndarray): Denoised 2D ultrasound image.
            
        Returns:
            np.ndarray: Continuous weight matrix W \in [0, 1].
        """
        _, fan_mask = cv2.threshold(gray_img, 5, 1, cv2.THRESH_BINARY)
        kernel = np.ones((15, 15), np.uint8)
        fan_mask = cv2.morphologyEx(fan_mask, cv2.MORPH_CLOSE, kernel).astype(np.float64)
        
        smooth_gray = cv2.GaussianBlur(gray_img, (5, 5), 0)
        grad_x = cv2.Sobel(smooth_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(smooth_gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag_sq = grad_x**2 + grad_y**2
        
        H_mask = (smooth_gray > self.cfg['masking']['noise_floor']).astype(np.float64)
        sigma_sq = self.cfg['masking']['sigma_sq']
        
        W = np.exp(-grad_mag_sq / (2 * sigma_sq)) * H_mask * fan_mask
        return W

    def calculate_affine_flow(self, prev_img: np.ndarray, curr_img: np.ndarray, W_mask: np.ndarray) -> np.ndarray:
        """
        Calculates the dense optical flow and extracts affine differential kinematics.
        Divergence (D) represents out-of-plane axial compression.
        
        Args:
            prev_img: Previous clean frame.
            curr_img: Current clean frame.
            W_mask: Confidence mask isolating valid tissue regions.
            
        Returns:
            np.ndarray: Vector [tx, ty, Divergence, Curl].
        """
        if prev_img is None:
            return np.array([0.0, 0.0, 0.0, 0.0])
            
        flow_args = self.cfg['optical_flow']
        flow = cv2.calcOpticalFlowFarneback(
            prev_img, curr_img, None,
            flow_args['pyr_scale'], flow_args['levels'], flow_args['winsize'],
            flow_args['iterations'], flow_args['poly_n'], flow_args['poly_sigma'],
            flow_args['flags']
        )
        
        vx, vy = flow[..., 0], flow[..., 1]
        valid_pixels = W_mask > 0.5
        
        if not np.any(valid_pixels):
            return np.array([0.0, 0.0, 0.0, 0.0])
            
        tx, ty = np.mean(vx[valid_pixels]), np.mean(vy[valid_pixels])
        
        dvx_dx = cv2.Sobel(vx, cv2.CV_64F, 1, 0, ksize=3)
        dvx_dy = cv2.Sobel(vx, cv2.CV_64F, 0, 1, ksize=3)
        dvy_dx = cv2.Sobel(vy, cv2.CV_64F, 1, 0, ksize=3)
        dvy_dy = cv2.Sobel(vy, cv2.CV_64F, 0, 1, ksize=3)
        
        div = np.mean((dvx_dx + dvy_dy)[valid_pixels])
        curl = np.mean((dvy_dx - dvx_dy)[valid_pixels])
        
        return np.array([tx, ty, div, curl])

    def process_episode(self, images: np.ndarray, poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main perception pipeline: Extract twist and visual divergence.
        """
        xi = self.extract_kinematics(poses)
        
        s_dot = []
        prev_clean = None
        nlm_cfg = self.cfg['nlm']
        
        for i in range(len(images)):
            curr_img = images[i]
            curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY) if len(curr_img.shape) == 3 else curr_img
            
            # Robust Edge-Preserving Denoising (NLM)
            curr_clean = cv2.fastNlMeansDenoising(
                curr_gray, None, 
                h=nlm_cfg['h'], 
                templateWindowSize=nlm_cfg['template_window'], 
                searchWindowSize=nlm_cfg['search_window']
            )
            
            W = self.get_confidence_mask(curr_clean)
            
            if prev_clean is not None:
                affine_feat = self.calculate_affine_flow(prev_clean, curr_clean, W)
                s_dot.append(affine_feat)
            else:
                s_dot.append(np.array([0.0, 0.0, 0.0, 0.0]))
                
            prev_clean = curr_clean
            
        s_dot = np.array(s_dot)[1:]  # Match difference dimensionality
        
        for dim in range(4):
            s_dot[:, dim] = savgol_filter(
                s_dot[:, dim], 
                window_length=self.cfg['filter_win_feat'], 
                polyorder=self.cfg['poly_order_feat']
            )
            
        m = self.cfg['trim_edge']
        return xi[m:-m], s_dot[m:-m]