# core/controller.py

import numpy as np
import cv2
from scipy.linalg import pinv, norm
from core.config_loader import IDUQConfig

class IDUQController:
    """
    Sensitivity-Aware Active Visual Servoing Controller.
    Implements Riemannian Null-Space Control and Dual-Track Jacobian Architecture.
    """
    def __init__(self, target_features: np.ndarray, config: IDUQConfig):
        self.s_star = target_features
        self.dt = config.kinematics['dt']
        self.cfg = config.controller  # 假设在 yaml 中新增了 controller 配置块
        
        self.Z_current = self.cfg.get('Z_init', 0.03)
        self.lambda_p = self.cfg.get('lambda_p', 1.5)
        self.kappa = self.cfg.get('kappa', 0.02)
        self.eta = self.cfg.get('eta', 0.8)
        
        self.lambda_task = self.cfg.get('lambda_task', 1.0)
        self.alpha_gain = self.cfg.get('alpha_gain', 0.8)
        self.beta_gain = self.cfg.get('beta_gain', 0.3)
        
        self.L_phys_integral = 0.0 
        self.prev_image = None

    def get_analytical_jacobian(self, s: np.ndarray) -> np.ndarray:
        """Computes the analytical image Jacobian based on geometric moments and depth."""
        xg, yg, a, alpha = s
        Z = self.Z_current
        a_ref = self.s_star[2] if self.s_star is not None else 1e-4
        a_norm = a / a_ref

        J = np.zeros((4, 6))
        J[0, :] = [-1/Z, 0, xg/Z, xg*yg, -(1+xg**2), yg]
        J[1, :] = [0, -1/Z, yg/Z, 1+yg**2, -xg*yg, -xg]
        J[2, :] = [0, 0, 2*a_norm/Z, 3*a_norm*yg, -3*a_norm*xg, 0]
        J[3, :] = [0, 0, 0, 0, 0, -1]
        return J

    def compute_elastodynamic_jacobian(self, s: np.ndarray) -> np.ndarray:
        """Applies the interaction compliance tensor (Lambda) to the geometric Jacobian."""
        J_geo = self.get_analytical_jacobian(s)
        Lambda = np.diag([self.eta, self.eta, self.kappa, 1.0])
        return Lambda @ J_geo

    # ... (保留 compute_physical_gradient 和 compute_geometric_gradient 逻辑) ...

    def step(self, s_t: np.ndarray, gray_img: np.ndarray, xi_last: np.ndarray):
        """Main Riemannian Control Synthesis Step."""
        if s_t is None:
            return np.zeros(6), 0, 0 
            
        e_t = s_t - self.s_star
        J_ed = self.compute_elastodynamic_jacobian(s_t)
        
        # Uncertainty Quantification
        h_phys, val_phys = self.compute_physical_gradient(gray_img, xi_last)
        h_geo, val_geo = self.compute_geometric_gradient(s_t, J_ed)
        
        # A. Primary Task Flow
        J_pinv = pinv(J_ed)
        xi_task = -self.lambda_task * (J_pinv @ e_t)
        
        # B. Secondary Null-Space Exploration Flow
        P_N = np.eye(6) - J_pinv @ J_ed
        dir_phys = h_phys / (norm(h_phys) + 1e-6)
        dir_geo = h_geo / (norm(h_geo) + 1e-6)
        
        xi_active = P_N @ (-self.alpha_gain * dir_phys - self.beta_gain * dir_geo)
        
        # Synthesis
        xi_star = xi_task + xi_active
        
        # Update internal depth state
        self.Z_current = np.clip(self.Z_current - xi_last[2] * self.dt, 0.005, 0.10)
        
        return xi_star, val_phys, val_geo