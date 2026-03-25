# tools/offline_calibration.py

import zarr
import numpy as np
from sklearn.linear_model import HuberRegressor
from core.config_loader import IDUQConfig
from core.perception import PhysicsAwarePerception

def calibrate_robust_prior_jacobian():
    """
    Offline tool to calibrate the robust prior Jacobian J_prior using Huber Regression.
    Generates the baseline interaction matrix for the QP Controller.
    """
    print("🛠️ Starting Offline Robust Jacobian Calibration...")
    cfg = IDUQConfig.from_yaml("configs/default_config.yaml")
    root = zarr.open(cfg.io['data_path'], mode='r')
    perception = PhysicsAwarePerception(cfg)
    
    # 抽取前几个 Episode 进行联合标定
    episodes = sorted(list(root.group_keys()))[:3]
    all_xi, all_s_dot = [], []
    trim = cfg.perception['trim_edge']
    
    for ep in episodes:
        images, poses = root[ep]['images'][:], root[ep]['poses'][:]
        xi = perception.extract_kinematics(poses)
        
        s_dot = []
        prev_img = None
        for img in images:
            curr_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
            if prev_img is not None:
                W = perception.get_confidence_mask(curr_gray)
                s_dot.append(perception.calculate_affine_flow(prev_img, curr_gray, W))
            else:
                s_dot.append(np.array([0,0,0,0]))
            prev_img = curr_gray
            
        all_xi.append(xi[trim:-trim])
        all_s_dot.append(np.array(s_dot)[trim:-trim])
        
    X_raw = np.vstack(all_xi)     
    Y_raw = np.vstack(all_s_dot)  

    # (此处可以接入你原有的 Time Alignment 逻辑寻找最佳 offset)
    X, Y = X_raw, Y_raw 
    
    print("\nPerforming Robust Huber Regression...")
    J_prior = np.zeros((4, 6))
    for i in range(4):
        huber = HuberRegressor(epsilon=1.35, alpha=10.0, fit_intercept=False)
        huber.fit(X, Y[:, i]) 
        J_prior[i, :] = huber.coef_

    # 施加稀疏结构先验 (Sparsity Thresholding)
    threshold = 0.05 * np.max(np.abs(J_prior))
    J_prior[np.abs(J_prior) < threshold] = 0.0
    
    print("\n===============================================================")
    print("✅ FINAL CLINICAL-GRADE 4x6 JACOBIAN (J_prior)")
    print("===============================================================")
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    print(J_prior)
    print("===============================================================")
    print(">> You can now copy this matrix into your config.yaml or controller.")

if __name__ == "__main__":
    calibrate_robust_prior_jacobian()