

# ID-UQ: Interaction-Driven Uncertainty Quantification for Ultrasound Visual Servoing

[![Paper](https://img.shields.io/badge/Paper-ICRA%2FIROS-blue.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Official codebase for the paper: **"Methodology: Sensitivity-Aware Active Visual Servoing via Interaction-Driven Uncertainty Quantification"**.

## 📖 Abstract

Autonomous robotic ultrasound frequently suffers from a critical failure mode: the inability to robustly differentiate "loss of contact" from "in-plane slip" under severe acoustic artifacts. Standard computer vision algorithms treat out-of-plane tissue compression as tracking noise, leading to catastrophic control failures.

This repository implements the **Interaction-Driven Uncertainty Quantification (ID-UQ)** framework. Drawing inspiration from continuous mechanics, we extract the **affine flow divergence** as a physically meaningful proxy for volumetric tissue strain. By embedding this into a novel **Dual-Track Jacobian Architecture** and explicitly decoupling uncertainty into **Physical Risk** and **Geometric Sensitivity**, we provide a mathematically rigorous, sensorless anomaly metric for safe human-robot interaction.

---

## 🔥 Core Theoretical Innovations

### 1. Dual-Track Jacobian Architecture (解决估计与控制的恶性耦合)
In dynamic soft tissue interaction, using a noise-polluted online Jacobian directly for feedback control induces high-frequency actuator oscillation (Estimator-Controller Malignant Coupling). We solve this by strictly separating the Control and Assessment domains:

*   **Control Domain (Stability Baseline):** The closed-loop QP optimizer strictly uses a fixed prior affine Jacobian ($\mathbf{J}_{prior}$). Its constant anisotropy severs the transmission of estimation noise to control commands, guaranteeing basic Lyapunov asymptotic stability.
*   **Assessment Domain (Digital Twin):** A dynamic affine Jacobian ($\hat{\mathbf{M}}_k$) is estimated online via PE-Gated Recursive Least Squares (RLS). It acts strictly as an observer to infer the expected affine kinematics and compute the physical residual, bounding interaction uncertainty.

### 2. Uncertainty Decoupling: The "Two Eyes" of the System (不确定性正交解耦)
We formalize the problem as a constrained optimization on a Riemannian manifold, governed by a dual-layer interaction model:

*   👁️ **The Physical Eye (Contact Risk $\mathcal{R}_{phys}$):** 
    Evaluates the **Phase-Aligned Affine-Kinematic Residual**. It computes the Mahalanobis distance between the delayed visual flow divergence and the expected tissue response projected from the robot's delayed $Z$-axis twist, modulated by acoustic SSIM:
    $$ \mathcal{R}_{phys} = \left\| \dot{\mathbf{s}}_{meas} - [\hat{\mathbf{M}}_k] \boldsymbol{\xi}_e \right\|_{\Sigma^{-1}} \cdot \exp\big( \gamma \cdot (1 - \text{SSIM}) \big) $$
    This metric isolates genuine contact failures (probe slip/air gaps) from benign tissue viscoelastic lag.

*   👁️ **The Geometric Eye (Acoustic Observability $\mathbf{h}_{geo}$):**
    Utilizes **Inertial Gradient Inference** to actively seek optimal acoustic windows. Instead of fragile numerical differentiation, it uses action-feedback logic (Extremum Seeking Control) to guide the probe's null-space redundant degrees of freedom (e.g., Roll/Pitch) to maximize image quality.

---

## 📂 Repository Structure

The codebase is organized into a modular core engine (continuous mechanics pipeline) and independent experiment scripts for reproducing all paper figures.

```text
ID-UQ_Framework/
├── configs/
│   └── default_config.yaml        # Unified parameter center (Zero Magic Numbers)
├── core/
│   ├── config_loader.py           # Type-safe YAML configuration parsing
│   ├── perception.py              # GPU-accelerated continuum mechanics: NLM, Adjoint Twist, Affine Divergence
│   ├── alignment.py               # Cross-correlation phase alignment & residual generation
│   └── data_loader.py             # Robust Zarr I/O with metadata healing for V2/V3 compatibility
├── experiments/
│   ├── exp0_synthetic_stress_test.py   # Synthetic validation: rotation decoupling & SNR breakdown
│   ├── exp1_ablation.py               # Ablation study: feature observability across motion complexity
│   ├── exp1_diagnostic_tails.py        # Diagnostic analysis of failure modes (tail pathology)
│   ├── exp1_multidof.py               # Multi-DOF kinematic-affine tracking visualization
│   ├── exp2_anomaly_roc.py            # Single-episode anomaly detection ROC with SSIM baseline
│   ├── exp3_flow_visualization.py     # Dense flow & divergence heatmap evolution (GPU-accelerated)
│   ├── exp4_kinematic_residual.py     # Forward tracking with 3σ statistical confidence bounds
│   ├── exp5_roc_auc_evaluation.py     # Global ROC-AUC evaluation across 300+ episodes
│   └── exp6_dual_eye_decoupling.py    # Dual-track orthogonal decoupling phase space analysis
└── tools/
    └── offline_calibration.py         # Offline robust Jacobian calibration (Huber regression)
```

---

## 🛠️ Installation & Setup

**1. Clone the repository and install dependencies:**
```bash
git clone https://github.com/YourUsername/ID-UQ-Visual-Servoing.git
cd ID-UQ-Visual-Servoing
pip install -r requirements.txt
```
*(Dependencies: `numpy`, `scipy`, `opencv-python`, `scikit-learn`, `zarr`, `matplotlib`, `seaborn`, `scikit-image`)*

**2. Dataset Preparation:**
Place your collected robotic ultrasound dataset (`.zarr` format) in the `data/` directory. The default path is `data/servo_dataset_dp.zarr` (configurable in `configs/default_config.yaml`). Each episode should contain synchronized robot `poses` (or `ee_pose`) and ultrasound `images`.

---

## 🧪 Reproducing Paper Experiments

Reviewers and researchers can effortlessly reproduce all figures presented in the manuscript using the scripts in the `experiments/` directory.

### [EXP 0] Synthetic Stress Test (Methodology Validation)
Validates the core decoupling hypothesis under controlled synthetic conditions: pure rotation (kinematic singularity) and multiplicative speckle noise injection. Confirms that divergence (D) remains near zero under rotation while curl (R) responds linearly, and identifies the numerical SNR breakdown threshold.
```bash
python experiments/exp0_synthetic_stress_test.py
```

### [EXP 1] Robustness of Affine Divergence (Ablation Study)
Demonstrates that affine divergence ($D$) strongly correlates with the Adjoint $Z$-velocity, completely outperforming traditional threshold-based geometric area features under acoustic shadowing.
```bash
python experiments/exp1_ablation.py
python experiments/exp1_multidof.py
python experiments/exp1_diagnostic_tails.py
```

### [EXP 2 & 5] Contact Failure Detection (GLRT & ROC-AUC)
Validates the diagnostic performance of the Phase-Aligned Physical Residual ($\mathcal{R}_{phys}$) as a generalized likelihood ratio test for detecting anomalous slips, achieving high AUC without force sensors.
```bash
python experiments/exp2_anomaly_roc.py
python experiments/exp5_roc_auc_evaluation.py
```

### [EXP 3] Spatiotemporal Flow & Continuum Mechanics Visualization
Visualizes the elastodynamic principles: NLM-denoised B-mode images alongside edge-preserved HSV flow and dense divergence heatmaps.
```bash
python experiments/exp3_flow_visualization.py
```

### [EXP 6] Dual-Eye Orthogonal Decoupling (Phase Space Analysis)
Validates the core theoretical claim that the Physical Eye (R_phys) and Geometric Eye (S_geo) are orthogonal uncertainty dimensions. Plots a four-quadrant phase space showing distinct failure modes: pure sliding, severe decoupling, acoustic shadowing, and the ideal servoing envelope.
```bash
python experiments/exp6_dual_eye_decoupling.py
```

### [EXP 4] Kinematic Residual & Uncertainty Tube
Plots the forward model tracking with a $3\sigma$ statistical confidence bound to validate the nominal physical model ($H_0$: In-Distribution) vs. slips ($H_1$: Out-of-Distribution).
```bash
python experiments/exp4_kinematic_residual.py
```

---

## 📐 Mathematical Mapping (Code to Paper)

To facilitate code review, here is the exact mapping between the LaTeX manuscript's formulations and our Python implementation:

| Concept in Paper | Code Implementation | Location |
| :--- | :--- | :--- |
| **Adjoint Transformation** $\boldsymbol{\xi}_e = \text{Ad}_{\mathbf{T}_b^e} \boldsymbol{\xi}_b$ | `v_e = rot_matrix.T @ v_b` | `core/perception.py` |
| **Affine Divergence** $D = \frac{\partial \dot{u}}{\partial x} + \frac{\partial \dot{v}}{\partial y}$ | `div = np.mean(dvx_dx + dvy_dy)` | `core/perception.py` |
| **Phase Alignment (Smith Predictor)** | `correlate(Y_norm, X_norm, mode='full')` | `core/alignment.py` |
| **Physical Risk Residual** $\mathcal{R}_{phys}$ | `mahalanobis_dist * np.exp(gamma*(1-ssim))` | `experiments/exp2_anomaly_roc.py` |
| **Dual-Track: Control Prior** $\mathbf{J}_{prior}$ | `Ridge(alpha=1.0).fit(X, Y)` (Offline) | `experiments/exp4_kinematic_residual.py` |
| **Dual-Track: Assessment** $\hat{\mathbf{M}}_k$ | `HuberRegressor().fit(X, Y)` (Online Digital Twin) | `experiments/exp2_anomaly_roc.py` |
| **Dual-Eye: Physical Risk** $\mathcal{R}_{phys}$ | Kinematic-affine residual Z-score | `experiments/exp6_dual_eye_decoupling.py` |
| **Dual-Eye: Geometric Eye** $\mathcal{S}_{geo}$ | Normalized acoustic observability (gradient magnitude) | `experiments/exp6_dual_eye_decoupling.py` |

---

## 📝 Citation

If you find this code or methodology useful in your research, please cite our paper:

```bibtex
@article{Yi2026SensitivityAware,
  title={Methodology: Sensitivity-Aware Active Visual Servoing via Interaction-Driven Uncertainty Quantification},
  author={Yi, Fan},
  journal={IEEE International Conference on Robotics and Automation (ICRA) / IROS},
  year={2026}
}
```

## 🤝 License
Released under the MIT License.