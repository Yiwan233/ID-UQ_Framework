# experiments/exp7_closing_the_loop.py
"""
Closing the Loop: Uncertainty-Guided Active Visual Servoing.

Demonstrates that ID-UQ uncertainty signals (R_phys, S_geo) can drive
a closed-loop decision policy, transforming uncertainty from a diagnostic
byproduct into an actionable gradient for robot control.

Three-part experiment:
  Part A — Predictive Lead Time: R_phys as early warning vs. SSIM baseline.
  Part B — Empirical Dynamics: Learn action→state transition model from data.
  Part C — Policy Simulation: Blind (open-loop) vs. UQ-guided (closed-loop)
            phase-space trajectories, showing convergence to the ideal
            servoing envelope (Quadrant IV).
"""

import os
import sys
import numpy as np
import cv2
import cupy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import concurrent.futures
from functools import partial
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.config_loader import IDUQConfig
from core.perception import PhysicsAwarePerception
from core.data_loader import safe_open_zarr, get_episode_data


# ---------------------------------------------------------------------------
# Per-episode feature extraction (runs in subprocess)
# ---------------------------------------------------------------------------
def extract_episode_state(ep_id, config_path):
    cv2.setNumThreads(1)
    try:
        cfg = IDUQConfig.from_yaml(config_path)
        root = safe_open_zarr(cfg.io['data_path'])
        perception = PhysicsAwarePerception(cfg)

        images, poses = get_episode_data(root, ep_id)
        if len(images) < 100:
            return None

        step = cfg.perception.get('step', 1)
        trim = cfg.perception.get('trim_edge', 20)

        xi_trim, s_dot_trim = perception.process_episode(images, poses)
        min_len = min(len(xi_trim), len(s_dot_trim))

        S_geo_list, contact_list, ssim_list = [], [], []
        v_z_list, omega_list = [], []

        for k in range(min_len):
            curr_idx = step + trim + 1 + k
            prev_idx = curr_idx - step
            if curr_idx >= len(images) or prev_idx < 0:
                break

            img_curr = images[curr_idx]
            img_prev = images[prev_idx]

            gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_RGB2GRAY) if img_curr.ndim == 3 else img_curr
            gray_prev = cv2.cvtColor(img_prev, cv2.COLOR_RGB2GRAY) if img_prev.ndim == 3 else img_prev

            ssim_list.append(ssim(gray_prev, gray_curr, data_range=255))

            gray_blur = cv2.GaussianBlur(gray_curr, (7, 7), 0)
            gray_cp = cp.array(gray_blur, dtype=cp.float64)
            W_cp = perception.get_confidence_mask_gpu(gray_cp)
            mask_roi = W_cp > 0.5

            contact_list.append(float(cp.mean(mask_roi)))

            grad_x = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = cp.array(np.sqrt(grad_x**2 + grad_y**2))
            S_geo_list.append(float(cp.mean(grad_mag[mask_roi])) if cp.any(mask_roi) else 0.0)

            v_z_list.append(xi_trim[k, 2])
            omega_list.append(np.linalg.norm(xi_trim[k, 3:6]))

        N = len(S_geo_list)
        contact_arr = np.array(contact_list)
        S_geo_arr = np.array(S_geo_list)
        ssim_arr = np.array(ssim_list)

        max_contact = np.max(contact_arr)
        if max_contact < 0.1:
            return None

        calib_threshold = max_contact * 0.85
        calib_idx = np.where(contact_arr > calib_threshold)[0][:150]
        if len(calib_idx) < 30:
            return None

        # ---- R_phys computation ----
        X_norm = StandardScaler().fit_transform(xi_trim[:N, 2].reshape(-1, 1))
        Y_norm = StandardScaler().fit_transform(s_dot_trim[:N, 2].reshape(-1, 1))

        model = Ridge(alpha=1.0).fit(X_norm[calib_idx], Y_norm[calib_idx])
        R_raw = np.abs(Y_norm.flatten() - model.predict(X_norm).flatten())
        sigma_calib = np.std(R_raw[calib_idx]) + 1e-6
        R_phys = R_raw / sigma_calib

        return {
            "ep_id": ep_id,
            "R_phys": R_phys[:N],
            "S_geo": S_geo_arr,
            "contact": contact_arr,
            "ssim": ssim_arr,
            "v_z": np.array(v_z_list),
            "omega": np.array(omega_list),
        }
    except Exception:
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Part A — Predictive Early Warning
# ---------------------------------------------------------------------------
def predictive_lead_time_analysis(all_data, out_dir):
    """
    For each episode, detect contact-loss events (contact drops below 0.5x max).
    Measure how many frames R_phys crosses its alarm threshold BEFORE the event,
    compared to SSIM crossing its own threshold.
    """
    lead_R, lead_SSIM = [], []
    alarm_threshold_R = 4.0       # R_phys alarm (same as exp6 Q1/Q2 boundary)
    alarm_threshold_S = 0.92      # SSIM alarm (typical "good" threshold)

    for d in all_data:
        contact = d["contact"]
        max_c = np.max(contact)
        if max_c < 0.1:
            continue

        loss_threshold = max_c * 0.5
        below = contact < loss_threshold

        # Find transition: first frame in each contiguous loss segment
        loss_events = []
        in_loss = False
        for i in range(len(below)):
            if below[i] and not in_loss:
                loss_events.append(i)
                in_loss = True
            elif not below[i]:
                in_loss = False

        for event_frame in loss_events:
            if event_frame < 10:
                continue
            # Look backwards from event_frame for first alarm
            R = d["R_phys"]
            S = d["ssim"]

            lead_R_frames = 0
            for j in range(event_frame - 1, max(event_frame - 60, 0), -1):
                if R[j] >= alarm_threshold_R:
                    lead_R_frames = event_frame - j
                else:
                    break

            lead_S_frames = 0
            for j in range(event_frame - 1, max(event_frame - 60, 0), -1):
                if S[j] <= alarm_threshold_S:
                    lead_S_frames = event_frame - j
                else:
                    break

            if lead_R_frames > 0:
                lead_R.append(lead_R_frames)
            if lead_S_frames > 0:
                lead_SSIM.append(lead_S_frames)

    lead_R = np.array(lead_R)
    lead_SSIM = np.array(lead_SSIM)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Part A: Predictive Lead Time — R_phys as Early Warning System",
                 fontsize=15, fontweight='bold')

    # Histogram
    bins = np.arange(0, 61, 3)
    ax = axes[0]
    ax.hist(lead_R, bins=bins, alpha=0.7, color='crimson', label=f'R_phys (mean={lead_R.mean():.1f} frames)')
    ax.hist(lead_SSIM, bins=bins, alpha=0.7, color='gray', label=f'SSIM (mean={lead_SSIM.mean():.1f} frames)')
    ax.set_xlabel('Lead Time (frames before contact loss)')
    ax.set_ylabel('Event Count')
    ax.set_title('Early Warning Lead Time Distribution')
    ax.legend(loc='upper right')
    ax.axvline(lead_R.mean(), color='crimson', linestyle='--', alpha=0.6)
    ax.axvline(lead_SSIM.mean(), color='gray', linestyle='--', alpha=0.6)

    # Box plot
    ax = axes[1]
    box_data = [lead_R, lead_SSIM]
    bp = ax.boxplot(box_data, labels=['R_phys', 'SSIM'], patch_artist=True,
                    widths=0.4, medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor('crimson')
    bp['boxes'][1].set_facecolor('gray')
    for b in bp['boxes']:
        b.set_alpha(0.6)
    ax.set_ylabel('Lead Time (frames)')
    ax.set_title('Predictive Lead Time Comparison')

    # Significance test
    if len(lead_R) > 5 and len(lead_SSIM) > 5:
        from scipy.stats import mannwhitneyu
        stat, p = mannwhitneyu(lead_R, lead_SSIM, alternative='greater')
        ax.text(0.5, 0.95, f'Mann-Whitney p = {p:.4f}\n({"R_phys leads earlier" if p < 0.05 else "No significant difference"})',
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'PartA_Predictive_Lead_Time.png'), dpi=300)
    plt.close(fig)
    print(f"  Part A: R_phys lead = {lead_R.mean():.1f} ± {lead_R.std():.1f} frames, "
          f"SSIM lead = {lead_SSIM.mean():.1f} ± {lead_SSIM.std():.1f} frames")

    return {"lead_R_mean": float(lead_R.mean()), "lead_S_mean": float(lead_SSIM.mean())}


# ---------------------------------------------------------------------------
# Part B — Empirical Dynamics Model
# ---------------------------------------------------------------------------
def learn_empirical_dynamics(all_data):
    """
    Learn first-order state transition model from pooled data:
        R_{t+1} = a0 + a1*R_t + a2*v_z_t
        S_{t+1} = b0 + b1*S_t + b2*omega_t
    """
    dR, Rt, vz = [], [], []
    dS, St, om = [], [], []

    for d in all_data:
        R, S, v, w = d["R_phys"], d["S_geo"], d["v_z"], d["omega"]
        n = min(len(R), len(v)) - 1
        if n < 10:
            continue
        dR.extend(R[1:n+1] - R[:n])
        Rt.extend(R[:n])
        vz.extend(v[:n])
        dS.extend(S[1:n+1] - S[:n])
        St.extend(S[:n])
        om.extend(w[:n])

    dR, Rt, vz = map(np.array, [dR, Rt, vz])
    dS, St, om = map(np.array, [dS, St, om])

    # Ridge regression: ΔR = f(R, v_z)
    X_R = np.column_stack([Rt, vz, Rt * vz])
    model_R = Ridge(alpha=0.1).fit(X_R, dR)

    # Ridge regression: ΔS = f(S, ω)
    X_S = np.column_stack([St, om, St * om])
    model_S = Ridge(alpha=0.1).fit(X_S, dS)

    print(f"  Part B: Dynamics model fitted on {len(dR)} transitions.")
    print(f"    ΔR model: coef = {model_R.coef_}, intercept = {model_R.intercept_:.4f}")
    print(f"    ΔS model: coef = {model_S.coef_}, intercept = {model_S.intercept_:.4f}")

    return {"R_model": model_R, "S_model": model_S, "R_scaler": StandardScaler().fit(X_R),
            "S_scaler": StandardScaler().fit(X_S)}


# ---------------------------------------------------------------------------
# Part C — Closed-Loop Policy Simulation
# ---------------------------------------------------------------------------
def simulate_trajectory(dynamics, R0, S0, policy, n_steps):
    """Simulate one trajectory given initial state and policy function."""
    model_R, model_S = dynamics["R_model"], dynamics["S_model"]
    R_traj, S_traj = [R0], [S0]

    R, S = R0, S0
    for _ in range(n_steps):
        v_z, omega = policy(R, S)
        X_R = np.array([[R, v_z, R * v_z]])
        X_S = np.array([[S, omega, S * omega]])
        dR = model_R.predict(X_R)[0] + np.random.normal(0, 0.1)
        dS = model_S.predict(X_S)[0] + np.random.normal(0, 0.02)
        R = max(0, R + dR)
        S = max(0, S + dS)
        R_traj.append(R)
        S_traj.append(S)

    return np.array(R_traj), np.array(S_traj)


def run_policy_simulation(all_data, dynamics, out_dir):
    """
    Compare two policies:
      Blind:    v_z = constant, omega = 0
      UQ:       v_z = v_nom - k_r*(R - R_target), omega = k_s*(S_target - S)
    """
    # Pool empirical ranges for initialisation
    all_R = np.concatenate([d["R_phys"] for d in all_data])
    all_S = np.concatenate([d["S_geo"] for d in all_data])
    all_vz = np.concatenate([d["v_z"] for d in all_data])

    # Normalise S_geo to [0,1] for consistent policy
    S_min, S_max = np.percentile(all_S, 1), np.percentile(all_S, 99)
    S_norm = lambda s: (s - S_min) / (S_max - S_min + 1e-6)

    v_nom = np.mean(np.abs(all_vz))
    R_target = 1.5        # desired R_phys (safely below alarm)
    k_r = 0.003            # R_phys feedback gain
    k_s = 0.0008           # S_geo feedback gain

    def blind_policy(R, S):
        return v_nom, 0.0

    def uq_policy(R, S):
        S_n = S_norm(S)
        v_z = v_nom - k_r * (R - R_target)
        omega = k_s * max(0, 1.0 - S_n)   # seek harder when S_geo is low
        return v_z, omega

    n_steps = 300
    n_trials = 80
    phys_thresh, geo_thresh = 4.0, np.percentile(all_S, 40)

    # Sample diverse initial states
    rng = np.random.RandomState(42)
    idx = rng.choice(len(all_R), size=n_trials, replace=False)

    blind_trajs, uq_trajs = [], []
    blind_q4, uq_q4 = [], []

    for i in idx:
        R0, S0 = all_R[i], all_S[i]
        Rb, Sb = simulate_trajectory(dynamics, R0, S0, blind_policy, n_steps)
        Ru, Su = simulate_trajectory(dynamics, R0, S0, uq_policy, n_steps)
        blind_trajs.append((Rb, Sb))
        uq_trajs.append((Ru, Su))
        blind_q4.append(np.mean((Rb < phys_thresh) & (Sb > geo_thresh)))
        uq_q4.append(np.mean((Ru < phys_thresh) & (Su > geo_thresh)))

    blind_q4 = np.array(blind_q4)
    uq_q4 = np.array(uq_q4)

    # ---- Plotting ----
    fig = plt.figure(figsize=(18, 13))
    fig.suptitle("Part C: Closing the Loop — Uncertainty-Guided Active Servoing",
                 fontsize=16, fontweight='bold', y=0.97)

    # Subplot 1: Phase-space trajectories (first 6 trials)
    ax1 = fig.add_subplot(2, 3, (1, 2))
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    for j in range(6):
        Rb, Sb = blind_trajs[j]
        Ru, Su = uq_trajs[j]
        ax1.plot(Sb, Rb, color=colors[j], alpha=0.3, linewidth=0.8)
        ax1.plot(Su, Ru, color=colors[j], alpha=0.9, linewidth=1.5)
        ax1.scatter(Sb[0], Rb[0], color=colors[j], marker='o', s=40, edgecolors='black', linewidth=0.5)
        ax1.scatter(Su[-1], Ru[-1], color=colors[j], marker='*', s=80, edgecolors='black', linewidth=0.8)

    ax1.axhline(y=phys_thresh, color='crimson', linestyle='--', linewidth=2)
    ax1.axvline(x=geo_thresh, color='darkorange', linestyle='--', linewidth=2)
    ax1.set_xlabel('Geometric Eye S_geo (Acoustic Observability)', fontsize=12)
    ax1.set_ylabel('Physical Eye R_phys (Interaction Risk)', fontsize=12)
    ax1.set_title('Phase-Space Trajectories\n(faint = blind, bold = UQ-guided, ★ = final state)', fontsize=13, fontweight='bold')

    # Quadrant labels
    x_lim = ax1.get_xlim()
    y_lim = ax1.get_ylim()
    ax1.text((geo_thresh + x_lim[1])/2, (phys_thresh + y_lim[1])/2, 'Q1: Sliding', ha='center', alpha=0.4, fontsize=10)
    ax1.text((x_lim[0] + geo_thresh)/2, (phys_thresh + y_lim[1])/2, 'Q2: Decoupled', ha='center', alpha=0.4, fontsize=10)
    ax1.text((x_lim[0] + geo_thresh)/2, (0 + phys_thresh)/2, 'Q3: Shadow', ha='center', alpha=0.4, fontsize=10)
    ax1.text((geo_thresh + x_lim[1])/2, (0 + phys_thresh)/2, 'Q4: IDEAL', ha='center', alpha=0.5, fontsize=10, fontweight='bold', color='green')

    # Subplot 2: Q4 occupancy bar chart
    ax2 = fig.add_subplot(2, 3, 3)
    methods = ['Blind\n(Open-Loop)', 'UQ-Guided\n(Closed-Loop)']
    means = [np.mean(blind_q4) * 100, np.mean(uq_q4) * 100]
    stds = [np.std(blind_q4) * 100, np.std(uq_q4) * 100]
    bars = ax2.bar(methods, means, yerr=stds, color=['gray', 'crimson'], capsize=8, width=0.5, alpha=0.8)
    ax2.set_ylabel('Q4 Occupancy (%)', fontsize=12)
    ax2.set_title('Ideal Envelope Occupancy\n(Quadrant IV: Safe + Clear)', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%',
                 ha='center', fontsize=13, fontweight='bold')

    # Subplot 3: R_phys time series (single trial)
    ax3 = fig.add_subplot(2, 3, 4)
    Rb, Sb = blind_trajs[0]
    Ru, Su = uq_trajs[0]
    ax3.plot(Rb, color='gray', alpha=0.6, label='Blind')
    ax3.plot(Ru, color='crimson', alpha=0.9, label='UQ-Guided')
    ax3.axhline(y=phys_thresh, color='crimson', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Simulation Step')
    ax3.set_ylabel('R_phys')
    ax3.set_title('R_phys Evolution (Single Trial)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)

    # Subplot 4: S_geo time series (single trial)
    ax4 = fig.add_subplot(2, 3, 5)
    ax4.plot(Sb, color='gray', alpha=0.6, label='Blind')
    ax4.plot(Su, color='darkorange', alpha=0.9, label='UQ-Guided')
    ax4.axhline(y=geo_thresh, color='darkorange', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Simulation Step')
    ax4.set_ylabel('S_geo')
    ax4.set_title('S_geo Evolution (Single Trial)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)

    # Subplot 5: Q4 occupancy histogram
    ax5 = fig.add_subplot(2, 3, 6)
    ax5.hist(blind_q4 * 100, bins=20, alpha=0.6, color='gray', label='Blind')
    ax5.hist(uq_q4 * 100, bins=20, alpha=0.6, color='crimson', label='UQ-Guided')
    ax5.set_xlabel('Q4 Occupancy (%)')
    ax5.set_ylabel('Trial Count')
    ax5.set_title('Distribution of Q4 Occupancy\n(80 trials × 300 steps)', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)

    # Significance
    from scipy.stats import wilcoxon
    stat, p = wilcoxon(uq_q4, blind_q4, alternative='greater')
    ax5.text(0.95, 0.95, f'Wilcoxon p = {p:.2e}\nΔ = {(means[1] - means[0]):.1f} pp',
             transform=ax5.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(out_dir, 'PartC_Closed_Loop_Simulation.png'), dpi=300)
    plt.close(fig)

    print(f"  Part C: Blind Q4 = {means[0]:.1f}%, UQ Q4 = {means[1]:.1f}%")
    return {"blind_q4_mean": means[0], "uq_q4_mean": means[1], "wilcoxon_p": float(p)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_closed_loop_experiment():
    print("=" * 70)
    print("EXP7: Closing the Loop — Uncertainty-Guided Active Servoing")
    print("=" * 70)

    config_path = "configs/default_config.yaml"
    cfg = IDUQConfig.from_yaml(config_path)
    out_dir = os.path.join(cfg.io.get('output_dir', 'Results'), 'EXP7_Closed_Loop')
    os.makedirs(out_dir, exist_ok=True)

    root = safe_open_zarr(cfg.io['data_path'])
    episodes = sorted(list(root.group_keys()))[:50]

    # ---- Extract state-action data from all episodes ----
    print(f"\nExtracting state-action features from {len(episodes)} episodes...")
    all_data = []
    process_func = partial(extract_episode_state, config_path=config_path)
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        for res in tqdm(executor.map(process_func, episodes), total=len(episodes),
                        desc="Extracting states"):
            if res is not None:
                all_data.append(res)

    print(f"  Successfully extracted {len(all_data)} episodes.\n")

    if len(all_data) < 5:
        print("Insufficient data for analysis.")
        return

    # ---- Part A: Predictive Lead Time ----
    print("Part A: Predictive Early Warning Analysis...")
    lead_results = predictive_lead_time_analysis(all_data, out_dir)

    # ---- Part B: Empirical Dynamics ----
    print("\nPart B: Learning Empirical Action-State Dynamics...")
    dynamics = learn_empirical_dynamics(all_data)

    # ---- Part C: Policy Simulation ----
    print("\nPart C: Simulating Blind vs. UQ-Guided Policies...")
    sim_results = run_policy_simulation(all_data, dynamics, out_dir)

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("EXP7 Complete. Summary:")
    print(f"  R_phys lead time: {lead_results['lead_R_mean']:.1f} frames")
    print(f"  SSIM   lead time: {lead_results['lead_S_mean']:.1f} frames")
    print(f"  Blind Q4 occupancy:  {sim_results['blind_q4_mean']:.1f}%")
    print(f"  UQ    Q4 occupancy:  {sim_results['uq_q4_mean']:.1f}%")
    print(f"  Wilcoxon p:          {sim_results['wilcoxon_p']:.2e}")
    print(f"\n  All figures saved to: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    run_closed_loop_experiment()
