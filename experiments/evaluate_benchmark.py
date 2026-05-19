import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    average_precision_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.config_loader import IDUQConfig
from core.data_loader import get_episode_data, safe_open_zarr
from core.perception import PhysicsAwarePerception


METHOD_COLUMNS = {
    "ssim": "score_ssim",
    "intensity": "score_intensity",
    "divergence": "score_divergence",
    "iduq": "score_iduq",
}


@dataclass
class CalibrationBundle:
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    ridge: Ridge
    intensity_mean: float
    intensity_std: float
    divergence_mean: float
    divergence_std: float
    gamma: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified benchmark evaluation for ID-UQ.")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Path to the project config.")
    parser.add_argument("--labels", default="data/eval_labels.csv", help="Frame-level or interval label CSV.")
    parser.add_argument("--splits", default="data/eval_splits.json", help="Episode split JSON.")
    parser.add_argument("--output", default=None, help="Output directory for reports.")
    parser.add_argument("--threshold-percentile", type=float, default=95.0, help="Calibration percentile for score thresholds.")
    parser.add_argument("--lead-window", type=int, default=60, help="Frames to search backward for lead-time events.")
    parser.add_argument(
        "--methods",
        default="ssim,intensity,divergence,iduq",
        help="Comma-separated list from: ssim,intensity,divergence,iduq",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip saving ROC/PR figures.")
    return parser.parse_args()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in {"ep", "episode_id"}:
            rename_map[col] = "episode"
        elif key in {"frame_idx", "frame_index", "index"}:
            rename_map[col] = "frame"
        elif key in {"start_frame", "frame_start", "begin_frame", "start"}:
            rename_map[col] = "frame_start"
        elif key in {"end_frame", "frame_end", "finish_frame", "end"}:
            rename_map[col] = "frame_end"
        elif key in {"label", "y", "y_true", "anomaly"}:
            rename_map[col] = "label"
        elif key in {"split", "subset", "partition"}:
            rename_map[col] = "split"
        elif key in {"event_type", "failure_mode", "mode"}:
            rename_map[col] = "event_type"
    return df.rename(columns=rename_map)


def load_labels(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing label file: {path}. Create it first using the schema in docs/evaluation_protocol.md."
        )

    df = normalize_columns(pd.read_csv(path))
    if "episode" not in df.columns or "label" not in df.columns:
        raise ValueError("Label CSV must contain at least `episode` and `label`.")

    if "frame" in df.columns:
        df["frame"] = df["frame"].astype(int)
        df["label"] = df["label"].astype(int)
        return df

    if {"frame_start", "frame_end"}.issubset(df.columns):
        rows = []
        for row in df.itertuples(index=False):
            episode = getattr(row, "episode")
            start = int(getattr(row, "frame_start"))
            end = int(getattr(row, "frame_end"))
            label = int(getattr(row, "label"))
            split = getattr(row, "split", None) if hasattr(row, "split") else None
            event_type = getattr(row, "event_type", None) if hasattr(row, "event_type") else None
            for frame in range(start, end + 1):
                rows.append(
                    {
                        "episode": episode,
                        "frame": frame,
                        "label": label,
                        "split": split,
                        "event_type": event_type,
                    }
                )
        return pd.DataFrame(rows)

    raise ValueError("Label CSV must have either `frame` or (`frame_start`, `frame_end`).")


def load_splits(path: str, labels_df: pd.DataFrame) -> Dict[str, List[str]]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        splits = {str(k).lower(): list(v) for k, v in raw.items()}
    elif "split" in labels_df.columns:
        splits = {}
        for split_name, group in labels_df.groupby("split"):
            split_key = str(split_name).lower()
            splits[split_key] = sorted(group["episode"].dropna().astype(str).unique().tolist())
    else:
        raise FileNotFoundError(
            f"Missing split file: {path}. Provide a JSON split file or add a `split` column to the labels."
        )

    if "train" in splits and "calibration" not in splits:
        splits["calibration"] = splits["train"]
    if "val" in splits and "validation" not in splits:
        splits["validation"] = splits["val"]

    if "calibration" not in splits or "test" not in splits:
        raise ValueError("Split file must define at least `calibration` and `test` episodes.")

    return splits


def build_label_lookup(labels_df: pd.DataFrame) -> Dict[str, Dict[int, int]]:
    lookup: Dict[str, Dict[int, int]] = {}
    for episode, group in labels_df.groupby("episode"):
        frame_map = group.groupby("frame")["label"].max().astype(int).to_dict()
        lookup[str(episode)] = frame_map
    return lookup


def safe_std(values: pd.Series | np.ndarray) -> float:
    std = float(np.std(np.asarray(values), ddof=0))
    return std if std > 1e-9 else 1.0


def extract_episode_rows(
    root,
    episode: str,
    perception: PhysicsAwarePerception,
    cfg: IDUQConfig,
    label_lookup: Dict[str, Dict[int, int]],
) -> pd.DataFrame:
    images, poses = get_episode_data(root, episode)
    if len(images) < 50:
        return pd.DataFrame()

    step = int(cfg.perception.get("step", 1))
    trim = int(cfg.perception.get("trim_edge", 20))
    xi_trim, s_dot_trim = perception.process_episode(images, poses)
    min_len = min(len(xi_trim), len(s_dot_trim))

    rows = []
    episode_labels = label_lookup.get(str(episode), {})
    for k in range(min_len):
        curr_idx = step + trim + 1 + k
        prev_idx = curr_idx - step
        if curr_idx >= len(images) or prev_idx < 0:
            break

        img_curr = images[curr_idx]
        img_prev = images[prev_idx]
        gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_RGB2GRAY) if img_curr.ndim == 3 else img_curr
        gray_prev = cv2.cvtColor(img_prev, cv2.COLOR_RGB2GRAY) if img_prev.ndim == 3 else img_prev

        rows.append(
            {
                "episode": episode,
                "frame": curr_idx,
                "label": int(episode_labels.get(curr_idx, 0)),
                "ssim": float(ssim(gray_prev, gray_curr, data_range=255)),
                "intensity": float(np.mean(gray_curr)),
                "xi_z": float(xi_trim[k, 2]) if xi_trim.shape[1] > 2 else float(xi_trim[k, 0]),
                "divergence": float(s_dot_trim[k, 2]) if s_dot_trim.shape[1] > 2 else float(s_dot_trim[k, 0]),
            }
        )

    return pd.DataFrame(rows)


def fit_calibration_bundle(calibration_df: pd.DataFrame, gamma: float) -> CalibrationBundle:
    normal_df = calibration_df[calibration_df["label"] == 0].copy()
    if len(normal_df) < 30:
        raise ValueError(
            f"Need at least 30 normal calibration frames, got {len(normal_df)}. Add more labeled calibration data."
        )

    x_scaler = StandardScaler().fit(normal_df[["xi_z"]])
    y_scaler = StandardScaler().fit(normal_df[["divergence"]])
    ridge = Ridge(alpha=1.0).fit(
        x_scaler.transform(normal_df[["xi_z"]]),
        y_scaler.transform(normal_df[["divergence"]]).ravel(),
    )

    return CalibrationBundle(
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        ridge=ridge,
        intensity_mean=float(normal_df["intensity"].mean()),
        intensity_std=safe_std(normal_df["intensity"]),
        divergence_mean=float(normal_df["divergence"].mean()),
        divergence_std=safe_std(normal_df["divergence"]),
        gamma=gamma,
    )


def add_scores(df: pd.DataFrame, bundle: CalibrationBundle) -> pd.DataFrame:
    scored = df.copy()
    scored["score_ssim"] = 1.0 - scored["ssim"]
    scored["score_intensity"] = np.abs((scored["intensity"] - bundle.intensity_mean) / bundle.intensity_std)
    scored["score_divergence"] = np.abs((scored["divergence"] - bundle.divergence_mean) / bundle.divergence_std)

    x_norm = bundle.x_scaler.transform(scored[["xi_z"]])
    y_norm = bundle.y_scaler.transform(scored[["divergence"]]).ravel()
    pred = bundle.ridge.predict(x_norm)
    scored["score_iduq"] = np.abs(y_norm - pred) * np.exp(bundle.gamma * (1.0 - scored["ssim"].to_numpy()))
    return scored


def contiguous_segments(labels: np.ndarray) -> List[Tuple[int, int]]:
    positive = np.flatnonzero(labels > 0)
    if len(positive) == 0:
        return []

    segments = []
    start = prev = int(positive[0])
    for idx in positive[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        segments.append((start, prev))
        start = prev = idx
    segments.append((start, prev))
    return segments


def summarize_method(
    scored_df: pd.DataFrame,
    score_col: str,
    threshold: float,
    fps: float,
    lead_window: int,
) -> Dict[str, float]:
    y_true = scored_df["label"].to_numpy(dtype=int)
    scores = scored_df[score_col].to_numpy(dtype=float)
    y_pred = (scores >= threshold).astype(int)

    n_pos = int(np.sum(y_true))
    n_neg = int(len(y_true) - n_pos)
    roc_value = float("nan")
    pr_value = float("nan")
    if n_pos > 0 and n_neg > 0:
        roc_value = float(roc_auc_score(y_true, scores))
        pr_value = float(average_precision_score(y_true, scores))

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    neg_minutes = max(n_neg / fps / 60.0, 1e-9)
    false_alarms_per_min = float(np.sum((y_true == 0) & (y_pred == 1)) / neg_minutes)

    lead_times = []
    event_hits = 0
    total_events = 0
    for episode, group in scored_df.groupby("episode"):
        labels = group["label"].to_numpy(dtype=int)
        scores_ep = group[score_col].to_numpy(dtype=float)
        for start, _end in contiguous_segments(labels):
            total_events += 1
            search_start = max(0, start - lead_window)
            pre_scores = scores_ep[search_start:start]
            hit = np.where(pre_scores >= threshold)[0]
            if len(hit) == 0:
                continue
            alarm_idx = search_start + int(hit[0])
            if alarm_idx < start:
                lead_times.append(start - alarm_idx)
                event_hits += 1

    lead_time_mean = float(np.mean(lead_times)) if lead_times else float("nan")
    lead_time_median = float(np.median(lead_times)) if lead_times else float("nan")
    event_recall = float(event_hits / total_events) if total_events else float("nan")

    return {
        "roc_auc": roc_value,
        "pr_auc": pr_value,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarms_per_min": false_alarms_per_min,
        "lead_time_mean_frames": lead_time_mean,
        "lead_time_median_frames": lead_time_median,
        "event_recall": event_recall,
    }


def plot_curves(scored_df: pd.DataFrame, methods: List[str], out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for method in methods:
        col = METHOD_COLUMNS[method]
        y_true = scored_df["label"].to_numpy(dtype=int)
        scores = scored_df[col].to_numpy(dtype=float)
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, scores)
        precision, recall, _ = precision_recall_curve(y_true, scores)
        axes[0].plot(fpr, tpr, linewidth=2, label=f"{method} AUC={auc(fpr, tpr):.3f}")
        axes[1].plot(recall, precision, linewidth=2, label=method)

    axes[0].plot([0, 1], [0, 1], color="navy", linestyle=":")
    axes[0].set_title("ROC Curves")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")

    axes[1].set_title("Precision-Recall Curves")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "benchmark_curves.png"), dpi=300)
    plt.close(fig)


def run_benchmark() -> None:
    args = parse_args()
    cfg = IDUQConfig.from_yaml(args.config)
    out_dir = args.output or os.path.join(cfg.io.get("output_dir", "Results"), "Benchmark")
    os.makedirs(out_dir, exist_ok=True)

    labels_df = load_labels(args.labels)
    split_map = load_splits(args.splits, labels_df)
    label_lookup = build_label_lookup(labels_df)
    gamma = float(cfg.alignment.get("gamma", 15.0))
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    for method in methods:
        if method not in METHOD_COLUMNS:
            raise ValueError(f"Unknown method: {method}. Choose from {sorted(METHOD_COLUMNS)}")

    root = safe_open_zarr(cfg.io["data_path"])
    perception = PhysicsAwarePerception(cfg)

    calibration_episodes = split_map["calibration"]
    test_episodes = split_map["test"]

    print(f"Calibration episodes: {len(calibration_episodes)}")
    print(f"Test episodes: {len(test_episodes)}")

    calib_frames = []
    for episode in tqdm(calibration_episodes, desc="Calibration"):
        calib_frames.append(extract_episode_rows(root, episode, perception, cfg, label_lookup))
    calib_df = pd.concat([df for df in calib_frames if not df.empty], ignore_index=True)
    if calib_df.empty:
        raise RuntimeError("No calibration frames were extracted. Check the split file and labels.")

    bundle = fit_calibration_bundle(calib_df, gamma=gamma)
    calib_scored = add_scores(calib_df, bundle)
    normal_calib = calib_scored[calib_scored["label"] == 0]
    thresholds = {
        method: float(np.percentile(normal_calib[METHOD_COLUMNS[method]], args.threshold_percentile))
        for method in methods
    }

    test_frames = []
    for episode in tqdm(test_episodes, desc="Test"):
        test_frames.append(extract_episode_rows(root, episode, perception, cfg, label_lookup))
    test_df = pd.concat([df for df in test_frames if not df.empty], ignore_index=True)
    if test_df.empty:
        raise RuntimeError("No test frames were extracted. Check the split file and labels.")

    test_scored = add_scores(test_df, bundle)
    test_scored.to_csv(os.path.join(out_dir, "test_frame_scores.csv"), index=False)

    summary_rows = []
    fps = 1.0 / float(cfg.kinematics.get("dt", 1.0))
    for method in methods:
        row = summarize_method(
            test_scored,
            METHOD_COLUMNS[method],
            thresholds[method],
            fps=fps,
            lead_window=args.lead_window,
        )
        row.update(
            {
                "method": method,
                "threshold": thresholds[method],
                "test_frames": int(len(test_scored)),
                "test_positive_frames": int(test_scored["label"].sum()),
                "test_negative_frames": int((test_scored["label"] == 0).sum()),
            }
        )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("method")
    summary_df.to_csv(os.path.join(out_dir, "benchmark_summary.csv"), index=False)
    with open(os.path.join(out_dir, "benchmark_summary.md"), "w", encoding="utf-8") as f:
        f.write(summary_df.to_markdown(index=False))
        f.write("\n")

    report_path = os.path.join(out_dir, "benchmark_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Calibration episodes: {len(calibration_episodes)}\n")
        f.write(f"Test episodes: {len(test_episodes)}\n")
        f.write(f"Threshold percentile: {args.threshold_percentile}\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n")

    if not args.no_plots:
        plot_curves(test_scored, methods, out_dir)

    print(summary_df.to_string(index=False))
    print(f"\nReports written to: {out_dir}")


if __name__ == "__main__":
    run_benchmark()
