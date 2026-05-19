# Evaluation Protocol

This repository now uses one shared protocol for benchmark claims, instead of letting each experiment script define its own labels and thresholds.

## Required inputs

1. `data/servo_dataset_dp.zarr`
2. `data/eval_labels.csv`
3. `data/eval_splits.json`

## Label schema

Use either frame-level labels:

```csv
episode,frame,label,event_type
episode_1000,248,0,normal
episode_1000,249,1,slip
```

or interval labels:

```csv
episode,frame_start,frame_end,label,event_type
episode_1000,240,280,1,slip
```

Rules:

1. `label` is `0` for normal and `1` for failure.
2. Frame indices are raw Zarr frame indices, not trimmed indices.
3. Positive intervals should be contiguous when possible.
4. A split tag may be added as `split = calibration | validation | test`.

## Split policy

Use explicit episode-level splits:

```json
{
  "calibration": ["episode_1", "episode_2"],
  "validation": ["episode_3"],
  "test": ["episode_4", "episode_5"]
}
```

Rules:

1. Calibration data is used only to fit thresholds and model parameters.
2. Validation data is optional but recommended for threshold selection.
3. Test data is never used for threshold tuning.
4. Split by session, day, phantom, or operator when possible.

## Metrics to report

Report at least:

1. ROC-AUC
2. PR-AUC
3. F1 at the calibration threshold
4. False alarms per minute
5. Event-level lead time
6. Event recall

## Recommended baselines

1. `1 - SSIM`
2. Global brightness deviation
3. Divergence deviation
4. ID-UQ residual score

## Running the benchmark

```bash
python experiments/evaluate_benchmark.py \
  --labels data/eval_labels.csv \
  --splits data/eval_splits.json
```

The script writes:

1. `benchmark_summary.csv`
2. `benchmark_summary.md`
3. `test_frame_scores.csv`
4. `benchmark_curves.png`

## Minimum publication bar

For a claim of method validity, the paper should include:

1. Independent labels that are not derived from the same score being evaluated.
2. At least one held-out test split.
3. A real closed-loop or hardware result if the paper claims control utility.
4. Cross-session or cross-operator generalization.
5. Failure-mode breakdown, not only a pooled average.

## Practical interpretation

More data helps, but only if it adds one of these:

1. New failure modes.
2. New operators or days.
3. New phantoms or tissues.
4. Independent ground truth, especially force or contact labels.

If the new data only repeats the current distribution without stronger labels, it will not move the publication case much.
