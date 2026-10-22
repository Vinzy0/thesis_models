# Per-symptom anxiety detection models

Seven binary LightGBM detectors (fidgeting, hand tremor, knee tremor, avoidant gaze,
facial tension, heavy breathing, fear) trained on MediaPipe landmarks, blendshapes and
face matrices extracted per video frame.

## Quick start

```
python run_all.py          # builds features.parquet if missing, trains all 7, writes summary
```

`run_all.log` holds the console output of the last full run.

Individual steps:

```
python build_features.py                                    # data.parquet -> features.parquet
python build_features.py --limit-participants 3 --out _test/features_test.parquet   # quick test
python train_fear.py                                        # one symptom
python train_fear.py --features _test/features_test.parquet --out _test   # against test features
python summary.py
```

## Files

| File | Purpose |
|---|---|
| `common.py` | All shared logic: constants, signal construction, window features, label check, `run_symptom()` |
| `build_features.py` | Reads only the ~159 needed columns of `data.parquet`, writes `features.parquet` |
| `train_<symptom>.py` | Calls `run_symptom("<symptom>")` |
| `run_all.py` | Whole pipeline |
| `summary.py` | `results/summary.csv`, `summary.md`, `summary_table.png`, `summary_by_angle.csv` |
| `data.parquet` | Copy of the frame-level dataset (the original is untouched) |
| `features.parquet` | One row per window: metadata + 822 `f_` features (float32) |
| `models/<symptom>_<mode>.joblib` | Dict: `model`, `features`, `window`, `bands`, `threshold`, ... |
| `results/<symptom>/` | `metrics.json`, confusion matrices, `what_fires.png`, importance plots |

## Features (build_features.py)

* Per clip: sorted by `frame`, time from `timestamp_ms`. Frames whose timestamp does not increase
  are dropped (none in the current data).
* First and last 1 s trimmed; 4 s windows with 2 s step.
* In each window every signal is resampled to a uniform 30 Hz grid (`np.interp` over non-NaN
  samples). A signal with < 50 % valid frames in the window gets NaN features.
* 63 signals:
  * Hands (each side): x, y of points 0, 4, 8, 12, 20, plus openness = dist(0, 12).
  * Pose: x, y of points 0, 11–16, 23–28, plus shoulder_mid_y, shoulder_width,
    torso_len = hip_mid_y − shoulder_mid_y.
  * Face (divided by inter-ocular distance dist(33, 263)): nose x/y (point 1); horizontal gaze ratio
    (iris 468 projected onto 33→133, iris 473 onto 362→263); vertical gaze ratio (468 onto lids
    159→145, 473 onto 386→374); head yaw/pitch proxy (nose minus eye-corner midpoint); mouth open
    (13–14); mouth width (61–291); brow height (105–159, 334–386). Infinite values become NaN.
    Gaze ratios are projections, so they are already scale-free.
* Per signal: mean, std, range (p95 − p5), mean |velocity|, p95 |velocity|, log10 total FFT power,
  dominant frequency, and fraction of power in breath 0.15–0.8 Hz, move 0.8–3 Hz, tremor 3–9 Hz.
  The FFT uses a Hann window on the mean-removed signal, DC bin excluded (0.25 Hz resolution).
* Blendshapes: nanmean / nanstd / nanmax. face_matrix: nanmean / nanstd. Detection flags: rate.
  These use the raw frames in the window (no resampling).
* `label_` columns are only checked against `symptom` (printed at the start), never used as features.

## Evaluation (common.run_symptom)

* `vs_control`: symptom windows vs control windows. `vs_everything`: symptom vs control + other 6.
* LightGBM, fixed hyperparameters (see `LGBM_PARAMS`), NaNs passed as-is, `class_weight="balanced"`.
* 5-fold `GroupKFold` by `participant_id`; the script asserts no participant is in both train and
  test of any fold. All metrics come from out-of-fold probabilities at threshold 0.5.
* Window-level, clip-level (mean window probability per `video_id`) and per-angle metrics, plus
  per-fold metrics and F1 std across folds, are stored in `metrics.json`.
* The saved model is refit on all data. Use the CV numbers when you report results, not this model's
  training fit.

## Data quirks handled

* Some participants have no control clips: they only contribute positives to `vs_control`.
* Some participant/angle/symptom combinations are missing or have more than one clip: windows
  are grouped by `video_id`, so neither needs special handling.
* Small subsets: folds drop to the number of participants. A training fold with only one class
  gets a constant prediction and a warning in the console and in `metrics.json`.

## Notes for interpretation

* `vs_everything` is imbalanced (~1:7). Accuracy is inflated by the negatives there, so report
  balanced accuracy, F1 and AUC.
* Fold-to-fold F1 std is 0.04–0.15, because each fold holds out only ~5 participants.
