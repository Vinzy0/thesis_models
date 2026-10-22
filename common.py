"""Shared logic for the per-symptom anxiety detection pipeline.

Contains:
  * constants (paths, window settings, frequency bands, model params)
  * raw-column selection + per-frame signal construction (used by build_features.py)
  * per-window feature extraction (used by build_features.py)
  * the label_ agreement check (used by build_features.py)
  * run_symptom(): cross-validated training/evaluation of the two binary models
    for one symptom (used by every train_<symptom>.py)
"""
import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data.parquet"
FEATURES_PATH = ROOT / "features.parquet"

SYMPTOMS = ["fidgeting", "hand_tremor", "knee_tremor", "avoidant_gaze",
            "facial_tension", "heavy_breathing", "fear"]
CLASSES = ["control"] + SYMPTOMS
ANGLES = ["Left", "Middle", "Right"]
MODES = ["vs_control", "vs_everything"]
META_COLS = ["participant_id", "angle", "symptom", "video_id", "win_start_s"]

WINDOW = dict(trim_s=1.0, win_s=4.0, step_s=2.0, fs=30.0, min_valid_frac=0.5)
# [lo, hi) in Hz; tremor upper edge is inclusive (9 Hz is an FFT bin at 0.25 Hz resolution)
BANDS = {"breath": (0.15, 0.8), "move": (0.8, 3.0), "tremor": (3.0, 9.0)}
SIGNAL_STATS = ["mean", "std", "range", "vel_mean", "vel_p95", "log_power", "dom_freq",
                "band_breath", "band_move", "band_tremor"]

HAND_POINTS = [0, 4, 8, 12, 20]
POSE_POINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
FACE_POINTS = [1, 13, 14, 33, 61, 105, 133, 145, 159, 263, 291, 334, 362, 374, 386, 468, 473]
FLAG_COLS = ["face_detected", "pose_detected", "left_hand_detected", "right_hand_detected"]
RAW_META = ["participant_id", "angle", "symptom", "video_id", "frame", "timestamp_ms", "fps"]

LGBM_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                   class_weight="balanced", random_state=42, verbose=-1)
THRESHOLD = 0.5
N_FOLDS = 5


# ----------------------------------------------------------------------------------------
# Raw columns and per-frame signals
# ----------------------------------------------------------------------------------------
def blendshape_cols(names):
    return [c for c in names if c.startswith("bs_")]


def face_matrix_cols(names):
    return [c for c in names if c.startswith("face_matrix")]


def landmark_cols():
    cols = []
    for side in ("left_hand", "right_hand"):
        cols += [f"{side}_{p}_{a}" for p in HAND_POINTS for a in "xy"]
    cols += [f"pose_{p}_{a}" for p in POSE_POINTS for a in "xy"]
    cols += [f"face_{p}_{a}" for p in FACE_POINTS for a in "xy"]
    return cols


def _dist(c, prefix, i, j):
    return np.hypot(c[f"{prefix}_{i}_x"] - c[f"{prefix}_{j}_x"],
                    c[f"{prefix}_{i}_y"] - c[f"{prefix}_{j}_y"])


def _proj_ratio(c, p, a, b):
    """Position of face point p along segment a->b (0 at a, 1 at b), 2D projection."""
    px, py = c[f"face_{p}_x"], c[f"face_{p}_y"]
    ax, ay = c[f"face_{a}_x"], c[f"face_{a}_y"]
    bx, by = c[f"face_{b}_x"], c[f"face_{b}_y"]
    dx, dy = bx - ax, by - ay
    return ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)


def compute_signals(c):
    """c: dict column -> float64 array (one value per frame). Returns (names, array[N, S])."""
    sig = {}
    with np.errstate(all="ignore"):
        for side in ("left_hand", "right_hand"):
            for p in HAND_POINTS:
                for a in "xy":
                    sig[f"{side}_{p}_{a}"] = c[f"{side}_{p}_{a}"]
            sig[f"{side}_openness"] = _dist(c, side, 0, 12)

        for p in POSE_POINTS:
            for a in "xy":
                sig[f"pose_{p}_{a}"] = c[f"pose_{p}_{a}"]
        shoulder_mid_y = (c["pose_11_y"] + c["pose_12_y"]) / 2
        hip_mid_y = (c["pose_23_y"] + c["pose_24_y"]) / 2
        sig["pose_shoulder_mid_y"] = shoulder_mid_y
        sig["pose_shoulder_width"] = _dist(c, "pose", 11, 12)
        sig["pose_torso_len"] = hip_mid_y - shoulder_mid_y

        iod = _dist(c, "face", 33, 263)
        eye_mid_x = (c["face_33_x"] + c["face_263_x"]) / 2
        eye_mid_y = (c["face_33_y"] + c["face_263_y"]) / 2
        sig["face_nose_x"] = c["face_1_x"] / iod
        sig["face_nose_y"] = c["face_1_y"] / iod
        sig["face_gaze_h_468"] = _proj_ratio(c, 468, 33, 133)
        sig["face_gaze_h_473"] = _proj_ratio(c, 473, 362, 263)
        sig["face_gaze_v_468"] = _proj_ratio(c, 468, 159, 145)
        sig["face_gaze_v_473"] = _proj_ratio(c, 473, 386, 374)
        sig["face_head_yaw"] = (c["face_1_x"] - eye_mid_x) / iod
        sig["face_head_pitch"] = (c["face_1_y"] - eye_mid_y) / iod
        sig["face_mouth_open"] = _dist(c, "face", 13, 14) / iod
        sig["face_mouth_width"] = _dist(c, "face", 61, 291) / iod
        sig["face_brow_105"] = _dist(c, "face", 105, 159) / iod
        sig["face_brow_334"] = _dist(c, "face", 334, 386) / iod

    names = list(sig)
    S = np.column_stack([np.asarray(v, dtype=np.float64) for v in sig.values()])
    S[~np.isfinite(S)] = np.nan  # inf from divisions -> NaN
    return names, S


# ----------------------------------------------------------------------------------------
# Window features
# ----------------------------------------------------------------------------------------
def feature_names(signal_names, bs_names, fm_names):
    names = [f"f_{s}_{st}" for s in signal_names for st in SIGNAL_STATS]
    names += [f"f_{b}_{st}" for b in bs_names for st in ("mean", "std", "max")]
    names += [f"f_{m}_{st}" for m in fm_names for st in ("mean", "std")]
    names += [f"f_{fl}_rate" for fl in FLAG_COLS]
    return names


def _signal_stats(G):
    """G: [W, S, T] uniformly resampled windows (rows all-NaN when invalid) -> [W, S, 10]."""
    fs = WINDOW["fs"]
    T = G.shape[-1]
    invalid = np.isnan(G).any(-1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = G.mean(-1)
        std = G.std(-1)
        p5, p95 = np.percentile(G, [5, 95], axis=-1)
        vel = np.abs(np.diff(G, axis=-1)) * fs
        vel_mean = vel.mean(-1)
        vel_p95 = np.percentile(vel, 95, axis=-1)

        x = (G - mean[..., None]) * np.hanning(T)
        P = np.abs(np.fft.rfft(x, axis=-1)) ** 2
        f = np.fft.rfftfreq(T, 1.0 / fs)
        P, f = P[..., 1:], f[1:]  # exclude DC
        tot = P.sum(-1)
        log_power = np.log10(tot + 1e-12)
        dom = f[np.argmax(np.nan_to_num(P, nan=-1.0), axis=-1)]
        dom = np.where(tot > 0, dom, np.nan)
        bands = []
        for name, (lo, hi) in BANDS.items():
            m = (f >= lo) & ((f <= hi) if name == "tremor" else (f < hi))
            bands.append(np.where(tot > 0, P[..., m].sum(-1) / tot, np.nan))

    out = np.stack([mean, std, p95 - p5, vel_mean, vel_p95, log_power, dom] + bands, axis=-1)
    out[invalid] = np.nan
    return out


def clip_window_features(t, S, B, M, F):
    """One clip. t: seconds from clip start (strictly increasing). S: signals [N, nS],
    B: blendshapes [N, nB], M: face_matrix [N, nM], F: flags [N, 4].
    Returns (win_starts, features [W, n_feat]) or (None, None) if the clip is too short."""
    trim, win, step, fs = WINDOW["trim_s"], WINDOW["win_s"], WINDOW["step_s"], WINDOW["fs"]
    first, last = t[0] + trim, t[-1] - trim
    starts = np.arange(first, last - win + 1e-9, step)
    if len(starts) == 0:
        return None, None

    T = int(round(win * fs))
    offsets = np.arange(T) / fs
    W, nS = len(starts), S.shape[1]
    G = np.full((W, nS, T), np.nan)
    extra = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for w, t0 in enumerate(starts):
            lo, hi = np.searchsorted(t, [t0, t0 + win])  # frames in [t0, t0 + win)
            n = hi - lo
            tt, seg = t[lo:hi], S[lo:hi]
            valid = ~np.isnan(seg)
            nvalid = valid.sum(0)
            grid = t0 + offsets
            for j in np.flatnonzero((nvalid >= WINDOW["min_valid_frac"] * n) & (nvalid >= 2)):
                m = valid[:, j]
                G[w, j] = np.interp(grid, tt[m], seg[m, j])
            b, mm, fl = B[lo:hi], M[lo:hi], F[lo:hi]
            extra.append(np.concatenate([
                np.stack([np.nanmean(b, 0), np.nanstd(b, 0), np.nanmax(b, 0)], 1).ravel(),
                np.stack([np.nanmean(mm, 0), np.nanstd(mm, 0)], 1).ravel(),
                np.nanmean(fl, 0),
            ]))
    sig_feats = _signal_stats(G).reshape(W, -1)
    return starts, np.hstack([sig_feats, np.vstack(extra)])


# ----------------------------------------------------------------------------------------
# label_ column check (labels are NOT used for training)
# ----------------------------------------------------------------------------------------
def label_agreement(data_path, participants=None):
    import pyarrow.parquet as pq
    names = pq.ParquetFile(data_path).schema_arrow.names
    label_cols = [c for c in names if c.startswith("label_")]
    filters = [("participant_id", "in", list(participants))] if participants is not None else None
    df = pq.read_table(data_path, columns=["symptom"] + label_cols, filters=filters).to_pandas()
    sym = df["symptom"].astype(str).values
    L = df[label_cols].fillna(0).values.astype(int)
    lab_names = [c[len("label_"):] for c in label_cols]
    expected = np.stack([(sym == s).astype(int) for s in lab_names], 1)  # control -> all zeros
    n_on = L.sum(1)

    print("\n=== label_ column agreement with 'symptom' (labels are NOT used for training) ===")
    print(f"frames: {len(df):,}")
    print(f"exact agreement (label vector == one-hot of symptom, control = all off): "
          f"{(L == expected).all(1).mean() * 100:.2f}%")
    print(f"frames with >1 label on: {(n_on > 1).sum():,} ({(n_on > 1).mean() * 100:.2f}%)")
    print(f"frames with 0 labels on: {(n_on == 0).sum():,}  "
          f"(of which control: {((n_on == 0) & (sym == 'control')).sum():,})")
    ctrl = sym == "control"
    print(f"control frames with any label on: {(ctrl & (n_on > 0)).sum():,} / {ctrl.sum():,}")
    print(f"{'label':<18}{'on | symptom==it':>18}{'on | symptom!=it':>18}")
    for k, s in enumerate(lab_names):
        own = sym == s
        a = L[own, k].mean() * 100 if own.any() else float("nan")
        b = L[~own, k].mean() * 100 if (~own).any() else float("nan")
        print(f"{s:<18}{a:>17.2f}%{b:>17.2f}%")
    print()


# ----------------------------------------------------------------------------------------
# Training / evaluation
# ----------------------------------------------------------------------------------------
def _f(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def binary_metrics(y, p, thr=THRESHOLD):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    n = len(y)
    recall = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    defined = [v for v in (recall, spec) if v is not None]
    both = len(np.unique(y)) == 2
    return dict(
        n=n, n_pos=int(y.sum()), n_neg=int(n - y.sum()), tp=tp, fp=fp, tn=tn, fn=fn,
        accuracy=(tp + tn) / n if n else None,
        balanced_accuracy=float(np.mean(defined)) if defined else None,
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=recall, specificity=spec,
        f1=2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        roc_auc=float(roc_auc_score(y, p)) if both else None,
    )


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return _f(float(o))
    return o


def _pretty(symptom):
    return symptom.replace("_", " ").title()


def _fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def plot_confusion(m, symptom, mode, path, n_folds=N_FOLDS):
    import matplotlib.pyplot as plt
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=float)
    rows = cm.sum(1, keepdims=True)
    pct = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0) * 100
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i, j]):,}\n({pct[i, j]:.1f}%)", ha="center", va="center",
                    fontsize=13, color="white" if pct[i, j] > 60 else "black")
    name = _pretty(symptom)
    neg = "Control" if mode == "vs_control" else "Not " + name
    ax.set_xticks([0, 1], [f"Pred: {neg}", f"Pred: {name}"])
    ax.set_yticks([0, 1], [f"True: {neg}", f"True: {name}"], rotation=90, va="center")
    ax.set_title(f"{name} — {mode.replace('_', ' ')}\n"
                 f"Acc {_fmt(m['accuracy'])} | F1 {_fmt(m['f1'])} | AUC {_fmt(m['roc_auc'])}")
    fig.text(0.5, 0.01, f"4 s windows, {n_folds}-fold CV, tested on unseen participants",
             ha="center", fontsize=9, color="dimgray")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_what_fires(fires, symptom, path, n_folds=N_FOLDS):
    import matplotlib.pyplot as plt
    classes = [c for c in CLASSES if c in fires]
    vals = [fires[c]["pct_flagged"] for c in classes]
    colors = ["#4caf50" if c == symptom else "#e57373" for c in classes]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar([_pretty(c) for c in classes], vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("% of windows flagged YES")
    ax.set_xlabel("True class")
    ax.set_title(f"What does the {_pretty(symptom)} model fire on?")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.text(0.5, 0.01, f"4 s windows, {n_folds}-fold CV, tested on unseen participants",
             ha="center", fontsize=9, color="dimgray")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(model, feats, symptom, mode, path, top=15):
    import matplotlib.pyplot as plt
    gain = model.booster_.feature_importance(importance_type="gain")
    order = np.argsort(gain)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feats[i] for i in order], gain[order], color="#5c9bd5")
    ax.set_xlabel("Total gain")
    ax.set_title(f"{_pretty(symptom)} — {mode.replace('_', ' ')}: top {top} features (gain)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [{"feature": feats[i], "gain": float(gain[i])} for i in order[::-1]]


def _run_mode(df, feats, symptom, mode, res_dir, mdl_dir):
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import GroupKFold
    import joblib

    d = df[df["symptom"].isin([symptom, "control"])] if mode == "vs_control" else df
    d = d.reset_index(drop=True)
    y = (d["symptom"] == symptom).astype(int).values
    groups = d["participant_id"].values
    X = d[feats]
    n_groups = len(np.unique(groups))
    info = dict(mode=mode, positives=[symptom],
                negatives=["control"] if mode == "vs_control" else [c for c in CLASSES if c != symptom],
                n_windows=len(d), n_pos=int(y.sum()), n_neg=int(len(y) - y.sum()),
                n_participants=n_groups,
                n_clips=int(d["video_id"].nunique()))
    if y.sum() == 0 or y.sum() == len(y) or n_groups < 2:
        print(f"[{symptom}] {mode:<14} SKIPPED: needs both classes and >=2 participants "
              f"(pos={y.sum()}, neg={len(y) - y.sum()}, participants={n_groups})")
        info["skipped"] = True
        return info

    n_splits = min(N_FOLDS, n_groups)
    if n_splits < N_FOLDS:
        print(f"[{symptom}] {mode}: only {n_groups} participants -> {n_splits}-fold CV")
    oof = np.full(len(d), np.nan)
    folds = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        tr_p, te_p = set(groups[tr].tolist()), set(groups[te].tolist())
        assert not (tr_p & te_p), f"participant leak in fold {k}: {tr_p & te_p}"
        note = None
        if len(np.unique(y[tr])) < 2:
            # cannot fit a classifier; predict the only training class (recorded, not hidden)
            oof[te] = float(y[tr][0])
            note = "training fold had a single class; constant prediction used"
            print(f"[{symptom}] {mode} fold {k}: WARNING {note}")
        else:
            model = LGBMClassifier(**LGBM_PARAMS)
            model.fit(X.iloc[tr], y[tr])
            oof[te] = model.predict_proba(X.iloc[te])[:, 1]
        fm = binary_metrics(y[te], oof[te])
        folds.append(dict(fold=k, test_participants=sorted(te_p), n_train=len(tr), n_test=len(te),
                          note=note, **fm))
    assert not np.isnan(oof).any()

    win = binary_metrics(y, oof)
    f1s = [f["f1"] for f in folds]
    win["f1_fold_std"] = float(np.std(f1s))
    win["f1_fold_mean"] = float(np.mean(f1s))

    clip_df = pd.DataFrame({"video_id": d["video_id"].astype(str).values, "y": y, "p": oof})
    clip_df = clip_df.groupby("video_id").agg(y=("y", "first"), p=("p", "mean"))
    clip = binary_metrics(clip_df["y"].values, clip_df["p"].values)

    per_angle = {}
    ang = d["angle"].astype(str).values
    for a in ANGLES:
        m = ang == a
        per_angle[a] = binary_metrics(y[m], oof[m]) if m.any() else None

    fires = None
    if mode == "vs_everything":
        sym = d["symptom"].astype(str).values
        fires = {}
        for c in CLASSES:
            m = sym == c
            if m.any():
                fires[c] = dict(n_windows=int(m.sum()),
                                pct_flagged=float((oof[m] >= THRESHOLD).mean() * 100))
        plot_what_fires(fires, symptom, res_dir / "what_fires.png", n_splits)

    plot_confusion(win, symptom, mode, res_dir / f"confusion_{mode}.png", n_splits)

    final = LGBMClassifier(**LGBM_PARAMS)
    final.fit(X, y)
    joblib.dump(dict(model=final, features=feats, window=WINDOW, bands=BANDS, threshold=THRESHOLD,
                     symptom=symptom, mode=mode, positives=info["positives"],
                     negatives=info["negatives"], lgbm_params=LGBM_PARAMS),
                mdl_dir / f"{symptom}_{mode}.joblib")
    top = plot_importance(final, feats, symptom, mode, res_dir / f"importance_{mode}.png")

    print(f"[{symptom}] {mode:<14} windows={len(d):>6} (pos={y.sum()})  "
          f"acc={_fmt(win['accuracy'])} bal_acc={_fmt(win['balanced_accuracy'])} "
          f"F1={_fmt(win['f1'])} AUC={_fmt(win['roc_auc'])} | "
          f"clip acc={_fmt(clip['accuracy'])} clip F1={_fmt(clip['f1'])}")

    info.update(n_folds=n_splits, window=win, clip=clip, per_angle=per_angle, folds=folds,
                top_features_gain=top)
    if fires is not None:
        info["what_fires"] = fires
    return info


def run_symptom(symptom):
    """Train + evaluate vs_control and vs_everything models for one symptom.
    Optional CLI args: --features <parquet> (default features.parquet),
                       --out <dir> (results/ and models/ are created inside; default this folder)."""
    import matplotlib
    matplotlib.use("Agg")
    assert symptom in SYMPTOMS, f"unknown symptom {symptom}"
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(FEATURES_PATH))
    ap.add_argument("--out", default=str(ROOT))
    args, _ = ap.parse_known_args()

    out = Path(args.out)
    res_dir = out / "results" / symptom
    mdl_dir = out / "models"
    res_dir.mkdir(parents=True, exist_ok=True)
    mdl_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.features)
    for c in ("angle", "symptom", "video_id"):
        df[c] = df[c].astype(str)
    feats = [c for c in df.columns if c.startswith("f_")]

    metrics = dict(symptom=symptom, features_file=str(Path(args.features).resolve()),
                   n_features=len(feats), window=WINDOW, bands=BANDS, threshold=THRESHOLD,
                   lgbm_params=LGBM_PARAMS, cv=f"GroupKFold by participant_id (n_splits<= {N_FOLDS})")
    for mode in MODES:
        metrics[mode] = _run_mode(df, feats, symptom, mode, res_dir, mdl_dir)

    with open(res_dir / "metrics.json", "w") as fh:
        json.dump(_clean(metrics), fh, indent=2)
    return metrics
