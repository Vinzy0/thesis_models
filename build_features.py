"""Build features.parquet (one row per 4 s window) from data.parquet.

Usage:
    python build_features.py                                   # full data -> features.parquet
    python build_features.py --limit-participants 3 --out features_test.parquet
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import (DATA_PATH, FEATURES_PATH, FLAG_COLS, META_COLS, RAW_META, WINDOW, ROOT,
                    blendshape_cols, clip_window_features, compute_signals, face_matrix_cols,
                    feature_names, label_agreement, landmark_cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_PATH))
    ap.add_argument("--out", default=str(FEATURES_PATH))
    ap.add_argument("--limit-participants", type=int, default=None,
                    help="only use the first N participant ids (for quick testing)")
    args = ap.parse_args()
    t_start = time.time()

    pf = pq.ParquetFile(args.data)
    names = pf.schema_arrow.names
    bs, fm = blendshape_cols(names), face_matrix_cols(names)
    print(f"blendshape columns: {len(bs)}, face_matrix columns: {len(fm)}")

    participants = None
    filters = None
    if args.limit_participants:
        all_p = np.unique(pq.read_table(args.data, columns=["participant_id"])
                          .column("participant_id").to_numpy())
        participants = sorted(all_p.tolist())[: args.limit_participants]
        filters = [("participant_id", "in", participants)]
        print(f"limiting to participants {participants}")

    label_agreement(args.data, participants)

    lm = landmark_cols()
    cols = RAW_META + FLAG_COLS + lm + bs + fm
    missing = [c for c in cols if c not in names]
    assert not missing, f"missing columns: {missing[:10]}"
    print(f"reading {len(cols)} of {len(names)} columns ...")
    tbl = pq.read_table(args.data, columns=cols, filters=filters)
    print(f"  {tbl.num_rows:,} frames read in {time.time() - t_start:.0f}s")

    meta = tbl.select(["participant_id", "angle", "symptom", "video_id"]).to_pandas()
    for c in ("angle", "symptom", "video_id"):
        meta[c] = meta[c].astype(str)
    frame = tbl.column("frame").to_numpy()
    ts = tbl.column("timestamp_ms").to_numpy().astype(np.float64)

    def num(cs):
        return np.column_stack([tbl.column(c).to_numpy().astype(np.float32) for c in cs])

    raw = {c: tbl.column(c).to_numpy().astype(np.float64) for c in lm}
    B, M, F = num(bs), num(fm), num(FLAG_COLS)
    del tbl
    sig_names, S = compute_signals(raw)
    del raw
    print(f"  {len(sig_names)} signals, {B.shape[1]} blendshapes, {M.shape[1]} face_matrix, "
          f"{F.shape[1]} flags")

    vid_codes, vid_uniques = pd.factorize(meta["video_id"])
    order = np.lexsort((frame, vid_codes))
    vid_sorted = vid_codes[order]
    bounds = np.flatnonzero(np.diff(vid_sorted)) + 1
    clip_slices = np.split(order, bounds)

    fnames = feature_names(sig_names, bs, fm)
    rows_meta, rows_feat = [], []
    n_short, n_dropped_ts = 0, 0
    for k, idx in enumerate(clip_slices, 1):
        t = ts[idx]
        # timestamps must be strictly increasing for interpolation; drop repeats/back-steps
        keep = t > np.maximum.accumulate(np.r_[-np.inf, t[:-1]])
        n_dropped_ts += int((~keep).sum())
        idx, t = idx[keep], t[keep]
        t = (t - t[0]) / 1000.0
        starts, feats = clip_window_features(t, S[idx], B[idx], M[idx], F[idx])
        m0 = meta.iloc[idx[0]]
        if starts is None:
            n_short += 1
            print(f"  skip clip {m0.video_id}: too short ({t[-1]:.1f} s)")
        else:
            for s in starts:
                rows_meta.append((m0.participant_id, m0.angle, m0.symptom, m0.video_id, s))
            rows_feat.append(feats)
        if k % 50 == 0 or k == len(clip_slices):
            print(f"  clip {k}/{len(clip_slices)}  windows so far: {len(rows_meta):,}  "
                  f"({time.time() - t_start:.0f}s)")

    out = pd.DataFrame(rows_meta, columns=META_COLS)
    out["participant_id"] = out["participant_id"].astype(np.int64)
    out["win_start_s"] = out["win_start_s"].astype(np.float32)
    feat_df = pd.DataFrame(np.vstack(rows_feat).astype(np.float32), columns=fnames)
    out = pd.concat([out, feat_df], axis=1)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print(f"\nwindow settings: {WINDOW}")
    print(f"clips: {len(clip_slices)}  (too short, skipped: {n_short}); "
          f"frames dropped for non-increasing timestamps: {n_dropped_ts}")
    print(f"windows: {len(out):,}   features: {len(fnames)}")
    print("\nwindows per class:")
    counts = out.groupby("symptom").agg(windows=("video_id", "size"), clips=("video_id", "nunique"),
                                        participants=("participant_id", "nunique"))
    print(counts.to_string())
    nan_frac = feat_df.isna().mean()
    print(f"\nfeatures >50% NaN: {(nan_frac > 0.5).sum()} / {len(fnames)}")
    worst = nan_frac.sort_values(ascending=False).head(5)
    print("most-missing features:\n" + worst.round(3).to_string())
    print(f"\nsaved {out_path}  ({time.time() - t_start:.0f}s total)")


if __name__ == "__main__":
    main()
