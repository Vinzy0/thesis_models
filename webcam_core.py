"""Live landmark adapter, shared-window inference, and score export. No UI imports."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import re
import tempfile
import urllib.request
import warnings
from collections import deque
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import (BANDS, FACE_POINTS, FLAG_COLS, HAND_POINTS, MODES, POSE_POINTS,
                    ROOT, SYMPTOMS, WINDOW, clip_window_features, compute_signals,
                    feature_names, landmark_cols)

ASSET_DIR = ROOT / "tracking_assets"
ASSETS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}
ASSET_SHA256 = {
    "face_landmarker.task": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    "pose_landmarker_lite.task": "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
    "hand_landmarker.task": "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
}
PRETTY = {s: s.replace("_", " ").capitalize() for s in SYMPTOMS}
MODE_LABELS = {"vs_everything": "Against all other categories", "vs_control": "Against control"}
TRACK_KEYS = ("face", "body", "left_hand", "right_hand", "knees", "reference")
EXPORT_FIELDS = ["session_id", "segment", "timestamp_utc", "elapsed_s", "window_start_s",
                 "window_end_s", "symptom", "mode", "model_score", "threshold",
                 "above_threshold", "valid", "reason", "warning", "processed_fps", "feature_coverage",
                 *[f"{key}_coverage" for key in TRACK_KEYS], "adapter"]
ADAPTER = "reconstructed_dataset_normalization_v1"


def check_cancelled(cancelled):
    if cancelled():
        raise InterruptedError("Tracking startup cancelled")


def ensure_assets(progress=lambda message: None, cancelled=lambda: False):
    """Download official, versioned bundles once, with atomic replacement."""
    check_cancelled(cancelled)
    ASSET_DIR.mkdir(exist_ok=True)
    manifest = {}
    for name, url in ASSETS.items():
        check_cancelled(cancelled)
        target = ASSET_DIR / name
        expected = ASSET_SHA256[name]
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            progress(f"Downloading {name.replace('_', ' ')}...")
            with tempfile.NamedTemporaryFile(dir=ASSET_DIR, suffix=".part", delete=False) as temp:
                part = Path(temp.name)
            try:
                with urllib.request.urlopen(url, timeout=5) as response, part.open("wb") as out:
                    while True:
                        check_cancelled(cancelled)
                        chunk = response.read(64 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                check_cancelled(cancelled)
                if hashlib.sha256(part.read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f"Tracking model checksum failed: {name}. Reconnect to download again.")
                part.replace(target)
            finally:
                part.unlink(missing_ok=True)
        manifest[name] = {"url": url, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    check_cancelled(cancelled)
    (ASSET_DIR / "sources.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def local_coordinates(points, origin, axis, scale):
    """Translate, rotate in image XY, and scale; never mirror inference input."""
    points = np.asarray(points, dtype=np.float64)
    if not np.isfinite(scale) or scale < 1e-6:
        return np.full_like(points, np.nan)
    x_axis = np.asarray(axis[:2], dtype=float)
    norm = np.linalg.norm(x_axis)
    if norm < 1e-6:
        return np.full_like(points, np.nan)
    x_axis /= norm
    centered = points - origin
    out = centered.copy()
    out[:, 0] = centered[:, :2] @ x_axis
    out[:, 1] = centered[:, :2] @ np.array([-x_axis[1], x_axis[0]])
    return out / scale


def xyz(landmarks):
    return np.asarray([[p.x, p.y, p.z] for p in landmarks], dtype=np.float64)


def visible(point):
    return (0 <= point.x <= 1 and 0 <= point.y <= 1
            and getattr(point, "visibility", 1) >= 0.5
            and getattr(point, "presence", 1) >= 0.5)


def adapt_landmarks(face_result, pose_result, hand_result, bs_names, fm_names):
    """Reconstruct dataset normalization, including pose-based hand assignment.

    Dataset invariants: face origin=midpoint(133,362), inner eyes horizontal,
    outer-eye XY distance=1; body origin=midpoint(23,24), shoulders horizontal
    with XY distance=1. Exact source-exporter parity remains unverified.
    """
    row = dict.fromkeys(landmark_cols() + bs_names + fm_names, np.nan)
    tracking = dict.fromkeys(TRACK_KEYS, False)
    raw = {}

    def store(prefix, points, indices):
        for i in indices:
            for j, axis in enumerate("xy"):
                row[f"{prefix}_{i}_{axis}"] = float(np.float32(points[i, j]))

    if face_result.face_landmarks:
        points = xyz(face_result.face_landmarks[0])
        origin = (points[133] + points[362]) / 2
        scale = np.linalg.norm(points[33, :2] - points[263, :2])
        normalized = local_coordinates(points, origin, points[362] - points[133], scale)
        store("face", normalized, FACE_POINTS)
        tracking["face"] = bool(np.isfinite(normalized[FACE_POINTS, :2]).all())
        raw["face"] = points
        if face_result.face_blendshapes:
            for category in face_result.face_blendshapes[0]:
                name = "bs_" + category.category_name.lstrip("_")
                if name in row:
                    row[name] = category.score
        if face_result.facial_transformation_matrixes:
            matrix = np.asarray(face_result.facial_transformation_matrixes[0])
            for name in fm_names:
                _, _, r, c = name.split("_")
                row[name] = float(matrix[int(r), int(c)])

    if pose_result.pose_landmarks:
        landmarks = pose_result.pose_landmarks[0]
        points = xyz(landmarks)
        origin = (points[23] + points[24]) / 2
        axis = points[12] - points[11]
        scale = np.linalg.norm(axis[:2])
        normalized = local_coordinates(points, origin, axis, scale)
        reference = bool(np.isfinite(normalized[POSE_POINTS, :2]).all())
        store("pose", normalized, POSE_POINTS)
        raw["body"] = points
        tracking["reference"] = reference
        tracking["body"] = reference and all(visible(landmarks[i]) for i in (11, 12))
        tracking["knees"] = reference and all(visible(landmarks[i]) for i in (25, 26))
        # Assign up to two hands to anatomical pose wrists, avoiding mirrored
        # handedness conventions between legacy and Tasks implementations.
        hands = [xyz(h) for h in hand_result.hand_landmarks[:2]]
        if hands and reference:
            wrists = points[[15, 16], :2]
            slots = min(itertools.permutations(range(2), len(hands)),
                        key=lambda order: sum(np.linalg.norm(hand[0, :2] - wrists[slot])
                                              for hand, slot in zip(hands, order)))
            for hand, slot in zip(hands, slots):
                side = ("left_hand", "right_hand")[slot]
                store(side, local_coordinates(hand, origin, axis, scale), HAND_POINTS)
                tracking[side] = True
                raw[side] = hand

    # Detection flags represent detector output, independently of quality gates.
    row["face_detected"] = float(bool(face_result.face_landmarks))
    row["pose_detected"] = float(bool(pose_result.pose_landmarks))
    row["left_hand_detected"] = float(tracking["left_hand"])
    row["right_hand_detected"] = float(tracking["right_hand"])
    return row, tracking, raw


def tracking_reason(symptom, coverage, fps=None):
    """Conservative display gates, not learned confidence or clinical validity."""
    enough = lambda key: coverage.get(key, 0) >= 0.5
    if fps is not None and fps < 8:
        return "Too few processed frames"
    if symptom in ("avoidant_gaze", "facial_tension", "fear") and not enough("face"):
        return "Keep your face in view"
    if symptom in ("fidgeting", "hand_tremor", "knee_tremor", "heavy_breathing"):
        if not enough("body") or not enough("reference"):
            return "Keep your shoulders and body in view"
    if symptom == "hand_tremor" and not (enough("left_hand") or enough("right_hand")):
        return "Bring a hand into view"
    if symptom == "knee_tremor" and not enough("knees"):
        return "Move back until both knees are visible"
    return ""


def sampling_warning(symptom, fps):
    if symptom in ("hand_tremor", "knee_tremor") and fps is not None and fps < 20:
        return f"Low FPS ({fps:.1f}); tremor score may be unreliable"
    return ""


class ModelBank:
    def __init__(self, root=ROOT):
        self.bundles = {}
        self.metrics = {}
        bundled_path = Path(root) / "model_metrics.json"
        bundled_metrics = json.loads(bundled_path.read_text(encoding="utf-8")) if bundled_path.exists() else {}
        for symptom, mode in itertools.product(SYMPTOMS, MODES):
            bundle = joblib.load(Path(root) / "models" / f"{symptom}_{mode}.joblib")
            if bundle.get("symptom") != symptom or bundle.get("mode") != mode:
                raise ValueError(f"Incorrect model identity: {symptom}/{mode}")
            if bundle.get("window") != WINDOW or bundle.get("bands") != BANDS:
                raise ValueError(f"Unsupported feature settings: {symptom}/{mode}")
            bundle["model"].set_params(n_jobs=1)
            self.bundles[(symptom, mode)] = bundle
            metric_path = Path(root) / "results" / symptom / "metrics.json"
            if metric_path.exists():
                self.metrics[symptom] = json.loads(metric_path.read_text(encoding="utf-8"))
            elif symptom in bundled_metrics:
                self.metrics[symptom] = bundled_metrics[symptom]
        self.features = list(next(iter(self.bundles.values()))["features"])
        if any(list(b["features"]) != self.features for b in self.bundles.values()):
            raise ValueError("Model feature layouts differ")
        self.bs_names = list(dict.fromkeys(re.sub(r"_(mean|std|max)$", "", f[2:])
                                          for f in self.features if f.startswith("f_bs_")))
        self.fm_names = list(dict.fromkeys(re.sub(r"_(mean|std)$", "", f[2:])
                                          for f in self.features if f.startswith("f_face_matrix_")))
        dummy = {c: np.zeros(1) for c in landmark_cols()}
        signals, _ = compute_signals(dummy)
        if feature_names(signals, self.bs_names, self.fm_names) != self.features:
            raise ValueError("Live feature names do not match saved model order")

    def extract(self, records, start):
        times = np.asarray([r[0] for r in records])
        rows = [r[1] for r in records]
        columns = {c: np.asarray([row[c] for row in rows], dtype=float) for c in landmark_cols()}
        _, signals = compute_signals(columns)
        matrix = lambda names: np.asarray([[row[n] for n in names] for row in rows], dtype=np.float32)
        _, values = clip_window_features(times, signals, matrix(self.bs_names), matrix(self.fm_names),
                                         matrix(FLAG_COLS), window_starts=[start])
        values = values.astype(np.float32)
        if np.isinf(values).any():
            raise ValueError("Non-finite live feature values")
        return pd.DataFrame(values, columns=self.features)

    def predict(self, records, start, mode, selected):
        end = start + WINDOW["win_s"]
        inside = [r for r in records if start <= r[0] < end]
        times = np.asarray([r[0] for r in inside])
        # Count unique processed observations, never the interpolated 30 Hz grid.
        fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0
        max_gap = float(np.diff(times).max()) if len(times) > 1 else float("inf")
        coverage = {key: float(np.mean([r[2][key] for r in inside])) if inside else 0
                    for key in TRACK_KEYS}
        features = self.extract(records, start)
        feature_coverage = float(np.isfinite(features.to_numpy()).mean())
        results = {}
        for symptom in SYMPTOMS if selected == "all" else [selected]:
            bundle = self.bundles[(symptom, mode)]
            reason = tracking_reason(symptom, coverage, fps)
            if max_gap > 0.5:
                reason = "Tracking was interrupted; collect a fresh window"
            score = None
            if not reason:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="X does not have valid feature names")
                    score = float(bundle["model"].predict_proba(features)[0, 1])
                if not np.isfinite(score):
                    raise ValueError("Model produced a non-finite score")
            results[symptom] = {"score": score, "threshold": float(bundle["threshold"]), "reason": reason,
                                "warning": sampling_warning(symptom, fps) if score is not None else ""}
        return {"results": results, "mode": mode, "start": start, "end": end,
                "fps": float(fps), "coverage": coverage, "feature_coverage": feature_coverage}


class LiveWindow:
    """Bounded rolling buffer with a one-second settling period and real timestamps."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.records = deque()
        self.began = self.start = self.last = None

    def push(self, timestamp, row, tracking):
        interrupted = self.last is not None and timestamp - self.last > 0.5
        if self.last is not None and timestamp <= self.last:
            return None, False
        if interrupted:
            self.reset()
        self.last = timestamp
        if self.began is None:
            self.began = timestamp
        if timestamp - self.began < WINDOW["trim_s"]:
            return None, interrupted
        if self.start is None:
            self.start = timestamp
        self.records.append((timestamp, row, tracking))
        if timestamp + 1e-9 < self.start + WINDOW["win_s"]:
            return None, interrupted
        start = self.start
        snapshot = list(self.records)
        self.start += WINDOW["step_s"]
        # Keep one predecessor so explicit window starts remain covered.
        while len(self.records) > 1 and self.records[1][0] <= self.start:
            self.records.popleft()
        return (snapshot, start), interrupted

    def progress(self):
        if self.began is None or self.last is None:
            return 0.0
        if self.start is None:
            return 0.0
        return min(1.0, max(0.0, (self.last - self.start) / WINDOW["win_s"]))


class LandmarkTracker:
    def __init__(self, bank, progress=lambda message: None, cancelled=lambda: False):
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        self.mp, self.bank = mp, bank
        ensure_assets(progress, cancelled)
        self.stack = ExitStack()
        try:
            options = dict(running_mode=vision.RunningMode.VIDEO)
            base = lambda name: mp.tasks.BaseOptions(model_asset_path=str(ASSET_DIR / name))
            check_cancelled(cancelled)
            self.face = self.stack.enter_context(vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(base_options=base("face_landmarker.task"), num_faces=1,
                    output_face_blendshapes=True, output_facial_transformation_matrixes=True, **options)))
            check_cancelled(cancelled)
            self.pose = self.stack.enter_context(vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(base_options=base("pose_landmarker_lite.task"), num_poses=1, **options)))
            check_cancelled(cancelled)
            self.hands = self.stack.enter_context(vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(base_options=base("hand_landmarker.task"), num_hands=2, **options)))
            check_cancelled(cancelled)
        except Exception:
            self.stack.close()
            raise
        self.last_ms = -1

    def process(self, rgb, timestamp, executor):
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        ms = max(self.last_ms + 1, int(timestamp * 1000))
        self.last_ms = ms
        futures = [executor.submit(task.detect_for_video, image, ms)
                   for task in (self.face, self.pose, self.hands)]
        results = [future.result() for future in futures]
        return adapt_landmarks(*results, self.bank.bs_names, self.bank.fm_names)

    def close(self):
        self.stack.close()


def export_rows(packet, session_id, segment, elapsed):
    timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    for symptom, result in packet["results"].items():
        score = result["score"]
        yield {"session_id": session_id, "segment": segment, "timestamp_utc": timestamp,
               "elapsed_s": round(elapsed, 3), "window_start_s": round(packet["start"], 3),
               "window_end_s": round(packet["end"], 3), "symptom": symptom, "mode": packet["mode"],
               "model_score": "" if score is None else round(score, 6), "threshold": result["threshold"],
               "above_threshold": "" if score is None else int(score >= result["threshold"]),
               "valid": int(score is not None), "reason": result["reason"], "warning": result.get("warning", ""),
               "processed_fps": round(packet["fps"], 2), "feature_coverage": round(packet["feature_coverage"], 4),
               **{f"{key}_coverage": round(packet["coverage"][key], 4) for key in TRACK_KEYS}, "adapter": ADAPTER}


def save_csv(path, rows):
    """An interrupted export never replaces an existing report with a partial one."""
    path = Path(path)
    part = path.with_name(path.name + ".part")
    try:
        with part.open("w", newline="", encoding="utf-8-sig") as out:
            writer = csv.DictWriter(out, fieldnames=EXPORT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)
