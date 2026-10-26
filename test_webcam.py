"""Run: python -m unittest test_webcam -v. No camera or network required."""
import csv
import shutil
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

from common import (FLAG_COLS, ROOT, SYMPTOMS, clip_window_features, compute_signals,
                    landmark_cols)
from webcam_core import (EXPORT_FIELDS, LiveWindow, ModelBank, adapt_landmarks,
                         ensure_assets, export_rows, local_coordinates, sampling_warning, save_csv, tracking_reason)


class StartupTests(unittest.TestCase):
    def test_bundled_models_and_metrics_work_without_research_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "models", root / "models")
            shutil.copyfile(ROOT / "model_metrics.json", root / "model_metrics.json")
            bank = ModelBank(root)
            self.assertEqual(len(bank.bundles), 14)
            self.assertEqual(set(bank.metrics), set(SYMPTOMS))
            self.assertFalse((root / "results").exists())
            self.assertFalse((root / "data.parquet").exists())
            for symptom in SYMPTOMS:
                for mode in ("vs_control", "vs_everything"):
                    self.assertTrue(0 <= bank.metrics[symptom][mode]["window"]["f1"] <= 1)

    def test_cancelled_download_does_not_continue_or_leave_partial_assets(self):
        cancel = False

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self, size):
                nonlocal cancel
                cancel = True
                return b"partial data"

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("webcam_core.ASSET_DIR", Path(tmp)), mock.patch(
                    "webcam_core.urllib.request.urlopen", return_value=Response()) as request:
                with self.assertRaises(InterruptedError):
                    ensure_assets(cancelled=lambda: cancel)
                self.assertEqual(request.call_count, 1)
                self.assertEqual(request.call_args.kwargs["timeout"], 5)
                self.assertEqual(list(Path(tmp).iterdir()), [])
                request.reset_mock()
                with self.assertRaises(InterruptedError):
                    ensure_assets(cancelled=lambda: True)
                request.assert_not_called()


class LiveInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = ModelBank()
        # A small slice of one real clip, not fabricated model scores.
        import pyarrow.parquet as pq
        columns = ["video_id", "timestamp_ms", *landmark_cols(), *cls.bank.bs_names,
                   *cls.bank.fm_names, *FLAG_COLS]
        raw = pq.ParquetFile(ROOT / "data.parquet").read_row_group(0, columns=columns).to_pandas()
        raw = raw[raw.video_id == raw.video_id.iloc[0]].sort_values("timestamp_ms")
        raw = raw.drop_duplicates("timestamp_ms").iloc[:240]
        cls.times = (raw.timestamp_ms.to_numpy() - raw.timestamp_ms.iloc[0]) / 1000
        cls.raw = raw
        cls.records = [(float(t), row, dict.fromkeys(("face", "body", "left_hand", "right_hand", "knees", "reference"), True))
                       for t, row in zip(cls.times, raw.to_dict("records"))]

    def test_live_features_equal_offline_and_cached_training_features(self):
        bank, raw = self.bank, self.raw
        _, signals = compute_signals({name: raw[name].to_numpy(dtype=np.float64) for name in landmark_cols()})
        starts, offline = clip_window_features(self.times, signals, raw[bank.bs_names].to_numpy(),
                                              raw[bank.fm_names].to_numpy(), raw[FLAG_COLS].to_numpy(dtype=np.float32))
        self.assertGreater(len(starts), 0)
        live = bank.extract(self.records, starts[0]).to_numpy()
        np.testing.assert_allclose(live[0], offline[0], rtol=1e-6, atol=1e-6, equal_nan=True)
        import pyarrow.parquet as pq
        cached = pq.read_table(ROOT / "features.parquet", filters=[("video_id", "=", raw.video_id.iloc[0])]).to_pandas()
        cached = cached[np.isclose(cached.win_start_s, starts[0])]
        self.assertEqual(len(cached), 1)
        np.testing.assert_allclose(live[0], cached[bank.features].to_numpy()[0], rtol=1e-5, atol=1e-5, equal_nan=True)

    def test_all_fourteen_saved_models_and_live_prediction_agree(self):
        bank = self.bank
        self.assertEqual(len(bank.bundles), 14)
        features = bank.extract(self.records, 1)
        for mode in ("vs_control", "vs_everything"):
            packet = bank.predict(self.records, 1, mode, "all")
            self.assertEqual(set(packet["results"]), set(SYMPTOMS))
            for symptom, result in packet["results"].items():
                expected = bank.bundles[(symptom, mode)]["model"].predict_proba(features)[0, 1]
                self.assertIsNotNone(result["score"], result["reason"])
                self.assertAlmostEqual(result["score"], expected)
                self.assertTrue(0 <= result["score"] <= 1)
        one = bank.predict(self.records, 1, "vs_control", "fear")
        self.assertEqual(list(one["results"]), ["fear"])

    def test_missing_tracking_and_extremely_low_rate_suppress_scores(self):
        records = [(t, row, dict.fromkeys(track, False)) for t, row, track in self.records]
        packet = self.bank.predict(records, 1, "vs_everything", "all")
        self.assertTrue(all(r["score"] is None and r["reason"] for r in packet["results"].values()))
        flags = dict.fromkeys(self.records[0][2], 1)
        self.assertEqual(tracking_reason("hand_tremor", flags, 15), "")
        self.assertIn("Too few", tracking_reason("hand_tremor", flags, 7))
        self.assertEqual(tracking_reason("fear", flags, 15), "")

    def test_low_fps_tremor_scores_are_visible_and_export_the_warning(self):
        # Preserve real timestamps: every other observation is about 15 fps.
        records = self.records[::2]
        for mode in ("vs_control", "vs_everything"):
            for symptom in ("hand_tremor", "knee_tremor"):
                packet = self.bank.predict(records, 1, mode, symptom)
                result = packet["results"][symptom]
                self.assertGreaterEqual(packet["fps"], 8)
                self.assertLess(packet["fps"], 20)
                self.assertIsNotNone(result["score"])
                self.assertEqual(result["reason"], "")
                self.assertIn("Low FPS", result["warning"])
                row = next(export_rows(packet, "low-fps", 1, 5))
                self.assertEqual(row["valid"], 1)
                self.assertEqual(row["warning"], result["warning"])
        self.assertEqual(sampling_warning("hand_tremor", 20), "")
        self.assertEqual(sampling_warning("fear", 15), "")

    def test_window_buffer_never_bridges_interruptions(self):
        window = LiveWindow()
        ready = []
        for i in range(241):
            result, _ = window.push(i / 30, {}, {})
            if result:
                ready.append(result[1])
        np.testing.assert_allclose(ready, [1, 3])
        result, interrupted = window.push(11, {}, {})
        self.assertTrue(interrupted)
        self.assertIsNone(result)
        self.assertEqual(len(window.records), 0)
        window.reset()
        self.assertIsNone(window.began)
        self.assertEqual(window.push(0, {}, {}), (None, False))
        self.assertEqual(window.push(0, {}, {}), (None, False))

    def test_incomplete_explicit_window_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "fully covered"):
            self.bank.extract(self.records, self.times[-1] - 1)

    def test_adapter_normalization_handedness_neutral_and_matrix(self):
        def points(array):
            return [NS(x=x, y=y, z=z, visibility=1, presence=1) for x, y, z in array]
        face = np.zeros((478, 3))
        face[133], face[362] = [-0.5, 0, 0], [0.5, 0, 0]
        face[33], face[263] = [-1, 0, 0], [1, 0, 0]
        face += [0.5, 0.5, 0]
        body = np.full((33, 3), [0.5, 0.7, 0])
        body[11], body[12] = [0.7, 0.3, 0], [0.3, 0.3, 0]
        body[23], body[24] = [0.6, 0.7, 0], [0.4, 0.7, 0]
        body[15], body[16] = [0.8, 0.6, 0], [0.2, 0.6, 0]
        left = np.tile(body[15], (21, 1))
        right = np.tile(body[16], (21, 1))
        result, flags, _ = adapt_landmarks(
            NS(face_landmarks=[points(face)], face_blendshapes=[[NS(category_name="_neutral", score=0.75)]],
               facial_transformation_matrixes=[np.arange(16).reshape(4, 4)]),
            NS(pose_landmarks=[points(body)]), NS(hand_landmarks=[points(right), points(left)]),
            self.bank.bs_names, self.bank.fm_names)
        self.assertAlmostEqual(result["bs_neutral"], 0.75)
        self.assertEqual(result["face_matrix_1_2"], 6)
        self.assertAlmostEqual(result["pose_12_x"] - result["pose_11_x"], 1)
        self.assertAlmostEqual(result["pose_11_y"], result["pose_12_y"])
        self.assertAlmostEqual(result["pose_23_x"] + result["pose_24_x"], 0)
        self.assertAlmostEqual(result["left_hand_0_x"], result["pose_15_x"])
        self.assertAlmostEqual(result["right_hand_0_x"], result["pose_16_x"])
        self.assertTrue(flags["left_hand"] and flags["right_hand"])
        self.assertTrue(np.isnan(local_coordinates(body, body[0], [0, 0, 0], 0)).all())
        missing, flags, raw = adapt_landmarks(NS(face_landmarks=[]), NS(pose_landmarks=[]), NS(hand_landmarks=[]),
                                              self.bank.bs_names, self.bank.fm_names)
        self.assertFalse(any(flags.values()))
        self.assertTrue(np.isnan(missing["face_1_x"]))

    def test_export_preserves_missing_values_and_comparison_mode(self):
        packet = self.bank.predict(self.records, 1, "vs_control", "fear")
        packet["results"]["fear"] = {"score": None, "threshold": 0.5, "reason": "Missing face"}
        rows = list(export_rows(packet, "test-session", 2, 5))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.csv"
            save_csv(path, rows)
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, EXPORT_FIELDS)
                row = next(reader)
            self.assertEqual(row["model_score"], "")
            self.assertEqual(row["valid"], "0")
            self.assertEqual(row["mode"], "vs_control")
            self.assertEqual(row["reason"], "Missing face")


if __name__ == "__main__":
    unittest.main()
