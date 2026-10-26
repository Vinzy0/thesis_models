"""Native UI state checks: python -m unittest test_webcam_ui -v."""
import time
import unittest

from PySide6.QtWidgets import QApplication
from common import SYMPTOMS
from webcam_app import MainWindow, STYLE


class WindowStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setStyle("Fusion")
        cls.app.setStyleSheet(STYLE)

    def setUp(self):
        self.window = MainWindow(start_camera=False)
        self.window.on_status({"kind": "ready", "message": "UI test; no camera attached"})
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.app.processEvents()

    def packet(self):
        w = self.window
        return {"epoch": w.epoch, "mode": w.mode_combo.currentData(),
                "start": time.monotonic() - 4, "end": time.monotonic(), "fps": 30,
                "coverage": dict.fromkeys(("face", "body", "left_hand", "right_hand", "knees", "reference"), 1),
                "feature_coverage": 1,
                "results": {s: {"score": 0.6, "threshold": 0.5, "reason": ""} for s in SYMPTOMS}}

    def test_pause_and_reset_cannot_show_old_results_as_live(self):
        w = self.window
        w.toggle_analysis()
        packet = self.packet()
        w.on_prediction(packet)
        self.assertEqual(len(w.rows), 7)
        self.assertTrue(w.export_button.isEnabled())
        w.toggle_analysis()
        self.assertTrue(w.score_rows["fear"].bar.stale)
        w.on_prediction(packet)
        self.assertEqual(len(w.rows), 7)
        w.toggle_analysis()
        self.assertIsNone(w.score_rows["fear"].result)
        w.on_prediction(packet)
        self.assertEqual(len(w.rows), 7)
        w.reset_session()
        self.assertEqual(w.rows, [])
        self.assertFalse(w.running)
        self.assertFalse(w.export_button.isEnabled())

    def test_selection_clears_results_but_retains_labeled_export_history(self):
        w = self.window
        w.toggle_analysis()
        packet = self.packet()
        w.on_prediction(packet)
        w.mode_combo.setCurrentIndex(w.mode_combo.findData("vs_control"))
        self.assertIsNone(w.last_packet)
        self.assertIsNone(w.score_rows["fear"].result)
        w.on_prediction(packet)
        self.assertEqual(len(w.rows), 7)
        self.assertTrue(all(row["mode"] == "vs_everything" for row in w.rows))
        w.symptom_combo.setCurrentIndex(w.symptom_combo.findData("fear"))
        self.app.processEvents()
        self.assertTrue(w.score_rows["fear"].isVisible())
        self.assertFalse(w.score_rows["fidgeting"].isVisible())
        self.assertEqual(w.history_combo.currentData(), "fear")

    def test_camera_error_disables_analysis_and_marks_results_stale(self):
        w = self.window
        w.toggle_analysis()
        w.on_prediction(self.packet())
        w.on_status({"kind": "camera_error", "message": "Test camera unavailable"})
        self.assertFalse(w.running)
        self.assertFalse(w.start_button.isEnabled())
        self.assertTrue(w.score_rows["fear"].bar.stale)
        self.assertIsNone(w.video.image)

    def test_low_fps_warning_keeps_tremor_bar_visible(self):
        w = self.window
        w.toggle_analysis()
        packet = self.packet()
        packet["fps"] = 15
        for symptom in ("hand_tremor", "knee_tremor"):
            packet["results"][symptom]["warning"] = "Low FPS (15.0); tremor score may be unreliable"
        w.on_prediction(packet)
        for symptom in ("hand_tremor", "knee_tremor"):
            row = w.score_rows[symptom]
            self.assertEqual(row.value.text(), "60%")
            self.assertEqual(row.bar.score, 0.6)
            self.assertFalse(row.bar.stale)
            self.assertIn("Low FPS", row.note.text())
            self.assertIn("Above threshold", row.note.text())
            row.show_result()
            self.assertIn("Low FPS", row.note.text())


if __name__ == "__main__":
    unittest.main()
