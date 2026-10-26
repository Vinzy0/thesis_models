"""Windowed launcher with a visible startup error if dependencies are missing."""
import ctypes
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
# pythonw has no standard streams; some native/ML packages still write to them.
log = (ROOT / "webcam_launcher.log").open("a", encoding="utf-8")
sys.stdout = sys.stderr = log
try:
    from webcam_app import main
    raise SystemExit(main())
except Exception:
    traceback.print_exc()
    log.flush()
    ctypes.windll.user32.MessageBoxW(
        0, "The webcam tester could not start.\n\n"
        "Open webcam_launcher.log in the project folder for the error.\n"
        "Follow the Windows setup steps in README.md,\nthen run Start Webcam Tester.cmd.",
        "Symptom Lab - Startup error", 0x10)
finally:
    log.close()
