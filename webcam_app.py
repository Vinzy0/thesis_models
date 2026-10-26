"""Symptom Lab: native Windows webcam testing for the existing fourteen models."""
from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QFileDialog,
                              QFrame, QGridLayout, QHBoxLayout, QLabel, QMainWindow,
                              QMessageBox, QProgressBar, QPushButton, QScrollArea,
                              QSizePolicy, QSplitter, QVBoxLayout, QWidget)

from common import ROOT, SYMPTOMS
from webcam_core import (MODE_LABELS, PRETTY, TRACK_KEYS, LandmarkTracker,
                         LiveWindow, ModelBank, export_rows, save_csv, tracking_reason)

INK = "#17232b"
MUTED = "#53616b"
BLUE = "#3156a3"
AMBER = "#94610b"
TRACK_LABELS = {"face": "Face", "body": "Shoulders", "left_hand": "Left hand",
                "right_hand": "Right hand", "knees": "Both knees", "reference": "Body reference"}
BODY_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23),
              (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
              (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
FACE_EDGES = [(33, 133), (362, 263), (33, 159), (159, 133), (33, 145), (145, 133),
              (362, 386), (386, 263), (362, 374), (374, 263), (61, 13), (13, 291), (61, 14), (14, 291)]


def annotate(rgb, raw):
    h, w = rgb.shape[:2]
    for group, points in raw.items():
        color = (124, 214, 179) if group == "body" else (159, 188, 255)
        edges = BODY_EDGES if group == "body" else FACE_EDGES if group == "face" else HAND_EDGES
        xy = [(int(np.clip(p[0], -2, 3) * w), int(np.clip(p[1], -2, 3) * h)) for p in points]
        for a, b in edges:
            cv2.line(rgb, xy[a], xy[b], color, 2, cv2.LINE_AA)
        ids = sorted(set(i for edge in edges for i in edge))
        if group == "face":
            ids += [1, 468, 473]
        for i in ids:
            cv2.circle(rgb, xy[i], 3, color, -1, cv2.LINE_AA)
    return rgb


class CameraWorker(QThread):
    status = Signal(object)
    bank_ready = Signal(object)
    prediction = Signal(object)

    def __init__(self):
        super().__init__()
        self.commands = queue.Queue()
        self.lock = threading.Lock()
        self.latest = None
        self.sequence = 0

    def command(self, *command):
        self.commands.put(command)

    def snapshot(self):
        with self.lock:
            return self.latest

    def run(self):
        cap = tracker = None
        active, epoch, overlay, mirror = False, 0, True, True
        mode, selected = "vs_everything", "all"
        window = LiveWindow()
        samples = deque(maxlen=60)
        failed_reads = 0

        def state(kind, message):
            self.status.emit({"kind": kind, "message": message})

        def open_camera(index):
            state("opening", f"Opening camera {index}...")
            camera = None
            for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF) if sys.platform == "win32" else (cv2.CAP_ANY,):
                camera = cv2.VideoCapture(index, backend)
                if camera.isOpened():
                    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    # This USB camera delivers ~10 fps at 720p but >20 at 480p.
                    # Favor enough real observations for the tremor band.
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    camera.set(cv2.CAP_PROP_FPS, 30)
                    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    state("ready", f"Camera {index} ready. Check your framing, then start analysis.")
                    return camera
                camera.release()
            state("camera_error", f"Camera {index} is unavailable. Close other camera apps, select a camera, then reconnect.")
            return None

        try:
            state("loading", "Loading the 14 symptom models...")
            bank = ModelBank()
            self.bank_ready.emit(bank)
            if self.isInterruptionRequested():
                return
            state("loading", "Preparing face, body, and hand tracking...")
            tracker = LandmarkTracker(bank, lambda message: state("loading", message), self.isInterruptionRequested)
            if self.isInterruptionRequested():
                return
            cap = open_camera(0)
            # One call per tracker per frame. Independent native detectors run in
            # parallel; the next frame waits for all three to preserve timestamps.
            with ThreadPoolExecutor(max_workers=3, thread_name_prefix="landmarks") as executor:
                while not self.isInterruptionRequested():
                    while not self.commands.empty():
                        command = self.commands.get_nowait()
                        if command[0] == "analysis":
                            _, active, epoch = command
                            window.reset()
                        elif command[0] == "select":
                            _, mode, selected, epoch = command
                            window.reset()
                        elif command[0] == "camera":
                            _, index, epoch = command
                            active = False
                            window.reset()
                            samples.clear()
                            if cap is not None:
                                cap.release()
                            with self.lock:
                                self.latest = None
                            cap = open_camera(index)
                            failed_reads = 0
                        elif command[0] == "overlay":
                            overlay = command[1]
                        elif command[0] == "mirror":
                            mirror = command[1]
                    if cap is None:
                        self.msleep(60)
                        continue
                    ok, bgr = cap.read()
                    captured = time.monotonic()
                    if not ok:
                        failed_reads += 1
                        if failed_reads >= 5:
                            cap.release()
                            cap, active = None, False
                            window.reset()
                            state("camera_error", "The camera stopped sending frames. Reconnect to try again.")
                        self.msleep(30)
                        continue
                    failed_reads = 0
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    row, tracking, raw = tracker.process(rgb, captured, executor)
                    samples.append(captured)
                    fps = (len(samples) - 1) / (samples[-1] - samples[0]) if len(samples) > 1 else 0
                    interrupted = False
                    if active:
                        ready, interrupted = window.push(captured, row, tracking)
                        if ready:
                            records, start = ready
                            packet = bank.predict(records, start, mode, selected)
                            packet["epoch"] = epoch
                            self.prediction.emit(packet)
                    if overlay:
                        rgb = annotate(rgb, raw)
                    if mirror:
                        rgb = cv2.flip(rgb, 1)
                    self.sequence += 1
                    with self.lock:
                        self.latest = {"seq": self.sequence, "rgb": np.ascontiguousarray(rgb),
                                       "tracking": tracking, "fps": fps, "active": active,
                                       "progress": window.progress(), "epoch": epoch,
                                       "settling": active and window.start is None,
                                       "captured": captured, "interrupted": interrupted}
                    delay = 1 / 30 - (time.monotonic() - captured)
                    if delay > 0:
                        self.msleep(max(1, int(delay * 1000)))
        except InterruptedError:
            logging.info("Tracking startup cancelled")
        except Exception:
            logging.exception("Camera worker failed")
            state("fatal", "Tracking could not start. See webcam_app.log for details; use Reconnect to retry.")
        finally:
            if cap is not None:
                cap.release()
            if tracker is not None:
                tracker.close()


class VideoView(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.message = "Preparing your camera..."
        self.setMinimumSize(360, 210)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAccessibleName("Live camera preview")

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#17232b"))
        if self.image is not None:
            scaled = self.image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
            target = QRectF((self.width() - scaled.width()) / 2, (self.height() - scaled.height()) / 2,
                            scaled.width(), scaled.height())
            p.drawImage(target, self.image)
        else:
            p.setPen(QColor("#e2e7e9"))
            p.setFont(QFont("Segoe UI", 12))
            p.drawText(self.rect().adjusted(24, 24, -24, -24), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                       self.message)


class ScoreBar(QWidget):
    def __init__(self):
        super().__init__()
        self.score = None
        self.threshold = 0.5
        self.stale = False
        self.setFixedHeight(12)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0, 2, self.width(), 8)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#e5e9ed"))
        p.drawRoundedRect(rect, 4, 4)
        if self.score is not None:
            color = "#a5afb8" if self.stale else AMBER if self.score >= self.threshold else BLUE
            p.setBrush(QColor(color))
            p.drawRoundedRect(QRectF(0, 2, self.width() * self.score, 8), 4, 4)
        p.setPen(QPen(QColor(INK), 1.5))
        x = self.width() * self.threshold
        p.drawLine(QPointF(x, 0), QPointF(x, 12))


class ScoreRow(QFrame):
    focused = Signal(str)

    def __init__(self, symptom):
        super().__init__()
        self.symptom = symptom
        self.result = None
        self.setObjectName("scoreRow")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(2)
        top = QHBoxLayout()
        self.name = QPushButton(PRETTY[symptom])
        self.name.setObjectName("rowName")
        self.name.setToolTip("Show this symptom in the history graph")
        self.name.clicked.connect(lambda: self.focused.emit(symptom))
        self.value = QLabel("--")
        self.value.setObjectName("scoreValue")
        top.addWidget(self.name)
        top.addStretch()
        top.addWidget(self.value)
        self.bar = ScoreBar()
        self.note = QLabel("Waiting for analysis")
        self.note.setObjectName("small")
        self.note.setWordWrap(True)
        layout.addLayout(top)
        layout.addWidget(self.bar)
        layout.addWidget(self.note)

    def mousePressEvent(self, event):
        self.focused.emit(self.symptom)
        super().mousePressEvent(event)

    def show_result(self, result=None, stale_reason=""):
        if result is not None:
            self.result = result
        result = self.result
        self.bar.score = result["score"] if result else None
        self.bar.threshold = result["threshold"] if result else 0.5
        self.bar.stale = bool(stale_reason)
        score = self.bar.score
        self.value.setText("--" if score is None else f"{score:.0%}")
        if stale_reason:
            text = stale_reason + (" · previous score" if score is not None else "")
        elif result and result["reason"]:
            text = result["reason"]
        elif score is not None:
            text = "Above threshold" if score >= self.bar.threshold else "Below threshold"
            if result.get("warning"):
                text += " · " + result["warning"]
        else:
            text = "Waiting for analysis"
        self.note.setText(text)
        self.value.setStyleSheet(f"color: {MUTED if stale_reason or score is None else AMBER if score >= self.bar.threshold else BLUE};")
        self.bar.update()
        self.setAccessibleName(f"{PRETTY[self.symptom]}, {self.value.text()}, {text}")

    def clear(self, text="Waiting for analysis"):
        self.result = None
        self.show_result()
        self.note.setText(text)


class HistoryPlot(QWidget):
    def __init__(self):
        super().__init__()
        self.points = deque(maxlen=300)
        self.current_time = 0
        self.threshold = 0.5
        self.setMinimumHeight(110)
        self.setAccessibleName("Model score over the last 60 seconds")

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setFont(QFont("Segoe UI", 9))
        r = QRectF(36, 9, max(10, self.width() - 49), max(10, self.height() - 33))
        for value in (0, 0.5, 1):
            y = r.bottom() - r.height() * value
            p.setPen(QColor("#dfe4e8"))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            p.setPen(QColor(MUTED))
            p.drawText(QRectF(0, y - 8, 30, 17), Qt.AlignmentFlag.AlignRight, f"{value:.0%}")
        p.setPen(QPen(QColor(AMBER), 1, Qt.PenStyle.DashLine))
        y = r.bottom() - r.height() * self.threshold
        p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
        now = self.current_time
        p.setPen(QColor(MUTED))
        p.drawText(QRectF(r.left(), r.bottom() + 6, 70, 18), "60 sec ago")
        p.drawText(QRectF(r.right() - 50, r.bottom() + 6, 50, 18), Qt.AlignmentFlag.AlignRight, "Now")
        path = QPainterPath()
        connected = False
        any_point = False
        previous_segment = None
        for t, score, segment in self.points:
            if t < now - 60:
                continue
            if score is None:
                connected = False
                continue
            x, y = r.left() + (t - now + 60) / 60 * r.width(), r.bottom() - score * r.height()
            if connected and segment == previous_segment:
                path.lineTo(x, y)
            else:
                path.moveTo(x, y)
            previous_segment, connected, any_point = segment, True, True
            p.setBrush(QColor(BLUE))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(x, y), 2.5, 2.5)
        p.setPen(QPen(QColor(BLUE), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        if not any_point:
            p.setPen(QColor(MUTED))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "Scores will appear after a complete window")


def label(text, name=""):
    widget = QLabel(text)
    if name:
        widget.setObjectName(name)
    return widget


class MainWindow(QMainWindow):
    def __init__(self, start_camera=True):
        super().__init__()
        self.setWindowTitle("Symptom Lab · Webcam model tester")
        self.resize(1320, 860)
        self.setMinimumSize(1080, 720)
        self.bank = None
        self.worker = None
        self.running = self.camera_ready = self.closing = False
        self.epoch = 0
        self.seen_seq = -1
        self.rows = []
        self.last_packet = None
        self.history = deque(maxlen=400)
        self.session_id = str(uuid.uuid4())[:8]
        self.session_start = time.monotonic()
        self.details = None
        self.build_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_preview)
        self.timer.start(33)
        if start_camera:
            QTimer.singleShot(0, self.start_worker)

    def build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(26, 18, 26, 16)
        outer.setSpacing(14)
        header = QHBoxLayout()
        title = QVBoxLayout()
        title.setSpacing(2)
        title.addWidget(label("Symptom Lab", "title"))
        title.addWidget(label("Live testing for your thesis models", "muted"))
        header.addLayout(title)
        header.addStretch()
        self.session_label = label(f"Session {self.session_id}", "muted")
        header.addWidget(self.session_label)
        self.export_button = QPushButton("Export CSV")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export)
        header.addWidget(self.export_button)
        outer.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter, 1)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 18, 0)
        left_layout.setSpacing(11)
        preview_header = QHBoxLayout()
        preview_header.addWidget(label("Camera preview", "sectionTitle"))
        preview_header.addStretch()
        self.fps_label = label("-- fps", "muted")
        preview_header.addWidget(self.fps_label)
        left_layout.addLayout(preview_header)
        self.video = VideoView()
        left_layout.addWidget(self.video, 1)
        camera_controls = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.camera_combo.setAccessibleName("Camera index")
        for i in range(10):
            self.camera_combo.addItem(f"Camera {i}", i)
        self.camera_combo.setToolTip("Camera indices depend on Windows. Try another index if the wrong camera opens.")
        self.camera_combo.currentIndexChanged.connect(self.change_camera)
        camera_controls.addWidget(self.camera_combo)
        self.reconnect_button = QPushButton("Reconnect")
        self.reconnect_button.clicked.connect(self.change_camera)
        camera_controls.addWidget(self.reconnect_button)
        camera_controls.addStretch()
        self.overlay = QCheckBox("Tracking overlay")
        self.overlay.setChecked(True)
        self.overlay.toggled.connect(lambda value: self.send("overlay", value))
        camera_controls.addWidget(self.overlay)
        self.mirror = QCheckBox("Mirror preview")
        self.mirror.setChecked(True)
        self.mirror.setToolTip("Mirrors the preview only. Model inputs stay unmirrored.")
        self.mirror.toggled.connect(lambda value: self.send("mirror", value))
        camera_controls.addWidget(self.mirror)
        left_layout.addLayout(camera_controls)
        self.guide = label("Sit in good light. Keep your face and shoulders visible; include hands and knees to test tremor.", "muted")
        self.guide.setWordWrap(True)
        left_layout.addWidget(self.guide)

        tracking = QGridLayout()
        tracking.setHorizontalSpacing(14)
        tracking.setVerticalSpacing(5)
        self.track_labels = {}
        for i, key in enumerate(TRACK_KEYS):
            widget = label(f"{TRACK_LABELS[key]}: waiting", "small")
            self.track_labels[key] = widget
            tracking.addWidget(widget, i // 3, i % 3)
        left_layout.addLayout(tracking)
        history_head = QHBoxLayout()
        history_head.addWidget(label("Recent scores", "sectionTitle"))
        history_head.addStretch()
        self.history_combo = QComboBox()
        self.history_combo.setAccessibleName("Symptom shown in history and model details")
        for symptom in SYMPTOMS:
            self.history_combo.addItem(PRETTY[symptom], symptom)
        self.history_combo.currentIndexChanged.connect(self.refresh_history)
        history_head.addWidget(self.history_combo)
        left_layout.addLayout(history_head)
        self.plot = HistoryPlot()
        self.plot.setFixedHeight(132)
        left_layout.addWidget(self.plot)
        splitter.addWidget(left)

        right = QFrame()
        right.setObjectName("resultsPane")
        right.setMinimumWidth(365)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(20, 17, 20, 16)
        right_layout.setSpacing(8)
        top = QHBoxLayout()
        top.addWidget(label("Model scores", "sectionTitle"))
        top.addStretch()
        self.loaded_label = label("Loading models", "small")
        top.addWidget(self.loaded_label)
        right_layout.addLayout(top)
        selectors = QGridLayout()
        selectors.setVerticalSpacing(6)
        selectors.setColumnStretch(1, 1)
        selectors.addWidget(label("Symptom", "small"), 0, 0)
        self.symptom_combo = QComboBox()
        self.symptom_combo.addItem("All seven symptoms", "all")
        for symptom in SYMPTOMS:
            self.symptom_combo.addItem(PRETTY[symptom], symptom)
        self.symptom_combo.setAccessibleName("Symptom model selection")
        self.symptom_combo.currentIndexChanged.connect(self.change_selection)
        selectors.addWidget(self.symptom_combo, 0, 1)
        selectors.addWidget(label("Compare", "small"), 1, 0)
        self.mode_combo = QComboBox()
        for mode, text in MODE_LABELS.items():
            self.mode_combo.addItem(text, mode)
        self.mode_combo.setAccessibleName("Model comparison mode")
        self.mode_combo.currentIndexChanged.connect(self.change_selection)
        selectors.addWidget(self.mode_combo, 1, 1)
        right_layout.addLayout(selectors)
        note = label("Each bar is an independent model score. The marker shows its saved threshold.", "small")
        note.setWordWrap(True)
        right_layout.addWidget(note)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        score_layout = QVBoxLayout(content)
        score_layout.setContentsMargins(0, 0, 4, 0)
        score_layout.setSpacing(0)
        self.score_rows = {}
        for symptom in SYMPTOMS:
            row = ScoreRow(symptom)
            row.focused.connect(self.focus_symptom)
            self.score_rows[symptom] = row
            score_layout.addWidget(row)
        score_layout.addStretch()
        scroll.setWidget(content)
        right_layout.addWidget(scroll, 1)
        self.details_button = QPushButton("Model details and validation")
        self.details_button.setEnabled(False)
        self.details_button.clicked.connect(self.show_details)
        right_layout.addWidget(self.details_button)
        interpretation = label("Scores describe model output, not an overall anxiety percentage.", "small")
        interpretation.setWordWrap(True)
        right_layout.addWidget(interpretation)
        splitter.addWidget(right)
        splitter.setSizes([780, 430])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        footer = QHBoxLayout()
        self.start_button = QPushButton("Start analysis")
        self.start_button.setObjectName("primary")
        self.start_button.setEnabled(False)
        self.start_button.setShortcut("Alt+S")
        self.start_button.clicked.connect(self.toggle_analysis)
        footer.addWidget(self.start_button)
        self.reset_button = QPushButton("Reset session")
        self.reset_button.clicked.connect(self.reset_session)
        footer.addWidget(self.reset_button)
        self.status_label = label("Loading models and tracking resources...", "muted")
        self.status_label.setWordWrap(True)
        footer.addWidget(self.status_label, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.progress.setFixedSize(110, 6)
        self.progress.setAccessibleName("Movement window collection progress")
        footer.addWidget(self.progress)
        outer.addLayout(footer)

    def send(self, *command):
        if self.worker is not None and self.worker.isRunning():
            self.worker.command(*command)

    def start_worker(self):
        if self.worker is not None and self.worker.isRunning():
            return
        self.worker = CameraWorker()
        self.worker.bank_ready.connect(self.on_bank)
        self.worker.status.connect(self.on_status)
        self.worker.prediction.connect(self.on_prediction)
        self.worker.finished.connect(self.worker_finished)
        self.worker.start()
        self.send("overlay", self.overlay.isChecked())
        self.send("mirror", self.mirror.isChecked())
        self.send("select", self.mode_combo.currentData(), self.symptom_combo.currentData(), self.epoch)
        if self.camera_combo.currentData() != 0:
            self.send("camera", self.camera_combo.currentData(), self.epoch)

    def on_bank(self, bank):
        self.bank = bank
        self.loaded_label.setText("14 models loaded")
        self.details_button.setEnabled(True)
        self.refresh_history()

    def on_status(self, state):
        self.status_label.setText(state["message"])
        kind = state["kind"]
        self.camera_ready = kind == "ready"
        self.start_button.setEnabled(self.camera_ready and not self.closing)
        if kind != "ready":
            self.running = False
            self.start_button.setText("Start analysis")
        if kind in ("camera_error", "fatal", "opening"):
            self.video.image = None
            self.video.message = state["message"]
            self.video.update()
            self.mark_stale("Camera unavailable" if kind != "opening" else "Switching camera")
            self.fps_label.setText("-- fps")
            for key, widget in self.track_labels.items():
                widget.setText(f"{TRACK_LABELS[key]}: waiting")
                widget.setStyleSheet(f"color: {MUTED};")

    def worker_finished(self):
        self.camera_ready = False
        self.start_button.setEnabled(False)
        if self.closing:
            self.close()

    def change_camera(self, *_):
        self.epoch += 1
        self.running = self.camera_ready = False
        self.start_button.setText("Start analysis")
        self.start_button.setEnabled(False)
        self.clear_scores("Waiting for the camera")
        self.progress.setValue(0)
        self.video.image = None
        self.video.message = "Opening camera..."
        self.video.update()
        if self.worker is None or not self.worker.isRunning():
            self.start_worker()
        else:
            self.send("camera", self.camera_combo.currentData(), self.epoch)

    def toggle_analysis(self):
        if not self.camera_ready:
            return
        self.running = not self.running
        self.epoch += 1
        self.send("analysis", self.running, self.epoch)
        self.start_button.setText("Pause analysis" if self.running else "Resume analysis")
        self.progress.setValue(0)
        if self.running:
            self.clear_scores("Collecting a fresh movement window")
            self.status_label.setText("Settling for 1 second, then collecting 4 seconds of movement...")
        else:
            self.mark_stale("Paused")
            self.status_label.setText("Analysis paused. Camera preview is still live.")

    def reset_session(self):
        self.running = False
        self.epoch += 1
        self.send("analysis", False, self.epoch)
        self.rows.clear()
        self.history.clear()
        self.session_id = str(uuid.uuid4())[:8]
        self.session_start = time.monotonic()
        self.session_label.setText(f"Session {self.session_id}")
        self.start_button.setText("Start analysis")
        self.export_button.setEnabled(False)
        self.clear_scores()
        self.refresh_history()
        self.progress.setValue(0)
        self.status_label.setText("Session cleared. Start analysis when you are ready.")

    def change_selection(self):
        if not hasattr(self, "score_rows"):
            return
        self.epoch += 1
        selected = self.symptom_combo.currentData()
        mode = self.mode_combo.currentData()
        self.send("select", mode, selected, self.epoch)
        for symptom, row in self.score_rows.items():
            row.setVisible(selected == "all" or selected == symptom)
        self.clear_scores("Collecting a fresh window" if self.running else "Waiting for analysis")
        if selected != "all":
            self.focus_symptom(selected)
        self.refresh_history()
        self.progress.setValue(0)
        self.status_label.setText("Selection changed. " + ("Collecting a fresh window..." if self.running else "Start analysis when ready."))

    def focus_symptom(self, symptom):
        index = self.history_combo.findData(symptom)
        self.history_combo.setCurrentIndex(index)
        self.refresh_history()

    def clear_scores(self, message="Waiting for analysis"):
        self.last_packet = None
        for row in self.score_rows.values():
            row.clear(message)

    def mark_stale(self, reason):
        for row in self.score_rows.values():
            row.show_result(stale_reason=reason)

    def on_prediction(self, packet):
        if packet["epoch"] != self.epoch or not self.running:
            return
        self.last_packet = packet
        elapsed = packet["end"] - self.session_start
        export_packet = dict(packet, start=packet["start"] - self.session_start,
                             end=packet["end"] - self.session_start)
        self.rows.extend(export_rows(export_packet, self.session_id, self.epoch, elapsed))
        self.history.append((elapsed, self.epoch, packet))
        for symptom, result in packet["results"].items():
            self.score_rows[symptom].show_result(result)
        self.export_button.setEnabled(bool(self.rows))
        self.refresh_history()

    def refresh_history(self, *_):
        if not hasattr(self, "plot"):
            return
        symptom, mode = self.history_combo.currentData(), self.mode_combo.currentData()
        self.plot.points.clear()
        self.plot.current_time = time.monotonic() - self.session_start
        if self.bank:
            self.plot.threshold = self.bank.bundles[(symptom, mode)]["threshold"]
        for elapsed, segment, packet in self.history:
            if packet["mode"] == mode and symptom in packet["results"]:
                self.plot.points.append((elapsed, packet["results"][symptom]["score"], segment))
        self.plot.update()

    def refresh_preview(self):
        if self.worker is None or not self.camera_ready:
            return
        packet = self.worker.snapshot()
        if packet is None:
            return
        age = time.monotonic() - packet["captured"]
        if packet["seq"] == self.seen_seq:
            if age > 3 and self.camera_ready:
                self.mark_stale("Camera frames are stale")
                self.status_label.setText("Waiting for fresh camera frames. Reconnect if this continues.")
            return
        self.seen_seq = packet["seq"]
        rgb = packet["rgb"]
        self.video.image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.video.update()
        self.fps_label.setText(f"{packet['fps']:.1f} processed fps")
        for key, widget in self.track_labels.items():
            good = packet["tracking"][key]
            widget.setText(f"{TRACK_LABELS[key]}: {'tracked' if good else 'missing'}")
            widget.setStyleSheet(f"color: {'#17664c' if good else MUTED};")
        if packet["epoch"] != self.epoch:
            return
        if self.running:
            self.progress.setValue(int(packet["progress"] * 100))
            if packet["interrupted"]:
                self.clear_scores("Tracking interrupted; collecting again")
            for symptom, row in self.score_rows.items():
                if row.result is None:
                    continue
                reason = tracking_reason(symptom, packet["tracking"], packet["fps"])
                if not reason and self.last_packet and packet["captured"] - self.last_packet["end"] > 3:
                    reason = "Waiting for a new score"
                row.show_result(stale_reason=reason)
            if self.last_packet:
                covered = self.last_packet["feature_coverage"]
                self.status_label.setText(f"Analyzing · updates every 2 seconds · {covered:.0%} feature coverage")
            else:
                self.status_label.setText(("Settling tracking for 1 second..." if packet["settling"] else
                                          "Collecting 4 seconds of movement...") +
                                          (" Low FPS may reduce tremor accuracy." if 0 < packet["fps"] < 20 else ""))
        self.refresh_history()

    def export(self):
        if not self.rows:
            return
        default = ROOT / f"webcam_scores_{datetime.now():%Y%m%d_%H%M%S}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export session scores", str(default), "CSV files (*.csv)")
        if path:
            if not path.lower().endswith(".csv"):
                path += ".csv"
            try:
                save_csv(path, self.rows)
            except OSError as error:
                QMessageBox.warning(self, "Export failed", f"Could not save scores. Choose a writable location.\n\n{error}")
            else:
                self.status_label.setText(f"Saved {len(self.rows)} score records to {Path(path).name}")

    def show_details(self):
        if not self.bank:
            return
        symptom, mode = self.history_combo.currentData(), self.mode_combo.currentData()
        bundle = self.bank.bundles[(symptom, mode)]
        metric = self.bank.metrics.get(symptom, {}).get(mode, {}).get("window", {})
        if self.details:
            self.details.close()
        self.details = QDialog(self)
        self.details.setWindowTitle(f"{PRETTY[symptom]} · Model details")
        self.details.resize(580, 480)
        layout = QVBoxLayout(self.details)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.addWidget(label(PRETTY[symptom], "sectionTitle"))
        layout.addWidget(label(MODE_LABELS[mode], "muted"))
        grid = QGridLayout()
        facts = [("Saved threshold", f"{bundle['threshold']:.0%}"), ("Input features", str(len(bundle["features"]))),
                 ("Movement window", "4 seconds; 2-second step"), ("Evaluation", "5-fold CV; held-out participants")]
        for key, title in [("balanced_accuracy", "Balanced accuracy"), ("f1", "F1"), ("roc_auc", "ROC AUC")]:
            value = metric.get(key)
            facts.append((title, "Unavailable" if value is None else f"{value:.3f}"))
        for i, (title, value) in enumerate(facts):
            grid.addWidget(label(title, "muted"), i, 0)
            grid.addWidget(label(value), i, 1)
        layout.addLayout(grid)
        notes = label("These evaluation numbers describe the existing dataset, not measured webcam accuracy.\n\n"
                      "The live adapter reproduces the dataset's observed coordinate conventions. Its exact parity with the original exporter is unverified.\n\n"
                      "Missing inputs are passed as NaN as in training; coverage is reported separately. Tracking gates suppress scores with missing key body areas. Below 20 processed fps, tremor scores remain visible with a sampling warning. Below 8 fps, all scores are withheld.\n\n"
                      "Inference stays on this computer. No video is recorded. CSV export saves scores and tracking quality.", "muted")
        notes.setWordWrap(True)
        layout.addWidget(notes)
        close = QPushButton("Close")
        close.clicked.connect(self.details.close)
        layout.addWidget(close)
        self.details.show()

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.closing = True
            self.start_button.setEnabled(False)
            self.reconnect_button.setEnabled(False)
            self.status_label.setText("Closing the camera and tracking workers...")
            self.worker.requestInterruption()
            event.ignore()
        else:
            event.accept()


STYLE = """
QMainWindow, QDialog { background: #f4f3ef; }
QWidget { color: #17232b; font-family: 'Segoe UI'; font-size: 13px; }
QLabel#title { font-size: 27px; font-weight: 650; }
QLabel#sectionTitle { font-size: 17px; font-weight: 600; }
QLabel#muted, QLabel#small { color: #53616b; }
QLabel#small { font-size: 11px; }
QLabel#scoreValue { font-size: 19px; font-weight: 600; }
QFrame#resultsPane { background: #ffffff; border: 1px solid #dfe3e5; border-radius: 10px; }
QFrame#scoreRow { border-bottom: 1px solid #e8ebed; }
QScrollArea, QScrollArea > QWidget > QWidget { background: #ffffff; }
QPushButton { background: #ffffff; border: 1px solid #c4ccd2; border-radius: 5px; padding: 7px 13px; min-height: 19px; }
QPushButton:hover { background: #e9eef7; border-color: #3156a3; }
QPushButton:pressed { background: #dbe4f4; }
QPushButton:focus, QComboBox:focus { border: 2px solid #3156a3; }
QPushButton:disabled { color: #6f7a84; background: #e8ecef; border-color: #cbd2d8; }
QPushButton#primary { background: #3156a3; color: #ffffff; border-color: #3156a3; font-weight: 600; }
QPushButton#primary:hover { background: #264687; }
QPushButton#primary:disabled { background: #d5dce5; border-color: #c4ccd2; color: #566273; }
QPushButton#rowName { border: none; border-radius: 3px; text-align: left; padding: 0; min-height: 20px; font-weight: 600; background: transparent; }
QPushButton#rowName:hover, QPushButton#rowName:focus { color: #3156a3; background: #e9eef7; }
QComboBox { background: #ffffff; border: 1px solid #c4ccd2; border-radius: 4px; padding: 5px 22px 5px 8px; min-height: 19px; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView { background: #ffffff; selection-background-color: #3156a3; selection-color: #ffffff; }
QCheckBox { spacing: 6px; }
QCheckBox::indicator { width: 15px; height: 15px; }
QProgressBar { background: #dfe4e9; border: none; border-radius: 3px; }
QProgressBar::chunk { background: #3156a3; border-radius: 3px; }
QSplitter::handle { background: #e1e4e5; width: 1px; }
QScrollBar:vertical { width: 8px; background: #f1f3f4; }
QScrollBar::handle:vertical { background: #b9c3cc; border-radius: 4px; min-height: 20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-camera", action="store_true", help="Open the UI without accessing camera hardware")
    parser.add_argument("--screenshot", type=Path, help="Save a UI-only screenshot and exit; does not open the camera")
    args = parser.parse_args()
    logging.basicConfig(filename=ROOT / "webcam_app.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    palette = app.palette()
    for role, color in [(QPalette.ColorRole.Window, "#f4f3ef"), (QPalette.ColorRole.Base, "#ffffff"),
                        (QPalette.ColorRole.WindowText, INK), (QPalette.ColorRole.Text, INK),
                        (QPalette.ColorRole.Button, "#ffffff"), (QPalette.ColorRole.ButtonText, INK),
                        (QPalette.ColorRole.Highlight, BLUE), (QPalette.ColorRole.HighlightedText, "#ffffff")]:
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    app.setStyleSheet(STYLE)
    app.setWindowIcon(QIcon(str(ROOT / "app_assets" / "symptom_lab.svg")))
    window = MainWindow(start_camera=not (args.no_camera or args.screenshot))
    window.show()
    if args.screenshot:
        def capture():
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(args.screenshot))
            app.quit()
        QTimer.singleShot(500, capture)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
