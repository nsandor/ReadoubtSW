from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from PySide6 import QtCore, QtWidgets, QtGui

from Drivers.stage_driver import (
    DummyRotationStage,
    DummyXYStage,
    NewportRotationStageAdapter,
    NewportXYStageAdapter,
)
from ScannerWin.acquisition import AcquisitionSettings


@dataclass
class StageGeometry:
    sensor_width_mm: float
    sensor_height_mm: float
    scan_width_mm: float
    scan_height_mm: float

    def columns(self) -> int:
        return max(1, math.ceil(self.scan_width_mm / self.sensor_width_mm))

    def rows(self) -> int:
        return max(1, math.ceil(self.scan_height_mm / self.sensor_height_mm))


class StageScanWorker(QtCore.QObject):
    sectionStarted = QtCore.Signal(int, int, float, float)
    sectionFinished = QtCore.Signal(int, int, float, float)
    rotationStarted = QtCore.Signal(int, float)
    rotationFinished = QtCore.Signal(int, float, object)
    progressChanged = QtCore.Signal(int)
    compositeUpdated = QtCore.Signal(int, object)
    etaUpdated = QtCore.Signal(float)
    manualMoveRequested = QtCore.Signal(int, int, float, float)
    error = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(
        self,
        *,
        stage: DummyXYStage,
        rotation_stage: DummyRotationStage,
        rotation_angles: Sequence[float],
        acquisition: AcquisitionSettings,
        geometry: StageGeometry,
        serpentine: bool,
        pixel_mask: np.ndarray,
        tile_row_slice: Optional[slice] = None,
        tile_col_slice: Optional[slice] = None,
        pixels: Sequence[int],
        inactive_channels: Sequence[int],
        switch,
        read_smu,
        bias_sm,
        manual_stage_move: bool = False,
    ) -> None:
        super().__init__()
        self._stage = stage
        self._rotation_stage = rotation_stage
        self._rotation_angles = [float(a) % 360.0 for a in rotation_angles] or [0.0]
        self._acq = acquisition
        self._geometry = geometry
        self._serpentine = serpentine
        self._pixel_mask = np.asarray(pixel_mask, dtype=bool)
        self._pixels = list(pixels)
        self._inactive = set(int(p) for p in inactive_channels)
        self._switch = switch
        self._sm = read_smu
        self._bias_sm = bias_sm
        self._rows = geometry.rows()
        self._cols = geometry.columns()
        self._step_x = geometry.sensor_width_mm
        self._step_y = geometry.sensor_height_mm
        (
            self._tile_row_slice,
            self._tile_col_slice,
        ) = self._resolve_tile_slices(
            self._pixel_mask, tile_row_slice, tile_col_slice
        )
        self._tile_rows = max(
            1, self._tile_row_slice.stop - self._tile_row_slice.start
        )
        self._tile_cols = max(
            1, self._tile_col_slice.stop - self._tile_col_slice.start
        )
        self._composite = np.full(
            (self._rows * self._tile_rows, self._cols * self._tile_cols), np.nan
        )
        self._stop = False
        self._start_time = 0.0
        self._manual_stage_move = bool(manual_stage_move)
        self._manual_wait = QtCore.QWaitCondition()
        self._manual_mutex = QtCore.QMutex()
        self._manual_ready = False

    @QtCore.Slot()
    def run(self) -> None:
        try:
            self._configure_instruments()
        except Exception as exc:
            self.error.emit(f"Instrument setup failed: {exc}")
            self.finished.emit()
            return

        total_sections = self._rows * self._cols * len(self._rotation_angles)
        processed = 0
        self._start_time = time.monotonic()
        try:
            for rot_idx, angle in enumerate(self._rotation_angles):
                if self._stop:
                    break
                try:
                    self._rotation_stage.move_to(angle)
                except Exception as exc:
                    raise RuntimeError(f"Rotation move failed: {exc}") from exc
                self.rotationStarted.emit(rot_idx, float(angle))
                try:
                    if not self._manual_stage_move:
                        self._stage.home()
                except Exception:
                    pass
                self._composite.fill(np.nan)
                for row in range(self._rows):
                    if self._stop:
                        break
                    col_iter: Sequence[int]
                    if self._serpentine and row % 2 == 1:
                        col_iter = reversed(range(self._cols))
                    else:
                        col_iter = range(self._cols)
                    for col in col_iter:
                        if self._stop:
                            break
                        x = col * self._step_x
                        y = row * self._step_y
                        if self._manual_stage_move:
                            if not self._await_manual_stage(row, col, x, y):
                                break
                        else:
                            try:
                                self._stage.move_to(x, y)
                            except Exception as exc:
                                raise RuntimeError(f"Stage move failed: {exc}") from exc
                        self.sectionStarted.emit(row, col, float(x), float(y))
                        frame = self._capture_tile()
                        self._insert_frame(row, col, frame)
                        processed += 1
                        self.sectionFinished.emit(row, col, float(x), float(y))
                        pct = int(processed * 100 / max(1, total_sections))
                        self.progressChanged.emit(pct)
                        self.compositeUpdated.emit(rot_idx, self._composite.copy())
                        self._update_eta(processed, total_sections)
                self.rotationFinished.emit(rot_idx, float(angle), self._composite.copy())
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self._teardown_instruments()
            self.finished.emit()

    def stop(self) -> None:
        self._stop = True
        if self._manual_stage_move:
            self._manual_mutex.lock()
            try:
                self._manual_wait.wakeAll()
            finally:
                self._manual_mutex.unlock()

    def _capture_tile(self) -> np.ndarray:
        loops = max(1, int(self._acq.loops))
        tile = np.zeros((10, 10), dtype=float)
        captures = 0
        for idx in range(loops):
            if self._stop:
                break
            frame = self._capture_frame_once()
            tile += frame
            captures += 1
            if (
                idx < loops - 1
                and self._acq.inter_loop_delay_s > 0
                and not self._stop
            ):
                self._sleep(self._acq.inter_loop_delay_s)
        if captures > 0:
            tile /= captures
        return tile

    def _capture_frame_once(self) -> np.ndarray:
        if self._acq.use_local_readout:
            return self._capture_local_frame()
        return self._capture_remote_frame()

    def _capture_local_frame(self) -> np.ndarray:
        samples = max(1, int(self._acq.samples_per_pixel))
        accum = None
        for _ in range(samples):
            if self._stop:
                break
            currents, _runtime = self._switch.measure_local()
            arr = np.asarray(currents, dtype=float)
            accum = arr if accum is None else accum + arr
        if accum is None:
            raise RuntimeError("Local measurement returned no data.")
        avg_na = accum / samples
        frame = avg_na.reshape((10, 10)) * 1e-9
        frame = np.where(self._pixel_mask, frame, np.nan)
        return frame

    def _capture_remote_frame(self) -> np.ndarray:
        samples = max(1, int(self._acq.samples_per_pixel))
        frame = np.full((10, 10), np.nan, dtype=float)
        for pixel in self._pixels:
            if self._stop:
                break
            if pixel in self._inactive:
                continue
            try:
                self._switch.route(pixel)
            except Exception as exc:
                raise RuntimeError(f"Switch route failed for pixel {pixel}: {exc}") from exc
            vals: List[float] = []
            for _ in range(samples):
                if self._stop:
                    break
                try:
                    vals.append(float(self._sm.current))
                except Exception as exc:
                    raise RuntimeError(f"Sourcemeter read failed: {exc}") from exc
            if not vals:
                continue
            r, c = divmod(pixel - 1, 10)
            frame[r, c] = float(np.mean(vals))
        frame = np.where(self._pixel_mask, frame, np.nan)
        return frame

    def _insert_frame(self, row: int, col: int, frame: np.ndarray) -> None:
        r0 = row * self._tile_rows
        c0 = col * self._tile_cols
        tile = frame[self._tile_row_slice, self._tile_col_slice]
        self._composite[r0 : r0 + self._tile_rows, c0 : c0 + self._tile_cols] = tile

    def _configure_instruments(self) -> None:
        if not self._acq.use_local_readout:
            self._sm.reset()
            self._sm.enable_source()
            try:
                self._sm.measure_current(
                    nplc=self._acq.nplc, auto_range=self._acq.auto_range
                )
            except TypeError:
                self._sm.measure_current(nplc=self._acq.nplc)
            if not self._acq.auto_range:
                try:
                    self._sm.current_range = self._acq.current_range
                except Exception:
                    pass
        if self._acq.constant_bias_voltage is not None:
            self._set_bias(self._acq.constant_bias_voltage)

    def _teardown_instruments(self) -> None:
        try:
            if not self._acq.use_local_readout:
                self._sm.disable_source()
        except Exception:
            pass
        if self._acq.constant_bias_voltage is not None and self._bias_sm:
            try:
                self._bias_sm.source_voltage = 0.0
                if hasattr(self._bias_sm, "disable_source"):
                    self._bias_sm.disable_source()
            except Exception:
                pass

    def _set_bias(self, voltage: float) -> None:
        if self._acq.use_local_bias:
            self._switch.set_local_voltage(voltage)
            return
        if not self._bias_sm:
            raise RuntimeError("No bias SMU configured for stage scan.")
        self._bias_sm.source_voltage = float(voltage)
        self._bias_sm.enable_source()
        try:
            _ = self._bias_sm.current
        except Exception:
            pass

    def _update_eta(self, processed: int, total: int) -> None:
        if processed <= 0 or total <= 0:
            return
        elapsed = max(0.0, time.monotonic() - self._start_time)
        if elapsed <= 0:
            return
        rate = processed / elapsed
        if rate <= 0:
            return
        remaining = max(0.0, (total - processed) / rate)
        self.etaUpdated.emit(remaining)

    @staticmethod
    def _sleep(seconds: float) -> None:
        QtCore.QThread.msleep(int(max(0.0, seconds) * 1000))

    @staticmethod
    def _resolve_tile_slices(
        mask: np.ndarray,
        row_slice: Optional[slice],
        col_slice: Optional[slice],
    ) -> Tuple[slice, slice]:
        default_row, default_col = StageScanWorker._mask_bounds(mask)
        resolved_row = StageScanWorker._normalize_slice(
            row_slice or default_row, mask.shape[0]
        )
        resolved_col = StageScanWorker._normalize_slice(
            col_slice or default_col, mask.shape[1]
        )
        return resolved_row, resolved_col

    @staticmethod
    def _mask_bounds(mask: np.ndarray) -> Tuple[slice, slice]:
        row_indices = np.where(np.any(mask, axis=1))[0]
        col_indices = np.where(np.any(mask, axis=0))[0]
        return (
            StageScanWorker._build_slice(row_indices, mask.shape[0]),
            StageScanWorker._build_slice(col_indices, mask.shape[1]),
        )

    @staticmethod
    def _build_slice(indices: np.ndarray, size: int) -> slice:
        if indices.size == 0:
            return slice(0, size)
        start = int(np.min(indices))
        stop = int(np.max(indices)) + 1
        start = max(0, min(size, start))
        stop = max(start + 1, min(size, stop))
        return slice(start, stop)

    @staticmethod
    def _normalize_slice(candidate: slice, size: int) -> slice:
        start = candidate.start if candidate.start is not None else 0
        stop = candidate.stop if candidate.stop is not None else size
        start = max(0, min(size, int(start)))
        stop = max(0, min(size, int(stop)))
        if stop <= start:
            return slice(0, size)
        return slice(start, stop)

    def _await_manual_stage(self, row: int, col: int, x: float, y: float) -> bool:
        self._manual_mutex.lock()
        try:
            self._manual_ready = False
            self.manualMoveRequested.emit(row, col, float(x), float(y))
            while not self._manual_ready and not self._stop:
                self._manual_wait.wait(self._manual_mutex)
            return self._manual_ready and not self._stop
        finally:
            self._manual_mutex.unlock()

    @QtCore.Slot()
    def acknowledge_manual_move(self) -> None:
        if not self._manual_stage_move:
            return
        self._manual_mutex.lock()
        try:
            self._manual_ready = True
            self._manual_wait.wakeAll()
        finally:
            self._manual_mutex.unlock()


class StageScanWindow(QtWidgets.QMainWindow):
    """Secondary window that orchestrates XY stage mosaics."""

    def __init__(self, main_window) -> None:
        super().__init__(parent=main_window)
        self._main = main_window
        self._stage_driver = getattr(main_window, "stage_driver", None)
        xy_stage = getattr(main_window, "xy_stage", None)
        rot_stage = getattr(main_window, "rotation_stage", None)
        if isinstance(xy_stage, (DummyXYStage, NewportXYStageAdapter)):
            self._stage = xy_stage
        elif self._stage_driver:
            self._stage = NewportXYStageAdapter(self._stage_driver)
        else:
            self._stage = DummyXYStage()
        if isinstance(rot_stage, (DummyRotationStage, NewportRotationStageAdapter)):
            self._rotation_stage = rot_stage
        elif self._stage_driver:
            self._rotation_stage = NewportRotationStageAdapter(self._stage_driver)
        else:
            self._rotation_stage = DummyRotationStage()
        if isinstance(self._stage, DummyXYStage):
            try:
                self._stage.connect()
            except Exception:
                pass
        if isinstance(self._rotation_stage, DummyRotationStage):
            try:
                self._rotation_stage.connect()
            except Exception:
                pass
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[StageScanWorker] = None
        self._composite = np.full((10, 10), np.nan)
        self._active_rows = 1
        self._active_cols = 1
        self._tile_rows = 10
        self._tile_cols = 10
        self._tile_row_slice = slice(0, 10)
        self._tile_col_slice = slice(0, 10)
        self._last_pixel_mask = np.ones((10, 10), dtype=bool)
        self._rotation_angles: List[float] = [0.0]
        self._rotation_results: List[Optional[np.ndarray]] = []
        self._view_rotation_index = 0
        self._current_rotation_index = 0
        self._manual_waiting_for_stage = False
        self._manual_pending_text = ""
        self._mirror_tile_x = False
        self._mirror_tile_y = False
        self._stage_autosave_enabled = bool(getattr(self._main, "autosave_enabled", True))
        self._stage_save_processed = True
        self._stage_output_folder = Path(getattr(self._main, "output_folder", Path.cwd()))
        self._stage_run_folder: Optional[Path] = None
        self._stage_data_dir: Optional[Path] = None
        self._stage_heatmap_dir: Optional[Path] = None
        self._active_scan_label = self._default_scan_label()
        self._active_geometry: Optional[StageGeometry] = None
        self._active_acquisition: Optional[AcquisitionSettings] = None
        self._build_ui()
        self._connect_main_heatmap_controls()
        self._update_manual_stage_controls()
        self._update_geometry_preview()
        self._stage_status_timer = QtCore.QTimer(self)
        self._stage_status_timer.setInterval(1500)
        self._stage_status_timer.timeout.connect(self._refresh_stage_status_labels)
        self._stage_status_timer.start()
        QtCore.QTimer.singleShot(0, self._refresh_stage_status_labels)

    # ------------------------------------------------------------------ UI setup
    def _build_ui(self) -> None:
        self.setWindowTitle("Stage Scan")
        self.resize(1200, 700)
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)

        # Left controls
        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls)
        controls_layout.setSpacing(12)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        controls_layout.addWidget(self._build_dimension_group())
        controls_layout.addWidget(self._build_stage_group())
        controls_layout.addWidget(self._build_tile_orientation_group())
        controls_layout.addWidget(self._build_output_group())

        self.status_label = QtWidgets.QLabel("Stage ready.")
        controls_layout.addWidget(self.status_label)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setValue(0)
        controls_layout.addWidget(self.progress)
        self.eta_label = QtWidgets.QLabel("ETA: --")
        controls_layout.addWidget(self.eta_label)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        controls_layout.addWidget(self.log, 1)

        controls_layout.addWidget(self._build_rotation_scrub_group())

        controls_layout.addStretch()
        layout.addWidget(controls, 0)

        # Right plot
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self._heatmap = self.ax.imshow(
            self._composite, cmap="inferno", interpolation="nearest"
        )
        self.cbar = self.figure.colorbar(
            self._heatmap, ax=self.ax, fraction=0.046, pad=0.04, label="Current (A)"
        )
        layout.addWidget(self.canvas, 1)

    def _build_dimension_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Scan Geometry")
        form = QtWidgets.QFormLayout(box)
        form.setLabelAlignment(QtCore.Qt.AlignLeft)

        self.sensor_width = self._create_length_spin(10.0)
        self.sensor_height = self._create_length_spin(10.0)
        self.scan_width = self._create_length_spin(40.0)
        self.scan_height = self._create_length_spin(40.0)

        form.addRow("Sensor width (mm):", self.sensor_width)
        form.addRow("Sensor height (mm):", self.sensor_height)
        form.addRow("Scan width (mm):", self.scan_width)
        form.addRow("Scan height (mm):", self.scan_height)

        self.serpentine = QtWidgets.QCheckBox("Serpentine path")
        self.serpentine.setChecked(True)
        form.addRow(self.serpentine)

        self.geometry_summary = QtWidgets.QLabel("")
        form.addRow("Layout:", self.geometry_summary)

        for spin in (
            self.sensor_width,
            self.sensor_height,
            self.scan_width,
            self.scan_height,
        ):
            spin.valueChanged.connect(self._update_geometry_preview)

        return box

    def _open_stage_controller(self) -> None:
        if hasattr(self._main, "_open_stage_controller_window"):
            self._main._open_stage_controller_window()

    def _refresh_stage_status_labels(self) -> None:
        driver = getattr(self._main, "stage_driver", None)
        if driver and driver.is_connected():
            xy_parts = []
            for axis in ("x", "y"):
                stage_name = driver.axis_assignment(axis) or "(unassigned)"
                zero = "zeroed" if driver.axis_zeroed(axis) else "zero not set"
                xy_parts.append(f"{axis.upper()} → {stage_name} ({zero})")
            xy_text = "XY Stage: " + ", ".join(xy_parts)
            theta_stage = driver.axis_assignment("theta") or "(unassigned)"
            theta_zero = "zeroed" if driver.axis_zeroed("theta") else "zero not set"
            rotation_text = f"Rotation Stage: {theta_stage} ({theta_zero})"
        else:
            xy_text = "XY Stage: controller disconnected"
            rotation_text = "Rotation Stage: controller disconnected"
            if driver is None:
                xy_text = "XY Stage: dummy stage"
                rotation_text = "Rotation Stage: dummy stage"
        self.stage_status_label.setText(xy_text)
        self.rotation_status_label.setText(rotation_text)

    def _build_stage_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Stage Control")
        layout = QtWidgets.QVBoxLayout(box)

        self.btn_open_stage_controller = QtWidgets.QPushButton("Open Stage Controller…")
        self.btn_open_stage_controller.clicked.connect(self._open_stage_controller)
        layout.addWidget(self.btn_open_stage_controller)

        self.stage_status_label = QtWidgets.QLabel("XY Stage: (not configured)")
        layout.addWidget(self.stage_status_label)
        self.rotation_status_label = QtWidgets.QLabel("Rotation Stage: (not configured)")
        layout.addWidget(self.rotation_status_label)

        self.btn_refresh_stage_status = QtWidgets.QPushButton("Refresh Stage Status")
        self.btn_refresh_stage_status.clicked.connect(self._refresh_stage_status_labels)
        layout.addWidget(self.btn_refresh_stage_status)

        self.btn_home_stage = QtWidgets.QPushButton("Home XY Stage")
        self.btn_home_stage.clicked.connect(self._home_stage)
        layout.addWidget(self.btn_home_stage)

        self.btn_start = QtWidgets.QPushButton("Start Stage Scan")
        self.btn_start.clicked.connect(self._start_stage_scan)
        layout.addWidget(self.btn_start)

        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_stage_scan)
        layout.addWidget(self.btn_stop)

        self.btn_home_rotation = QtWidgets.QPushButton("Home Rotation Stage")
        self.btn_home_rotation.clicked.connect(self._home_rotation)
        layout.addWidget(self.btn_home_rotation)

        rotation_row = QtWidgets.QHBoxLayout()
        rotation_row.addWidget(QtWidgets.QLabel("Rotation steps:"))
        self.rotation_steps_spin = QtWidgets.QSpinBox()
        self.rotation_steps_spin.setMinimum(0)
        self.rotation_steps_spin.setMaximum(360)
        self.rotation_steps_spin.setValue(0)
        rotation_row.addWidget(self.rotation_steps_spin, 1)
        rotation_widget = QtWidgets.QWidget()
        rotation_widget.setLayout(rotation_row)
        layout.addWidget(rotation_widget)

        self.check_manual_stage = QtWidgets.QCheckBox("Manual stage movement")
        self.check_manual_stage.setToolTip(
            "When enabled, the user moves the XY stage between tiles."
        )
        self.check_manual_stage.toggled.connect(self._update_manual_stage_controls)
        layout.addWidget(self.check_manual_stage)

        self.manual_prompt = QtWidgets.QLabel("")
        self.manual_prompt.setWordWrap(True)
        layout.addWidget(self.manual_prompt)

        self.btn_manual_next = QtWidgets.QPushButton("Capture next tile")
        self.btn_manual_next.clicked.connect(self._on_manual_next_clicked)
        layout.addWidget(self.btn_manual_next)

        self.manual_prompt.setVisible(False)
        self.btn_manual_next.setVisible(False)
        self.btn_manual_next.setEnabled(False)

        return box

    def _build_tile_orientation_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Tile Orientation")
        layout = QtWidgets.QVBoxLayout(box)
        self.check_mirror_x = QtWidgets.QCheckBox("Mirror tiles in X (left/right)")
        self.check_mirror_x.setToolTip("Flip each tile horizontally before stitching.")
        self.check_mirror_y = QtWidgets.QCheckBox("Mirror tiles in Y (up/down)")
        self.check_mirror_y.setToolTip("Flip each tile vertically before stitching.")
        layout.addWidget(self.check_mirror_x)
        layout.addWidget(self.check_mirror_y)
        self.check_mirror_x.toggled.connect(self._update_mirror_flags)
        self.check_mirror_y.toggled.connect(self._update_mirror_flags)
        self._update_mirror_flags()
        return box

    def _update_mirror_flags(self) -> None:
        self._mirror_tile_x = bool(
            getattr(self, "check_mirror_x", None)
            and self.check_mirror_x.isChecked()
        )
        self._mirror_tile_y = bool(
            getattr(self, "check_mirror_y", None)
            and self.check_mirror_y.isChecked()
        )
        if hasattr(self, "_heatmap"):
            self._update_heatmap()

    def _build_output_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Output")
        form = QtWidgets.QFormLayout(box)
        form.setLabelAlignment(QtCore.Qt.AlignLeft)

        self.check_stage_autosave = QtWidgets.QCheckBox("Autosave stage scan results")
        self.check_stage_autosave.setChecked(self._stage_autosave_enabled)
        self.check_stage_autosave.toggled.connect(self._update_output_controls)
        form.addRow(self.check_stage_autosave)

        folder_row = QtWidgets.QHBoxLayout()
        self.edit_stage_output = QtWidgets.QLineEdit(str(self._stage_output_folder))
        self.edit_stage_output.setPlaceholderText("Directory for stage runs")
        folder_row.addWidget(self.edit_stage_output, 1)
        self.btn_stage_output = QtWidgets.QPushButton("Browse…")
        self.btn_stage_output.clicked.connect(self._select_output_folder)
        folder_row.addWidget(self.btn_stage_output)
        folder_widget = QtWidgets.QWidget()
        folder_widget.setLayout(folder_row)
        form.addRow("Folder:", folder_widget)

        self.edit_stage_label = QtWidgets.QLineEdit(self._active_scan_label)
        form.addRow("Scan label:", self.edit_stage_label)

        self.check_stage_save_processed = QtWidgets.QCheckBox(
            "Save processed CSV (apply reference math)"
        )
        self.check_stage_save_processed.setChecked(self._stage_save_processed)
        self.check_stage_save_processed.toggled.connect(
            lambda checked: setattr(self, "_stage_save_processed", bool(checked))
        )
        form.addRow(self.check_stage_save_processed)

        self._output_group = box
        self._update_output_controls()
        return box

    def _select_output_folder(self) -> None:
        start_dir = str(self._stage_output_folder)
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Stage Output Folder", start_dir
        )
        if not folder:
            return
        path = Path(folder)
        self._stage_output_folder = path
        if hasattr(self, "edit_stage_output"):
            self.edit_stage_output.setText(str(path))

    def _update_output_controls(self) -> None:
        autosave = bool(
            getattr(self, "check_stage_autosave", None)
            and self.check_stage_autosave.isChecked()
        )
        running = self._worker is not None
        self._stage_autosave_enabled = autosave
        editable = bool(autosave and not running)
        for widget in (
            getattr(self, "edit_stage_output", None),
            getattr(self, "btn_stage_output", None),
            getattr(self, "edit_stage_label", None),
        ):
            if widget:
                widget.setEnabled(editable)
        if hasattr(self, "check_stage_save_processed"):
            self.check_stage_save_processed.setEnabled(not running)

    def _main_has_capability(self, name: str) -> bool:
        func = getattr(self._main, name, None)
        if callable(func):
            try:
                return bool(func())
            except Exception:
                return False
        return False

    def _dummy_stage_active(self) -> bool:
        flag = getattr(self._main, "dummy_stage_enabled", None)
        if callable(flag):
            try:
                return bool(flag())
            except Exception:
                return False
        return False

    def _manual_stage_enabled(self) -> bool:
        return bool(
            getattr(self, "check_manual_stage", None)
            and self.check_manual_stage.isChecked()
        )

    def _update_manual_stage_controls(self) -> None:
        if not hasattr(self, "manual_prompt"):
            return
        manual = self._manual_stage_enabled()
        running = self._worker is not None
        self.manual_prompt.setVisible(manual)
        self.btn_manual_next.setVisible(manual)
        if not manual:
            self.manual_prompt.setText("")
            self.btn_manual_next.setEnabled(False)
            return
        if not running:
            self.manual_prompt.setText(
                "Manual stage mode enabled. Start a scan to receive move prompts."
            )
            self.btn_manual_next.setEnabled(False)
            return
        if self._manual_waiting_for_stage:
            self.manual_prompt.setText(self._manual_pending_text)
            self.btn_manual_next.setEnabled(True)
        else:
            self.manual_prompt.setText("Capturing tile…")
            self.btn_manual_next.setEnabled(False)

    def _on_manual_next_clicked(self) -> None:
        if not self._worker:
            self.btn_manual_next.setEnabled(False)
            return
        self._manual_waiting_for_stage = False
        self.manual_prompt.setText("Capturing tile…")
        self.btn_manual_next.setEnabled(False)
        QtCore.QMetaObject.invokeMethod(
            self._worker,
            "acknowledge_manual_move",
            QtCore.Qt.QueuedConnection,
        )

    def _build_rotation_scrub_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Rotation Results")
        layout = QtWidgets.QVBoxLayout(box)
        self.rotation_info_label = QtWidgets.QLabel("No rotation data yet.")
        layout.addWidget(self.rotation_info_label)

        controls = QtWidgets.QHBoxLayout()
        self.rotation_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.rotation_slider.setMinimum(0)
        self.rotation_slider.setMaximum(0)
        self.rotation_slider.setEnabled(False)
        self.rotation_slider.valueChanged.connect(self._on_rotation_slider_changed)
        controls.addWidget(self.rotation_slider, 1)

        self.rotation_spin = QtWidgets.QSpinBox()
        self.rotation_spin.setMinimum(0)
        self.rotation_spin.setMaximum(0)
        self.rotation_spin.setEnabled(False)
        self.rotation_spin.valueChanged.connect(self._on_rotation_spin_changed)
        controls.addWidget(self.rotation_spin)
        layout.addLayout(controls)
        return box

    @staticmethod
    def _create_length_spin(default: float) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(2)
        spin.setMinimum(0.10)
        spin.setMaximum(1000.0)
        spin.setSingleStep(0.5)
        spin.setValue(default)
        return spin

    # ------------------------------------------------------------------ slots / helpers
    def _home_stage(self) -> None:
        if not self._ensure_stage_connected():
            return
        try:
            self._stage.home()
            self._append_log("Stage homed.")
            self._refresh_stage_status_labels()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Stage", f"Failed to home: {exc}")

    def _home_rotation(self) -> None:
        if not self._rotation_stage.is_connected():
            QtWidgets.QMessageBox.warning(
                self,
                "Rotation Stage",
                "Configure the rotation stage in the Stage Controller first.",
            )
            return
        try:
            self._rotation_stage.home()
            self._append_log("Rotation stage homed.")
            self._refresh_stage_status_labels()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Rotation Stage", f"Failed to home rotation stage: {exc}"
            )

    def _start_stage_scan(self) -> None:
        if self._worker is not None:
            return
        if getattr(self._main, "_thread", None):
            QtWidgets.QMessageBox.warning(
                self,
                "Readout Busy",
                "Stop the main acquisition before starting a stage scan.",
            )
            return
        try:
            acquisition = self._main.export_acquisition_settings()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Scan Settings", f"{exc}")
            return
        if acquisition.measurement_mode != "time":
            QtWidgets.QMessageBox.warning(
                self,
                "Unsupported Mode",
                "Stage scanning currently uses the time-mode acquisition path. "
                "Switch the Measurement Type to a time scan before launching.",
            )
            return
        geometry = self._current_geometry()
        rows, cols = geometry.rows(), geometry.columns()
        if rows * cols <= 0:
            QtWidgets.QMessageBox.critical(
                self, "Geometry", "Scan geometry produces no sections."
            )
            return

        rotation_steps = int(self.rotation_steps_spin.value())
        if not self._validate_stage_ready(require_rotation=True):
            return
        rotation_angles = self._build_rotation_angles(rotation_steps)
        self._rotation_angles = rotation_angles
        self._rotation_results = [None for _ in rotation_angles]
        self._view_rotation_index = 0
        self._current_rotation_index = 0
        self._manual_waiting_for_stage = False
        self._manual_pending_text = ""
        self._update_mirror_flags()
        label_text = (
            self.edit_stage_label.text().strip()
            if hasattr(self, "edit_stage_label")
            else ""
        )
        if not label_text:
            label_text = self._default_scan_label()
            if hasattr(self, "edit_stage_label"):
                self.edit_stage_label.setText(label_text)
        self._active_scan_label = label_text
        self._stage_autosave_enabled = bool(
            getattr(self, "check_stage_autosave", None)
            and self.check_stage_autosave.isChecked()
        )
        self._stage_save_processed = bool(
            getattr(self, "check_stage_save_processed", None)
            and self.check_stage_save_processed.isChecked()
        )

        self._active_geometry = geometry
        self._active_acquisition = acquisition

        pixel_mask = self._build_pixel_mask(acquisition.pixels)
        self._last_pixel_mask = np.array(pixel_mask, copy=True)
        (
            row_slice,
            col_slice,
        ) = self._pixel_slices_from_mask(self._last_pixel_mask)
        self._tile_row_slice = row_slice
        self._tile_col_slice = col_slice
        self._tile_rows = max(1, row_slice.stop - row_slice.start)
        self._tile_cols = max(1, col_slice.stop - col_slice.start)
        self._composite = np.full(
            (rows * self._tile_rows, cols * self._tile_cols), np.nan
        )
        self._active_rows, self._active_cols = rows, cols
        self._update_heatmap()
        self._set_rotation_scrub_enabled(False)
        self._update_eta_label(0.0)
        if self._stage_autosave_enabled:
            try:
                self._stage_output_folder = self._resolve_output_folder()
                run_folder = self._ensure_stage_run_folder(force_new=True)
                self._write_stage_metadata(
                    acquisition, geometry, rotation_angles, pixel_mask
                )
                self._append_log(
                    f"Autosave enabled – stage results will be saved to {run_folder}."
                )
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    self, "Output Folder", f"Failed to prepare output: {exc}"
                )
                self._worker = None
                return
        else:
            self._stage_run_folder = None
            self._stage_data_dir = None
            self._stage_heatmap_dir = None
        bias_device = (
            None if acquisition.use_local_bias else getattr(self._main, "bias_sm", None)
        )

        self._worker = StageScanWorker(
            stage=self._stage,
            rotation_stage=self._rotation_stage,
            rotation_angles=rotation_angles,
            acquisition=acquisition,
            geometry=geometry,
            serpentine=self.serpentine.isChecked(),
            pixel_mask=pixel_mask,
            tile_row_slice=row_slice,
            tile_col_slice=col_slice,
            pixels=acquisition.pixels,
            inactive_channels=self._main.inactive_channels or [],
            switch=self._main.switch,
            read_smu=self._main.sm,
            bias_sm=bias_device,
            manual_stage_move=self._manual_stage_enabled(),
        )
        self._thread = QtCore.QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.sectionStarted.connect(self._on_section_started)
        self._worker.sectionFinished.connect(self._on_section_finished)
        self._worker.rotationStarted.connect(self._on_rotation_started)
        self._worker.rotationFinished.connect(self._on_rotation_finished)
        self._worker.compositeUpdated.connect(self._handle_composite_update)
        self._worker.progressChanged.connect(self.progress.setValue)
        self._worker.etaUpdated.connect(self._update_eta_label)
        self._worker.error.connect(self._handle_worker_error)
        self._worker.finished.connect(self._on_worker_finished)
        if self._manual_stage_enabled():
            self._worker.manualMoveRequested.connect(self._on_manual_move_requested)

        self._set_busy_state(True)
        self._append_log(
            f"Starting stage scan ({rows}x{cols} sections, {rows*cols} total tiles)."
        )
        if self._manual_stage_enabled():
            self._append_log(
                "Manual stage movement enabled – waiting for user confirmation between tiles."
            )
        self._thread.start()

    def _stop_stage_scan(self) -> None:
        if self._worker:
            self._worker.stop()
        self._append_log("Stopping stage scan…")

    def _on_worker_finished(self) -> None:
        self._append_log("Stage scan finished.")
        self._set_busy_state(False)
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None
        all_done = bool(self._rotation_results) and all(
            isinstance(arr, np.ndarray) for arr in self._rotation_results
        )
        self._set_rotation_scrub_enabled(all_done)
        self._update_eta_label(0.0)
        self._manual_waiting_for_stage = False
        self._manual_pending_text = ""
        self._update_manual_stage_controls()

    def _handle_worker_error(self, message: str) -> None:
        self._append_log(f"Error: {message}")
        QtWidgets.QMessageBox.critical(self, "Stage Scan", message)

    def _on_manual_move_requested(self, row: int, col: int, x: float, y: float) -> None:
        self._manual_waiting_for_stage = True
        self._manual_pending_text = (
            f"Move stage to section ({row+1}, {col+1}) at X={x:.2f} mm, Y={y:.2f} mm, "
            "then click 'Capture next tile'."
        )
        self._append_log(
            f"Awaiting manual move to section ({row+1}, {col+1}) "
            f"(X={x:.2f} mm, Y={y:.2f} mm)."
        )
        self._update_manual_stage_controls()

    def _on_section_started(self, row: int, col: int, x: float, y: float) -> None:
        rot_total = max(1, len(self._rotation_angles))
        rot_idx = min(self._current_rotation_index, rot_total - 1)
        angle = self._rotation_angles[rot_idx] if self._rotation_angles else 0.0
        self.status_label.setText(
            f"Rotation {rot_idx + 1}/{rot_total} ({angle:.1f}°) – "
            f"section ({row+1}, {col+1}) at X={x:.2f} mm, Y={y:.2f} mm"
        )

    def _on_section_finished(self, row: int, col: int, _x: float, _y: float) -> None:
        rot_idx = min(self._current_rotation_index, max(0, len(self._rotation_angles) - 1))
        angle = self._rotation_angles[rot_idx] if self._rotation_angles else 0.0
        self._append_log(
            f"Completed rotation {rot_idx + 1} section ({row+1}, {col+1}) at {angle:.1f}°."
        )

    def _handle_composite_update(self, rotation_idx: int, array) -> None:
        if self.rotation_slider.isEnabled() and rotation_idx != self._view_rotation_index:
            return
        self._composite = np.array(array, copy=True)
        self._update_heatmap()

    def _on_rotation_started(self, index: int, angle: float) -> None:
        self._current_rotation_index = index
        if not self.rotation_slider.isEnabled():
            self._view_rotation_index = index
        self._update_rotation_label(angle, index, suffix="(scanning)")
        total = max(1, len(self._rotation_angles))
        self._append_log(
            f"Rotation {index + 1}/{total} at {angle:.1f}° started."
        )

    def _on_rotation_finished(self, index: int, angle: float, array) -> None:
        data = np.array(array, copy=True)
        if 0 <= index < len(self._rotation_results):
            self._rotation_results[index] = data
        self._save_rotation_result(index, angle, data)
        self._append_log(
            f"Rotation {index + 1}/{len(self._rotation_angles)} at {angle:.1f}° completed."
        )
        if not self.rotation_slider.isEnabled() and self._view_rotation_index == index:
            self._composite = data
            self._update_heatmap()
        self._update_rotation_label(angle, index)

    def _on_rotation_slider_changed(self, value: int) -> None:
        if value <= 0:
            return
        self.rotation_spin.blockSignals(True)
        self.rotation_spin.setValue(value)
        self.rotation_spin.blockSignals(False)
        self._apply_rotation_selection(value - 1)

    def _on_rotation_spin_changed(self, value: int) -> None:
        if value <= 0:
            return
        self.rotation_slider.blockSignals(True)
        self.rotation_slider.setValue(value)
        self.rotation_slider.blockSignals(False)
        self._apply_rotation_selection(value - 1)

    def _apply_rotation_selection(self, index: int) -> None:
        if not (0 <= index < len(self._rotation_angles)):
            return
        self._view_rotation_index = index
        angle = self._rotation_angles[index]
        data = self._rotation_results[index] if index < len(self._rotation_results) else None
        suffix = "(pending)" if data is None else ""
        self._update_rotation_label(angle, index, suffix=suffix)
        if data is not None:
            self._composite = np.array(data, copy=True)
            self._update_heatmap()

    def _set_rotation_scrub_enabled(self, enabled: bool) -> None:
        slider = self.rotation_slider
        spin = self.rotation_spin
        if not enabled:
            slider.blockSignals(True)
            spin.blockSignals(True)
            slider.setMinimum(0)
            slider.setMaximum(0)
            slider.setValue(0)
            spin.setMinimum(0)
            spin.setMaximum(0)
            spin.setValue(0)
            slider.blockSignals(False)
            spin.blockSignals(False)
            slider.setEnabled(False)
            spin.setEnabled(False)
            self.rotation_info_label.setText("Rotation results will appear after the scan.")
            return
        if not self._rotation_results:
            return
        slider.blockSignals(True)
        spin.blockSignals(True)
        slider.setMinimum(1)
        slider.setMaximum(len(self._rotation_results))
        target = min(len(self._rotation_results), max(1, self._view_rotation_index + 1))
        slider.setValue(target)
        spin.setMinimum(1)
        spin.setMaximum(len(self._rotation_results))
        spin.setValue(target)
        slider.blockSignals(False)
        spin.blockSignals(False)
        slider.setEnabled(True)
        spin.setEnabled(True)
        self._apply_rotation_selection(target - 1)

    def _update_rotation_label(self, angle: float, index: int, *, suffix: str = "") -> None:
        total = max(1, len(self._rotation_angles))
        text = f"Rotation {index + 1}/{total} – {angle:.1f}° {suffix}".strip()
        self.rotation_info_label.setText(text)

    def _update_heatmap(self) -> None:
        display, norm, cmap_name, label = self._prepare_display_payload(self._composite)
        self._heatmap.set_data(display)
        self._heatmap.set_cmap(cmap_name)
        self._heatmap.set_norm(norm)
        self.cbar.set_label(label)
        self.cbar.update_normal(self._heatmap)
        angle = 0.0
        if 0 <= self._view_rotation_index < len(self._rotation_angles):
            angle = self._rotation_angles[self._view_rotation_index]
        self.ax.set_title(
            f"Mosaic ({max(1, self._active_cols*self._tile_cols)}x"
            f"{max(1, self._active_rows*self._tile_rows)} pixels) – {angle:.1f}°"
        )
        self.canvas.draw_idle()

    def _set_busy_state(self, running: bool) -> None:
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.btn_open_stage_controller.setEnabled(not running)
        self.btn_refresh_stage_status.setEnabled(not running)
        self.btn_home_stage.setEnabled(not running)
        self.btn_home_rotation.setEnabled(not running)
        self.rotation_steps_spin.setEnabled(not running)
        if running:
            self._set_rotation_scrub_enabled(False)
        self.check_manual_stage.setEnabled(not running)
        self._update_manual_stage_controls()
        self._update_output_controls()

    def _build_pixel_mask(self, pixels: Sequence[int]) -> np.ndarray:
        mask = np.zeros((10, 10), dtype=bool)
        for idx in pixels:
            r, c = divmod(int(idx) - 1, 10)
            if 0 <= r < 10 and 0 <= c < 10:
                mask[r, c] = True
        for idx in getattr(self._main, "inactive_channels", []) or []:
            r, c = divmod(int(idx) - 1, 10)
            if 0 <= r < 10 and 0 <= c < 10:
                mask[r, c] = False
        return mask

    def _pixel_slices_from_mask(self, mask: np.ndarray) -> Tuple[slice, slice]:
        row_indices = np.where(np.any(mask, axis=1))[0]
        col_indices = np.where(np.any(mask, axis=0))[0]
        return (
            self._slice_from_indices(row_indices, mask.shape[0]),
            self._slice_from_indices(col_indices, mask.shape[1]),
        )

    @staticmethod
    def _slice_from_indices(indices: np.ndarray, size: int) -> slice:
        if indices.size == 0:
            return slice(0, size)
        start = int(np.min(indices))
        stop = int(np.max(indices)) + 1
        start = max(0, min(size, start))
        stop = max(start + 1, min(size, stop))
        return slice(start, stop)

    @staticmethod
    def _pixel_indices_from_mask(mask: np.ndarray) -> List[int]:
        rows, cols = mask.shape
        indices: List[int] = []
        for r in range(rows):
            for c in range(cols):
                if mask[r, c]:
                    indices.append(r * cols + c + 1)
        return indices

    def _prepare_display_payload(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, Normalize, str, str]:
        processed = self._apply_reference_math(data)
        processed = self._apply_tile_mirror(processed)
        units, scale = self._heatmap_unit_scale()
        display = processed * scale
        cmap_widget = self._ui_widget("combo_colormap")
        cmap_name = cmap_widget.currentText() if cmap_widget else "inferno"
        log_widget = self._ui_widget("check_log_scale_heatmap")
        use_log = bool(log_widget.isChecked()) if log_widget else False
        auto_widget = self._ui_widget("check_auto_scale")
        auto_scale = True if auto_widget is None else bool(auto_widget.isChecked())
        valid = display[np.isfinite(display)]
        vmin: float
        vmax: float
        if valid.size == 0:
            vmin, vmax = 1e-12, 1e-9
        elif auto_scale:
            vmin = float(np.nanmin(valid))
            vmax = float(np.nanmax(valid))
            if vmin == vmax:
                delta = max(abs(vmin) * 0.1, 1e-12)
                vmin -= delta
                vmax += delta
        else:
            try:
                vmin_widget = self._ui_widget("edit_vmin")
                vmax_widget = self._ui_widget("edit_vmax")
                vmin = float(vmin_widget.text()) if vmin_widget else 1e-12
                vmax = float(vmax_widget.text()) if vmax_widget else 1e-9
            except Exception:
                vmin, vmax = 1e-12, 1e-9
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = 1e-12, 1e-9
        if vmin == vmax:
            vmax = vmin + max(abs(vmin) * 0.1, 1e-12)
        label = f"Current ({units})"
        if use_log:
            vmin = max(vmin, 1e-12)
            vmax = max(vmax, vmin * 1.01)
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
        return display, norm, cmap_name, label

    def _apply_reference_math(self, data: np.ndarray) -> np.ndarray:
        result = np.array(data, copy=True)
        eps = float(getattr(self._main, "math_eps", 1e-12))
        for ref, mode in self._reference_sources():
            result = self._apply_single_reference(result, ref, mode, eps)
        return result

    def _reference_sources(self) -> List[Tuple[np.ndarray, str]]:
        sources: List[Tuple[np.ndarray, str]] = []
        ref1 = getattr(self._main, "ref_matrix", None)
        mode1 = getattr(self._main, "math_mode", "none")
        if ref1 is not None and mode1 != "none":
            sources.append((np.asarray(ref1), str(mode1)))
        ref2 = getattr(self._main, "ref_matrix2", None)
        mode2 = getattr(self._main, "math_mode2", "none")
        if ref2 is not None and mode2 != "none":
            sources.append((np.asarray(ref2), str(mode2)))
        return sources

    def _apply_single_reference(
        self, data: np.ndarray, ref_matrix: np.ndarray, mode: str, eps: float
    ) -> np.ndarray:
        tile_ref = ref_matrix[self._tile_row_slice, self._tile_col_slice]
        if tile_ref.size == 0:
            return data
        rows = max(1, math.ceil(data.shape[0] / tile_ref.shape[0]))
        cols = max(1, math.ceil(data.shape[1] / tile_ref.shape[1]))
        ref_mosaic = np.tile(tile_ref, (rows, cols))[: data.shape[0], : data.shape[1]]
        result = np.array(data, copy=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            if mode == "divide":
                denom = ref_mosaic + eps
                np.divide(result, denom, out=result, where=~np.isnan(data))
            else:
                result = result - ref_mosaic
        mask = np.isnan(data)
        result[mask] = np.nan
        return result

    def _apply_tile_mirror(
        self, data: np.ndarray, *, mirror_x: Optional[bool] = None, mirror_y: Optional[bool] = None
    ) -> np.ndarray:
        mx = self._mirror_tile_x if mirror_x is None else bool(mirror_x)
        my = self._mirror_tile_y if mirror_y is None else bool(mirror_y)
        if not mx and not my:
            return np.array(data, copy=True)
        arr = np.array(data, copy=True)
        tile_rows = max(1, self._tile_rows)
        tile_cols = max(1, self._tile_cols)
        total_rows = max(1, self._active_rows)
        total_cols = max(1, self._active_cols)
        for row in range(total_rows):
            r0 = row * tile_rows
            r1 = r0 + tile_rows
            if r1 > arr.shape[0]:
                break
            for col in range(total_cols):
                c0 = col * tile_cols
                c1 = c0 + tile_cols
                if c1 > arr.shape[1]:
                    break
                tile = arr[r0:r1, c0:c1]
                if mx:
                    tile = tile[:, ::-1]
                if my:
                    tile = tile[::-1, :]
                arr[r0:r1, c0:c1] = tile
        return arr

    def _heatmap_unit_scale(self) -> Tuple[str, float]:
        widget = self._ui_widget("combo_units")
        units = widget.currentText() if widget else "A"
        factors = {"pA": 1e12, "nA": 1e9, "\u00B5A": 1e6, "mA": 1e3}
        return units, factors.get(units, 1.0)

    def _ui_widget(self, name: str):
        ui = getattr(self._main, "ui", None)
        return getattr(ui, name, None) if ui else None

    def _default_scan_label(self) -> str:
        collector = getattr(self._main, "_collect_experiment_name", None)
        base = "scan"
        if callable(collector):
            try:
                base = collector() or "scan"
            except Exception:
                base = "scan"
        if not base.lower().endswith("stage"):
            return f"{base}_stage"
        return base

    def _sanitize_scan_label(self, text: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip())
        return slug or "stage_scan"

    def _resolve_output_folder(self) -> Path:
        text = (
            self.edit_stage_output.text().strip()
            if hasattr(self, "edit_stage_output")
            else ""
        )
        if text:
            return Path(text).expanduser()
        return self._stage_output_folder

    def _ensure_stage_run_folder(self, *, force_new: bool = False) -> Path:
        if (
            self._stage_run_folder
            and self._stage_run_folder.exists()
            and not force_new
        ):
            return self._stage_run_folder
        root = self._stage_output_folder
        root.mkdir(parents=True, exist_ok=True)
        slug = self._sanitize_scan_label(self._active_scan_label)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = root / f"{slug}_stage_{timestamp}"
        data_dir = run_folder / "data"
        heatmap_dir = run_folder / "heatmaps"
        for folder in (run_folder, data_dir, heatmap_dir):
            folder.mkdir(parents=True, exist_ok=True)
        self._stage_run_folder = run_folder
        self._stage_data_dir = data_dir
        self._stage_heatmap_dir = heatmap_dir
        logging.info("Prepared stage scan folder at %s", run_folder)
        return run_folder

    def _write_stage_metadata(
        self,
        acquisition: AcquisitionSettings,
        geometry: StageGeometry,
        rotation_angles: Sequence[float],
        pixel_mask: np.ndarray,
    ) -> None:
        if not self._stage_run_folder:
            return
        metadata = {
            "stage_scan": {
                "label": self._active_scan_label,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "serpentine": bool(self.serpentine.isChecked()),
                "manual_stage_move": bool(self._manual_stage_enabled()),
                "rotation_angles_deg": [float(a) for a in rotation_angles],
                "sections": {"rows": geometry.rows(), "cols": geometry.columns()},
                "geometry_mm": asdict(geometry),
                "tile_shape": [self._tile_rows, self._tile_cols],
                "tile_orientation": {
                    "mirror_x": bool(self._mirror_tile_x),
                    "mirror_y": bool(self._mirror_tile_y),
                },
                "pixel_indices": self._pixel_indices_from_mask(pixel_mask),
            },
            "acquisition": asdict(acquisition),
            "reference": {
                "primary": {
                    "path": str(getattr(self._main, "ref_path", "") or ""),
                    "mode": getattr(self._main, "math_mode", "none"),
                },
                "secondary": {
                    "path": str(getattr(self._main, "ref_path2", "") or ""),
                    "mode": getattr(self._main, "math_mode2", "none"),
                },
                "epsilon": float(getattr(self._main, "math_eps", 1e-12)),
                "save_processed": bool(self._stage_save_processed),
            },
            "devices": {
                "read_smu": getattr(self._main, "read_sm_idn", "Read SMU: (unknown)"),
                "bias_smu": getattr(self._main, "bias_sm_idn", "Bias SMU: (unknown)"),
                "switch": getattr(self._main, "switch_idn", "Switch: (unknown)"),
            },
            "output": {
                "root": str(self._stage_run_folder),
                "data": str(self._stage_data_dir or self._stage_run_folder),
                "heatmaps": str(self._stage_heatmap_dir or self._stage_run_folder),
            },
        }
        meta_path = self._stage_run_folder / "metadata.json"
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        logging.info("Wrote stage scan metadata to %s", meta_path)

    def _rotation_base_name(self, index: int, angle: float) -> str:
        slug = self._sanitize_scan_label(self._active_scan_label)
        angle_str = f"{angle:.1f}".replace("-", "m").replace(".", "p")
        return f"{slug}_rot{index + 1:03d}_{angle_str}"

    def _save_rotation_result(self, index: int, angle: float, data: np.ndarray) -> None:
        if not (
            self._stage_autosave_enabled
            and self._stage_data_dir
            and self._stage_run_folder
        ):
            return
        base = self._rotation_base_name(index, angle)
        metadata = self._build_rotation_metadata(index, angle)
        data_dir = self._stage_data_dir or self._stage_run_folder
        raw_path = data_dir / f"{base}_raw.csv"
        raw_arr = np.array(data, copy=True)
        raw_meta = dict(metadata)
        raw_meta["data_type"] = "raw"
        save_kwargs = dict(delimiter=",", fmt="%.6e")
        self._write_stage_csv(raw_path, raw_arr, raw_meta, save_kwargs, index, angle, "raw")
        if self._stage_save_processed:
            processed = self._apply_reference_math(data)
            processed = self._apply_tile_mirror(processed)
            proc_path = data_dir / f"{base}_processed.csv"
            proc_meta = dict(metadata)
            proc_meta["data_type"] = "processed"
            self._write_stage_csv(
                proc_path, processed, proc_meta, save_kwargs, index, angle, "processed"
            )
        try:
            self._save_rotation_heatmap(base, data, index, angle)
        except Exception as exc:
            logging.warning("Failed to save rotation %s heatmap: %s", index + 1, exc)

    def _build_rotation_metadata(self, index: int, angle: float) -> dict:
        return {
            "rotation_index": int(index),
            "rotation_angle_deg": float(angle),
            "label": self._active_scan_label,
            "tile_shape": [self._tile_rows, self._tile_cols],
            "tile_orientation": {
                "mirror_x": bool(self._mirror_tile_x),
                "mirror_y": bool(self._mirror_tile_y),
            },
            "sections": {"rows": self._active_rows, "cols": self._active_cols},
            "pixel_indices": self._pixel_indices_from_mask(self._last_pixel_mask),
            "references": {
                "primary": {
                    "path": str(getattr(self._main, "ref_path", "") or ""),
                    "mode": getattr(self._main, "math_mode", "none"),
                },
                "secondary": {
                    "path": str(getattr(self._main, "ref_path2", "") or ""),
                    "mode": getattr(self._main, "math_mode2", "none"),
                },
                "epsilon": float(getattr(self._main, "math_eps", 1e-12)),
            },
            "save_processed": bool(self._stage_save_processed),
        }

    def _write_stage_csv(
        self,
        path: Path,
        array: np.ndarray,
        metadata: dict,
        save_kwargs: dict,
        index: int,
        angle: float,
        mode: str,
    ) -> None:
        header = self._format_csv_metadata(metadata)
        kwargs = dict(save_kwargs)
        if header:
            kwargs["header"] = header
            kwargs["comments"] = "# "
        try:
            np.savetxt(path, array, **kwargs)
            self._append_log(
                f"Saved rotation {index + 1} ({angle:.1f}°) {mode} data to {path}"
            )
        except Exception as exc:
            logging.warning("Failed to save rotation %s %s CSV: %s", index + 1, mode, exc)

    def _save_rotation_heatmap(
        self, base_name: str, data: np.ndarray, index: int, angle: float
    ) -> None:
        if not self._stage_heatmap_dir:
            return
        path = self._stage_heatmap_dir / f"{base_name}_heatmap.png"
        display, norm, cmap_name, label = self._prepare_display_payload(data)
        fig = Figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        ax.set_xticks([])
        ax.set_yticks([])
        heat = ax.imshow(display, cmap=cmap_name, norm=norm, interpolation="nearest")
        fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04, label=label)
        ax.set_title(
            f"{self._active_scan_label} – rotation {index + 1} ({angle:.1f}°)"
        )
        fig.savefig(path, dpi=300, bbox_inches="tight")
        fig.clear()
        logging.info("Saved stage rotation heatmap to %s", path)

    @staticmethod
    def _format_csv_metadata(metadata: dict) -> str:
        try:
            payload = json.dumps(metadata, separators=(",", ":"), sort_keys=True)
            return f"READOUT_METADATA {payload}"
        except Exception as exc:
            logging.warning("Failed to encode stage CSV metadata: %s", exc)
            return ""

    def _connect_main_heatmap_controls(self) -> None:
        ui = getattr(self._main, "ui", None)
        if not ui:
            return
        mappings = [
            ("combo_units", "currentTextChanged"),
            ("combo_colormap", "currentTextChanged"),
            ("check_log_scale_heatmap", "toggled"),
            ("check_auto_scale", "toggled"),
            ("edit_vmin", "editingFinished"),
            ("edit_vmax", "editingFinished"),
            ("combo_math", "currentIndexChanged"),
            ("btn_load_ref", "clicked"),
        ]
        for attr, signal_name in mappings:
            widget = getattr(ui, attr, None)
            if not widget:
                continue
            signal = getattr(widget, signal_name, None)
            if not signal:
                continue
            signal.connect(self._update_heatmap)

    def _update_geometry_preview(self) -> None:
        geometry = self._current_geometry()
        rows, cols = geometry.rows(), geometry.columns()
        coverage_x = cols * geometry.sensor_width_mm
        coverage_y = rows * geometry.sensor_height_mm
        self.geometry_summary.setText(
            f"{rows} x {cols} sections  (covers {coverage_x:.1f} x {coverage_y:.1f} mm)"
        )

    def _current_geometry(self) -> StageGeometry:
        return StageGeometry(
            sensor_width_mm=float(self.sensor_width.value()),
            sensor_height_mm=float(self.sensor_height.value()),
            scan_width_mm=float(self.scan_width.value()),
            scan_height_mm=float(self.scan_height.value()),
        )

    def _build_rotation_angles(self, steps: int) -> List[float]:
        if steps <= 0:
            return [0.0]
        step_deg = 360.0 / max(1, steps)
        return [round(i * step_deg, 6) for i in range(steps)]

    def _append_log(self, text: str) -> None:
        self.log.appendPlainText(text)
        cursor = self.log.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()

    def _update_eta_label(self, seconds: float) -> None:
        if seconds <= 0:
            self.eta_label.setText("ETA: --")
            return
        self.eta_label.setText(f"ETA: {self._format_duration(seconds)}")

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total = int(round(seconds))
        minutes, sec = divmod(total, 60)
        hours, minutes = divmod(minutes, 60)
        parts = []
        if hours:
            parts.append(f"{hours}h")
        if minutes or hours:
            parts.append(f"{minutes}m")
        parts.append(f"{sec}s")
        return " ".join(parts)

    def _ensure_stage_connected(self) -> bool:
        if self._manual_stage_enabled():
            return True
        if self._stage.is_connected():
            return True
        QtWidgets.QMessageBox.warning(
            self,
            "Stage",
            "Configure the stage in the Stage Controller window before starting a scan.",
        )
        return False

    def _validate_stage_ready(self, require_rotation: bool) -> bool:
        if self._manual_stage_enabled():
            return True
        driver = getattr(self._main, "stage_driver", None)
        if driver:
            ok, message = driver.ready_for_scan(require_rotation=require_rotation)
            if ok:
                return True
            QtWidgets.QMessageBox.warning(self, "Stage", message)
            return False
        if not self._ensure_stage_connected():
            return False
        if require_rotation and not self._rotation_stage.is_connected():
            QtWidgets.QMessageBox.warning(
                self,
                "Rotation Stage",
                "Configure the rotation stage in the Stage Controller before scanning.",
            )
            return False
        return True

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._worker:
            self._stop_stage_scan()
            event.ignore()
            return
        super().closeEvent(event)
