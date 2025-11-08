from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from PySide6 import QtCore, QtWidgets, QtGui

from Drivers.stage_driver import DummyXYStage, DummyRotationStage
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
        pixels: Sequence[int],
        inactive_channels: Sequence[int],
        switch,
        read_smu,
        bias_sm,
    ) -> None:
        super().__init__()
        self._stage = stage
        self._rotation_stage = rotation_stage
        self._rotation_angles = [float(a) % 360.0 for a in rotation_angles] or [0.0]
        self._acq = acquisition
        self._geometry = geometry
        self._serpentine = serpentine
        self._pixel_mask = pixel_mask
        self._pixels = list(pixels)
        self._inactive = set(int(p) for p in inactive_channels)
        self._switch = switch
        self._sm = read_smu
        self._bias_sm = bias_sm
        self._rows = geometry.rows()
        self._cols = geometry.columns()
        self._step_x = geometry.sensor_width_mm
        self._step_y = geometry.sensor_height_mm
        self._composite = np.full((self._rows * 10, self._cols * 10), np.nan)
        self._stop = False
        self._start_time = 0.0

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
        r0 = row * 10
        c0 = col * 10
        self._composite[r0 : r0 + 10, c0 : c0 + 10] = frame

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


class StageScanWindow(QtWidgets.QMainWindow):
    """Secondary window that orchestrates XY stage mosaics."""

    def __init__(self, main_window) -> None:
        super().__init__(parent=main_window)
        self._main = main_window
        self._stage = DummyXYStage()
        self._rotation_stage = DummyRotationStage()
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[StageScanWorker] = None
        self._composite = np.full((10, 10), np.nan)
        self._active_rows = 1
        self._active_cols = 1
        self._rotation_angles: List[float] = [0.0]
        self._rotation_results: List[Optional[np.ndarray]] = []
        self._view_rotation_index = 0
        self._current_rotation_index = 0
        self._build_ui()
        self._update_geometry_preview()

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

    def _build_stage_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Stage Control")
        layout = QtWidgets.QVBoxLayout(box)

        self.btn_connect_stage = QtWidgets.QPushButton("Connect Dummy Stage")
        self.btn_connect_stage.clicked.connect(self._toggle_stage_connection)
        layout.addWidget(self.btn_connect_stage)

        self.btn_home_stage = QtWidgets.QPushButton("Home Stage")
        self.btn_home_stage.clicked.connect(self._home_stage)
        layout.addWidget(self.btn_home_stage)

        self.btn_start = QtWidgets.QPushButton("Start Stage Scan")
        self.btn_start.clicked.connect(self._start_stage_scan)
        layout.addWidget(self.btn_start)

        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_stage_scan)
        layout.addWidget(self.btn_stop)

        self.btn_connect_rotation = QtWidgets.QPushButton("Connect Rotation Stage")
        self.btn_connect_rotation.clicked.connect(self._toggle_rotation_connection)
        layout.addWidget(self.btn_connect_rotation)

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

        return box

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
    def _toggle_stage_connection(self) -> None:
        if self._stage.is_connected():
            self._stage.disconnect()
            self.btn_connect_stage.setText("Connect Dummy Stage")
            self._append_log("Stage disconnected.")
        else:
            self._stage.connect()
            self.btn_connect_stage.setText("Disconnect Stage")
            self._append_log("Stage connected.")

    def _home_stage(self) -> None:
        if not self._ensure_stage_connected():
            return
        try:
            self._stage.home()
            self._append_log("Stage homed.")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Stage", f"Failed to home: {exc}")

    def _toggle_rotation_connection(self) -> None:
        if self._rotation_stage.is_connected():
            self._rotation_stage.disconnect()
            self.btn_connect_rotation.setText("Connect Rotation Stage")
            self._append_log("Rotation stage disconnected.")
        else:
            self._rotation_stage.connect()
            self.btn_connect_rotation.setText("Disconnect Rotation Stage")
            self._append_log("Rotation stage connected.")

    def _home_rotation(self) -> None:
        if not self._rotation_stage.is_connected():
            QtWidgets.QMessageBox.warning(
                self, "Rotation Stage", "Connect the rotation stage first."
            )
            return
        try:
            self._rotation_stage.home()
            self._append_log("Rotation stage homed.")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Rotation Stage", f"Failed to home rotation stage: {exc}"
            )

    def _start_stage_scan(self) -> None:
        if self._worker is not None:
            return
        if not self._ensure_stage_connected():
            return
        if not self._rotation_stage.is_connected():
            QtWidgets.QMessageBox.warning(
                self, "Rotation Stage", "Connect the rotation stage before scanning."
            )
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
        rotation_angles = self._build_rotation_angles(rotation_steps)
        self._rotation_angles = rotation_angles
        self._rotation_results = [None for _ in rotation_angles]
        self._view_rotation_index = 0
        self._current_rotation_index = 0

        self._composite = np.full((rows * 10, cols * 10), np.nan)
        self._active_rows, self._active_cols = rows, cols
        self._update_heatmap()
        self._set_rotation_scrub_enabled(False)
        self._update_eta_label(0.0)

        pixel_mask = self._build_pixel_mask(acquisition.pixels)
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
            pixels=acquisition.pixels,
            inactive_channels=self._main.inactive_channels or [],
            switch=self._main.switch,
            read_smu=self._main.sm,
            bias_sm=bias_device,
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

        self._set_busy_state(True)
        self._append_log(
            f"Starting stage scan ({rows}x{cols} sections, {rows*cols} total tiles)."
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

    def _handle_worker_error(self, message: str) -> None:
        self._append_log(f"Error: {message}")
        QtWidgets.QMessageBox.critical(self, "Stage Scan", message)

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
        data = self._composite
        valid = data[np.isfinite(data)]
        if valid.size:
            vmin = float(np.nanmin(valid))
            vmax = float(np.nanmax(valid))
            if vmin == vmax:
                if vmin == 0.0:
                    vmin, vmax = -1e-12, 1e-12
                else:
                    vmin *= 0.9
                    vmax *= 1.1
        else:
            vmin, vmax = 1e-12, 1e-9
        self._heatmap.set_data(data)
        self._heatmap.set_norm(Normalize(vmin=vmin, vmax=vmax))
        angle = 0.0
        if 0 <= self._view_rotation_index < len(self._rotation_angles):
            angle = self._rotation_angles[self._view_rotation_index]
        self.ax.set_title(
            f"Mosaic ({self._active_cols*10}x{self._active_rows*10} pixels) – {angle:.1f}°"
        )
        self.canvas.draw_idle()

    def _set_busy_state(self, running: bool) -> None:
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.btn_connect_stage.setEnabled(not running)
        self.btn_home_stage.setEnabled(not running)
        self.btn_connect_rotation.setEnabled(not running)
        self.btn_home_rotation.setEnabled(not running)
        self.rotation_steps_spin.setEnabled(not running)
        if running:
            self._set_rotation_scrub_enabled(False)

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
        if self._stage.is_connected():
            return True
        QtWidgets.QMessageBox.warning(
            self, "Stage", "Connect the stage before starting a scan."
        )
        return False

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._worker:
            self._stop_stage_scan()
            event.ignore()
            return
        super().closeEvent(event)
