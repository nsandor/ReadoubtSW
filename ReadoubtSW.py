#!/usr/bin/env python3
# main.py — Readout app with PySide6 UI (from pyside6-uic)

import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Iterable, List, Optional

# ---------- Matplotlib (QtAgg backend works with PySide6) ----------
import matplotlib
import numpy as np
import pyvisa
from PySide6 import QtCore, QtWidgets, QtGui

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas  # works for PySide6 too
from matplotlib.colors import LogNorm, Normalize
from pymeasure.adapters import PrologixAdapter, VISAAdapter
from pymeasure.instruments.keithley import Keithley2400
from DevicePicker import DevicePicker
from Drivers.Keithley2400_drv import (
    DummyBias2400,
    DummyKeithley2400,
    ReadoutSafe2400,
)
from Drivers.Switchboard_drv import DummySwitchBoard, SwitchBoard
from Readoubt_ui import Ui_MainWindow
from Scanner import ScanWorker


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # ---- load UI ----
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Readoubt")

        # ---- paths/state ----
        self.output_folder: Path = Path.home()
        self._run_folder: Optional[Path] = None
        self._current_loop_tag: str = ""
        self.data = np.full((10, 10), np.nan)
        self.ref_matrix: Optional[np.ndarray] = None
        self.ref_path: Optional[Path] = None
        self.math_mode: str = "none"  # none|divide|subtract
        self.math_eps: float = 1e-12
        self.save_processed: bool = False
        self.inactive_channels: List[int] = []
        self._is_paused = False
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[ScanWorker] = None
        self.measurement_mode: str = "time"
        self._loop_label: str = "Loop"
        self._current_voltage: Optional[float] = None
        self._voltage_sequence: List[float] = []
        self._constant_bias_voltage: Optional[float] = None
        self._voltage_control_widgets: List[QtWidgets.QWidget] = []
        self._loop_control_widgets: List[QtWidgets.QWidget] = []
        self._time_bias_widgets: List[QtWidgets.QWidget] = []

        # ---- instruments (dummy by default) ----
        self.sm = DummyKeithley2400()
        self.bias_sm = DummyBias2400()
        self.switch = DummySwitchBoard()
        self.read_sm_idn = "Read SMU: (not connected)"
        self.bias_sm_idn = "Bias SMU: (not connected)"  # stub for future
        self.switch_idn = "Switch: (not connected)"

        # ---- matplotlib canvases into layouts ----
        self.figure_heatmap = plt.figure()
        self.canvas_heatmap = FigureCanvas(self.figure_heatmap)
        self.ax_heatmap = self.figure_heatmap.add_subplot(111)
        self.im = self.ax_heatmap.imshow(
            np.full((10, 10), np.nan),
            cmap="inferno",
            norm=LogNorm(vmin=1e-10, vmax=1e-7),
        )
        self.ax_heatmap.set_xticks([])
        self.ax_heatmap.set_yticks([])
        self.cbar = self.figure_heatmap.colorbar(
            self.im, ax=self.ax_heatmap, fraction=0.046, pad=0.04, label="Current (A)"
        )
        self.figure_heatmap.tight_layout(pad=2.5)
        self.ui.layout_canvas_heatmap.addWidget(self.canvas_heatmap)

        self.figure_hist = plt.figure()
        self.canvas_hist = FigureCanvas(self.figure_hist)
        self.ax_hist = self.figure_hist.add_subplot(111)
        self.ax_hist.set_xlabel("Current (A)")
        self.ax_hist.set_ylabel("Pixel Count")
        self.ax_hist.grid(True, linestyle="--", alpha=0.6)
        self.figure_hist.tight_layout(pad=2.5)
        self.ui.layout_canvas_hist.addWidget(self.canvas_hist)

        # ---- default values reflecting UI ----
        self.ui.edit_output_folder.setText(str(self.output_folder))
        self.ui.edit_pixel_spec.setText("1-100")
        self.ui.spin_loops.setValue(1)
        self.ui.spin_nplc.setValue(10.0)  # your UI default
        self.ui.spin_nsamp.setValue(1)
        self.ui.loop_delay.setText("1")
        self._update_integration_time_label()
        self._loop_control_widgets = [
            self.ui.label_loops,
            self.ui.spin_loops,
            self.ui.loop_delay_label,
            self.ui.loop_delay,
        ]

        # ---- time-mode bias controls ----
        self.ui.check_bias_enable = QtWidgets.QCheckBox(
            "Apply bias during time measurement"
        )
        self.ui.check_bias_enable.setObjectName("check_bias_enable")
        bias_insert_row = 4
        self.ui.formLayout_2.insertRow(bias_insert_row, self.ui.check_bias_enable)

        self.ui.label_bias_voltage = QtWidgets.QLabel("Bias level:")
        self.ui.label_bias_voltage.setObjectName("label_bias_voltage")
        self.ui.spin_bias_voltage = QtWidgets.QDoubleSpinBox()
        self.ui.spin_bias_voltage.setObjectName("spin_bias_voltage")
        self.ui.spin_bias_voltage.setDecimals(3)
        self.ui.spin_bias_voltage.setMinimum(-200.0)
        self.ui.spin_bias_voltage.setMaximum(200.0)
        self.ui.spin_bias_voltage.setSingleStep(0.1)
        self.ui.spin_bias_voltage.setValue(0.0)

        self.ui.combo_bias_units = QtWidgets.QComboBox()
        self.ui.combo_bias_units.setObjectName("combo_bias_units")
        self.ui.combo_bias_units.addItems(["V", "V/cm"])

        bias_value_widget = QtWidgets.QWidget()
        self._bias_value_widget = bias_value_widget
        bias_value_layout = QtWidgets.QHBoxLayout(bias_value_widget)
        bias_value_layout.setContentsMargins(0, 0, 0, 0)
        bias_value_layout.setSpacing(6)
        bias_value_layout.addWidget(self.ui.spin_bias_voltage)
        bias_value_layout.addWidget(self.ui.combo_bias_units)

        self.ui.formLayout_2.insertRow(
            bias_insert_row + 1,
            self.ui.label_bias_voltage,
            bias_value_widget,
        )

        self.ui.label_bias_thickness = QtWidgets.QLabel("Sample thickness (cm):")
        self.ui.label_bias_thickness.setObjectName("label_bias_thickness")
        self.ui.spin_bias_thickness = QtWidgets.QDoubleSpinBox()
        self.ui.spin_bias_thickness.setObjectName("spin_bias_thickness")
        self.ui.spin_bias_thickness.setDecimals(3)
        self.ui.spin_bias_thickness.setMinimum(0.001)
        self.ui.spin_bias_thickness.setMaximum(1000.0)
        self.ui.spin_bias_thickness.setSingleStep(0.01)
        self.ui.spin_bias_thickness.setValue(1.0)

        self.ui.formLayout_2.insertRow(
            bias_insert_row + 2,
            self.ui.label_bias_thickness,
            self.ui.spin_bias_thickness,
        )

        self._time_bias_widgets = [
            self.ui.check_bias_enable,
            self.ui.label_bias_voltage,
            self._bias_value_widget,
            self.ui.label_bias_thickness,
            self.ui.spin_bias_thickness,
        ]
        self._voltage_control_widgets = [
            self.ui.label_voltage_start,
            self.ui.spin_voltage_start,
            self.ui.label_voltage_end,
            self.ui.spin_voltage_end,
            self.ui.label_voltage_step,
            self.ui.spin_voltage_step,
            self.ui.label_voltage_settle,
            self.ui.spin_voltage_settle,
        ]

        # ---- signal wiring ----
        self.ui.btn_run_abort.clicked.connect(self.on_run_abort_clicked)
        self.ui.btn_pause_resume.clicked.connect(self.on_pause_resume_clicked)

        self.ui.btn_browse_folder.clicked.connect(self._select_output_folder)
        self.ui.btn_export_heatmap.clicked.connect(self._export_heatmap)
        self.ui.btn_export_hist.clicked.connect(self._export_histogram)

        self.ui.btn_load_ref.clicked.connect(self._load_reference_csv)
        self.ui.combo_math.currentIndexChanged.connect(self._math_mode_changed)
        self.ui.check_save_processed.toggled.connect(
            lambda b: setattr(self, "save_processed", bool(b))
        )

        self.ui.plot_selector.currentIndexChanged.connect(self._on_plot_selected)
        self.ui.check_auto_scale.toggled.connect(self._update_plot_controls_state)
        for w in (self.ui.edit_heatmap_title, self.ui.edit_vmin, self.ui.edit_vmax):
            w.editingFinished.connect(self._update_plots)
        self.ui.combo_colormap.currentIndexChanged.connect(self._update_plots)
        self.ui.combo_units.currentIndexChanged.connect(self._update_plots)
        self.ui.check_log_scale_heatmap.toggled.connect(self._update_plots)
        self.ui.check_show_values.toggled.connect(self._update_plots)

        self.ui.spin_nplc.valueChanged.connect(self._update_integration_time_label)
        self.ui.spin_nsamp.valueChanged.connect(self._update_integration_time_label)
        self.ui.loop_delay.editingFinished.connect(self._update_integration_time_label)
        self.ui.Measurement_type_combobox.currentIndexChanged.connect(
            self._on_measurement_type_changed
        )
        self.ui.check_bias_enable.toggled.connect(self._update_bias_controls_state)
        self.ui.combo_bias_units.currentIndexChanged.connect(
            self._update_bias_controls_state
        )

        # Menu actions
        self.ui.actionConnect_SMU.triggered.connect(self._connect_read_smu)
        self.ui.actionConnect_Bias_SMU.triggered.connect(
            self._connect_bias_smu
        )  # optional; not used in scan
        self.ui.actionConnect_SwitchBoard.triggered.connect(self._connect_switch)

        # Initialize plot settings panel selection tabs already exist in UI
        self._update_plot_controls_state()
        self._update_plots(reset=True)
        self._update_statusbar(text="Ready")
        self._update_bias_controls_state()
        self._on_measurement_type_changed(
            self.ui.Measurement_type_combobox.currentIndex()
        )

    # ---------------------- convenience ----------------------
    def _update_statusbar(self, text: str):
        self.statusBar().showMessage(text, 4000)

    def _bias_smu_connected(self) -> bool:
        return not isinstance(self.bias_sm, DummyBias2400)

    def _update_bias_controls_state(self):
        show_bias = self.measurement_mode != "voltage"
        for widget in self._time_bias_widgets:
            widget.setVisible(show_bias)

        bias_available = self._bias_smu_connected()
        checkbox_enabled = bias_available and show_bias
        if not bias_available and self.ui.check_bias_enable.isChecked():
            self.ui.check_bias_enable.blockSignals(True)
            self.ui.check_bias_enable.setChecked(False)
            self.ui.check_bias_enable.blockSignals(False)
        self.ui.check_bias_enable.setEnabled(checkbox_enabled)

        base_enabled = bias_available and show_bias
        apply_bias = base_enabled and self.ui.check_bias_enable.isChecked()

        self.ui.label_bias_voltage.setEnabled(base_enabled)
        self._bias_value_widget.setEnabled(base_enabled)
        self.ui.spin_bias_voltage.setEnabled(base_enabled)
        self.ui.combo_bias_units.setEnabled(base_enabled)

        show_thickness = (
            show_bias
            and self.ui.combo_bias_units.currentText().strip().lower().startswith("v/")
        )
        thickness_enabled = base_enabled and show_thickness
        self.ui.label_bias_thickness.setVisible(show_thickness)
        self.ui.spin_bias_thickness.setVisible(show_thickness)
        self.ui.label_bias_thickness.setEnabled(thickness_enabled)
        self.ui.spin_bias_thickness.setEnabled(thickness_enabled)

    def _on_measurement_type_changed(self, index: int):
        text = (
            self.ui.Measurement_type_combobox.itemText(index) or ""
        ).lower()
        show_voltage = "voltage" in text
        self.measurement_mode = "voltage" if show_voltage else "time"
        self._loop_label = "Voltage Step" if show_voltage else "Loop"
        for w in self._loop_control_widgets:
            w.setVisible(not show_voltage)
        for w in self._voltage_control_widgets:
            w.setVisible(show_voltage)
        self._update_bias_controls_state()
        if show_voltage:
            self._current_voltage = None
            if self.ui.spin_loops.value() != 1:
                self.ui.spin_loops.setValue(1)
        self._update_statusbar(
            "Voltage sweep mode ready" if show_voltage else "Time-mode scan"
        )

    @staticmethod
    def _voltage_display(voltage: float) -> str:
        return f"{voltage:+.3f} V"

    @staticmethod
    def _voltage_file_tag(voltage: float) -> str:
        return (
            f"{voltage:+.3f}V".replace("+", "p").replace("-", "m").replace(".", "p")
        )

    @staticmethod
    def _generate_voltage_steps(start: float, end: float, step: float) -> List[float]:
        if step <= 0:
            raise ValueError("Step size must be positive.")
        values: List[float] = []
        eps = max(abs(step) * 1e-6, 1e-9)
        if start <= end:
            val = start
            while val <= end + eps:
                values.append(round(val, 6))
                val += step
        else:
            val = start
            step = -abs(step)
            while val >= end - eps:
                values.append(round(val, 6))
                val += step
        if not values:
            raise ValueError("Voltage sweep produced no steps.")
        if len(values) > 2000:
            raise ValueError("Voltage sweep would produce more than 2000 steps.")
        return values

    def _collect_voltage_sweep_settings(self) -> tuple[List[float], float]:
        start = float(self.ui.spin_voltage_start.value())
        end = float(self.ui.spin_voltage_end.value())
        step = float(self.ui.spin_voltage_step.value())
        settle = float(self.ui.spin_voltage_settle.value())
        voltages = self._generate_voltage_steps(start, end, step)
        return voltages, max(0.0, settle)

    def _collect_time_bias_voltage(self) -> Optional[float]:
        if not getattr(self.ui, "check_bias_enable", None):
            return None
        if not self.ui.check_bias_enable.isChecked():
            return None
        if not self._bias_smu_connected():
            raise ValueError(
                "Connect a bias sourcemeter before enabling a bias voltage."
            )
        value = float(self.ui.spin_bias_voltage.value())
        units = (self.ui.combo_bias_units.currentText() or "V").strip().lower()
        if "cm" in units:
            thickness = float(self.ui.spin_bias_thickness.value())
            if thickness <= 0:
                raise ValueError("Sample thickness must be positive (cm).")
            value *= thickness
        return round(value, 6)

    # ---------------------- selection/parse ----------------------
    @staticmethod
    def _parse_pixel_spec(spec: str) -> List[int]:
        indices: set[int] = set()
        s = (spec or "").replace(" ", "").strip()
        if not s:
            return list(range(1, 101))
        for tok in s.split(","):
            if not tok:
                continue
            if "-" in tok:
                a_str, b_str = tok.split("-", 1)
                a, b = int(a_str), int(b_str)
                step = 1 if b >= a else -1
                for k in range(a, b + step, step):
                    if 1 <= k <= 100:
                        indices.add(k)
            else:
                k = int(tok)
                if 1 <= k <= 100:
                    indices.add(k)
        if not indices:
            raise ValueError("No valid pixel indices in selection")
        return sorted(indices)

    # ---------------------- run control ----------------------
    def on_run_abort_clicked(self):
        if self._thread is not None:
            self._abort_scan()
        else:
            self._start_scan()

    def on_pause_resume_clicked(self):
        if not self._worker:
            return
        self._is_paused = not self._is_paused
        if self._is_paused:
            self._worker.pause()
            self.ui.btn_pause_resume.setText("Resume")
        else:
            self._worker.resume()
            self.ui.btn_pause_resume.setText("Pause")

    def _ensure_run_folder(self) -> Path:
        if not self.output_folder or not Path(self.output_folder).is_dir():
            raise RuntimeError("Output folder is invalid. Choose a valid folder.")
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        exp_name = (self.ui.edit_exp_name.text().strip() or "Run").replace(" ", "_")
        self._run_folder = Path(self.output_folder) / f"{exp_name}_{timestamp}"
        self._run_folder.mkdir(parents=True, exist_ok=True)
        return self._run_folder

    def _start_scan(self):
        try:
            run_folder = self._ensure_run_folder()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Output Folder", str(e))
            return
        try:
            pixels = self._parse_pixel_spec(self.ui.edit_pixel_spec.text())
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Pixel Selection", f"{e}")
            return

        auto_rng = self.ui.check_auto_current_range.isChecked()
        try:
            cur_rng = float(self.ui.edit_current_range.text())
            if cur_rng <= 0:
                raise ValueError
        except Exception:
            if auto_rng:
                cur_rng = 1e-7
            else:
                QtWidgets.QMessageBox.critical(
                    self, "Current Range", "Manual range must be positive (A)."
                )
                return

        # reset data and UI
        self.data.fill(np.nan)
        self._update_plots(reset=True)
        nplc = self.ui.spin_nplc.value()
        nsamp = self.ui.spin_nsamp.value()

        measurement_text = (
            self.ui.Measurement_type_combobox.currentText() or ""
        ).lower()
        voltage_mode = "voltage" in measurement_text
        self.measurement_mode = "voltage" if voltage_mode else "time"
        self._loop_label = "Voltage Step" if voltage_mode else "Loop"
        self._current_voltage = None
        self._constant_bias_voltage = None

        voltage_steps = None
        settle_time = 0.0
        constant_bias_voltage: Optional[float] = None
        if voltage_mode:
            if isinstance(self.bias_sm, DummyBias2400):
                QtWidgets.QMessageBox.critical(
                    self,
                    "Bias SMU",
                    "Connect a bias sourcemeter before running a voltage sweep.",
                )
                return
            try:
                voltage_steps, settle_time = self._collect_voltage_sweep_settings()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Voltage Sweep", f"{e}")
                return
            loops = len(voltage_steps)
            inter_loop_delay = 0.0
            self._voltage_sequence = voltage_steps
        else:
            try:
                constant_bias_voltage = self._collect_time_bias_voltage()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Bias Voltage", f"{e}")
                return
            loops = self.ui.spin_loops.value()
            try:
                inter_loop_delay = float(self.ui.loop_delay.text())
            except Exception:
                inter_loop_delay = 0.0
            self._voltage_sequence = []
            self._constant_bias_voltage = constant_bias_voltage

        if loops <= 0:
            QtWidgets.QMessageBox.critical(
                self, "Scan", "At least one iteration is required."
            )
            return

        # worker
        bias_device = (
            self.bias_sm
            if voltage_mode or constant_bias_voltage is not None
            else None
        )

        self._worker = ScanWorker(
            self.sm,
            self.switch,
            n_samples=nsamp,
            nplc=nplc,
            pixel_indices=pixels,
            loops=loops,
            auto_range=auto_rng,
            current_range=cur_rng,
            inter_sample_delay_s=0,
            inter_loop_delay_s=inter_loop_delay,
            bias_sm=bias_device,
            voltage_steps=voltage_steps if voltage_mode else None,
            voltage_settle_s=settle_time if voltage_mode else 0.0,
            constant_bias_voltage=constant_bias_voltage if not voltage_mode else None,
        )
        self._thread = QtCore.QThread()
        self._worker.moveToThread(self._thread)

        # signals
        self._worker.pixelDone.connect(self._on_pixel)
        self._worker.loopStarted.connect(self._on_loop_started)
        self._worker.loopFinished.connect(self._on_loop_finished)
        self._worker.deviceError.connect(self._handle_device_error)
        self._worker.finished.connect(self._scan_finished)
        self._thread.started.connect(self._worker.run)

        # UI state
        self.ui.btn_run_abort.setText("Abort")
        self.ui.btn_pause_resume.setText("Pause")
        self.ui.btn_pause_resume.setEnabled(True)
        self._is_paused = False
        self._start_time = time.time()
        self._total_steps = loops * len(pixels)
        self._done_steps = 0
        self.ui.ScanprogressBar.setValue(0)
        self.ui.ScanTimers.setText(
            f"{self._loop_label} #:    Time Elapsed:    Predicted remaining time:"
        )
        summary = (
            f"{self._loop_label}s: {loops} × {len(pixels)} pixels"
        )
        self._update_statusbar(text=f"Running… {summary} → {run_folder}")
        self._thread.start()

    def _abort_scan(self):
        if self._worker:
            self.ui.btn_run_abort.setText("Aborting…")
            self.ui.btn_run_abort.setEnabled(False)
            self._worker.stop()

    # ---------------------- worker callbacks ----------------------
    def _on_loop_started(self, loop_idx: int, metadata: Optional[dict] = None):
        self.data.fill(np.nan)
        voltage = None
        if metadata and isinstance(metadata, dict):
            voltage = metadata.get("voltage")
        self._current_voltage = float(voltage) if voltage is not None else None
        if self._current_voltage is not None:
            self._current_loop_tag = self._voltage_file_tag(self._current_voltage)
        else:
            self._current_loop_tag = uuid.uuid4().hex[:6].upper()
        self._update_plots(reset=True)
        if self._current_voltage is not None:
            display = self._voltage_display(self._current_voltage)
            if self.measurement_mode == "voltage":
                total = len(self._voltage_sequence) or "?"
                msg = f"Voltage step {loop_idx}/{total}: {display}"
            else:
                msg = f"Loop {loop_idx} started – bias {display}"
            self._update_statusbar(msg)
        else:
            self._update_statusbar(text=f"Loop {loop_idx} started…")
        self._current_loop = loop_idx

    def _on_loop_finished(self, loop_idx: int, metadata: Optional[dict] = None):
        # save per-loop CSV (raw or processed according to checkbox)
        try:
            if not self._run_folder:
                self._ensure_run_folder()
            voltage = None
            if metadata and isinstance(metadata, dict):
                voltage = metadata.get("voltage")
            if self._current_voltage is not None and voltage is None:
                voltage = self._current_voltage
            if voltage is not None:
                tag = self._voltage_file_tag(float(voltage))
                out_name = f"voltage_{loop_idx:03d}_{tag}_data.csv"
            else:
                out_name = f"loop_{loop_idx:03d}_{self._current_loop_tag}_data.csv"
            arr = self._apply_math(self.data) if self.save_processed else self.data
            np.savetxt(self._run_folder / out_name, arr, delimiter=",", fmt="%.5e")
            mode = "processed" if self.save_processed else "raw"
            if voltage is not None:
                heatmap_name = f"voltage_{loop_idx:03d}_{tag}_heatmap.png"
                try:
                    self.figure_heatmap.savefig(
                        self._run_folder / heatmap_name,
                        dpi=300,
                        bbox_inches="tight",
                    )
                except Exception as exc:
                    logging.warning(f"Failed to save heatmap for {heatmap_name}: {exc}")
                self._update_statusbar(
                    f"Saved {mode} data at {self._voltage_display(float(voltage))}"
                )
            else:
                self._update_statusbar(text=f"Loop {loop_idx} saved ({mode})")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Loop Save Error", f"{e}")

    def _on_pixel(self, idx: int, i_avg: float):
        r, c = divmod(idx - 1, 10)
        self.data[r, c] = i_avg
        self._done_steps += 1
        # progress + timers
        pct = int(100 * self._done_steps / max(1, self._total_steps))
        self.ui.ScanprogressBar.setValue(pct)
        elapsed = time.time() - getattr(self, "_start_time", time.time())
        rate = self._done_steps / max(elapsed, 1e-9)
        remaining = (self._total_steps - self._done_steps) / max(rate, 1e-9)
        loop_idx = getattr(self, "_current_loop", 1)
        voltage_text = (
            f"  Bias: {self._voltage_display(self._current_voltage)}"
            if self._current_voltage is not None
            else ""
        )
        self.ui.ScanTimers.setText(
            f"{self._loop_label} {loop_idx}{voltage_text}    "
            f"Elapsed: {elapsed:5.1f}s    Remaining: {remaining:5.1f}s"
        )
        self._update_plots()

    def _scan_finished(self):
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None
        self.ui.btn_run_abort.setText("Run Scan")
        self.ui.btn_run_abort.setEnabled(True)
        self.ui.btn_pause_resume.setText("Pause")
        self.ui.btn_pause_resume.setEnabled(False)

        # optional: save end-of-run images of displayed plots
        try:
            if self._run_folder:
                self.figure_heatmap.savefig(
                    self._run_folder / "summary_heatmap.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                self.figure_hist.savefig(
                    self._run_folder / "summary_histogram.png",
                    dpi=300,
                    bbox_inches="tight",
                )
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Summary Save", f"Failed to save images: {e}"
            )

        self._current_voltage = None
        self._voltage_sequence = []
        self._constant_bias_voltage = None
        self._update_statusbar(text="Scan finished.")

    def _handle_device_error(self, message: str):
        logging.error(f"Device error: {message}")
        if self._worker:
            self._worker.stop()
        QtWidgets.QMessageBox.critical(
            self,
            "Device Error",
            f"A device disconnected or failed.\n\n{message}\n\nScan aborted.",
        )
        # We don’t have dedicated status widgets in this UI; reflect in statusbar.
        self._update_statusbar(text=message)

    # ---------------------- math/ref ----------------------
    def _load_reference_csv(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Reference CSV",
            str(self.output_folder),
            "CSV files (*.csv);;All files (*)",
        )
        if not fname:
            return
        try:
            arr = np.loadtxt(fname, delimiter=",", dtype=float)
            if arr.shape != (10, 10):
                if arr.size == 100:
                    arr = arr.reshape((10, 10))
                else:
                    raise ValueError(
                        f"Reference must be 10×10 (or 100 values). Got {arr.shape}."
                    )
            self.ref_matrix = arr
            self.ref_path = Path(fname)
            self.ui.lbl_ref_info.setText(
                f"{Path(fname).name}  [{arr.shape[0]}×{arr.shape[1]}]"
            )
            self._update_plots()
        except Exception as e:
            self.ref_matrix = None
            self.ref_path = None
            self.ui.lbl_ref_info.setText("(none)")
            QtWidgets.QMessageBox.critical(
                self, "Reference CSV", f"Failed to load: {e}"
            )

    def _math_mode_changed(self, _index: int):
        text = self.ui.combo_math.currentText().lower()
        if text.startswith("divide"):
            self.math_mode = "divide"
        elif text.startswith("subtract"):
            self.math_mode = "subtract"
        else:
            self.math_mode = "none"
        self._update_plots()

    def _apply_math(self, data: np.ndarray) -> np.ndarray:
        if self.math_mode == "none" or self.ref_matrix is None:
            return data
        ref = self.ref_matrix
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.math_mode == "divide":
                denom = ref + self.math_eps
                out = np.divide(data, denom, where=~np.isnan(data))
            else:  # subtract
                out = data - ref
        mask_nan = np.isnan(data)
        out = np.where(mask_nan, np.nan, out)
        return out

    # ---------------------- plotting ----------------------
    def _on_plot_selected(self, index: int):
        self.ui.plot_stack.setCurrentIndex(index)
        # plot_settings tabs are in a QTabWidget already; user controls selection

    def _update_plot_controls_state(self):
        is_manual = not self.ui.check_auto_scale.isChecked()
        self.ui.edit_vmin.setEnabled(is_manual)
        self.ui.edit_vmax.setEnabled(is_manual)
        self._update_plots()

    def _update_plots(self, reset: bool = False):
        units = self.ui.combo_units.currentText()
        if units == "pA":
            scale = 1e12
        elif units == "nA":
            scale = 1e9
        elif units == "µA":
            scale = 1e6
        elif units == "mA":
            scale = 1e3
        self.ax_heatmap.set_title(self.ui.edit_heatmap_title.text())
        display = np.full((10, 10), np.nan) if reset else self._apply_math(self.data)
        display = display * scale
        self.cbar.set_label(f"Current ({units})")
        self.im.set_data(display)
        valid = display[~np.isnan(display)]

        use_log = self.ui.check_log_scale_heatmap.isChecked()
        if valid.size > 0:
            if self.ui.check_auto_scale.isChecked():
                vmin, vmax = float(np.nanmin(valid)), float(np.nanmax(valid))
                if vmin == vmax:
                    vmin = max(1e-12, vmin * 0.9)
                    vmax = max(1e-12, vmax * 1.1)
                if vmin <= 0 and use_log:
                    vmin = 1e-12
                self.ui.edit_vmin.setText(f"{vmin:.3e}")
                self.ui.edit_vmax.setText(f"{vmax:.3e}")
            else:
                try:
                    vmin = float(self.ui.edit_vmin.text())
                    vmax = float(self.ui.edit_vmax.text())
                except Exception:
                    vmin, vmax = 1e-10, 1e-7
            self.im.set_cmap(self.ui.combo_colormap.currentText())
            if use_log:
                self.im.set_norm(LogNorm(vmin=max(vmin, 1e-12), vmax=max(vmax, 1e-11)))
            else:
                self.im.set_norm(Normalize(vmin=vmin, vmax=vmax))
        #self.figure_heatmap.tight_layout(pad=2.5)
        for artist in getattr(self, "_val_texts", []):
            try:
                artist.remove()
            except Exception:
                pass
        self._val_texts = []
        if self.ui.check_show_values.isChecked() and valid.size > 0:
            norm = self.im.norm
            for r in range(10):
                for c in range(10):
                    val = display[r, c]
                    if not np.isnan(val):
                        norm_v = (
                            float(norm(val))
                            if (norm and ((not use_log) or val > 0))
                            else 0.5
                        )
                        color = "white" if norm_v < 0.5 else "black"
                        self._val_texts.append(
                            self.ax_heatmap.text(
                                c,
                                r,
                                f"{val:.1f}",
                                ha="center",
                                va="center",
                                color=color,
                                fontsize=8,
                            )
                        )
        self.canvas_heatmap.draw_idle()

        # Histogram
        self.ax_hist.clear()
        self.ax_hist.set_title(self.ui.edit_hist_title.text())
        if valid.size > 0:
            self.ax_hist.hist(valid.flatten(), bins=self.ui.spin_bins.value())
        if self.ui.check_log_scale_hist.isChecked():
            self.ax_hist.set_yscale("log")
        self.ax_hist.grid(True, linestyle="--", alpha=0.6)
        self.ax_hist.set_xlabel(f"Current ({units})")
        self.ax_hist.set_ylabel("Pixel Count")
        self.figure_hist.tight_layout(pad=2.5)
        self.canvas_hist.draw_idle()

    # ---------------------- exports ----------------------
    def _export_heatmap(self):
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save PNG",
            str((self._run_folder or self.output_folder) / "heatmap.png"),
            "PNG files (*.png);;All files (*)",
        )
        if not fname:
            return
        try:
            self.figure_heatmap.savefig(fname, dpi=300, bbox_inches="tight")
            QtWidgets.QMessageBox.information(self, "Export", f"Saved to {fname}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export", f"Failed: {e}")

    def _export_histogram(self):
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save PNG",
            str((self._run_folder or self.output_folder) / "histogram.png"),
            "PNG files (*.png);;All files (*)",
        )
        if not fname:
            return
        try:
            self.figure_hist.savefig(fname, dpi=300, bbox_inches="tight")
            QtWidgets.QMessageBox.information(self, "Export", f"Saved to {fname}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export", f"Failed: {e}")

    # ---------------------- folders ----------------------
    def _select_output_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Folder", str(self.output_folder)
        )
        if folder:
            self.output_folder = Path(folder)
            self.ui.edit_output_folder.setText(str(self.output_folder))

    # ---------------------- menu: device connects ----------------------
    def _connect_read_smu(self):
        dlg = DevicePicker(self, title="Connect Read SMU", show_gpib=True)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        selection, gpib_addr = dlg.get()
        if not selection:
            return
        try:
            if (
                selection.upper().startswith(("USB", "GPIB", "TCPIP"))
            ):
                adapter = VISAAdapter(selection)
            elif PrologixAdapter:
                adapter = PrologixAdapter(
                    selection, gpib_addr or 5, gpib_read_timeout=3000
                )
                adapter.connection.timeout = 20000
                adapter.write("++mode 1")
                adapter.write("++auto 0")
                adapter.write("++eoi 1")
            else:
                raise RuntimeError("PyMeasure VISA/Prologix adapters not available.")
            self.sm = ReadoutSafe2400(adapter)
            try:
                ident = self.sm.adapter.ask("*IDN?")
            except Exception:
                ident = "Unknown"
            self.read_sm_idn = f"Read SMU: {ident.strip()}"
            QtWidgets.QMessageBox.information(
                self, "Sourcemeter", f"Connected: {ident}\nLocked at 0 V."
            )
        except Exception as e:
            self.sm = DummyKeithley2400()
            self.read_sm_idn = "Read SMU: DUMMY TEST DEVICE"
            QtWidgets.QMessageBox.critical(
                self, "Sourcemeter", f"Failed to connect: {e}"
            )
        self._update_statusbar(self.read_sm_idn)

    def _connect_bias_smu(self):
        dlg = DevicePicker(self, title="Connect Bias SMU", show_gpib=True)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        selection, gpib_addr = dlg.get()
        if not selection:
            return
        try:
            if selection.upper().startswith(("USB", "GPIB", "TCPIP")):
                adapter = VISAAdapter(selection)
            elif PrologixAdapter:
                adapter = PrologixAdapter(
                    selection, gpib_addr or 5, gpib_read_timeout=3000
                )
                adapter.connection.timeout = 20000
                adapter.write("++mode 1")
                adapter.write("++auto 0")
                adapter.write("++eoi 1")
            else:
                raise RuntimeError(
                    "PyMeasure VISA/Prologix adapters not available."
                )
            self.bias_sm = Keithley2400(adapter)
            ident = "Unknown"
            self.bias_sm.reset()
            self.bias_sm.apply_voltage()
            self.bias_sm.source_voltage_range = 100

            #self.bias_sm.compliance_current=0.1
            self.bias_sm.source_voltage = 0
            self.bias_sm.enable_source()
            self.bias_sm.measure_current(nplc=1, current=0.105, auto_range=False)
            self.bias_sm.current_range = 0.001
            print(self.bias_sm.current)
            self.bias_sm.disable_source()
            #self.bias_sm.sample_continuously()
            try:
                ident = self.bias_sm.ask("*IDN?")
            except Exception:
                pass
           # try:
                #self.bias_sm.source_voltage = 0.0
                #self.bias_sm.disable_source()
            #except Exception as exc:
            #    logging.warning(f"Failed to zero bias SMU on connect: {exc}")
            self.bias_sm_idn = f"Bias SMU: {ident.strip()}"
            QtWidgets.QMessageBox.information(
                self,
                "Bias SMU",
                f"Connected: {ident}\nOutput zeroed and disabled.",
            )
        except Exception as e:
            self.bias_sm = DummyBias2400()
            self.bias_sm_idn = "Bias SMU: (not connected)"
            QtWidgets.QMessageBox.critical(
                self, "Bias SMU", f"Failed to connect: {e}"
            )
        self._update_statusbar(self.bias_sm_idn)
        self._update_bias_controls_state()

    def _connect_switch(self):
        dlg = DevicePicker(self, title="Connect Switch Board", show_gpib=False)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        port, _ = dlg.get()
        if not port:
            return
        try:
            self.switch = SwitchBoard(port)
            ser = self.switch.ser
            ser.write(b"VERBOSE OFF\n")
            ser.readline()
            ser.write(b"IDN\n")
            idn = ser.readline().decode("utf-8").strip() or "(no IDN)"
            self.switch_idn = f"Switch: {idn}"
            ser.write(b"SWSTATUS\n")
            status = ser.readline().decode("utf-8").strip()
            self.inactive_channels = (
                [int(x) for x in status.split(",") if x.strip()] if status else []
            )
            QtWidgets.QMessageBox.information(
                self, "Switch", f"Connected on {port}.\nID: {idn}"
            )
        except Exception as e:
            self.switch = DummySwitchBoard()
            self.switch_idn = "Switch: (not connected)"
            self.inactive_channels = []
            QtWidgets.QMessageBox.critical(self, "Switch", f"Failed: {e}")
        self._update_statusbar(self.switch_idn)
        self._update_plots(reset=False)

    # ---------------------- misc ----------------------
    def _update_integration_time_label(self):
        # Approx integration per reading = NPLC / mains_freq (assume 60 Hz unless you want to make it user-selectable)
        mains = 60.0
        t_read = self.ui.spin_nplc.value() / mains
        self.ui.integration_time.setText(
            f"{t_read:.3f} s/read"
        )


# ======================================================================
#                                 main
# ======================================================================


def main():
    app = QtWidgets.QApplication(sys.argv)
    # Optional splash:
    splash = QtWidgets.QSplashScreen(QtGui.QPixmap("res/Splash.png"))
    splash.show()
    delay = 2000  # ms
    end = time.time() + delay / 1000.0
    while time.time() < end:
        app.processEvents()
    # Main window:
    win = MainWindow()
    win.show()
    if 'splash' in locals(): splash.finish(win)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
