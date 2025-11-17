#!/usr/bin/env python3
# main.py — Readout app with PySide6 UI (from pyside6-uic)

import json
import logging
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from logging.handlers import RotatingFileHandler
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
from ScannerWin.acquisition import AcquisitionSettings
from Analysis.analysis_window import AnalysisWindow
from StageScan.stage_scan_window import StageScanWindow


APP_DIR = Path(__file__).resolve().parent
LOG_FILE = APP_DIR / "ReadoubtSW.log"


def _configure_logging():
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers = [logging.StreamHandler()]
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3)
        )
    except Exception:
        pass
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=handlers)
    logging.getLogger("readoubt.devices").setLevel(logging.DEBUG)


_configure_logging()
DEVICE_LOGGER = logging.getLogger("readoubt.devices")


@dataclass
class LoopSnapshot:
    number: int
    label: str
    voltage: Optional[float]
    requested_voltage: Optional[float]
    data: np.ndarray
    runtime_ms: Optional[float] = None
    timestamp: float = field(default_factory=lambda: time.time())


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # ---- load UI ----
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Readoubt")
        self.analysis_window: Optional[AnalysisWindow] = None
        self.stage_window: Optional[StageScanWindow] = None
        self.menu_analysis = self.ui.menuBar.addMenu("Analysis")
        self.action_open_analysis = QtGui.QAction("Open Analysis Window", self)
        self.menu_analysis.addAction(self.action_open_analysis)
        self.action_open_analysis.triggered.connect(self._open_analysis_window)
        self.menu_stage = self.ui.menuBar.addMenu("Stage")
        self.action_open_stage = QtGui.QAction("Open Stage Scan Window", self)
        self.menu_stage.addAction(self.action_open_stage)
        self.action_open_stage.triggered.connect(self._open_stage_scan_window)

        # ---- paths/state ----
        self.output_folder: Path = APP_DIR
        self._run_folder: Optional[Path] = None
        self._heatmap_dir: Optional[Path] = None
        self._histogram_dir: Optional[Path] = None
        self._data_dir: Optional[Path] = None
        self._active_run_info: Optional[dict] = None
        self.data = np.full((10, 10), np.nan)
        self.ref_matrix: Optional[np.ndarray] = None
        self.ref_path: Optional[Path] = None
        self.math_mode: str = "none"  # none|divide|subtract
        self.math_eps: float = 1e-12
        self.save_processed: bool = False
        self.inactive_channels: List[int] = []
        self._loop_history: List[LoopSnapshot] = []
        self._selected_history_index: Optional[int] = None
        self._is_paused = False
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[ScanWorker] = None
        self.measurement_mode: str = "time"
        self._loop_label: str = "Loop"
        self._current_voltage: Optional[float] = None
        self._voltage_sequence: List[float] = []
        self._constant_bias_voltage: Optional[float] = None
        self._requested_voltage: Optional[float] = None
        self._voltage_control_widgets: List[QtWidgets.QWidget] = []
        self._loop_control_widgets: List[QtWidgets.QWidget] = []
        self._time_bias_widgets: List[QtWidgets.QWidget] = []
        self._switch_settle_ms: int = 4
        self.autosave_enabled: bool = True
        self.device_logger = DEVICE_LOGGER

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
        self.ui.check_auto_current_range.toggled.connect(
            self._toggle_manual_current_range
        )
        self._toggle_manual_current_range(
            self.ui.check_auto_current_range.isChecked()
        )
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
        self._init_switch_options_panel()
        self._init_loop_scrubber()

        # ---- signal wiring ----
        self.ui.btn_run_abort.clicked.connect(self._handle_run_abort_clicked)
        self.ui.btn_pause_resume.clicked.connect(self._handle_pause_resume_clicked)

        self.ui.btn_browse_folder.clicked.connect(self._select_output_folder)
        self.autosave_enabled = self.ui.check_autosave.isChecked()
        self.ui.check_autosave.toggled.connect(self._handle_autosave_toggled)
        self._register_run_name_watchers()
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
        self._update_switch_controls_state()
        self._on_measurement_type_changed(
            self.ui.Measurement_type_combobox.currentIndex()
        )

    @QtCore.Slot()
    def _handle_run_abort_clicked(self):
        if self._worker:
            self._abort_scan()
        else:
            self._start_scan()

    @QtCore.Slot()
    def _handle_pause_resume_clicked(self):
        if not self._worker:
            return
        if not self._is_paused:
            self._is_paused = True
            self.ui.btn_pause_resume.setText("Resume")
            try:
                QtCore.QMetaObject.invokeMethod(
                    self._worker, "pause", QtCore.Qt.QueuedConnection
                )
            except Exception:
                pass
            self._update_statusbar(text="Scan paused.")
        else:
            self._is_paused = False
            self.ui.btn_pause_resume.setText("Pause")
            try:
                QtCore.QMetaObject.invokeMethod(
                    self._worker, "resume", QtCore.Qt.QueuedConnection
                )
            except Exception:
                pass
            self._update_statusbar(text="Resuming scan…")

    def _init_switch_options_panel(self):
        self.switch_options_box = QtWidgets.QGroupBox("Switch Board Options")
        self.switch_options_box.setObjectName("switch_options_box")
        layout = QtWidgets.QFormLayout(self.switch_options_box)
        layout.setObjectName("switch_options_layout")

        self.ui.label_readout_source = QtWidgets.QLabel("Readout source:")
        self.ui.label_readout_source.setObjectName("label_readout_source")
        self.ui.combo_readout_source = QtWidgets.QComboBox()
        self.ui.combo_readout_source.setObjectName("combo_readout_source")
        self.ui.combo_readout_source.addItems(
            ["Read SMU", "Switch board (local)"]
        )
        layout.addRow(self.ui.label_readout_source, self.ui.combo_readout_source)

        self.ui.label_bias_source = QtWidgets.QLabel("Bias source:")
        self.ui.label_bias_source.setObjectName("label_bias_source")
        self.ui.combo_bias_source = QtWidgets.QComboBox()
        self.ui.combo_bias_source.setObjectName("combo_bias_source")
        self.ui.combo_bias_source.addItems(
            ["Bias SMU", "Switch board (local)"]
        )
        layout.addRow(self.ui.label_bias_source, self.ui.combo_bias_source)

        self.ui.label_led_control = QtWidgets.QLabel("Board LEDs:")
        self.ui.label_led_control.setObjectName("label_led_control")
        self.ui.check_led_enable = QtWidgets.QCheckBox("Enable LEDs")
        self.ui.check_led_enable.setObjectName("check_led_enable")
        layout.addRow(self.ui.label_led_control, self.ui.check_led_enable)

        self.ui.label_switch_settle = QtWidgets.QLabel("Routing settle (ms):")
        self.ui.label_switch_settle.setObjectName("label_switch_settle")
        self.ui.spin_switch_settle = QtWidgets.QSpinBox()
        self.ui.spin_switch_settle.setObjectName("spin_switch_settle")
        self.ui.spin_switch_settle.setRange(0, 5000)
        self.ui.spin_switch_settle.setSingleStep(1)
        self.ui.spin_switch_settle.setValue(self._switch_settle_ms)
        layout.addRow(self.ui.label_switch_settle, self.ui.spin_switch_settle)

        idx = self.ui.verticalLayout.indexOf(self.ui.scan_settings_box)
        insert_pos = max(0, idx + 1)
        self.ui.verticalLayout.insertWidget(insert_pos, self.switch_options_box)

        self.ui.combo_readout_source.currentIndexChanged.connect(
            self._on_readout_source_changed
        )
        self.ui.combo_bias_source.currentIndexChanged.connect(
            self._on_bias_source_changed
        )
        self.ui.check_led_enable.toggled.connect(self._handle_led_toggled)
        self.ui.spin_switch_settle.valueChanged.connect(
            self._on_switch_settle_changed
        )

    def _init_loop_scrubber(self):
        container = QtWidgets.QWidget()
        container.setObjectName("loopScrubWidget")
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        label = QtWidgets.QLabel("Loop viewer:")
        label.setObjectName("loop_scrub_label")
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setObjectName("loop_scrub_slider")
        slider.setMinimum(0)
        slider.setMaximum(0)
        slider.setSingleStep(1)
        slider.setPageStep(1)
        slider.setEnabled(False)
        spin = QtWidgets.QSpinBox()
        spin.setObjectName("loop_scrub_spin")
        spin.setMinimum(0)
        spin.setMaximum(0)
        spin.setEnabled(False)
        info = QtWidgets.QLabel("Loops will appear here after a scan.")
        info.setObjectName("loop_scrub_info")
        info.setMinimumWidth(200)
        layout.addWidget(label)
        layout.addWidget(slider, 1)
        layout.addWidget(spin)
        layout.addWidget(info)
        self.ui.plot_area_layout.insertWidget(1, container)
        slider.valueChanged.connect(self._on_loop_slider_changed)
        spin.valueChanged.connect(self._on_loop_spin_changed)
        self._loop_scrub_widget = container
        self._loop_scrub_slider = slider
        self._loop_scrub_spin = spin
        self._loop_scrub_info = info
        self._update_loop_scrub_state()

    def _clear_loop_history(self):
        self._loop_history.clear()
        self._selected_history_index = None
        self._update_loop_scrub_state()

    def _update_loop_scrub_state(self):
        slider = getattr(self, "_loop_scrub_slider", None)
        spin = getattr(self, "_loop_scrub_spin", None)
        if slider is None or spin is None:
            return
        count = len(self._loop_history)
        slider.blockSignals(True)
        spin.blockSignals(True)
        if count:
            slider.setMinimum(1)
            slider.setMaximum(count)
            if (
                self._selected_history_index is None
                or self._selected_history_index >= count
            ):
                self._selected_history_index = count - 1
            slider.setValue(self._selected_history_index + 1)
            spin.setMinimum(1)
            spin.setMaximum(count)
            spin.setValue(self._selected_history_index + 1)
        else:
            slider.setMinimum(0)
            slider.setMaximum(0)
            slider.setValue(0)
            spin.setMinimum(0)
            spin.setMaximum(0)
            spin.setValue(0)
        slider.blockSignals(False)
        spin.blockSignals(False)
        enable_controls = count > 0 and self._thread is None
        slider.setEnabled(enable_controls)
        spin.setEnabled(enable_controls)
        self._update_loop_scrub_info()

    def _update_loop_scrub_info(self):
        info = getattr(self, "_loop_scrub_info", None)
        if info is None:
            return
        if not self._loop_history:
            info.setText("No loops captured yet.")
            return
        if self._thread is not None:
            info.setText(
                f"{len(self._loop_history)} loops captured – scrub after scan finishes."
            )
            return
        if self._selected_history_index is None:
            info.setText(f"{len(self._loop_history)} loops captured.")
            return
        info.setText(self._format_loop_snapshot_text(self._selected_history_index))

    def _format_loop_snapshot_text(self, index: int) -> str:
        if not (0 <= index < len(self._loop_history)):
            return ""
        snapshot = self._loop_history[index]
        total = len(self._loop_history)
        label = snapshot.label or self._loop_label or "Loop"
        parts = [f"{label} {snapshot.number} of {total}"]
        if snapshot.voltage is not None:
            voltage_text = self._voltage_display(float(snapshot.voltage))
            if (
                snapshot.requested_voltage is not None
                and abs(float(snapshot.requested_voltage) - float(snapshot.voltage))
                > 5e-4
            ):
                voltage_text = (
                    f"{voltage_text} (req {self._voltage_display(float(snapshot.requested_voltage))})"
                )
            parts.append(voltage_text)
        if snapshot.runtime_ms is not None:
            parts.append(f"{snapshot.runtime_ms / 1000.0:.2f}s local read")
        return " – ".join(parts)

    def _set_loop_selection(self, index: int, source: str):
        if self._thread is not None:
            return
        if not (0 <= index < len(self._loop_history)):
            return
        if self._selected_history_index == index:
            self._update_loop_scrub_info()
            return
        self._selected_history_index = index
        if source != "slider":
            self._loop_scrub_slider.blockSignals(True)
            self._loop_scrub_slider.setValue(index + 1)
            self._loop_scrub_slider.blockSignals(False)
        if source != "spin":
            self._loop_scrub_spin.blockSignals(True)
            self._loop_scrub_spin.setValue(index + 1)
            self._loop_scrub_spin.blockSignals(False)
        snapshot = self._loop_history[index]
        self.data = np.array(snapshot.data, copy=True)
        self._current_voltage = (
            float(snapshot.voltage) if snapshot.voltage is not None else None
        )
        self._update_plots(reset=False)
        self._update_loop_scrub_info()

    def _on_loop_slider_changed(self, value: int):
        if value <= 0:
            return
        self._set_loop_selection(value - 1, source="slider")

    def _on_loop_spin_changed(self, value: int):
        if value <= 0:
            return
        self._set_loop_selection(value - 1, source="spin")

    def _store_loop_snapshot(self, loop_idx: int, metadata: Optional[dict]):
        try:
            data_copy = np.array(self.data, copy=True)
        except Exception:
            return
        voltage = None
        requested = None
        runtime_ms = None
        if metadata and isinstance(metadata, dict):
            voltage = metadata.get("voltage")
            requested = metadata.get("requested_voltage")
            runtime_ms = metadata.get("runtime_ms")
        if voltage is None and self._current_voltage is not None:
            voltage = self._current_voltage
        snapshot = LoopSnapshot(
            number=int(loop_idx),
            label=self._loop_label,
            voltage=float(voltage) if voltage is not None else None,
            requested_voltage=(
                float(requested) if requested is not None else None
            ),
            data=data_copy,
            runtime_ms=float(runtime_ms) if runtime_ms is not None else None,
        )
        self._loop_history.append(snapshot)
        self._selected_history_index = len(self._loop_history) - 1
        self._update_loop_scrub_state()

    def _register_run_name_watchers(self):
        fields = (
            "edit_exp_name",
            "edit_run_name",
            "edit_run_tag",
            "edit_experiment_name",
            "line_run_name",
        )
        for name in fields:
            widget = getattr(self.ui, name, None)
            if widget is None or not hasattr(widget, "textChanged"):
                continue
            widget.textChanged.connect(self._invalidate_run_folder)

    def _invalidate_run_folder(self):
        thread = getattr(self, "_thread", None)
        if thread is not None and thread.isRunning():
            return
        self._run_folder = None
        self._heatmap_dir = None
        self._histogram_dir = None
        self._data_dir = None
        self._active_run_info = None

    # ---------------------- convenience ----------------------
    def _update_statusbar(self, text: str):
        self.statusBar().showMessage(text, 4000)

    def _switch_connected(self) -> bool:
        return not isinstance(self.switch, DummySwitchBoard)

    def _using_local_readout(self) -> bool:
        combo = getattr(self.ui, "combo_readout_source", None)
        return bool(combo and combo.currentIndex() == 1)

    def _using_local_bias(self) -> bool:
        combo = getattr(self.ui, "combo_bias_source", None)
        return bool(combo and combo.currentIndex() == 1)

    @staticmethod
    def _set_combo_index(combo: QtWidgets.QComboBox, index: int):
        combo.blockSignals(True)
        combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def _set_led_checkbox(self, checked: bool):
        if not hasattr(self.ui, "check_led_enable"):
            return
        self.ui.check_led_enable.blockSignals(True)
        self.ui.check_led_enable.setChecked(checked)
        self.ui.check_led_enable.blockSignals(False)

    def _update_switch_controls_state(self):
        connected = self._switch_connected()
        if not connected:
            if self._using_local_readout():
                self._set_combo_index(self.ui.combo_readout_source, 0)
            if self._using_local_bias():
                self._set_combo_index(self.ui.combo_bias_source, 0)
            self._set_led_checkbox(False)
        self.ui.check_led_enable.setEnabled(connected)
        self._update_bias_controls_state()

    def _handle_led_toggled(self, checked: bool):
        if not self._switch_connected():
            self._set_led_checkbox(False)
            return
        try:
            self.device_logger.info(
                "Switch board LEDs %s", "ON" if checked else "OFF"
            )
            self.switch.set_led(bool(checked))
        except Exception as exc:
            logging.error(f"Failed to toggle LEDs: {exc}")
            QtWidgets.QMessageBox.warning(
                self, "Switch LEDs", f"Failed to toggle LEDs:\n{exc}"
            )
            self._set_led_checkbox(not checked)

    def _on_switch_settle_changed(self, value: int):
        value = max(0, int(value))
        self._switch_settle_ms = value
        if self._switch_connected():
            self._apply_switch_settle_time(show_error=True)

    def _apply_switch_settle_time(self, *, show_error: bool = False):
        if not self._switch_connected():
            return
        try:
            applied = self.switch.set_settle_time(self._switch_settle_ms)
            self.device_logger.info(
                "Switch board settle time set to %s ms", applied
            )
        except Exception as exc:
            logging.error(f"Failed to set switch settle time: {exc}")
            if show_error:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Switch Settle Time",
                    f"Failed to apply settle time:\n{exc}",
                )
    def _on_readout_source_changed(self, index: int):
        if index == 1 and not self._switch_connected():
            QtWidgets.QMessageBox.warning(
                self, "Local Readout", "Connect the switch board to use local readout."
            )
            self._set_combo_index(self.ui.combo_readout_source, 0)
            return
        self._update_switch_controls_state()

    def _on_bias_source_changed(self, index: int):
        if index == 1 and not self._switch_connected():
            QtWidgets.QMessageBox.warning(
                self, "Local Bias", "Connect the switch board to use local bias."
            )
            self._set_combo_index(self.ui.combo_bias_source, 0)
            return
        self._update_bias_controls_state()

    def _bias_smu_connected(self) -> bool:
        return not isinstance(self.bias_sm, DummyBias2400)

    def _update_bias_controls_state(self):
        show_bias = self.measurement_mode != "voltage"
        for widget in self._time_bias_widgets:
            widget.setVisible(show_bias)

        use_local_bias = self._using_local_bias()
        bias_available = (
            self._switch_connected() if use_local_bias else self._bias_smu_connected()
        )
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
        self._ensure_local_bias_supported(voltages)
        return voltages, max(0.0, settle)

    def _collect_time_bias_voltage(self) -> Optional[float]:
        if not getattr(self.ui, "check_bias_enable", None):
            return None
        if not self.ui.check_bias_enable.isChecked():
            return None
        if not self._bias_smu_connected() and not self._using_local_bias():
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
        value = round(value, 6)
        self._ensure_local_bias_supported([value])
        return value

    def _ensure_local_bias_supported(self, voltages: Iterable[float]):
        if not self._using_local_bias():
            return
        min_v, max_v = 6.0, 87.0
        for v in voltages:
            if v is None:
                continue
            if not (min_v <= float(v) <= max_v):
                raise ValueError(
                    f"Local bias via switch board supports {min_v:.0f}–{max_v:.0f} V."
                )

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

    @QtCore.Slot(bool)
    def _toggle_manual_current_range(self, auto_enabled: bool) -> None:
        if getattr(self.ui, "edit_current_range", None):
            self.ui.edit_current_range.setEnabled(not auto_enabled)

    def _collect_current_range(self) -> tuple[bool, float]:
        auto_range = self.ui.check_auto_current_range.isChecked()
        text = (self.ui.edit_current_range.text() or "").strip()
        if auto_range:
            try:
                value = float(text) if text else 1e-7
            except Exception:
                value = 1e-7
            return True, float(value)
        if not text:
            raise ValueError("Enter a manual current range (A).")
        try:
            value = float(text)
        except Exception as exc:
            raise ValueError("Manual range must be numeric (A).") from exc
        if value <= 0:
            raise ValueError("Manual range must be positive (A).")
        return False, float(value)

    def _build_acquisition_settings(self) -> AcquisitionSettings:
        try:
            pixels = self._parse_pixel_spec(self.ui.edit_pixel_spec.text())
        except Exception as exc:
            raise ValueError(f"Pixel selection invalid: {exc}") from exc

        auto_range, current_range = self._collect_current_range()
        samples = max(1, int(self.ui.spin_nsamp.value()))
        nplc = float(self.ui.spin_nplc.value())
        measurement_text = (
            self.ui.Measurement_type_combobox.currentText() or ""
        ).lower()
        voltage_mode = "voltage" in measurement_text

        voltage_steps: Optional[List[float]] = None
        settle_time = 0.0
        constant_bias_voltage: Optional[float] = None
        if voltage_mode:
            try:
                voltage_steps, settle_time = self._collect_voltage_sweep_settings()
            except Exception as exc:
                raise ValueError(f"Voltage sweep invalid: {exc}") from exc
            loops = len(voltage_steps)
            inter_loop_delay = 0.0
        else:
            try:
                constant_bias_voltage = self._collect_time_bias_voltage()
            except Exception as exc:
                raise ValueError(f"Bias voltage invalid: {exc}") from exc
            loops = int(self.ui.spin_loops.value())
            if loops <= 0:
                raise ValueError("Loop count must be positive.")
            try:
                inter_loop_delay = float(self.ui.loop_delay.text())
            except Exception:
                inter_loop_delay = 0.0

        loops = int(loops)
        if loops <= 0:
            raise ValueError("At least one loop is required.")

        use_local_readout = self._using_local_readout()
        use_local_bias = self._using_local_bias()

        return AcquisitionSettings(
            pixels=pixels,
            samples_per_pixel=samples,
            nplc=nplc,
            loops=loops,
            inter_loop_delay_s=max(0.0, float(inter_loop_delay)),
            auto_range=auto_range,
            current_range=float(current_range),
            measurement_mode="voltage" if voltage_mode else "time",
            voltage_steps=list(voltage_steps or []) if voltage_mode else None,
            voltage_settle_s=settle_time if voltage_mode else 0.0,
            constant_bias_voltage=(
                float(constant_bias_voltage)
                if constant_bias_voltage is not None and not voltage_mode
                else None
            ),
            use_local_readout=use_local_readout,
            use_local_bias=use_local_bias,
        )

    def export_acquisition_settings(self) -> AcquisitionSettings:
        return self._build_acquisition_settings()

    def _collect_experiment_name(self) -> str:
        for attr in (
            "edit_exp_name",
            "edit_run_name",
            "edit_run_tag",
            "edit_experiment_name",
            "line_run_name",
        ):
            widget = getattr(self.ui, attr, None)
            if widget and hasattr(widget, "text"):
                text = (widget.text() or "").strip()
                if text:
                    return text
        return "scan"

    def _ensure_run_folder(self, *, force_new: bool = False) -> Path:
        if (
            self._run_folder
            and self._run_folder.exists()
            and not force_new
        ):
            return self._run_folder
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = self._collect_experiment_name()
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", base_name).strip("_")
        if not slug:
            slug = "scan"
        run_id = f"{slug}_{timestamp}"
        output_root = Path(self.output_folder)
        output_root.mkdir(parents=True, exist_ok=True)
        run_folder = output_root / run_id
        run_folder.mkdir(parents=True, exist_ok=True)
        heatmap_dir = run_folder / "heatmaps"
        histogram_dir = run_folder / "histograms"
        data_dir = run_folder / "data"
        for folder in (heatmap_dir, histogram_dir, data_dir):
            folder.mkdir(parents=True, exist_ok=True)
        self._run_folder = run_folder
        self._heatmap_dir = heatmap_dir
        self._histogram_dir = histogram_dir
        self._data_dir = data_dir
        self._active_run_info = {
            "id": run_id,
            "experiment_name": base_name or "scan",
            "timestamp": timestamp,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "root": str(run_folder),
            "run_uuid": uuid.uuid4().hex,
        }
        logging.info("Created run folder at %s", run_folder)
        return run_folder

    def _write_run_metadata(self, acquisition: AcquisitionSettings):
        if not self._run_folder:
            return
        metadata = {
            "run": {
                **(self._active_run_info or {}),
                "autosave": self.autosave_enabled,
            },
            "devices": {
                "read_smu": self.read_sm_idn,
                "bias_smu": self.bias_sm_idn,
                "switch": self.switch_idn,
            },
            "acquisition": asdict(acquisition),
            "pixel_spec_text": self.ui.edit_pixel_spec.text(),
            "math": {
                "mode": self.math_mode,
                "epsilon": self.math_eps,
                "save_processed": self.save_processed,
            },
            "output": {
                "root": str(self._run_folder),
                "heatmaps": str(self._heatmap_dir or self._run_folder),
                "histograms": str(self._histogram_dir or self._run_folder),
                "data": str(self._data_dir or self._run_folder),
            },
            "inactive_channels": list(self.inactive_channels),
            "plot": {
                "title": self.ui.edit_heatmap_title.text(),
                "units": self.ui.combo_units.currentText(),
                "colormap": self.ui.combo_colormap.currentText(),
                "log_scale": self.ui.check_log_scale_heatmap.isChecked(),
            },
            "software": {
                "log_file": str(LOG_FILE),
                "python": sys.version.split()[0],
                "app_dir": str(APP_DIR),
            },
        }
        metadata_path = self._run_folder / "metadata.json"
        try:
            with metadata_path.open("w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2)
            logging.info("Wrote run metadata to %s", metadata_path)
        except Exception as exc:
            logging.error(f"Failed to write metadata to {metadata_path}: {exc}")

    def _start_scan(self):
        try:
            acquisition = self._build_acquisition_settings()
        except ValueError as exc:
            QtWidgets.QMessageBox.critical(self, "Scan Settings", str(exc))
            return
        run_folder = None
        if self.autosave_enabled:
            try:
                run_folder = self._ensure_run_folder(force_new=True)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Output Folder", str(e))
                return
            self._write_run_metadata(acquisition)
        else:
            logging.info(
                "Autosave disabled – scan results will not be written to disk."
            )
        logging.info(
            "Starting scan (mode=%s, loops=%s, samples/pixel=%s, autosave=%s)",
            acquisition.measurement_mode,
            acquisition.loops,
            acquisition.samples_per_pixel,
            self.autosave_enabled,
        )

        pixels = acquisition.pixels
        auto_rng = acquisition.auto_range
        cur_rng = acquisition.current_range

        # reset data and UI
        self.data.fill(np.nan)
        self._clear_loop_history()
        self._update_plots(reset=True)
        nplc = acquisition.nplc
        nsamp = acquisition.samples_per_pixel

        voltage_mode = acquisition.measurement_mode == "voltage"
        self.measurement_mode = acquisition.measurement_mode
        self._loop_label = "Voltage Step" if voltage_mode else "Loop"
        self._current_voltage = None
        self._constant_bias_voltage = None
        self._requested_voltage = None
        use_local_readout = acquisition.use_local_readout
        use_local_bias = acquisition.use_local_bias
        if use_local_readout and not self._switch_connected():
            QtWidgets.QMessageBox.critical(
                self, "Switch Board", "Connect the switch board for local readout."
            )
            return
        if use_local_bias and not self._switch_connected():
            QtWidgets.QMessageBox.critical(
                self, "Switch Board", "Connect the switch board for local bias."
            )
            return

        voltage_steps = acquisition.voltage_steps if voltage_mode else None
        settle_time = acquisition.voltage_settle_s if voltage_mode else 0.0
        constant_bias_voltage = (
            None if voltage_mode else acquisition.constant_bias_voltage
        )
        loops = acquisition.loops
        inter_loop_delay = (
            0.0 if voltage_mode else acquisition.inter_loop_delay_s
        )
        if voltage_mode:
            if not use_local_bias and isinstance(self.bias_sm, DummyBias2400):
                QtWidgets.QMessageBox.critical(
                    self,
                    "Bias SMU",
                    "Connect a bias sourcemeter before running a voltage sweep.",
                )
                return
            self._voltage_sequence = list(voltage_steps or [])
        else:
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
            if (voltage_mode or constant_bias_voltage is not None) and not use_local_bias
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
            use_local_readout=use_local_readout,
            use_local_bias=use_local_bias,
        )
        self._thread = QtCore.QThread()
        self._worker.moveToThread(self._thread)

        # signals
        self._worker.loopDataReady.connect(self._on_loop_data)
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
        requested_voltage = None
        if metadata and isinstance(metadata, dict):
            voltage = metadata.get("voltage")
            requested_voltage = metadata.get("requested_voltage")
        self._current_voltage = float(voltage) if voltage is not None else None
        self._requested_voltage = (
            float(requested_voltage)
            if requested_voltage is not None
            else self._current_voltage
        )
        self._update_plots(reset=True)
        if self._current_voltage is not None:
            display = self._voltage_display(self._current_voltage)
            if (
                self._requested_voltage is not None
                and abs(self._requested_voltage - self._current_voltage) > 5e-4
            ):
                display = (
                    f"{display} (req {self._voltage_display(self._requested_voltage)})"
                )
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
        try:
            voltage = None
            runtime_ms = None
            if metadata and isinstance(metadata, dict):
                voltage = metadata.get("voltage")
                runtime_ms = metadata.get("runtime_ms")
            if self._current_voltage is not None and voltage is None:
                voltage = self._current_voltage
            if voltage is not None:
                base_name = f"voltage_{loop_idx:03d}"
            else:
                base_name = f"loop_{loop_idx:03d}"
            out_name = f"{base_name}_data.csv"
            if not self.autosave_enabled:
                logging.info("Autosave disabled – skipping files for loop %s", loop_idx)
                self._update_statusbar(
                    f"Autosave off – loop {loop_idx} complete (not saved)"
                )
                return
            if not self._run_folder:
                self._ensure_run_folder()
            if not self._run_folder:
                return
            arr = self._apply_math(self.data) if self.save_processed else self.data
            data_dir = self._data_dir or self._run_folder
            csv_path = data_dir / out_name
            np.savetxt(csv_path, arr, delimiter=",", fmt="%.5e")
            logging.info("Saved loop data to %s", csv_path)
            mode = "processed" if self.save_processed else "raw"
            runtime_text = ""
            if runtime_ms is not None:
                runtime_text = f" – local read {float(runtime_ms) / 1000.0:.2f}s"
            heatmap_name = f"{base_name}_heatmap.png"
            if voltage is not None:
                display = self._voltage_display(float(voltage))
                status_msg = f"Saved {mode} data at {display}{runtime_text}"
            else:
                status_msg = f"Loop {loop_idx} saved ({mode}){runtime_text}"
            heatmap_dir = self._heatmap_dir or self._run_folder
            heatmap_path = heatmap_dir / heatmap_name
            try:
                self.figure_heatmap.savefig(
                    heatmap_path,
                    dpi=300,
                    bbox_inches="tight",
                )
                logging.info("Saved loop heatmap to %s", heatmap_path)
            except Exception as exc:
                logging.warning(f"Failed to save heatmap for {heatmap_name}: {exc}")
            histogram_dir = self._histogram_dir or self._run_folder
            histogram_name = f"{base_name}_histogram.png"
            histogram_path = histogram_dir / histogram_name
            try:
                self.figure_hist.savefig(
                    histogram_path,
                    dpi=300,
                    bbox_inches="tight",
                )
                logging.info("Saved loop histogram to %s", histogram_path)
            except Exception as exc:
                logging.warning(
                    f"Failed to save histogram for {histogram_name}: {exc}"
                )
            self._update_statusbar(status_msg)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Loop Save Error", f"{e}")
        finally:
            self._store_loop_snapshot(loop_idx, metadata)

    def _on_loop_data(self, loop_idx: int, loop_values):
        entries = list(loop_values) if loop_values is not None else []
        updated = 0
        for idx, i_avg in entries:
            try:
                r, c = divmod(int(idx) - 1, 10)
            except Exception:
                continue
            if not (0 <= r < 10 and 0 <= c < 10):
                continue
            try:
                self.data[r, c] = float(i_avg)
            except Exception:
                self.data[r, c] = np.nan
            updated += 1
        if updated == 0:
            return
        self._done_steps = min(
            self._total_steps, self._done_steps + updated
        )
        pct = int(100 * self._done_steps / max(1, self._total_steps))
        self.ui.ScanprogressBar.setValue(pct)
        elapsed = time.time() - getattr(self, "_start_time", time.time())
        rate = self._done_steps / max(elapsed, 1e-9)
        remaining = (self._total_steps - self._done_steps) / max(rate, 1e-9)
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
                summary_heatmap = self._run_folder / "summary_heatmap.png"
                summary_hist = self._run_folder / "summary_histogram.png"
                self.figure_heatmap.savefig(
                    summary_heatmap,
                    dpi=300,
                    bbox_inches="tight",
                )
                self.figure_hist.savefig(
                    summary_hist,
                    dpi=300,
                    bbox_inches="tight",
                )
                logging.info("Saved summary heatmap to %s", summary_heatmap)
                logging.info("Saved summary histogram to %s", summary_hist)
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Summary Save", f"Failed to save images: {e}"
            )

        self._current_voltage = None
        self._voltage_sequence = []
        self._constant_bias_voltage = None
        self._update_statusbar(text="Scan finished.")
        self._update_loop_scrub_state()
        if self.analysis_window and self._run_folder:
            try:
                self.analysis_window.set_run_folder(self._run_folder, auto_load=True)
            except Exception as exc:
                logging.warning(f"Analysis window update failed: {exc}")

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
            self._invalidate_run_folder()
            logging.info("Output folder changed to %s", self.output_folder)

    def _handle_autosave_toggled(self, checked: bool):
        self.autosave_enabled = bool(checked)
        state = "enabled" if self.autosave_enabled else "disabled"
        logging.info("Autosave %s", state)
        if self.autosave_enabled:
            self._update_statusbar("Autosave enabled – runs will be saved.")
        else:
            self._update_statusbar(
                "Autosave disabled – scans will not be written to disk."
            )

    # ---------------------- menu: device connects ----------------------
    def _connect_read_smu(self):
        dlg = DevicePicker(self, title="Connect Read SMU", show_gpib=True)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        selection, gpib_addr = dlg.get()
        if not selection:
            return
        logging.info("Connecting read SMU via %s (gpib=%s)", selection, gpib_addr)
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
        logging.info("Connecting bias SMU via %s (gpib=%s)", selection, gpib_addr)
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
            #print(self.bias_sm.current)
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
        logging.info("Connecting switch board on port %s", port)
        try:
            self.switch = SwitchBoard(port)
            ser = self.switch.ser
            ser.flush()  # clear buffer
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
            self._set_led_checkbox(True)
            self._apply_switch_settle_time()
            QtWidgets.QMessageBox.information(
                self, "Switch", f"Connected on {port}.\nID: {idn}"
            )
        except Exception as e:
            self.switch = DummySwitchBoard()
            self.switch_idn = "Switch: (not connected)"
            self.inactive_channels = []
            self._set_led_checkbox(False)
            QtWidgets.QMessageBox.critical(self, "Switch", f"Failed: {e}")
        self._update_statusbar(self.switch_idn)
        self._update_switch_controls_state()
        self._update_plots(reset=False)

    def _open_analysis_window(self):
        if self.analysis_window is None:
            try:
                self.analysis_window = AnalysisWindow(self, pixel_parser=self._parse_pixel_spec)
            except Exception as exc:
                logging.error(f"Failed to create analysis window: {exc}")
                QtWidgets.QMessageBox.critical(
                    self, "Analysis", f"Failed to open analysis window:\n{exc}"
                )
                self.analysis_window = None
                return
        try:
            if self._run_folder and self._run_folder.exists():
                self.analysis_window.set_run_folder(self._run_folder)
            elif self.output_folder and Path(self.output_folder).is_dir():
                self.analysis_window.set_run_folder(Path(self.output_folder))
        except Exception as exc:
            logging.warning(f"Unable to update analysis window folder: {exc}")
        self.analysis_window.show()
        self.analysis_window.raise_()
        self.analysis_window.activateWindow()

    def _open_stage_scan_window(self):
        if self.stage_window is None:
            try:
                self.stage_window = StageScanWindow(self)
            except Exception as exc:
                logging.error(f"Failed to create stage scan window: {exc}")
                QtWidgets.QMessageBox.critical(
                    self, "Stage Scan", f"Failed to open stage scan window:\n{exc}"
                )
                self.stage_window = None
                return
        self.stage_window.show()
        self.stage_window.raise_()
        self.stage_window.activateWindow()

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
