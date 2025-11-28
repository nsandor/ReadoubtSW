#!/usr/bin/env python3
# main.py — Readout app with PySide6 UI (from pyside6-uic)

import json
import logging
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
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
from Drivers.stage_driver import (
    NewportRotationStageAdapter,
    NewportXPSStageDriver,
    NewportXYStageAdapter,
)
from Readoubt_ui import Ui_MainWindow
from Scanner import ScanWorker
from ScannerWin.acquisition import AcquisitionSettings
from Analysis.analysis_window import AnalysisWindow
from StageScan.stage_controller_window import StageControllerWindow
from StageScan.stage_scan_window import StageScanWindow
from characterization import CharacterizationSettings, CharacterizationWorker


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


class CharacterizationSettingsDialog(QtWidgets.QDialog):
    """Dialog to expose characterization suite options."""

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        settings: CharacterizationSettings,
        pixel_spec: str,
    ):
        super().__init__(parent)
        self.setWindowTitle("Characterization Settings")
        self.setModal(True)
        self.resize(700, 720)
        layout = QtWidgets.QVBoxLayout(self)

        def _maybe_scientific(spin: QtWidgets.QDoubleSpinBox):
            try:
                # Older PySide builds may not expose setNotation; ignore if unavailable.
                spin.setNotation(QtWidgets.QDoubleSpinBox.Notation.ScientificNotation)
            except Exception:
                try:
                    spin.setProperty("notation", QtWidgets.QDoubleSpinBox.Notation.ScientificNotation)
                except Exception:
                    pass

        def _config_nano_spin(
            spin: QtWidgets.QDoubleSpinBox,
            value_a: Optional[float],
            *,
            allow_zero: bool = False,
            single_step: float = 1.0,
        ):
            spin.setDecimals(6)
            spin.setSuffix(" nA")
            spin.setRange(0.0 if allow_zero else 1e-3, 1e9)
            spin.setSingleStep(single_step)
            _maybe_scientific(spin)
            try:
                spin.setValue(float(value_a) * 1e9 if value_a is not None else 0.0)
            except Exception:
                spin.setValue(0.0)

        def _set_nano_value(spin: QtWidgets.QDoubleSpinBox, value_a: Optional[float]):
            try:
                spin.setValue(float(value_a) * 1e9 if value_a is not None else 0.0)
            except Exception:
                spin.setValue(0.0)

        tabs = QtWidgets.QTabWidget(self)
        layout.addWidget(tabs)

        # General
        general_box = QtWidgets.QGroupBox("General")
        general_form = QtWidgets.QFormLayout(general_box)
        self.pixel_spec_edit = QtWidgets.QLineEdit(pixel_spec)
        general_form.addRow("Pixels:", self.pixel_spec_edit)

        self.samples_spin = QtWidgets.QSpinBox()
        self.samples_spin.setRange(1, 100)
        self.samples_spin.setValue(int(settings.samples_per_pixel))
        general_form.addRow("Samples per pixel:", self.samples_spin)

        self.nplc_spin = QtWidgets.QDoubleSpinBox()
        self.nplc_spin.setDecimals(3)
        self.nplc_spin.setRange(0.01, 25.0)
        self.nplc_spin.setValue(float(settings.nplc))
        general_form.addRow("NPLC:", self.nplc_spin)

        self.auto_range_check = QtWidgets.QCheckBox("Auto current range")
        self.auto_range_check.setChecked(settings.auto_range)
        general_form.addRow(self.auto_range_check)

        self.current_range_spin = QtWidgets.QDoubleSpinBox()
        _config_nano_spin(self.current_range_spin, settings.current_range)
        self.current_range_spin.setEnabled(not settings.auto_range)
        general_form.addRow("Manual range (nA):", self.current_range_spin)
        self.auto_range_check.toggled.connect(
            lambda checked: self.current_range_spin.setEnabled(not checked)
        )

        self.local_readout_check = QtWidgets.QCheckBox("Use switch board local readout")
        self.local_readout_check.setChecked(settings.use_local_readout)
        general_form.addRow(self.local_readout_check)

        self.local_bias_check = QtWidgets.QCheckBox("Use switch board local bias")
        self.local_bias_check.setChecked(settings.use_local_bias)
        general_form.addRow(self.local_bias_check)

        self.output_subdir_edit = QtWidgets.QLineEdit(settings.output_subdir)
        general_form.addRow("Output subfolder:", self.output_subdir_edit)

        self.route_settle_spin = QtWidgets.QSpinBox()
        self.route_settle_spin.setRange(0, 5000)
        self.route_settle_spin.setValue(int(settings.route_settle_ms))
        general_form.addRow("Routing settle (ms):", self.route_settle_spin)

        # Shorted pixel test
        short_box = QtWidgets.QGroupBox("Shorted pixel test")
        short_form = QtWidgets.QFormLayout(short_box)
        self.short_threshold_spin = QtWidgets.QDoubleSpinBox()
        _config_nano_spin(self.short_threshold_spin, settings.short_threshold_a, single_step=10.0)
        short_form.addRow("Short threshold (nA):", self.short_threshold_spin)

        pin_row = QtWidgets.QHBoxLayout()
        self.pin_map_edit = QtWidgets.QLineEdit(settings.pin_map_path or "")
        browse_pin = QtWidgets.QPushButton("Browse…")
        browse_pin.clicked.connect(self._browse_pin_map)
        pin_row.addWidget(self.pin_map_edit, 1)
        pin_row.addWidget(browse_pin)
        pin_widget = QtWidgets.QWidget()
        pin_widget.setLayout(pin_row)
        short_form.addRow("Pixel→pin map (optional):", pin_widget)

        # Dead short test
        dead_box = QtWidgets.QGroupBox("Dead short test")
        dead_form = QtWidgets.QFormLayout(dead_box)
        self.dead_voltage_spin = QtWidgets.QDoubleSpinBox()
        self.dead_voltage_spin.setDecimals(3)
        self.dead_voltage_spin.setRange(0.0, 10.0)
        self.dead_voltage_spin.setSingleStep(0.01)
        self.dead_voltage_spin.setValue(float(settings.dead_short_voltage_v))
        dead_form.addRow("Bias voltage (V):", self.dead_voltage_spin)

        self.dead_threshold_spin = QtWidgets.QDoubleSpinBox()
        _config_nano_spin(self.dead_threshold_spin, settings.dead_short_threshold_a, single_step=10.0)
        dead_form.addRow("Dead short threshold (nA):", self.dead_threshold_spin)

        self.stop_dead_check = QtWidgets.QCheckBox("Stop suite if dead shorts found")
        self.stop_dead_check.setChecked(settings.stop_on_dead_short)
        dead_form.addRow(self.stop_dead_check)

        # Resistance JV
        res_box = QtWidgets.QGroupBox("Resistance JV sweep")
        res_form = QtWidgets.QFormLayout(res_box)
        self.res_start_spin = QtWidgets.QDoubleSpinBox()
        self.res_start_spin.setDecimals(3)
        self.res_start_spin.setRange(-200.0, 200.0)
        self.res_start_spin.setSingleStep(0.05)
        self.res_start_spin.setValue(float(settings.resistance_start_v))
        res_form.addRow("Start voltage (V):", self.res_start_spin)

        self.res_end_spin = QtWidgets.QDoubleSpinBox()
        self.res_end_spin.setDecimals(3)
        self.res_end_spin.setRange(-200.0, 200.0)
        self.res_end_spin.setSingleStep(0.05)
        self.res_end_spin.setValue(float(settings.resistance_end_v))
        res_form.addRow("End voltage (V):", self.res_end_spin)

        self.res_step_spin = QtWidgets.QDoubleSpinBox()
        self.res_step_spin.setDecimals(3)
        self.res_step_spin.setRange(0.001, 200.0)
        self.res_step_spin.setSingleStep(0.01)
        self.res_step_spin.setValue(float(settings.resistance_step_v))
        res_form.addRow("Step (V):", self.res_step_spin)

        self.res_settle_spin = QtWidgets.QDoubleSpinBox()
        self.res_settle_spin.setDecimals(3)
        self.res_settle_spin.setRange(0.0, 600.0)
        self.res_settle_spin.setSingleStep(0.1)
        self.res_settle_spin.setValue(float(settings.resistance_settle_s))
        res_form.addRow("Settle time (s):", self.res_settle_spin)

        # Dark current @ operating bias
        op_box = QtWidgets.QGroupBox("Dark current @ operating bias")
        op_form = QtWidgets.QFormLayout(op_box)
        self.operating_field_spin = QtWidgets.QDoubleSpinBox()
        self.operating_field_spin.setDecimals(2)
        self.operating_field_spin.setRange(0.0, 1_000.0)
        self.operating_field_spin.setSingleStep(5.0)
        self.operating_field_spin.setValue(float(settings.operating_field_v_per_cm))
        op_form.addRow("Field (V/cm):", self.operating_field_spin)

        self.operating_thickness_spin = QtWidgets.QDoubleSpinBox()
        self.operating_thickness_spin.setDecimals(3)
        self.operating_thickness_spin.setRange(0.001, 1000.0)
        self.operating_thickness_spin.setSingleStep(0.01)
        self.operating_thickness_spin.setValue(float(settings.device_thickness_cm))
        op_form.addRow("Thickness (cm):", self.operating_thickness_spin)

        self.operating_settle_spin = QtWidgets.QDoubleSpinBox()
        self.operating_settle_spin.setDecimals(1)
        self.operating_settle_spin.setRange(0.0, 3600.0)
        self.operating_settle_spin.setSingleStep(1.0)
        self.operating_settle_spin.setValue(float(settings.operating_settle_s))
        op_form.addRow("Settle time (s):", self.operating_settle_spin)

        self.operating_limit_spin = QtWidgets.QDoubleSpinBox()
        _config_nano_spin(self.operating_limit_spin, settings.operating_current_limit_a, allow_zero=True, single_step=100.0)
        self.operating_limit_spin.setSpecialValueText("Disabled")
        op_form.addRow("Current limit (nA):", self.operating_limit_spin)

        # Wide dark JV
        wide_box = QtWidgets.QGroupBox("Wide dark JV")
        wide_form = QtWidgets.QFormLayout(wide_box)
        self.wide_start_spin = QtWidgets.QDoubleSpinBox()
        self.wide_start_spin.setDecimals(3)
        self.wide_start_spin.setRange(-300.0, 300.0)
        self.wide_start_spin.setSingleStep(1.0)
        self.wide_start_spin.setValue(float(settings.jv_dark_start_v))
        wide_form.addRow("Start voltage (V):", self.wide_start_spin)

        self.wide_end_spin = QtWidgets.QDoubleSpinBox()
        self.wide_end_spin.setDecimals(3)
        self.wide_end_spin.setRange(-300.0, 300.0)
        self.wide_end_spin.setSingleStep(1.0)
        self.wide_end_spin.setValue(float(settings.jv_dark_end_v))
        wide_form.addRow("End voltage (V):", self.wide_end_spin)

        self.wide_step_spin = QtWidgets.QDoubleSpinBox()
        self.wide_step_spin.setDecimals(3)
        self.wide_step_spin.setRange(0.001, 300.0)
        self.wide_step_spin.setSingleStep(0.5)
        self.wide_step_spin.setValue(float(settings.jv_dark_step_v))
        wide_form.addRow("Step (V):", self.wide_step_spin)

        self.wide_zero_pause_spin = QtWidgets.QDoubleSpinBox()
        self.wide_zero_pause_spin.setDecimals(2)
        self.wide_zero_pause_spin.setRange(0.0, 600.0)
        self.wide_zero_pause_spin.setSingleStep(0.1)
        self.wide_zero_pause_spin.setValue(float(settings.jv_dark_zero_pause_s))
        wide_form.addRow("0 V dwell (s):", self.wide_zero_pause_spin)

        self.wide_settle_spin = QtWidgets.QDoubleSpinBox()
        self.wide_settle_spin.setDecimals(2)
        self.wide_settle_spin.setRange(0.0, 600.0)
        self.wide_settle_spin.setSingleStep(0.1)
        self.wide_settle_spin.setValue(float(settings.jv_dark_settle_s))
        wide_form.addRow("Settle time (s):", self.wide_settle_spin)

        self.wide_current_limit_spin = QtWidgets.QDoubleSpinBox()
        _config_nano_spin(self.wide_current_limit_spin, settings.jv_dark_current_limit_a, allow_zero=True, single_step=100.0)
        self.wide_current_limit_spin.setSpecialValueText("Disabled")
        wide_form.addRow("Current limit (nA):", self.wide_current_limit_spin)

        self.wide_zero_center_check = QtWidgets.QCheckBox("Zero-centered sweep")
        self.wide_zero_center_check.setChecked(settings.jv_dark_zero_center)
        wide_form.addRow(self.wide_zero_center_check)

        # Light JT
        jt_box = QtWidgets.QGroupBox("Light JT")
        jt_form = QtWidgets.QFormLayout(jt_box)
        self.jt_bias_spin = QtWidgets.QDoubleSpinBox()
        self.jt_bias_spin.setDecimals(3)
        self.jt_bias_spin.setRange(-300.0, 300.0)
        self.jt_bias_spin.setSingleStep(1.0)
        self.jt_bias_spin.setValue(float(settings.jt_light_bias_v))
        jt_form.addRow("Bias (V):", self.jt_bias_spin)

        self.jt_settle_spin = QtWidgets.QDoubleSpinBox()
        self.jt_settle_spin.setDecimals(2)
        self.jt_settle_spin.setRange(0.0, 600.0)
        self.jt_settle_spin.setSingleStep(0.1)
        self.jt_settle_spin.setValue(float(settings.jt_light_settle_s))
        jt_form.addRow("Settle time (s):", self.jt_settle_spin)

        self.jt_samples_spin = QtWidgets.QSpinBox()
        self.jt_samples_spin.setRange(1, 100)
        self.jt_samples_spin.setValue(int(settings.jt_light_samples_per_pixel))
        jt_form.addRow("Samples per pixel:", self.jt_samples_spin)

        self.jt_threshold_spin = QtWidgets.QDoubleSpinBox()
        _config_nano_spin(self.jt_threshold_spin, settings.jt_light_threshold_a, single_step=1.0)
        jt_form.addRow("Active threshold (nA):", self.jt_threshold_spin)

        self.jt_led_check = QtWidgets.QCheckBox("Enable switch-board LEDs")
        self.jt_led_check.setChecked(settings.jt_light_use_led)
        jt_form.addRow(self.jt_led_check)

        self.jt_limit_spin = QtWidgets.QDoubleSpinBox()
        _config_nano_spin(self.jt_limit_spin, settings.jt_light_current_limit_a, allow_zero=True, single_step=100.0)
        self.jt_limit_spin.setSpecialValueText("Disabled")
        jt_form.addRow("Current limit (nA):", self.jt_limit_spin)

        # Analysis
        analysis_box = QtWidgets.QGroupBox("Analysis / plots")
        analysis_form = QtWidgets.QFormLayout(analysis_box)
        self.analysis_active_spin = QtWidgets.QDoubleSpinBox()
        _config_nano_spin(self.analysis_active_spin, settings.analysis_active_threshold_a, single_step=1.0)
        analysis_form.addRow("Active pixel threshold (nA):", self.analysis_active_spin)
        self.hist_bins_spin = QtWidgets.QSpinBox()
        self.hist_bins_spin.setRange(5, 500)
        self.hist_bins_spin.setValue(int(settings.histogram_bins))
        analysis_form.addRow("Histogram bins:", self.hist_bins_spin)
        self.hist_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.hist_sigma_spin.setDecimals(2)
        self.hist_sigma_spin.setRange(0.0, 10.0)
        self.hist_sigma_spin.setSingleStep(0.1)
        self.hist_sigma_spin.setValue(float(settings.histogram_sigma_clip))
        analysis_form.addRow("Outlier clip (σ, 0=off):", self.hist_sigma_spin)
        cmap_opts = ["viridis", "inferno", "plasma", "magma", "cividis", "turbo"]
        self.res_cmap_combo = QtWidgets.QComboBox()
        self.res_cmap_combo.addItems(cmap_opts)
        try:
            idx = cmap_opts.index(settings.heatmap_cmap_resistance)
            self.res_cmap_combo.setCurrentIndex(idx)
        except ValueError:
            pass
        analysis_form.addRow("Resistance heatmap cmap:", self.res_cmap_combo)
        self.dark_cmap_combo = QtWidgets.QComboBox()
        self.dark_cmap_combo.addItems(cmap_opts)
        try:
            idx = cmap_opts.index(settings.heatmap_cmap_dark)
            self.dark_cmap_combo.setCurrentIndex(idx)
        except ValueError:
            pass
        analysis_form.addRow("Dark heatmap cmap:", self.dark_cmap_combo)
        self.units_combo = QtWidgets.QComboBox()
        self.units_combo.addItems(["A", "mA", "µA", "nA", "pA"])
        try:
            idx = self.units_combo.findText(settings.plot_units_current)
            if idx >= 0:
                self.units_combo.setCurrentIndex(idx)
        except Exception:
            pass
        analysis_form.addRow("Current plot units:", self.units_combo)
        self.vmin_edit = QtWidgets.QLineEdit(
            "" if settings.plot_vmin_current is None else str(settings.plot_vmin_current)
        )
        self.vmax_edit = QtWidgets.QLineEdit(
            "" if settings.plot_vmax_current is None else str(settings.plot_vmax_current)
        )
        analysis_form.addRow("Dark heatmap min (unit):", self.vmin_edit)
        analysis_form.addRow("Dark heatmap max (unit):", self.vmax_edit)
        self.res_units_combo = QtWidgets.QComboBox()
        self.res_units_combo.addItems(["Ohm", "kOhm", "MOhm", "GOhm"])
        try:
            idx = self.res_units_combo.findText(settings.plot_units_resistance)
            if idx >= 0:
                self.res_units_combo.setCurrentIndex(idx)
        except Exception:
            pass
        analysis_form.addRow("Resistance units:", self.res_units_combo)
        self.res_vmin_edit = QtWidgets.QLineEdit(
            "" if settings.plot_vmin_resistance is None else str(settings.plot_vmin_resistance)
        )
        self.res_vmax_edit = QtWidgets.QLineEdit(
            "" if settings.plot_vmax_resistance is None else str(settings.plot_vmax_resistance)
        )
        analysis_form.addRow("Resistance heatmap min:", self.res_vmin_edit)
        analysis_form.addRow("Resistance heatmap max:", self.res_vmax_edit)

        # Tabs assembly to reduce required window height
        tab_general = QtWidgets.QWidget()
        tab_general_layout = QtWidgets.QVBoxLayout(tab_general)
        tab_general_layout.addWidget(general_box)
        tab_general_layout.addStretch(1)
        tabs.addTab(tab_general, "General")

        tab_screen = QtWidgets.QWidget()
        tab_screen_layout = QtWidgets.QVBoxLayout(tab_screen)
        tab_screen_layout.addWidget(short_box)
        tab_screen_layout.addWidget(dead_box)
        tab_screen_layout.addStretch(1)
        tabs.addTab(tab_screen, "Screening")

        tab_resistance = QtWidgets.QWidget()
        tab_res_layout = QtWidgets.QVBoxLayout(tab_resistance)
        tab_res_layout.addWidget(res_box)
        tab_res_layout.addWidget(op_box)
        tab_res_layout.addStretch(1)
        tabs.addTab(tab_resistance, "DC / Resistance")

        tab_wide = QtWidgets.QWidget()
        tab_wide_layout = QtWidgets.QVBoxLayout(tab_wide)
        tab_wide_layout.addWidget(wide_box)
        tab_wide_layout.addStretch(1)
        tabs.addTab(tab_wide, "Wide Dark JV")

        tab_light = QtWidgets.QWidget()
        tab_light_layout = QtWidgets.QVBoxLayout(tab_light)
        tab_light_layout.addWidget(jt_box)
        tab_light_layout.addStretch(1)
        tabs.addTab(tab_light, "Light JT")

        tab_analysis = QtWidgets.QWidget()
        tab_analysis_layout = QtWidgets.QVBoxLayout(tab_analysis)
        tab_analysis_layout.addWidget(analysis_box)
        tab_analysis_layout.addStretch(1)
        tabs.addTab(tab_analysis, "Analysis")

        # Save/load buttons
        button_row = QtWidgets.QHBoxLayout()
        self.btn_load_settings = QtWidgets.QPushButton("Load settings…")
        self.btn_save_settings = QtWidgets.QPushButton("Save settings…")
        button_row.addWidget(self.btn_load_settings)
        button_row.addWidget(self.btn_save_settings)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.btn_load_settings.clicked.connect(self._handle_load_settings)
        self.btn_save_settings.clicked.connect(self._handle_save_settings)

    def _browse_pin_map(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select pin map", "", "JSON/CSV (*.json *.csv *.txt);;All files (*)"
        )
        if fname:
            self.pin_map_edit.setText(fname)

    def _handle_load_settings(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load characterization settings", "", "JSON files (*.json);;All files (*)"
        )
        if not fname:
            return
        try:
            data = json.loads(Path(fname).read_text(encoding="utf-8"))
            base = CharacterizationSettings()
            merged = {**base.__dict__, **data} if isinstance(data, dict) else base.__dict__
            loaded = CharacterizationSettings(**merged)
            self._apply_settings_to_fields(loaded)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load settings", f"Failed to load settings:\n{exc}")

    def _handle_save_settings(self):
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save characterization settings", "characterization_settings.json", "JSON files (*.json);;All files (*)"
        )
        if not fname:
            return
        try:
            settings = self.get_settings()
            Path(fname).write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save settings", f"Failed to save settings:\n{exc}")

    def _apply_settings_to_fields(self, s: CharacterizationSettings):
        self.pixel_spec_edit.setText(s.pixel_spec)
        self.samples_spin.setValue(int(s.samples_per_pixel))
        self.nplc_spin.setValue(float(s.nplc))
        self.auto_range_check.setChecked(bool(s.auto_range))
        self.current_range_spin.setValue(max(0.0, float(s.current_range) * 1e9))
        self.current_range_spin.setEnabled(not s.auto_range)
        self.local_readout_check.setChecked(bool(s.use_local_readout))
        self.local_bias_check.setChecked(bool(s.use_local_bias))
        self.output_subdir_edit.setText(s.output_subdir)
        self.route_settle_spin.setValue(int(s.route_settle_ms))

        self.short_threshold_spin.setValue(max(0.0, float(s.short_threshold_a) * 1e9))
        self.pin_map_edit.setText(s.pin_map_path or "")

        self.dead_voltage_spin.setValue(float(s.dead_short_voltage_v))
        self.dead_threshold_spin.setValue(max(0.0, float(s.dead_short_threshold_a) * 1e9))
        self.stop_dead_check.setChecked(bool(s.stop_on_dead_short))

        self.res_start_spin.setValue(float(s.resistance_start_v))
        self.res_end_spin.setValue(float(s.resistance_end_v))
        self.res_step_spin.setValue(float(s.resistance_step_v))
        self.res_settle_spin.setValue(float(s.resistance_settle_s))

        self.operating_field_spin.setValue(float(s.operating_field_v_per_cm))
        self.operating_thickness_spin.setValue(float(s.device_thickness_cm))
        self.operating_settle_spin.setValue(float(s.operating_settle_s))
        self.operating_limit_spin.setValue(max(0.0, float(s.operating_current_limit_a or 0.0) * 1e9))

        self.wide_start_spin.setValue(float(s.jv_dark_start_v))
        self.wide_end_spin.setValue(float(s.jv_dark_end_v))
        self.wide_step_spin.setValue(float(s.jv_dark_step_v))
        self.wide_zero_pause_spin.setValue(float(s.jv_dark_zero_pause_s))
        self.wide_settle_spin.setValue(float(s.jv_dark_settle_s))
        self.wide_current_limit_spin.setValue(max(0.0, float(s.jv_dark_current_limit_a or 0.0) * 1e9))
        self.wide_zero_center_check.setChecked(bool(s.jv_dark_zero_center))

        self.jt_bias_spin.setValue(float(s.jt_light_bias_v))
        self.jt_samples_spin.setValue(int(s.jt_light_samples_per_pixel))
        self.jt_threshold_spin.setValue(max(0.0, float(s.jt_light_threshold_a) * 1e9))
        self.jt_led_check.setChecked(bool(s.jt_light_use_led))
        self.jt_limit_spin.setValue(max(0.0, float(s.jt_light_current_limit_a or 0.0) * 1e9))
        self.jt_settle_spin.setValue(float(s.jt_light_settle_s))

        self.analysis_active_spin.setValue(max(0.0, float(s.analysis_active_threshold_a) * 1e9))
        self.hist_bins_spin.setValue(int(s.histogram_bins))
        self.hist_sigma_spin.setValue(float(s.histogram_sigma_clip))
        cmap_opts = [self.res_cmap_combo.itemText(i) for i in range(self.res_cmap_combo.count())]
        if s.heatmap_cmap_resistance in cmap_opts:
            self.res_cmap_combo.setCurrentIndex(cmap_opts.index(s.heatmap_cmap_resistance))
        cmap_opts_dark = [self.dark_cmap_combo.itemText(i) for i in range(self.dark_cmap_combo.count())]
        if s.heatmap_cmap_dark in cmap_opts_dark:
            self.dark_cmap_combo.setCurrentIndex(cmap_opts_dark.index(s.heatmap_cmap_dark))
        idx_units = self.units_combo.findText(s.plot_units_current)
        if idx_units >= 0:
            self.units_combo.setCurrentIndex(idx_units)
        self.vmin_edit.setText("" if s.plot_vmin_current is None else str(s.plot_vmin_current))
        self.vmax_edit.setText("" if s.plot_vmax_current is None else str(s.plot_vmax_current))
        idx_res_units = self.res_units_combo.findText(s.plot_units_resistance)
        if idx_res_units >= 0:
            self.res_units_combo.setCurrentIndex(idx_res_units)
        self.res_vmin_edit.setText("" if s.plot_vmin_resistance is None else str(s.plot_vmin_resistance))
        self.res_vmax_edit.setText("" if s.plot_vmax_resistance is None else str(s.plot_vmax_resistance))
    @staticmethod
    def _spin_value_or_none(spin: QtWidgets.QDoubleSpinBox, scale: float = 1.0) -> Optional[float]:
        try:
            value = float(spin.value()) * scale
        except Exception:
            return None
        return None if value <= 0 else value

    def get_settings(self) -> CharacterizationSettings:
        s = CharacterizationSettings()
        s.pixel_spec = (self.pixel_spec_edit.text() or "1-100").strip()
        s.samples_per_pixel = int(self.samples_spin.value())
        s.nplc = float(self.nplc_spin.value())
        s.auto_range = bool(self.auto_range_check.isChecked())
        s.current_range = float(self.current_range_spin.value()) * 1e-9
        s.use_local_readout = bool(self.local_readout_check.isChecked())
        s.use_local_bias = bool(self.local_bias_check.isChecked())
        s.output_subdir = (self.output_subdir_edit.text() or s.output_subdir).strip()
        s.route_settle_ms = int(self.route_settle_spin.value())

        s.short_threshold_a = float(self.short_threshold_spin.value()) * 1e-9
        pin_text = (self.pin_map_edit.text() or "").strip()
        s.pin_map_path = pin_text or None

        s.dead_short_voltage_v = float(self.dead_voltage_spin.value())
        s.dead_short_threshold_a = float(self.dead_threshold_spin.value()) * 1e-9
        s.stop_on_dead_short = bool(self.stop_dead_check.isChecked())

        s.resistance_start_v = float(self.res_start_spin.value())
        s.resistance_end_v = float(self.res_end_spin.value())
        s.resistance_step_v = float(self.res_step_spin.value())
        s.resistance_settle_s = float(self.res_settle_spin.value())

        s.operating_field_v_per_cm = float(self.operating_field_spin.value())
        s.device_thickness_cm = float(self.operating_thickness_spin.value())
        s.operating_settle_s = float(self.operating_settle_spin.value())
        s.operating_current_limit_a = self._spin_value_or_none(self.operating_limit_spin, 1e-9)

        s.jv_dark_start_v = float(self.wide_start_spin.value())
        s.jv_dark_end_v = float(self.wide_end_spin.value())
        s.jv_dark_step_v = float(self.wide_step_spin.value())
        s.jv_dark_zero_pause_s = float(self.wide_zero_pause_spin.value())
        s.jv_dark_settle_s = float(self.wide_settle_spin.value())
        s.jv_dark_current_limit_a = self._spin_value_or_none(self.wide_current_limit_spin, 1e-9)
        s.jv_dark_zero_center = bool(self.wide_zero_center_check.isChecked())

        s.jt_light_bias_v = float(self.jt_bias_spin.value())
        s.jt_light_samples_per_pixel = int(self.jt_samples_spin.value())
        s.jt_light_threshold_a = float(self.jt_threshold_spin.value()) * 1e-9
        s.jt_light_use_led = bool(self.jt_led_check.isChecked())
        s.jt_light_current_limit_a = self._spin_value_or_none(self.jt_limit_spin, 1e-9)
        s.jt_light_settle_s = float(self.jt_settle_spin.value())

        s.analysis_active_threshold_a = float(self.analysis_active_spin.value()) * 1e-9
        s.histogram_bins = int(self.hist_bins_spin.value())
        s.histogram_sigma_clip = float(self.hist_sigma_spin.value())
        s.heatmap_cmap_resistance = self.res_cmap_combo.currentText()
        s.heatmap_cmap_dark = self.dark_cmap_combo.currentText()
        s.plot_units_current = self.units_combo.currentText()
        s.plot_units_resistance = self.res_units_combo.currentText()
        try:
            s.plot_vmin_current = float(self.vmin_edit.text()) if self.vmin_edit.text().strip() else None
        except Exception:
            s.plot_vmin_current = None
        try:
            s.plot_vmax_current = float(self.vmax_edit.text()) if self.vmax_edit.text().strip() else None
        except Exception:
            s.plot_vmax_current = None
        try:
            s.plot_vmin_resistance = float(self.res_vmin_edit.text()) if self.res_vmin_edit.text().strip() else None
        except Exception:
            s.plot_vmin_resistance = None
        try:
            s.plot_vmax_resistance = float(self.res_vmax_edit.text()) if self.res_vmax_edit.text().strip() else None
        except Exception:
            s.plot_vmax_resistance = None
        return s


class CharacterizationProgressDialog(QtWidgets.QDialog):
    """Lightweight progress popup with step breakdown."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Characterization Progress")
        self.setModal(False)
        self.setMinimumWidth(420)
        layout = QtWidgets.QVBoxLayout(self)
        self.overall_bar = QtWidgets.QProgressBar()
        self.overall_bar.setFormat("Overall: %p% (%v/%m)")
        layout.addWidget(self.overall_bar)
        self.steps_container = QtWidgets.QWidget()
        self.steps_layout = QtWidgets.QVBoxLayout(self.steps_container)
        self.steps_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.steps_container)
        self._step_bars: dict[str, QtWidgets.QProgressBar] = {}

    def set_steps(self, plan: object):
        # plan: list of (key, label, total)
        # Clear existing
        while self.steps_layout.count():
            item = self.steps_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self._step_bars.clear()
        if not isinstance(plan, (list, tuple)):
            return
        for entry in plan:
            try:
                key, label, total = entry
            except Exception:
                continue
            row = QtWidgets.QHBoxLayout()
            lbl = QtWidgets.QLabel(str(label))
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, max(1, int(total)))
            bar.setValue(0)
            bar.setFormat("%p% (%v/%m)")
            row.addWidget(lbl)
            row.addWidget(bar, 1)
            wrapper = QtWidgets.QWidget()
            wrapper.setLayout(row)
            self.steps_layout.addWidget(wrapper)
            self._step_bars[str(key)] = bar
        self.steps_layout.addStretch(1)

    def update_step(self, key: str, done: int, total: int):
        bar = self._step_bars.get(str(key))
        if not bar:
            return
        total = max(1, int(total))
        done = max(0, min(int(done), total))
        bar.setRange(0, total)
        bar.setValue(done)

    def update_overall(self, done: int, total: int, text: str = ""):
        total = max(1, int(total))
        done = max(0, min(int(done), total))
        self.overall_bar.setRange(0, total)
        self.overall_bar.setValue(done)
        if text:
            self.overall_bar.setFormat(f"{text} – %p% (%v/%m)")

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # ---- load UI ----
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Readoubt")
        self.analysis_window: Optional[AnalysisWindow] = None
        self.stage_window: Optional[StageScanWindow] = None
        self.stage_controller_window: Optional[StageControllerWindow] = None
        self.menu_analysis = self.ui.menuBar.addMenu("Analysis")
        self.action_open_analysis = QtGui.QAction("Open Analysis Window", self)
        self.menu_analysis.addAction(self.action_open_analysis)
        self.action_open_analysis.triggered.connect(self._open_analysis_window)
        self.menu_stage = self.ui.menuBar.addMenu("Stage")
        self.action_open_stage = QtGui.QAction("Open Stage Scan Window", self)
        self.menu_stage.addAction(self.action_open_stage)
        self.action_open_stage.triggered.connect(self._open_stage_scan_window)
        self.action_open_stage_controller = QtGui.QAction("Open Stage Controller", self)
        self.menu_stage.addAction(self.action_open_stage_controller)
        self.action_open_stage_controller.triggered.connect(self._open_stage_controller_window)

        self.menu_characterization = self.ui.menuBar.addMenu("Characterization")
        self.action_run_characterization = QtGui.QAction("Run Characterization Suite", self)
        self.action_characterization_settings = QtGui.QAction("Characterization Settings…", self)
        self.action_abort_characterization = QtGui.QAction("Abort Characterization", self)
        self.menu_characterization.addAction(self.action_run_characterization)
        self.menu_characterization.addAction(self.action_characterization_settings)
        self.menu_characterization.addAction(self.action_abort_characterization)
        self.action_run_characterization.triggered.connect(self._start_characterization_suite)
        self.action_characterization_settings.triggered.connect(self._open_characterization_settings)
        self.action_abort_characterization.triggered.connect(self._abort_characterization_suite)
        self.action_abort_characterization.setEnabled(False)

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
        self.ref_matrix2: Optional[np.ndarray] = None
        self.ref_path2: Optional[Path] = None
        self.math_mode: str = "none"  # none|divide|subtract
        self.math_mode2: str = "none"
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
        self._voltage_zero_center: bool = False
        self._voltage_zero_pause: float = 0.0
        self._voltage_settle_s: float = 0.0
        self._voltage_control_widgets: List[QtWidgets.QWidget] = []
        self._loop_control_widgets: List[QtWidgets.QWidget] = []
        self._time_bias_widgets: List[QtWidgets.QWidget] = []
        self._switch_settle_ms: int = 4
        self.autosave_enabled: bool = True
        self.device_logger = DEVICE_LOGGER
        self._loop_progress_total: int = 0
        self._scan_pixel_count: int = 0
        self._current_limit: Optional[float] = None
        self._excluded_pixels: set[int] = set()
        self._scan_loop_count: int = 0
        current_range_text = (self.ui.edit_current_range.text() or "").strip()
        try:
            current_range_value = float(current_range_text) if current_range_text else 1e-7
        except Exception:
            current_range_value = 1e-7
        thickness_default = (
            float(self.ui.spin_bias_thickness.value())
            if hasattr(self.ui, "spin_bias_thickness")
            else 1.0
        )
        self._characterization_settings = CharacterizationSettings(
            pixel_spec=self.ui.edit_pixel_spec.text(),
            samples_per_pixel=int(self.ui.spin_nsamp.value()),
            nplc=float(self.ui.spin_nplc.value()),
            auto_range=self.ui.check_auto_current_range.isChecked(),
            current_range=float(current_range_value),
            use_local_readout=self._using_local_readout(),
            use_local_bias=self._using_local_bias(),
            device_thickness_cm=thickness_default,
            route_settle_ms=self._switch_settle_ms,
            jt_light_settle_s=0.0,
        )
        self._characterization_thread: Optional[QtCore.QThread] = None
        self._characterization_worker: Optional[CharacterizationWorker] = None
        self._characterization_progress_dialog: Optional[CharacterizationProgressDialog] = None

        # ---- instruments (dummy by default) ----
        self.sm = DummyKeithley2400()
        self.bias_sm = DummyBias2400()
        self.switch = DummySwitchBoard()
        self.stage_driver = NewportXPSStageDriver()
        self.xy_stage = NewportXYStageAdapter(self.stage_driver)
        self.rotation_stage = NewportRotationStageAdapter(self.stage_driver)
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
        self.ui.label_current_limit = QtWidgets.QLabel("Current limit (A):")
        self.ui.label_current_limit.setObjectName("label_current_limit")
        self.ui.edit_current_limit = QtWidgets.QLineEdit()
        self.ui.edit_current_limit.setObjectName("edit_current_limit")
        self.ui.edit_current_limit.setPlaceholderText("Leave blank to disable")
        row_idx = self.ui.formLayout_2.rowCount()
        self.ui.formLayout_2.insertRow(
            row_idx, self.ui.label_current_limit, self.ui.edit_current_limit
        )
        self._update_integration_time_label()
        self._characterization_progress = QtWidgets.QProgressBar()
        self._characterization_progress.setMaximumWidth(260)
        self._characterization_progress.setFormat("Characterization: %p%")
        self._characterization_progress.setVisible(False)
        self.statusBar().addPermanentWidget(self._characterization_progress)
        self._loop_control_widgets = [
            self.ui.label_loops,
            self.ui.spin_loops,
            self.ui.loop_delay_label,
            self.ui.loop_delay,
        ]
        if hasattr(self.ui, "LoopProgressBar"):
            self.ui.LoopProgressBar.setRange(0, 1)
            self.ui.LoopProgressBar.setValue(0)
            self.ui.LoopProgressBar.setFormat("Loop progress: %p%")

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

        settle_index = self.ui.formLayout_2.indexOf(self.ui.label_voltage_settle)
        settle_row = (
            self.ui.formLayout_2.getItemPosition(settle_index)[0]
            if settle_index >= 0
            else self.ui.formLayout_2.rowCount()
        )
        self.ui.check_voltage_zero_center = QtWidgets.QCheckBox(
            "Zero-centered sweep (start at 0 V)"
        )
        self.ui.check_voltage_zero_center.setObjectName("check_voltage_zero_center")
        self.ui.formLayout_2.insertRow(settle_row + 1, self.ui.check_voltage_zero_center)

        self.ui.check_voltage_pulse = QtWidgets.QCheckBox(
            "Pulsed JV (return to 0 V between steps)"
        )
        self.ui.check_voltage_pulse.setObjectName("check_voltage_pulse")
        self.ui.formLayout_2.insertRow(settle_row + 2, self.ui.check_voltage_pulse)

        self.ui.label_voltage_zero_pause = QtWidgets.QLabel("0 V dwell between steps (s):")
        self.ui.label_voltage_zero_pause.setObjectName("label_voltage_zero_pause")
        self.ui.spin_voltage_zero_pause = QtWidgets.QDoubleSpinBox()
        self.ui.spin_voltage_zero_pause.setObjectName("spin_voltage_zero_pause")
        self.ui.spin_voltage_zero_pause.setDecimals(3)
        self.ui.spin_voltage_zero_pause.setMinimum(0.0)
        self.ui.spin_voltage_zero_pause.setMaximum(600.0)
        self.ui.spin_voltage_zero_pause.setSingleStep(0.1)
        self.ui.spin_voltage_zero_pause.setValue(0.0)
        self.ui.formLayout_2.insertRow(
            settle_row + 3,
            self.ui.label_voltage_zero_pause,
            self.ui.spin_voltage_zero_pause,
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
            self.ui.check_voltage_zero_center,
            self.ui.check_voltage_pulse,
            self.ui.label_voltage_zero_pause,
            self.ui.spin_voltage_zero_pause,
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

        self.ui.btn_load_ref.clicked.connect(lambda: self._load_reference_csv(slot=1))
        if hasattr(self.ui, "btn_load_ref2"):
            self.ui.btn_load_ref2.clicked.connect(
                lambda: self._load_reference_csv(slot=2)
            )
        self.ui.combo_math.currentIndexChanged.connect(
            lambda idx: self._math_mode_changed(1, idx)
        )
        if hasattr(self.ui, "combo_math2"):
            self.ui.combo_math2.currentIndexChanged.connect(
                lambda idx: self._math_mode_changed(2, idx)
            )
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
        self.ui.check_voltage_pulse.toggled.connect(self._update_pulsed_controls_state)

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

    def _reset_loop_progress_bar(self):
        bar = getattr(self.ui, "LoopProgressBar", None)
        if not bar:
            return
        bar.setRange(0, 1)
        bar.setValue(0)
        bar.setFormat("Loop progress: %p%")

    def _prepare_loop_progress_bar(self, *, use_local: bool):
        bar = getattr(self.ui, "LoopProgressBar", None)
        if not bar:
            return
        local_total = 100
        total = max(1, local_total if use_local else (self._scan_pixel_count or 1))
        bar.setRange(0, total)
        bar.setValue(0)
        label = (
            "Local board progress"
            if use_local
            else ("Voltage step progress" if self.measurement_mode == "voltage" else "Loop progress")
        )
        bar.setFormat(f"{label}: %p% (%v/%m)")

    def _set_loop_progress_value(self, loop_idx: int, done: int, total: int):
        bar = getattr(self.ui, "LoopProgressBar", None)
        if not bar:
            return
        total = max(1, int(total))
        done = max(0, min(int(done), total))
        if bar.maximum() != total or bar.minimum() != 0:
            bar.setRange(0, total)
        bar.setValue(done)
        label = "Voltage step" if self.measurement_mode == "voltage" else "Loop"
        bar.setFormat(f"{label} {loop_idx}: %p% (%v/%m)")

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

    # ---------------------- characterization suite ----------------------
    def _open_characterization_settings(self):
        dlg = CharacterizationSettingsDialog(
            self, self._characterization_settings, self.ui.edit_pixel_spec.text()
        )
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        self._characterization_settings = dlg.get_settings()
        # Keep source selections in sync with the main UI.
        self._characterization_settings.use_local_readout = self._using_local_readout()
        self._characterization_settings.use_local_bias = self._using_local_bias()
        self._update_statusbar("Characterization settings updated.")

    def _create_characterization_root(self, settings: CharacterizationSettings) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = self._collect_experiment_name()
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", base_name).strip("_") or "characterization"
        subdir = settings.output_subdir or "characterization"
        root = Path(self.output_folder) / subdir
        root.mkdir(parents=True, exist_ok=True)
        folder = root / f"{slug}_characterization_{timestamp}"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _start_characterization_suite(self):
        if self._worker is not None:
            QtWidgets.QMessageBox.warning(
                self,
                "Characterization",
                "Finish or abort the active scan before running the characterization suite.",
            )
            return
        if self._characterization_thread is not None:
            QtWidgets.QMessageBox.information(
                self, "Characterization", "Characterization suite is already running."
            )
            return
        run_settings = CharacterizationSettings(**vars(self._characterization_settings))
        run_settings.use_local_readout = self._using_local_readout()
        run_settings.use_local_bias = self._using_local_bias()
        try:
            pixels = self._parse_pixel_spec(run_settings.pixel_spec)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Characterization", f"Pixel selection invalid:\n{exc}")
            return
        if run_settings.use_local_readout and not self._switch_connected():
            QtWidgets.QMessageBox.critical(
                self, "Characterization", "Connect the switch board to use local readout."
            )
            return
        if run_settings.use_local_bias and not self._switch_connected():
            QtWidgets.QMessageBox.critical(
                self, "Characterization", "Connect the switch board to use local bias."
            )
            return
        if (
            not run_settings.use_local_bias
            and isinstance(self.bias_sm, DummyBias2400)
        ):
            QtWidgets.QMessageBox.critical(
                self,
                "Characterization",
                "Connect a bias sourcemeter or enable local bias before running the suite.",
            )
            return
        try:
            root = self._create_characterization_root(run_settings)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Characterization", f"Output folder error:\n{exc}")
            return

        worker = CharacterizationWorker(
            settings=run_settings,
            pixel_indices=pixels,
            sm=self.sm,
            bias_sm=self.bias_sm,
            switch=self.switch,
            output_root=root,
        )
        thread = QtCore.QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progressChanged.connect(self._on_characterization_progress)
        worker.stepsPlanned.connect(self._on_characterization_steps_planned)
        worker.stepProgress.connect(self._on_characterization_step_progress)
        worker.statusMessage.connect(self._update_statusbar)
        worker.error.connect(self._on_characterization_error)
        worker.finished.connect(self._on_characterization_finished)
        self._characterization_worker = worker
        self._characterization_thread = thread
        self._set_characterization_ui_state(running=True)
        if getattr(self, "_characterization_progress", None):
            self._characterization_progress.setRange(0, 0)
            self._characterization_progress.setValue(0)
            self._characterization_progress.setFormat("Characterization: %p%")
            self._characterization_progress.setVisible(True)
        try:
            if self._characterization_progress_dialog:
                self._characterization_progress_dialog.close()
            self._characterization_progress_dialog = CharacterizationProgressDialog(self)
            self._characterization_progress_dialog.show()
        except Exception:
            self._characterization_progress_dialog = None
        self._update_statusbar(f"Characterization suite started → {root}")
        thread.start()

    def _abort_characterization_suite(self):
        if not self._characterization_worker:
            return
        try:
            self._characterization_worker.stop()
        except Exception:
            pass
        self.action_abort_characterization.setEnabled(False)
        self._update_statusbar("Stopping characterization suite…")
        if self._characterization_progress_dialog:
            try:
                self._characterization_progress_dialog.close()
            except Exception:
                pass
            self._characterization_progress_dialog = None

    def _on_characterization_progress(self, done: int, total: int, text: str):
        bar = getattr(self, "_characterization_progress", None)
        if bar:
            total = max(1, int(total))
            done = max(0, min(int(done), total))
            bar.setRange(0, total)
            bar.setValue(done)
            bar.setFormat(f"Characterization: %p% ({done}/{total})")
            bar.setVisible(True)
        dlg = getattr(self, "_characterization_progress_dialog", None)
        if dlg:
            dlg.update_overall(done, total, "Overall")
        self.statusBar().showMessage(text, 4000)

    def _on_characterization_steps_planned(self, plan: object):
        dlg = getattr(self, "_characterization_progress_dialog", None)
        if dlg:
            dlg.set_steps(plan)

    def _on_characterization_step_progress(self, key: str, done: int, total: int):
        dlg = getattr(self, "_characterization_progress_dialog", None)
        if dlg:
            dlg.update_step(key, done, total)

    def _on_characterization_error(self, message: str):
        QtWidgets.QMessageBox.critical(self, "Characterization", message)
        if self._characterization_progress_dialog:
            try:
                self._characterization_progress_dialog.close()
            except Exception:
                pass
            self._characterization_progress_dialog = None

    def _on_characterization_finished(self, success: bool, summary_obj):
        thread = getattr(self, "_characterization_thread", None)
        if thread:
            try:
                thread.quit()
                thread.wait()
            except Exception:
                pass
        self._characterization_thread = None
        self._characterization_worker = None
        bar = getattr(self, "_characterization_progress", None)
        if bar:
            bar.setValue(bar.maximum())
            bar.setVisible(False)
        self._set_characterization_ui_state(running=False)

        root_path = None
        try:
            root_path = getattr(summary_obj, "suite_root", None)
        except Exception:
            root_path = None
        if success:
            msg = "Characterization suite finished."
        else:
            msg = "Characterization suite stopped or was aborted."
        try:
            dead = getattr(summary_obj, "dead_short_pixels", set()) or set()
        except Exception:
            dead = set()
        if dead and self._characterization_settings.stop_on_dead_short:
            msg = f"{msg}\nDead shorts detected: {sorted(dead)}"
        if root_path:
            msg = f"{msg}\nResults: {root_path}"
        QtWidgets.QMessageBox.information(self, "Characterization", msg)
        self._update_statusbar(msg)
        try:
            if self._characterization_progress_dialog:
                self._characterization_progress_dialog.close()
        finally:
            self._characterization_progress_dialog = None

    def _set_characterization_ui_state(self, *, running: bool):
        self.action_run_characterization.setEnabled(not running)
        self.action_characterization_settings.setEnabled(not running)
        self.action_abort_characterization.setEnabled(running)
        if running:
            self.ui.btn_run_abort.setEnabled(False)
            self.ui.btn_pause_resume.setEnabled(False)
        else:
            self.ui.btn_run_abort.setEnabled(True)
            self.ui.btn_pause_resume.setEnabled(bool(self._worker))

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
        self._update_pulsed_controls_state()

    def _update_pulsed_controls_state(self):
        show_voltage = self.measurement_mode == "voltage"
        checkbox_enabled = show_voltage
        self.ui.check_voltage_zero_center.setEnabled(show_voltage)
        self.ui.check_voltage_pulse.setEnabled(show_voltage)
        pulsed_enabled = (
            show_voltage
            and self.ui.check_voltage_pulse.isChecked()
        )
        self.ui.label_voltage_zero_pause.setEnabled(pulsed_enabled)
        self.ui.spin_voltage_zero_pause.setEnabled(pulsed_enabled)

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
        self._update_pulsed_controls_state()
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

    @staticmethod
    def _apply_zero_centering(voltages: List[float], enabled: bool) -> List[float]:
        if not enabled:
            return voltages
        if not voltages:
            return [0.0]
        eps = 1e-9
        result: List[float] = [0.0]
        if abs(voltages[0]) > eps:
            result.extend(voltages)
        else:
            deduped = [0.0]
            for v in voltages[1:]:
                if abs(v) < eps and abs(deduped[-1]) < eps:
                    continue
                deduped.append(v)
            result = deduped
        return result

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

    def _collect_current_limit(self) -> Optional[float]:
        text = (self.ui.edit_current_limit.text() or "").strip()
        if not text:
            return None
        try:
            value = float(text)
        except Exception as exc:
            raise ValueError("Current limit must be numeric (A).") from exc
        if value <= 0:
            raise ValueError("Current limit must be positive (A).")
        return float(value)

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
        zero_center = bool(
            getattr(self.ui, "check_voltage_zero_center", None)
        ) and bool(self.ui.check_voltage_zero_center.isChecked())
        pulsed_jv = bool(
            getattr(self.ui, "check_voltage_pulse", None)
        ) and bool(self.ui.check_voltage_pulse.isChecked())
        zero_pause = 0.0
        if pulsed_jv and getattr(self.ui, "spin_voltage_zero_pause", None):
            try:
                zero_pause = max(0.0, float(self.ui.spin_voltage_zero_pause.value()))
            except Exception:
                zero_pause = 0.0

        voltage_steps: Optional[List[float]] = None
        settle_time = 0.0
        constant_bias_voltage: Optional[float] = None
        if voltage_mode:
            try:
                voltage_steps, settle_time = self._collect_voltage_sweep_settings()
            except Exception as exc:
                raise ValueError(f"Voltage sweep invalid: {exc}") from exc
            voltage_steps = self._apply_zero_centering(
                list(voltage_steps or []), zero_center
            )
            self._ensure_local_bias_supported(voltage_steps)
            loops = len(voltage_steps)
            inter_loop_delay = 0.0
        else:
            zero_center = False
            zero_pause = 0.0
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
        try:
            current_limit = self._collect_current_limit()
        except Exception as exc:
            raise ValueError(f"Current limit invalid: {exc}") from exc

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
            current_limit=current_limit,
            measurement_mode="voltage" if voltage_mode else "time",
            voltage_steps=list(voltage_steps or []) if voltage_mode else None,
            voltage_settle_s=settle_time if voltage_mode else 0.0,
            voltage_zero_center=bool(zero_center) if voltage_mode else False,
            voltage_zero_pause_s=zero_pause if voltage_mode and pulsed_jv else 0.0,
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

    def _loop_csv_metadata(
        self,
        loop_idx: int,
        voltage: Optional[float],
        runtime_ms: Optional[float],
    ) -> dict:
        dataset = "voltage" if self.measurement_mode == "voltage" else "time"
        elapsed = max(
            0.0, float(time.time() - getattr(self, "_start_time", time.time()))
        )
        metadata: dict = {
            "dataset": dataset,
            "loop_index": int(loop_idx),
            "measurement_mode": self.measurement_mode,
            "timestamp": datetime.utcnow().isoformat(),
            "elapsed_s": elapsed,
        }
        if runtime_ms is not None:
            metadata["runtime_ms"] = float(runtime_ms)
        if voltage is not None:
            metadata["voltage"] = float(voltage)
            if self._requested_voltage is not None:
                metadata["requested_voltage"] = float(self._requested_voltage)
        if self._constant_bias_voltage is not None:
            metadata["constant_bias_voltage"] = float(self._constant_bias_voltage)
        if self._voltage_sequence:
            metadata["voltage_sequence_count"] = len(self._voltage_sequence)
        if self.measurement_mode == "voltage":
            if self._voltage_zero_center:
                metadata["voltage_zero_center"] = True
            if self._voltage_zero_pause > 0:
                metadata["voltage_zero_pause_s"] = float(self._voltage_zero_pause)
            if self._voltage_settle_s > 0:
                metadata["voltage_settle_s"] = float(self._voltage_settle_s)
        if self._current_limit is not None:
            metadata["current_limit_a"] = float(self._current_limit)
        if self._excluded_pixels:
            metadata["excluded_pixels"] = sorted(int(x) for x in self._excluded_pixels)
        metadata["pixel_spec"] = self.ui.edit_pixel_spec.text()
        return metadata

    @staticmethod
    def _format_csv_metadata(metadata: dict) -> str:
        try:
            payload = json.dumps(metadata, separators=(",", ":"), sort_keys=True)
            return f"READOUT_METADATA {payload}"
        except Exception as exc:
            logging.warning(f"Failed to encode CSV metadata: {exc}")
            return ""

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
                "mode_primary": self.math_mode,
                "mode_secondary": self.math_mode2,
                "reference_primary": str(self.ref_path or ""),
                "reference_secondary": str(self.ref_path2 or ""),
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
        if self._characterization_thread is not None:
            QtWidgets.QMessageBox.warning(
                self,
                "Characterization",
                "Wait for the characterization suite to finish before starting a manual scan.",
            )
            return
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
        self._excluded_pixels = set()
        self._current_limit = acquisition.current_limit

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
        self._voltage_zero_center = bool(
            acquisition.voltage_zero_center if voltage_mode else False
        )
        self._voltage_zero_pause = (
            float(acquisition.voltage_zero_pause_s) if voltage_mode else 0.0
        )
        self._voltage_settle_s = acquisition.voltage_settle_s if voltage_mode else 0.0
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
        self._scan_loop_count = loops
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
        self._scan_pixel_count = len(pixels)
        self._loop_progress_total = (
            100 if use_local_readout else max(1, self._scan_pixel_count or 1)
        )
        self._prepare_loop_progress_bar(use_local=use_local_readout)

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
            voltage_zero_pause_s=self._voltage_zero_pause if voltage_mode else 0.0,
            constant_bias_voltage=constant_bias_voltage if not voltage_mode else None,
            use_local_readout=use_local_readout,
            use_local_bias=use_local_bias,
            current_limit=acquisition.current_limit,
        )
        self._thread = QtCore.QThread()
        self._worker.moveToThread(self._thread)

        # signals
        self._worker.loopDataReady.connect(self._on_loop_data)
        self._worker.loopStarted.connect(self._on_loop_started)
        self._worker.loopFinished.connect(self._on_loop_finished)
        self._worker.loopProgress.connect(self._on_loop_progress)
        if hasattr(self._worker, "pixelExcluded"):
            self._worker.pixelExcluded.connect(self._on_pixel_excluded)
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
        self._total_steps = max(1, self._total_steps)
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
        base_total = self._loop_progress_total or self._scan_pixel_count or 1
        self._set_loop_progress_value(loop_idx, 0, base_total)
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
            csv_metadata = self._loop_csv_metadata(loop_idx, voltage, runtime_ms)
            header = self._format_csv_metadata(csv_metadata)
            save_kwargs = dict(delimiter=",", fmt="%.5e")
            if header:
                save_kwargs["header"] = header
                save_kwargs["comments"] = "# "
            np.savetxt(csv_path, arr, **save_kwargs)
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

    def _on_pixel_excluded(self, pixel_idx: int, measured: float, loop_idx: int):
        try:
            pixel_idx = int(pixel_idx)
        except Exception:
            return
        self._excluded_pixels.add(pixel_idx)
        remaining_loops = max(0, int(self._scan_loop_count or 0) - int(loop_idx))
        if remaining_loops > 0:
            try:
                self._total_steps = max(
                    1, max(self._done_steps, self._total_steps - remaining_loops)
                )
            except Exception:
                pass
        val_text = (
            f"{measured:.3e} A"
            if measured is not None and not np.isnan(measured)
            else "limit hit"
        )
        msg = (
            f"Pixel {pixel_idx} exceeded current limit ({val_text}) – "
            "excluded from remaining loops."
        )
        logging.warning(msg)
        self.device_logger.warning(msg)
        self._update_statusbar(msg)

    def _on_loop_progress(self, loop_idx: int, done: int, total: int):
        self._set_loop_progress_value(loop_idx, done, total)

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
        self._voltage_zero_center = False
        self._voltage_zero_pause = 0.0
        self._voltage_settle_s = 0.0
        self._current_limit = None
        self._excluded_pixels = set()
        self._scan_loop_count = 0
        self._update_statusbar(text="Scan finished.")
        self._update_loop_scrub_state()
        self._reset_loop_progress_bar()
        data_ready = (
            self._run_folder
            and (self._run_folder / "data").exists()
        )
        if self.analysis_window and data_ready:
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
    def _load_reference_csv(self, slot: int = 1):
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
            path = Path(fname)
            label = f"{path.name}  [{arr.shape[0]}×{arr.shape[1]}]"
            if slot == 2:
                self.ref_matrix2 = arr
                self.ref_path2 = path
                if hasattr(self.ui, "lbl_ref2_info"):
                    self.ui.lbl_ref2_info.setText(label)
            else:
                self.ref_matrix = arr
                self.ref_path = path
                self.ui.lbl_ref_info.setText(label)
            self._update_plots()
        except Exception as e:
            if slot == 2:
                self.ref_matrix2 = None
                self.ref_path2 = None
                if hasattr(self.ui, "lbl_ref2_info"):
                    self.ui.lbl_ref2_info.setText("(none)")
            else:
                self.ref_matrix = None
                self.ref_path = None
                self.ui.lbl_ref_info.setText("(none)")
            QtWidgets.QMessageBox.critical(
                self, "Reference CSV", f"Failed to load: {e}"
            )

    def _math_mode_changed(self, slot: int, _index: int):
        combo = self.ui.combo_math if slot == 1 else getattr(self.ui, "combo_math2", None)
        if combo is None:
            return
        text = combo.currentText().lower()
        if text.startswith("divide"):
            mode = "divide"
        elif text.startswith("subtract"):
            mode = "subtract"
        else:
            mode = "none"
        if slot == 2:
            self.math_mode2 = mode
        else:
            self.math_mode = mode
        self._update_plots()

    def _apply_math(self, data: np.ndarray) -> np.ndarray:
        result = np.array(data, copy=True)
        result = self._apply_reference_operation(result, self.ref_matrix, self.math_mode)
        result = self._apply_reference_operation(result, self.ref_matrix2, self.math_mode2)
        return result

    def _apply_reference_operation(
        self, data: np.ndarray, ref_matrix: Optional[np.ndarray], mode: str
    ) -> np.ndarray:
        if mode == "none" or ref_matrix is None:
            return data
        ref = np.array(ref_matrix, copy=False)
        result = np.array(data, copy=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            if mode == "divide":
                denom = ref + self.math_eps
                np.divide(result, denom, out=result, where=~np.isnan(data))
            else:  # subtract
                result = result - ref
        mask_nan = np.isnan(data)
        result = np.where(mask_nan, np.nan, result)
        return result

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

    def _open_stage_controller_window(self):
        if self.stage_controller_window is None:
            try:
                self.stage_controller_window = StageControllerWindow(self, self.stage_driver)
            except Exception as exc:
                logging.error(f"Failed to create stage controller window: {exc}")
                QtWidgets.QMessageBox.critical(
                    self, "Stage Controller", f"Failed to open stage controller window:\n{exc}"
                )
                self.stage_controller_window = None
                return
            self.stage_controller_window.destroyed.connect(self._clear_stage_controller_window)
        self.stage_controller_window.show()
        self.stage_controller_window.raise_()
        self.stage_controller_window.activateWindow()

    def _clear_stage_controller_window(self) -> None:
        self.stage_controller_window = None

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
