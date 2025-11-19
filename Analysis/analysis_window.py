from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, to_hex
import matplotlib.pyplot as plt
from PySide6 import QtCore, QtWidgets

# Type aliases
PixelParser = Callable[[str], Sequence[int]]


@dataclass
class VoltageEntry:
    index: int
    voltage: float
    data: np.ndarray
    path: Path
    timestamp: datetime


@dataclass
class TimeEntry:
    index: int
    elapsed_s: float
    data: np.ndarray
    path: Path
    timestamp: datetime


class MatplotlibCanvas(FigureCanvas):
    """Simple Matplotlib canvas with a single axes."""

    def __init__(self, width: float = 5.0, height: float = 4.0, dpi: int = 100) -> None:
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.figure)
        self.ax = self.figure.add_subplot(111)

    def clear(self) -> None:
        self.figure.clf()
        self.ax = self.figure.add_subplot(111)


COLOR_SCHEME_MAP = {
    "Default": None,
    "Tab10": "tab10",
    "Tab20": "tab20",
    "Viridis": "viridis",
    "Plasma": "plasma",
    "Inferno": "inferno",
    "Magma": "magma",
    "Cividis": "cividis",
    "Rainbow": "rainbow",
}


def load_csv_with_metadata(path: Path) -> tuple[np.ndarray, dict]:
    metadata: dict = {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                if not stripped.startswith("#"):
                    break
                content = stripped.lstrip("#").strip()
                if content.startswith("READOUT_METADATA"):
                    payload = content[len("READOUT_METADATA") :].strip()
                    if payload:
                        try:
                            metadata = json.loads(payload)
                        except json.JSONDecodeError:
                            metadata = {}
                    break
    except Exception:
        metadata = {}
    array = np.loadtxt(path, delimiter=",", dtype=float)
    return array, metadata


class AnalysisWindow(QtWidgets.QMainWindow):
    """Standalone window providing post-run analysis utilities."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, *, pixel_parser: Optional[PixelParser] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Readoubt Analysis")
        self.resize(1200, 800)

        self._pixel_parser: PixelParser = pixel_parser or self._default_pixel_parser

        # Runtime state
        self._current_folder: Optional[Path] = None
        self._voltage_entries: list[VoltageEntry] = []
        self._time_entries: list[TimeEntry] = []
        self._resistivity_map: Optional[np.ndarray] = None
        self._heatmap_images: list[Path] = []
        self._hist_images: list[Path] = []
        self.pixel_checkboxes: dict[int, QtWidgets.QCheckBox] = {}
        self._suppress_curve_warnings = False

        # UI
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        self.tabs = QtWidgets.QTabWidget()
        main_layout.addWidget(self.tabs)

        self._build_curve_tab()
        self._build_resistivity_tab()
        self._build_video_tab()

    # ------------------------------------------------------------------ UI builders
    def _build_curve_tab(self) -> None:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        # Folder selector
        folder_row = QtWidgets.QHBoxLayout()
        self.curve_folder_edit = QtWidgets.QLineEdit()
        self.curve_folder_edit.setPlaceholderText("Select a run folder containing loop_*.csv or voltage_*.csv files")
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_curve_folder)
        self.curve_load_btn = QtWidgets.QPushButton("Load Folder")
        self.curve_load_btn.clicked.connect(self._load_curve_folder)
        folder_row.addWidget(QtWidgets.QLabel("Run Folder:"))
        folder_row.addWidget(self.curve_folder_edit, stretch=1)
        folder_row.addWidget(browse_btn)
        folder_row.addWidget(self.curve_load_btn)
        layout.addLayout(folder_row)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(splitter, stretch=1)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # Settings group
        settings_group = QtWidgets.QGroupBox("Curve Settings")
        settings_form = QtWidgets.QFormLayout(settings_group)

        self.dataset_combo = QtWidgets.QComboBox()
        self.dataset_combo.addItem("Time (Loop)", userData="time")
        self.dataset_combo.addItem("Voltage Sweep", userData="voltage")
        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_changed)
        settings_form.addRow("Dataset:", self.dataset_combo)

        self.time_axis_combo = QtWidgets.QComboBox()
        self.time_axis_combo.addItem("Elapsed seconds", userData="elapsed")
        self.time_axis_combo.addItem("Loop index", userData="index")
        settings_form.addRow("Time Axis:", self.time_axis_combo)

        self.pixel_spec_edit = QtWidgets.QLineEdit("1-100")
        settings_form.addRow("Pixels:", self.pixel_spec_edit)

        self.color_scheme_combo = QtWidgets.QComboBox()
        for label in ["Default", "Tab10", "Tab20", "Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Rainbow"]:
            self.color_scheme_combo.addItem(label)
        settings_form.addRow("Color Scheme:", self.color_scheme_combo)

        self.legend_checkbox = QtWidgets.QCheckBox("Show legend")
        self.legend_checkbox.setChecked(True)
        settings_form.addRow(self.legend_checkbox)

        self.line_style_combo = QtWidgets.QComboBox()
        self.line_style_combo.addItem("Solid (-)", userData="-")
        self.line_style_combo.addItem("Dashed (--)", userData="--")
        self.line_style_combo.addItem("Dash-dot (-.)", userData="-.")
        self.line_style_combo.addItem("Dotted (:)", userData=":")
        settings_form.addRow("Line Style:", self.line_style_combo)

        self.marker_combo = QtWidgets.QComboBox()
        self.marker_combo.addItem("Circle (o)", userData="o")
        self.marker_combo.addItem("Square (s)", userData="s")
        self.marker_combo.addItem("Triangle (^)", userData="^")
        self.marker_combo.addItem("Point (.)", userData=".")
        self.marker_combo.addItem("None", userData="")
        settings_form.addRow("Marker:", self.marker_combo)

        self.line_width_spin = QtWidgets.QDoubleSpinBox()
        self.line_width_spin.setRange(0.1, 10.0)
        self.line_width_spin.setSingleStep(0.1)
        self.line_width_spin.setValue(1.5)
        settings_form.addRow("Line Width:", self.line_width_spin)

        self.marker_size_spin = QtWidgets.QDoubleSpinBox()
        self.marker_size_spin.setRange(1.0, 20.0)
        self.marker_size_spin.setSingleStep(1.0)
        self.marker_size_spin.setValue(6.0)
        settings_form.addRow("Marker Size:", self.marker_size_spin)

        self.plot_title_edit = QtWidgets.QLineEdit("")
        self.plot_title_edit.setPlaceholderText("Leave blank for automatic title")
        settings_form.addRow("Plot Title:", self.plot_title_edit)

        grid_box = QtWidgets.QWidget()
        grid_layout = QtWidgets.QHBoxLayout(grid_box)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_major_checkbox = QtWidgets.QCheckBox("Major")
        self.grid_major_checkbox.setChecked(True)
        self.grid_minor_checkbox = QtWidgets.QCheckBox("Minor")
        grid_layout.addWidget(self.grid_major_checkbox)
        grid_layout.addWidget(self.grid_minor_checkbox)
        grid_layout.addStretch(1)
        settings_form.addRow("Grid:", grid_box)

        axis_scale_widget = QtWidgets.QWidget()
        axis_scale_layout = QtWidgets.QHBoxLayout(axis_scale_widget)
        axis_scale_layout.setContentsMargins(0, 0, 0, 0)
        self.log_x_checkbox = QtWidgets.QCheckBox("Log X")
        self.log_y_checkbox = QtWidgets.QCheckBox("Log Y")
        axis_scale_layout.addWidget(self.log_x_checkbox)
        axis_scale_layout.addWidget(self.log_y_checkbox)
        axis_scale_layout.addStretch(1)
        settings_form.addRow("Axis Scale:", axis_scale_widget)

        self.xmin_edit = QtWidgets.QLineEdit()
        self.xmin_edit.setPlaceholderText("auto")
        self.xmax_edit = QtWidgets.QLineEdit()
        self.xmax_edit.setPlaceholderText("auto")
        x_limits_widget = QtWidgets.QWidget()
        x_limits_layout = QtWidgets.QHBoxLayout(x_limits_widget)
        x_limits_layout.setContentsMargins(0, 0, 0, 0)
        x_limits_layout.addWidget(QtWidgets.QLabel("Min"))
        x_limits_layout.addWidget(self.xmin_edit)
        x_limits_layout.addWidget(QtWidgets.QLabel("Max"))
        x_limits_layout.addWidget(self.xmax_edit)
        settings_form.addRow("X Limits:", x_limits_widget)

        self.ymin_edit = QtWidgets.QLineEdit()
        self.ymin_edit.setPlaceholderText("auto")
        self.ymax_edit = QtWidgets.QLineEdit()
        self.ymax_edit.setPlaceholderText("auto")
        y_limits_widget = QtWidgets.QWidget()
        y_limits_layout = QtWidgets.QHBoxLayout(y_limits_widget)
        y_limits_layout.setContentsMargins(0, 0, 0, 0)
        y_limits_layout.addWidget(QtWidgets.QLabel("Min"))
        y_limits_layout.addWidget(self.ymin_edit)
        y_limits_layout.addWidget(QtWidgets.QLabel("Max"))
        y_limits_layout.addWidget(self.ymax_edit)
        settings_form.addRow("Y Limits:", y_limits_widget)

        self.auto_refresh_checkbox = QtWidgets.QCheckBox("Auto-update plot on change")
        self.auto_refresh_checkbox.setChecked(True)
        settings_form.addRow(self.auto_refresh_checkbox)

        self.curve_refresh_btn = QtWidgets.QPushButton("Plot Curve")
        self.curve_refresh_btn.clicked.connect(self._plot_curve)
        settings_form.addRow(self.curve_refresh_btn)

        left_layout.addWidget(settings_group)

        pixel_group = QtWidgets.QGroupBox("Pixel Visibility")
        pixel_layout = QtWidgets.QVBoxLayout(pixel_group)
        grid_container = QtWidgets.QWidget()
        grid_container_layout = QtWidgets.QGridLayout(grid_container)
        grid_container_layout.setSpacing(2)
        for row in range(10):
            for col in range(10):
                idx = row * 10 + col + 1
                checkbox = QtWidgets.QCheckBox(f"{idx:02d}")
                checkbox.setChecked(True)
                checkbox.toggled.connect(self._on_pixel_checkbox_toggled)
                self.pixel_checkboxes[idx] = checkbox
                grid_container_layout.addWidget(checkbox, row, col)
        pixel_layout.addWidget(grid_container)
        pixel_button_row = QtWidgets.QHBoxLayout()
        select_all_btn = QtWidgets.QPushButton("Select All")
        select_all_btn.clicked.connect(lambda: self._set_all_pixel_checkboxes(True))
        clear_btn = QtWidgets.QPushButton("Clear All")
        clear_btn.clicked.connect(lambda: self._set_all_pixel_checkboxes(False))
        pixel_button_row.addWidget(select_all_btn)
        pixel_button_row.addWidget(clear_btn)
        pixel_button_row.addStretch(1)
        pixel_layout.addLayout(pixel_button_row)
        self.pixel_summary_label = QtWidgets.QLabel("")
        pixel_layout.addWidget(self.pixel_summary_label)
        left_layout.addWidget(pixel_group, stretch=1)
        left_layout.addStretch(1)

        splitter.addWidget(left_panel)

        # Plot canvas and stats on the right
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.curve_canvas = MatplotlibCanvas(width=6.0, height=4.5, dpi=100)
        right_layout.addWidget(self.curve_canvas, stretch=1)
        self.curve_stats_label = QtWidgets.QLabel("")
        self.curve_stats_label.setWordWrap(True)
        right_layout.addWidget(self.curve_stats_label)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self._initialize_auto_refresh_connections()
        self._update_pixel_summary()

        self.tabs.addTab(tab, "Curves")
        self._on_dataset_changed()
        self._auto_refresh_curve()

    def _initialize_auto_refresh_connections(self) -> None:
        controls = [
            self.color_scheme_combo,
            self.legend_checkbox,
            self.time_axis_combo,
            self.line_style_combo,
            self.marker_combo,
            self.line_width_spin,
            self.marker_size_spin,
            self.grid_major_checkbox,
            self.grid_minor_checkbox,
            self.log_x_checkbox,
            self.log_y_checkbox,
            self.xmin_edit,
            self.xmax_edit,
            self.ymin_edit,
            self.ymax_edit,
            self.pixel_spec_edit,
            self.plot_title_edit,
        ]
        for widget in controls:
            self._connect_auto_refresh(widget)
        self.auto_refresh_checkbox.toggled.connect(self._auto_refresh_curve)

    def _connect_auto_refresh(self, widget: QtWidgets.QWidget) -> None:
        if isinstance(widget, QtWidgets.QComboBox):
            widget.currentIndexChanged.connect(self._auto_refresh_curve)
        elif isinstance(widget, QtWidgets.QCheckBox):
            widget.toggled.connect(self._auto_refresh_curve)
        elif isinstance(widget, QtWidgets.QLineEdit):
            widget.editingFinished.connect(self._auto_refresh_curve)
        elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            widget.valueChanged.connect(self._auto_refresh_curve)

    def _build_resistivity_tab(self) -> None:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        controls = QtWidgets.QHBoxLayout()
        self.force_origin_checkbox = QtWidgets.QCheckBox("Force fit through origin")
        controls.addWidget(self.force_origin_checkbox)
        controls.addStretch(1)
        self.compute_res_btn = QtWidgets.QPushButton("Compute Resistivity Map")
        self.compute_res_btn.clicked.connect(self._compute_resistivity)
        controls.addWidget(self.compute_res_btn)
        layout.addLayout(controls)

        canvases_layout = QtWidgets.QHBoxLayout()
        self.res_heatmap_canvas = MatplotlibCanvas(width=5.0, height=4.5, dpi=100)
        canvases_layout.addWidget(self.res_heatmap_canvas, stretch=1)
        self.res_hist_canvas = MatplotlibCanvas(width=5.0, height=4.5, dpi=100)
        canvases_layout.addWidget(self.res_hist_canvas, stretch=1)
        layout.addLayout(canvases_layout, stretch=1)

        export_row = QtWidgets.QHBoxLayout()
        self.export_heatmap_btn = QtWidgets.QPushButton("Export Heatmap PNG…")
        self.export_heatmap_btn.clicked.connect(self._export_resistivity_heatmap)
        self.export_heatmap_btn.setEnabled(False)
        self.export_hist_btn = QtWidgets.QPushButton("Export Histogram PNG…")
        self.export_hist_btn.clicked.connect(self._export_resistivity_histogram)
        self.export_hist_btn.setEnabled(False)
        export_row.addWidget(self.export_heatmap_btn)
        export_row.addWidget(self.export_hist_btn)
        export_row.addStretch(1)
        layout.addLayout(export_row)

        self.res_status_label = QtWidgets.QLabel("Load a voltage sweep and compute to view results.")
        self.res_status_label.setWordWrap(True)
        layout.addWidget(self.res_status_label)

        self.tabs.addTab(tab, "Resistivity")

    def _build_video_tab(self) -> None:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        form = QtWidgets.QFormLayout()

        self.video_folder_edit = QtWidgets.QLineEdit()
        folder_row = QtWidgets.QHBoxLayout()
        folder_row.addWidget(self.video_folder_edit)
        browse_images_btn = QtWidgets.QPushButton("Browse…")
        browse_images_btn.clicked.connect(self._browse_video_folder)
        folder_row.addWidget(browse_images_btn)
        form.addRow("Image Folder:", folder_row)

        self.video_pattern_edit = QtWidgets.QLineEdit("voltage_*_heatmap.png")
        form.addRow("Filename Pattern:", self.video_pattern_edit)

        self.video_output_edit = QtWidgets.QLineEdit()
        output_row = QtWidgets.QHBoxLayout()
        output_row.addWidget(self.video_output_edit)
        browse_output_btn = QtWidgets.QPushButton("Choose…")
        browse_output_btn.clicked.connect(self._browse_video_output)
        output_row.addWidget(browse_output_btn)
        form.addRow("Output Video:", output_row)

        self.video_fps_spin = QtWidgets.QSpinBox()
        self.video_fps_spin.setRange(1, 120)
        self.video_fps_spin.setValue(12)
        form.addRow("Frame Rate (fps):", self.video_fps_spin)

        layout.addLayout(form)

        actions_row = QtWidgets.QHBoxLayout()
        self.populate_from_run_btn = QtWidgets.QPushButton("Use Loaded Run Images")
        self.populate_from_run_btn.clicked.connect(self._populate_video_defaults)
        actions_row.addWidget(self.populate_from_run_btn)
        actions_row.addStretch(1)
        self.video_create_btn = QtWidgets.QPushButton("Create Video with ffmpeg")
        self.video_create_btn.clicked.connect(self._create_video)
        actions_row.addWidget(self.video_create_btn)
        layout.addLayout(actions_row)

        self.video_status_label = QtWidgets.QLabel("")
        self.video_status_label.setWordWrap(True)
        layout.addWidget(self.video_status_label)

        self.tabs.addTab(tab, "Video Builder")

    def _on_pixel_checkbox_toggled(self) -> None:
        self._update_pixel_summary()

    def _set_all_pixel_checkboxes(self, checked: bool) -> None:
        for checkbox in self.pixel_checkboxes.values():
            checkbox.blockSignals(True)
            checkbox.setChecked(checked)
            checkbox.blockSignals(False)
        self._update_pixel_summary()

    def _update_pixel_summary(self) -> None:
        active = len(self._active_pixel_indices())
        self.pixel_summary_label.setText(f"Enabled pixels: {active}/100")
        if active == 0:
            self.pixel_summary_label.setStyleSheet("color: crimson;")
        else:
            self.pixel_summary_label.setStyleSheet("")
        self._auto_refresh_curve()

    def _active_pixel_indices(self) -> List[int]:
        return [idx for idx, checkbox in self.pixel_checkboxes.items() if checkbox.isChecked()]

    def _resolve_pixel_selection(self) -> List[int]:
        spec_text = (self.pixel_spec_edit.text() or "").strip()
        try:
            parsed = list(self._pixel_parser(spec_text or "1-100"))
        except Exception as exc:
            raise ValueError(f"Invalid pixel selection: {exc}") from exc
        parsed = [idx for idx in parsed if 1 <= idx <= 100]
        active = self._active_pixel_indices()
        if not parsed:
            selected = list(active)
        else:
            selected = [idx for idx in parsed if idx in active]
        if not selected:
            if not active:
                raise ValueError("No pixels are enabled in the grid.")
            raise ValueError("Selected pixels are disabled in the visibility grid.")
        return selected

    def _auto_refresh_curve(self) -> None:
        if not self.auto_refresh_checkbox.isChecked():
            return
        if not (self._time_entries or self._voltage_entries):
            return
        self._suppress_curve_warnings = True
        try:
            self._plot_curve()
        finally:
            self._suppress_curve_warnings = False

    def _show_curve_warning(self, title: str, message: str) -> None:
        if self._suppress_curve_warnings:
            return
        QtWidgets.QMessageBox.warning(self, title, message)

    def _apply_axis_formatting(self, ax: Axes, x_values: Sequence[float], y_values: Sequence[float]) -> None:
        if self.grid_major_checkbox.isChecked():
            ax.grid(True, which="major", linestyle="--", alpha=0.4)
        else:
            ax.grid(False, which="major")
        if self.grid_minor_checkbox.isChecked():
            ax.minorticks_on()
            ax.grid(True, which="minor", linestyle=":", alpha=0.25)
        else:
            ax.minorticks_off()
            ax.grid(False, which="minor")

        if self.log_x_checkbox.isChecked():
            if x_values and all(val > 0 for val in x_values):
                ax.set_xscale("log")
            else:
                self._show_curve_warning("Axis Scale", "Cannot enable log X axis when values are zero or negative.")
                self.log_x_checkbox.setChecked(False)
                ax.set_xscale("linear")
        else:
            ax.set_xscale("linear")

        if self.log_y_checkbox.isChecked():
            if y_values and all(val > 0 for val in y_values):
                ax.set_yscale("log")
            else:
                self._show_curve_warning("Axis Scale", "Cannot enable log Y axis when values are zero or negative.")
                self.log_y_checkbox.setChecked(False)
                ax.set_yscale("linear")
        else:
            ax.set_yscale("linear")

        x_min = self._extract_limit_value(self.xmin_edit)
        x_max = self._extract_limit_value(self.xmax_edit)
        if x_min is not None or x_max is not None:
            current = ax.get_xlim()
            ax.set_xlim(x_min if x_min is not None else current[0], x_max if x_max is not None else current[1])

        y_min = self._extract_limit_value(self.ymin_edit)
        y_max = self._extract_limit_value(self.ymax_edit)
        if y_min is not None or y_max is not None:
            current = ax.get_ylim()
            ax.set_ylim(y_min if y_min is not None else current[0], y_max if y_max is not None else current[1])

    def _extract_limit_value(self, edit: QtWidgets.QLineEdit) -> Optional[float]:
        text = edit.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            self._show_curve_warning("Axis Limits", f"Invalid numeric limit: {text}")
            edit.clear()
            return None

    @staticmethod
    def _metadata_timestamp(metadata: dict) -> Optional[datetime]:
        stamp = metadata.get("timestamp")
        if not stamp:
            return None
        try:
            return datetime.fromisoformat(stamp)
        except ValueError:
            return None

    # ------------------------------------------------------------------ public helpers
    def set_run_folder(self, folder: Path, *, auto_load: bool = False) -> None:
        """Expose ability to inject a folder from the main window."""
        folder = Path(folder)
        data_folder = folder / "data"
        target = data_folder if data_folder.is_dir() else folder
        self.curve_folder_edit.setText(str(target))
        self.video_folder_edit.setText(str(folder))
        if auto_load:
            self._load_run_folder(target)

    # ------------------------------------------------------------------ events / slots
    def _browse_curve_folder(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Run Folder", self.curve_folder_edit.text() or str(Path.home())
        )
        if directory:
            self.curve_folder_edit.setText(directory)

    def _load_curve_folder(self) -> None:
        path_text = self.curve_folder_edit.text().strip()
        if not path_text:
            QtWidgets.QMessageBox.warning(self, "Analysis", "Enter a folder path first.")
            return
        path = Path(path_text).expanduser()
        if not path.is_dir():
            QtWidgets.QMessageBox.critical(self, "Analysis", f"Folder not found: {path}")
            return
        self._load_run_folder(path)

    def _load_run_folder(self, folder: Path) -> None:
        voltage_entries: list[VoltageEntry] = []
        time_temp: list[tuple[int, Optional[float], np.ndarray, Path, datetime]] = []
        heatmap_images: list[Path] = []
        hist_images: list[Path] = []

        csv_files = sorted(folder.glob("*_data.csv"))
        if not csv_files:
            QtWidgets.QMessageBox.warning(
                self, "Analysis", f"No *_data.csv files found in {folder}"
            )
            return

        for csv_path in csv_files:
            name = csv_path.name
            try:
                arr, metadata = load_csv_with_metadata(csv_path)
                if arr.size == 100:
                    arr = arr.reshape((10, 10))
                if arr.shape != (10, 10):
                    raise ValueError(f"Unexpected array shape {arr.shape}")
                timestamp = (
                    self._metadata_timestamp(metadata)
                    or datetime.fromtimestamp(csv_path.stat().st_mtime)
                )
                dataset_hint = (metadata.get("dataset") or "").lower()
                loop_index_meta = metadata.get("loop_index")
                idx_meta = int(loop_index_meta) if isinstance(loop_index_meta, (int, float)) else None

                if name.startswith("voltage_") and name.endswith("_data.csv") or dataset_hint == "voltage":
                    idx = idx_meta
                    if idx is None:
                        parts = name.split("_")
                        if len(parts) < 2:
                            raise ValueError("voltage filename missing index")
                        idx = int(parts[1])
                    voltage_val = metadata.get("voltage")
                    if voltage_val is None:
                        parts = name.split("_")
                        if len(parts) < 3:
                            raise ValueError("voltage filename missing tag")
                        tag = parts[2]
                        voltage_val = self._decode_voltage_tag(tag)
                    voltage_entries.append(
                        VoltageEntry(
                            index=int(idx),
                            voltage=float(voltage_val),
                            data=arr,
                            path=csv_path,
                            timestamp=timestamp,
                        )
                    )
                    guess_heatmap = csv_path.with_name(csv_path.stem.replace("_data", "_heatmap") + ".png")
                    if guess_heatmap.exists():
                        heatmap_images.append(guess_heatmap)
                elif name.startswith("loop_") and name.endswith("_data.csv") or dataset_hint == "time":
                    idx = idx_meta
                    if idx is None:
                        parts = name.split("_")
                        if len(parts) < 2:
                            raise ValueError("loop filename missing index")
                        idx = int(parts[1])
                    elapsed_val = metadata.get("elapsed_s")
                    elapsed = float(elapsed_val) if isinstance(elapsed_val, (int, float)) else None
                    time_temp.append((int(idx), elapsed, arr, csv_path, timestamp))
                    guess_heatmap = csv_path.with_name(csv_path.stem.replace("_data", "_heatmap") + ".png")
                    if guess_heatmap.exists():
                        heatmap_images.append(guess_heatmap)
                else:
                    continue
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self, "Analysis", f"Failed to load {csv_path.name}: {exc}"
                )

        time_entries: list[TimeEntry] = []
        if time_temp:
            time_temp.sort(key=lambda item: item[0])
            base_ts = min(ts for _, _, _, _, ts in time_temp)
            for idx, elapsed_opt, arr, path, timestamp in time_temp:
                if elapsed_opt is not None:
                    elapsed = max(float(elapsed_opt), 0.0)
                else:
                    elapsed = max((timestamp - base_ts).total_seconds(), 0.0)
                time_entries.append(
                    TimeEntry(
                        index=idx,
                        elapsed_s=elapsed,
                        data=arr,
                        path=path,
                        timestamp=timestamp,
                    )
                )

        if voltage_entries:
            voltage_entries.sort(key=lambda entry: entry.index)
        if heatmap_images:
            heatmap_images.sort()
        hist_candidates = list(folder.glob("*histogram*.png"))
        if hist_candidates:
            hist_images.extend(sorted(hist_candidates))

        self._current_folder = folder
        self._voltage_entries = voltage_entries
        self._time_entries = time_entries
        self._heatmap_images = heatmap_images
        self._hist_images = hist_images
        self.curve_stats_label.clear()
        self.res_status_label.setText("Voltage sweep loaded." if voltage_entries else "Load a voltage sweep to compute resistivity.")
        self.video_status_label.clear()
        self.video_folder_edit.setText(str(folder))
        self._populate_video_defaults()
        self._update_dataset_availability()
        self._plot_curve()

    def _update_dataset_availability(self) -> None:
        has_voltage = bool(self._voltage_entries)
        self.compute_res_btn.setEnabled(has_voltage and len(self._voltage_entries) >= 2)

    def _on_dataset_changed(self) -> None:
        is_time = self.dataset_combo.currentData() == "time"
        self.time_axis_combo.setVisible(is_time)
        self._auto_refresh_curve()

    def _plot_curve(self) -> None:
        dataset_kind = self.dataset_combo.currentData()
        if dataset_kind == "time":
            entries = self._time_entries
        elif dataset_kind == "voltage":
            entries = self._voltage_entries
        else:
            entries = []

        if not entries:
            self.curve_canvas.clear()
            self.curve_canvas.draw_idle()
            self.curve_stats_label.setText("Dataset not available for the selected type.")
            return

        try:
            pixel_indices = self._resolve_pixel_selection()
        except ValueError as exc:
            self._show_curve_warning("Pixels", str(exc))
            return

        self.curve_canvas.clear()
        ax = self.curve_canvas.ax

        stats_message = ""
        legend_labels: list[str] = []
        axis_x: list[float] = []
        axis_y: list[float] = []
        per_pixel = self._compute_per_pixel_curves(entries, dataset_kind, pixel_indices)
        if not per_pixel:
            self.curve_canvas.draw_idle()
            self.curve_stats_label.setText("No finite values for selected pixels.")
            return
        colors = self._get_colors(len(per_pixel))
        all_values: list[float] = []
        line_style = self.line_style_combo.currentData() or "-"
        marker = self.marker_combo.currentData() or None
        line_width = float(self.line_width_spin.value())
        marker_size = float(self.marker_size_spin.value())
        plot_kwargs = dict(
            marker=marker or None,
            linestyle=line_style,
            linewidth=line_width,
            markersize=marker_size,
        )
        for (idx, (xs, ys)), color in zip(per_pixel.items(), colors):
            if color:
                ax.plot(xs, ys, color=color, label=f"Pixel {idx}", **plot_kwargs)
            else:
                ax.plot(xs, ys, label=f"Pixel {idx}", **plot_kwargs)
            legend_labels.append(f"Pixel {idx}")
            all_values.extend(ys)
            axis_x.extend(xs)
            axis_y.extend(ys)
        if all_values:
            arr = np.array(all_values, dtype=float)
            stats_message = (
                f"Curves: {len(per_pixel)}    Samples: {arr.size}    "
                f"Min: {np.nanmin(arr):.3e} A    Max: {np.nanmax(arr):.3e} A"
            )
        else:
            stats_message = "All pixel curves are empty."

        if dataset_kind == "voltage":
            ax.set_xlabel("Bias Voltage (V)")
        else:
            axis_mode = self.time_axis_combo.currentData()
            ax.set_xlabel("Elapsed time (s)" if axis_mode == "elapsed" else "Loop index")
        ax.set_ylabel("Current (A)")
        default_title = f"Current vs {'Voltage' if dataset_kind == 'voltage' else 'Time'} (Individual Pixels)"
        custom_title = (self.plot_title_edit.text() or "").strip()
        ax.set_title(custom_title or default_title)

        if self.legend_checkbox.isChecked() and legend_labels:
            ax.legend()
        else:
            leg = ax.get_legend()
            if leg:
                leg.remove()

        self._apply_axis_formatting(ax, axis_x, axis_y)
        self.curve_canvas.figure.tight_layout()
        self.curve_canvas.draw_idle()

        self.curve_stats_label.setText(stats_message or "No statistics available.")

    def _compute_curve_points(
        self,
        entries: Iterable[VoltageEntry | TimeEntry],
        dataset_kind: str,
        pixel_indices: Sequence[int],
        aggregator: Callable[[np.ndarray], float],
    ) -> tuple[list[float], list[float]]:
        x_values: list[float] = []
        y_values: list[float] = []
        axis_mode = self.time_axis_combo.currentData()

        mask = np.zeros((10, 10), dtype=bool)
        for idx in pixel_indices:
            if idx < 1 or idx > 100:
                continue
            r, c = divmod(idx - 1, 10)
            mask[r, c] = True

        for entry in entries:
            data = entry.data
            selected = data[mask]
            if selected.size == 0:
                continue
            y = aggregator(selected)
            if not math.isfinite(y):
                continue
            if dataset_kind == "voltage" and isinstance(entry, VoltageEntry):
                x = float(entry.voltage)
            elif dataset_kind == "time" and isinstance(entry, TimeEntry):
                x = float(entry.elapsed_s) if axis_mode == "elapsed" else float(entry.index)
            else:
                continue
            x_values.append(x)
            y_values.append(y)

        return x_values, y_values

    def _compute_per_pixel_curves(
        self,
        entries: Iterable[VoltageEntry | TimeEntry],
        dataset_kind: str,
        pixel_indices: Sequence[int],
    ) -> dict[int, tuple[list[float], list[float]]]:
        result: dict[int, tuple[list[float], list[float]]] = {}
        axis_mode = self.time_axis_combo.currentData()
        for idx in pixel_indices:
            if idx < 1 or idx > 100:
                continue
            xs: list[float] = []
            ys: list[float] = []
            r, c = divmod(idx - 1, 10)
            for entry in entries:
                if dataset_kind == "voltage" and isinstance(entry, VoltageEntry):
                    x_value = float(entry.voltage)
                elif dataset_kind == "time" and isinstance(entry, TimeEntry):
                    x_value = float(entry.elapsed_s) if axis_mode == "elapsed" else float(entry.index)
                else:
                    continue
                value = entry.data[r, c]
                if not math.isfinite(value):
                    continue
                xs.append(x_value)
                ys.append(float(value))
            if xs:
                result[idx] = (xs, ys)
        return result

    def _get_colors(self, count: int) -> List[Optional[str]]:
        if count <= 0:
            return []
        scheme_label = self.color_scheme_combo.currentText() or "Default"
        cmap_name = COLOR_SCHEME_MAP.get(scheme_label, None)
        if not cmap_name:
            return [None] * count
        try:
            cmap = plt.get_cmap(cmap_name)
        except ValueError:
            return [None] * count
        colors: List[str] = []
        palette = getattr(cmap, "colors", None)
        if palette:
            palette = list(palette)
            for i in range(count):
                colors.append(to_hex(palette[i % len(palette)]))
        else:
            if count == 1:
                colors.append(to_hex(cmap(0.0)))
            else:
                for i in range(count):
                    frac = i / (count - 1)
                    colors.append(to_hex(cmap(frac)))
        return colors

    def _compute_resistivity(self) -> None:
        if len(self._voltage_entries) < 2:
            QtWidgets.QMessageBox.warning(
                self, "Resistivity", "Load at least two voltage steps to perform a fit."
            )
            return

        voltages = np.array([entry.voltage for entry in self._voltage_entries], dtype=float)
        current_stack = np.array([entry.data for entry in self._voltage_entries], dtype=float)  # shape (n,10,10)
        if current_stack.shape[1:] != (10, 10):
            QtWidgets.QMessageBox.warning(self, "Resistivity", "Voltage data is not 10x10.")
            return

        flat_currents = current_stack.reshape(current_stack.shape[0], -1)
        slopes = np.full(flat_currents.shape[1], np.nan, dtype=float)
        intercepts = np.full_like(slopes, np.nan)
        force_origin = self.force_origin_checkbox.isChecked()

        for idx in range(flat_currents.shape[1]):
            y = flat_currents[:, idx]
            valid_mask = np.isfinite(y) & np.isfinite(voltages)
            if valid_mask.sum() < 2:
                continue
            x_valid = voltages[valid_mask]
            y_valid = y[valid_mask]
            if force_origin:
                denom = np.sum(x_valid * x_valid)
                if denom <= 0:
                    continue
                slope = np.sum(x_valid * y_valid) / denom
                intercept = 0.0
            else:
                try:
                    coeff = np.polyfit(x_valid, y_valid, 1)
                    slope, intercept = float(coeff[0]), float(coeff[1])
                except Exception:
                    continue
            slopes[idx] = slope
            intercepts[idx] = intercept

        slopes = slopes.reshape((10, 10))
        intercepts = intercepts.reshape((10, 10))
        resistivity = np.full_like(slopes, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            mask = np.isfinite(slopes) & (np.abs(slopes) > 1e-12)
            resistivity[mask] = 1.0 / slopes[mask]

        self._resistivity_map = resistivity
        self._render_resistivity(resistivity)
        valid = resistivity[np.isfinite(resistivity)]
        if valid.size:
            self.res_status_label.setText(
                f"Computed {valid.size} resistivity values. "
                f"Mean={valid.mean():.3e} Ω, Median={np.median(valid):.3e} Ω."
            )
        else:
            self.res_status_label.setText("Resistivity calculation produced no finite values.")
        self.export_heatmap_btn.setEnabled(valid.size > 0)
        self.export_hist_btn.setEnabled(valid.size > 0)

    def _render_resistivity(self, resistivity: np.ndarray) -> None:
        self.res_heatmap_canvas.clear()
        ax_hm = self.res_heatmap_canvas.ax
        finite_values = resistivity[np.isfinite(resistivity)]
        if finite_values.size:
            vmin = float(np.nanmin(finite_values))
            vmax = float(np.nanmax(finite_values))
            norm = Normalize(vmin=vmin, vmax=vmax)
            im = ax_hm.imshow(resistivity, cmap="viridis", norm=norm)
            cbar = self.res_heatmap_canvas.figure.colorbar(im, ax=ax_hm)
            cbar.set_label("Resistivity (Ω)")
        else:
            ax_hm.imshow(np.zeros((10, 10)), cmap="viridis")
        ax_hm.set_title("Resistivity Heatmap")
        ax_hm.set_xticks([])
        ax_hm.set_yticks([])
        self.res_heatmap_canvas.figure.tight_layout()
        self.res_heatmap_canvas.draw_idle()

        self.res_hist_canvas.clear()
        ax_hist = self.res_hist_canvas.ax
        valid = resistivity[np.isfinite(resistivity)]
        if valid.size:
            ax_hist.hist(valid, bins=min(40, max(10, valid.size // 3)), color="#ef6c00", alpha=0.8)
        ax_hist.set_xlabel("Resistivity (Ω)")
        ax_hist.set_ylabel("Pixel Count")
        ax_hist.set_title("Resistivity Distribution")
        ax_hist.grid(True, linestyle="--", alpha=0.4)
        self.res_hist_canvas.figure.tight_layout()
        self.res_hist_canvas.draw_idle()

    def _export_resistivity_heatmap(self) -> None:
        if self._resistivity_map is None:
            return
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Resistivity Heatmap",
            str(self._current_folder or Path.home() / "resistivity_heatmap.png"),
            "PNG Files (*.png)",
        )
        if fname:
            self.res_heatmap_canvas.figure.savefig(fname, dpi=300, bbox_inches="tight")

    def _export_resistivity_histogram(self) -> None:
        if self._resistivity_map is None:
            return
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Resistivity Histogram",
            str(self._current_folder or Path.home() / "resistivity_histogram.png"),
            "PNG Files (*.png)",
        )
        if fname:
            self.res_hist_canvas.figure.savefig(fname, dpi=300, bbox_inches="tight")

    def _browse_video_folder(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Image Folder", self.video_folder_edit.text() or str(Path.home())
        )
        if directory:
            self.video_folder_edit.setText(directory)

    def _browse_video_output(self) -> None:
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select Output Video",
            self.video_output_edit.text() or str(Path.home() / "heatmap_sequence.mp4"),
            "MP4 Video (*.mp4);;AVI Video (*.avi);;All Files (*)",
        )
        if fname:
            self.video_output_edit.setText(fname)

    def _populate_video_defaults(self) -> None:
        images = self._heatmap_images or self._hist_images
        if not images:
            return
        folder = images[0].parent
        self.video_folder_edit.setText(str(folder))
        if images is self._heatmap_images:
            pattern = "voltage_*_heatmap.png"
            default_name = f"{folder.name}_heatmap.mp4"
        else:
            pattern = "*histogram*.png"
            default_name = f"{folder.name}_histogram.mp4"
        self.video_pattern_edit.setText(pattern)
        default_out = folder / default_name
        self.video_output_edit.setText(str(default_out))

    def _create_video(self) -> None:
        folder = Path(self.video_folder_edit.text().strip()).expanduser()
        if not folder.is_dir():
            QtWidgets.QMessageBox.critical(self, "Video", f"Image folder not found: {folder}")
            return
        pattern = self.video_pattern_edit.text().strip() or "*.png"
        images = sorted(folder.glob(pattern))
        if not images:
            QtWidgets.QMessageBox.warning(
                self, "Video", f"No images match pattern '{pattern}' in {folder}"
            )
            return

        output_text = self.video_output_edit.text().strip()
        if not output_text:
            QtWidgets.QMessageBox.warning(self, "Video", "Specify an output video filename.")
            return
        output_path = Path(output_text).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fps = self.video_fps_spin.value()
        glob_pattern = (folder / pattern).as_posix()
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-pattern_type",
            "glob",
            "-i",
            glob_pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.video_status_label.setText(f"Video created: {output_path}")
        except FileNotFoundError:
            QtWidgets.QMessageBox.critical(
                self,
                "Video",
                "ffmpeg executable not found. Install ffmpeg and ensure it is on your PATH.",
            )
        except subprocess.CalledProcessError as exc:
            msg = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            QtWidgets.QMessageBox.critical(
                self,
                "Video",
                f"ffmpeg failed with error:\n{msg}",
            )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _decode_voltage_tag(tag: str) -> float:
        cleaned = tag.upper().strip()
        if cleaned.endswith("V"):
            cleaned = cleaned[:-1]
        if not cleaned:
            raise ValueError("Empty voltage tag")
        sign_char = cleaned[0]
        rest = cleaned[1:] or "0"
        if sign_char == "P":
            sign = 1.0
        elif sign_char == "M":
            sign = -1.0
        else:
            # No explicit sign encoded; treat as positive
            sign = 1.0
            rest = cleaned
        numeric_str = rest.replace("P", ".")
        try:
            magnitude = float(numeric_str)
        except ValueError as exc:
            raise ValueError(f"Cannot decode voltage tag '{tag}'") from exc
        return sign * magnitude

    @staticmethod
    def _default_pixel_parser(spec: str) -> List[int]:
        indices: set[int] = set()
        s = (spec or "").replace(" ", "").strip()
        if not s:
            return list(range(1, 101))
        for token in s.split(","):
            if not token:
                continue
            if "-" in token:
                a_str, b_str = token.split("-", 1)
                a, b = int(a_str), int(b_str)
                step = 1 if b >= a else -1
                for k in range(a, b + step, step):
                    if 1 <= k <= 100:
                        indices.add(k)
            else:
                k = int(token)
                if 1 <= k <= 100:
                    indices.add(k)
        if not indices:
            raise ValueError("No valid pixel indices parsed.")
        return sorted(indices)
