from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import Normalize, to_hex
import matplotlib.pyplot as plt
from PySide6 import QtCore, QtWidgets, QtGui

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


def _agg_average(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return math.nan
    return float(valid.mean())


def _agg_sum(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return math.nan
    return float(valid.sum())


def _agg_median(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return math.nan
    return float(np.median(valid))


def _agg_min(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return math.nan
    return float(valid.min())


def _agg_max(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return math.nan
    return float(valid.max())


AGGREGATORS: dict[str, Callable[[np.ndarray], float]] = {
    "Average": _agg_average,
    "Sum": _agg_sum,
    "Median": _agg_median,
    "Min": _agg_min,
    "Max": _agg_max,
}

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

        # Settings group
        settings_group = QtWidgets.QGroupBox("Curve Settings")
        settings_form = QtWidgets.QFormLayout(settings_group)

        self.dataset_combo = QtWidgets.QComboBox()
        self.dataset_combo.addItem("Time (Loop)", userData="time")
        self.dataset_combo.addItem("Voltage Sweep", userData="voltage")
        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_changed)
        settings_form.addRow("Dataset:", self.dataset_combo)

        self.plot_mode_combo = QtWidgets.QComboBox()
        self.plot_mode_combo.addItem("Aggregate selection", userData="aggregate")
        self.plot_mode_combo.addItem("Individual pixel curves", userData="individual")
        self.plot_mode_combo.currentIndexChanged.connect(self._on_plot_mode_changed)
        settings_form.addRow("Plot Mode:", self.plot_mode_combo)

        self.time_axis_combo = QtWidgets.QComboBox()
        self.time_axis_combo.addItem("Elapsed seconds", userData="elapsed")
        self.time_axis_combo.addItem("Loop index", userData="index")
        settings_form.addRow("Time Axis:", self.time_axis_combo)

        self.pixel_spec_edit = QtWidgets.QLineEdit("1-100")
        settings_form.addRow("Pixels:", self.pixel_spec_edit)

        self.aggregate_combo = QtWidgets.QComboBox()
        for name in AGGREGATORS:
            self.aggregate_combo.addItem(name)
        settings_form.addRow("Combine:", self.aggregate_combo)

        self.color_scheme_combo = QtWidgets.QComboBox()
        for label in ["Default", "Tab10", "Tab20", "Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Rainbow"]:
            self.color_scheme_combo.addItem(label)
        settings_form.addRow("Color Scheme:", self.color_scheme_combo)

        self.legend_checkbox = QtWidgets.QCheckBox("Show legend")
        settings_form.addRow(self.legend_checkbox)

        self.curve_refresh_btn = QtWidgets.QPushButton("Plot Curve")
        self.curve_refresh_btn.clicked.connect(self._plot_curve)
        settings_form.addRow(self.curve_refresh_btn)

        layout.addWidget(settings_group)

        # Plot canvas
        self.curve_canvas = MatplotlibCanvas(width=6.0, height=4.5, dpi=100)
        layout.addWidget(self.curve_canvas, stretch=1)

        self.curve_stats_label = QtWidgets.QLabel("")
        self.curve_stats_label.setWordWrap(True)
        layout.addWidget(self.curve_stats_label)

        self.tabs.addTab(tab, "Curves")
        self._on_dataset_changed()
        self._on_plot_mode_changed()

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

    # ------------------------------------------------------------------ public helpers
    def set_run_folder(self, folder: Path, *, auto_load: bool = False) -> None:
        """Expose ability to inject a folder from the main window."""
        self.curve_folder_edit.setText(str(folder))
        self.video_folder_edit.setText(str(folder))
        if auto_load:
            self._load_run_folder(folder)

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
        time_temp: list[tuple[int, datetime, np.ndarray, Path]] = []
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
                arr = np.loadtxt(csv_path, delimiter=",", dtype=float)
                if arr.size == 100:
                    arr = arr.reshape((10, 10))
                if arr.shape != (10, 10):
                    raise ValueError(f"Unexpected array shape {arr.shape}")
                timestamp = datetime.fromtimestamp(csv_path.stat().st_mtime)

                if name.startswith("voltage_") and name.endswith("_data.csv"):
                    parts = name.split("_")
                    if len(parts) < 3:
                        raise ValueError("voltage filename missing tag")
                    idx = int(parts[1])
                    tag = parts[2]
                    voltage = self._decode_voltage_tag(tag)
                    voltage_entries.append(
                        VoltageEntry(
                            index=idx,
                            voltage=voltage,
                            data=arr,
                            path=csv_path,
                            timestamp=timestamp,
                        )
                    )
                    guess_heatmap = csv_path.with_name(csv_path.stem.replace("_data", "_heatmap") + ".png")
                    if guess_heatmap.exists():
                        heatmap_images.append(guess_heatmap)
                elif name.startswith("loop_") and name.endswith("_data.csv"):
                    parts = name.split("_")
                    if len(parts) < 3:
                        raise ValueError("loop filename missing tag")
                    idx = int(parts[1])
                    time_temp.append((idx, timestamp, arr, csv_path))
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
            base_ts = min(t for _, t, _, _ in time_temp)
            for idx, timestamp, arr, path in time_temp:
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
        has_time = bool(self._time_entries)
        has_voltage = bool(self._voltage_entries)
        dataset_flags = (has_time, has_voltage)
        model = self.dataset_combo.model()
        if isinstance(model, QtGui.QStandardItemModel):
            for row, enabled in enumerate(dataset_flags):
                item = model.item(row)
                if item is not None:
                    item.setEnabled(bool(enabled))
        view = self.dataset_combo.view()
        if view is not None:
            for row, enabled in enumerate(dataset_flags):
                view.setRowHidden(row, not enabled)

        if not has_time and self.dataset_combo.currentData() == "time" and has_voltage:
            self.dataset_combo.setCurrentIndex(1)
        elif not has_voltage and self.dataset_combo.currentData() == "voltage" and has_time:
            self.dataset_combo.setCurrentIndex(0)
        elif not has_time and not has_voltage:
            self.curve_canvas.clear()
            self.curve_canvas.draw_idle()

        self.compute_res_btn.setEnabled(has_voltage and len(self._voltage_entries) >= 2)

    def _on_dataset_changed(self) -> None:
        is_time = self.dataset_combo.currentData() == "time"
        self.time_axis_combo.setVisible(is_time)
        self._on_plot_mode_changed()

    def _on_plot_mode_changed(self) -> None:
        plot_mode = self.plot_mode_combo.currentData()
        aggregate = plot_mode == "aggregate"
        self.aggregate_combo.setEnabled(aggregate)
        self.legend_checkbox.setEnabled(True)
        self.color_scheme_combo.setEnabled(True)

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
            pixel_indices = list(self._pixel_parser(self.pixel_spec_edit.text()))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Pixels", f"Invalid pixel selection: {exc}")
            return

        if not pixel_indices:
            QtWidgets.QMessageBox.warning(self, "Pixels", "Select at least one pixel.")
            return

        agg_name = self.aggregate_combo.currentText()
        plot_mode = self.plot_mode_combo.currentData()

        self.curve_canvas.clear()
        ax = self.curve_canvas.ax
        ax.grid(True, linestyle="--", alpha=0.4)

        stats_message = ""
        legend_labels: list[str] = []
        if plot_mode == "aggregate":
            agg_fn = AGGREGATORS.get(agg_name)
            if not agg_fn:
                QtWidgets.QMessageBox.warning(self, "Combine", f"Unknown aggregator: {agg_name}")
                return
            x_vals, y_vals = self._compute_curve_points(entries, dataset_kind, pixel_indices, agg_fn)
            if not x_vals:
                self.curve_canvas.draw_idle()
                self.curve_stats_label.setText("No valid data points for the current selection.")
                return
            color = self._get_colors(1)[0]
            if color:
                ax.plot(x_vals, y_vals, marker="o", linestyle="-", color=color, label="Selection")
            else:
                ax.plot(x_vals, y_vals, marker="o", linestyle="-", label="Selection")
            stats = np.array(y_vals, dtype=float)
            stats = stats[np.isfinite(stats)]
            if stats.size:
                stats_message = (
                    f"Points: {stats.size}    Min: {stats.min():.3e} A    "
                    f"Max: {stats.max():.3e} A    Mean: {stats.mean():.3e} A"
                )
            else:
                stats_message = "All curve points are NaN."
            legend_labels = ["Selection"]
        else:
            per_pixel = self._compute_per_pixel_curves(entries, dataset_kind, pixel_indices)
            if not per_pixel:
                self.curve_canvas.draw_idle()
                self.curve_stats_label.setText("No finite values for selected pixels.")
                return
            colors = self._get_colors(len(per_pixel))
            all_values: list[float] = []
            for (idx, (xs, ys)), color in zip(per_pixel.items(), colors):
                if color:
                    ax.plot(xs, ys, marker="o", linestyle="-", color=color, label=f"Pixel {idx}")
                else:
                    ax.plot(xs, ys, marker="o", linestyle="-", label=f"Pixel {idx}")
                legend_labels.append(f"Pixel {idx}")
                all_values.extend(ys)
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
        if plot_mode == "aggregate":
            mode_title = f"Aggregate • {agg_name}"
        else:
            mode_title = "Individual Pixels"
        ax.set_title(f"Current vs {'Voltage' if dataset_kind == 'voltage' else 'Time'} ({mode_title})")

        if self.legend_checkbox.isChecked() and legend_labels:
            ax.legend()
        else:
            leg = ax.get_legend()
            if leg:
                leg.remove()

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
