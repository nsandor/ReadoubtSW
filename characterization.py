from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from PySide6 import QtCore

from Scanner import ScanWorker
from ScannerWin.acquisition import AcquisitionSettings

LOG = logging.getLogger("readoubt.characterization")


# ------------------------------ data classes ------------------------------
@dataclass
class CharacterizationSettings:
    pixel_spec: str = "1-100"
    samples_per_pixel: int = 1
    nplc: float = 10.0
    auto_range: bool = True
    current_range: float = 1e-7
    use_local_readout: bool = False
    use_local_bias: bool = False
    route_settle_ms: int = 4

    short_threshold_a: float = 1e-7
    pin_map_path: Optional[str] = None

    dead_short_voltage_v: float = 0.1
    dead_short_threshold_a: float = 1e-7
    stop_on_dead_short: bool = True

    resistance_start_v: float = -1.0
    resistance_end_v: float = 1.0
    resistance_step_v: float = 0.05
    resistance_settle_s: float = 0.0

    operating_field_v_per_cm: float = 100.0
    device_thickness_cm: float = 1.0
    operating_settle_s: float = 60.0
    operating_current_limit_a: Optional[float] = None

    jv_dark_start_v: float = -150.0
    jv_dark_end_v: float = 150.0
    jv_dark_step_v: float = 5.0
    jv_dark_zero_pause_s: float = 2.0
    jv_dark_settle_s: float = 0.0
    jv_dark_current_limit_a: float = 5e-6
    jv_dark_zero_center: bool = True

    jt_light_bias_v: float = 150.0
    jt_light_samples_per_pixel: int = 1
    jt_light_threshold_a: float = 0.2e-9
    jt_light_use_led: bool = True
    jt_light_current_limit_a: Optional[float] = None
    jt_light_settle_s: float = 0.0

    analysis_active_threshold_a: float = 0.2e-9
    output_subdir: str = "characterization"
    histogram_bins: int = 30
    histogram_sigma_clip: float = 0.0
    heatmap_cmap_resistance: str = "viridis"
    heatmap_cmap_dark: str = "inferno"
    plot_units_current: str = "nA"
    plot_units_resistance: str = "Ohm"
    plot_vmin_current: Optional[float] = None
    plot_vmax_current: Optional[float] = None
    plot_vmin_resistance: Optional[float] = None
    plot_vmax_resistance: Optional[float] = None


@dataclass
class LoopResult:
    index: int
    voltage: Optional[float]
    data: np.ndarray
    metadata: dict


@dataclass
class ScanResult:
    name: str
    loops: List[LoopResult]
    excluded_pixels: Set[int] = field(default_factory=set)
    output_dir: Optional[Path] = None


@dataclass
class CharacterizationSummary:
    suite_root: Path
    shorted_pixels: Set[int] = field(default_factory=set)
    dead_short_pixels: Set[int] = field(default_factory=set)
    wide_jv_over_limit: Set[int] = field(default_factory=set)
    active_pixels: Set[int] = field(default_factory=set)
    resistance_map: Optional[np.ndarray] = None
    dark_current_map: Optional[np.ndarray] = None
    r2_values: Dict[int, float] = field(default_factory=dict)
    plot_paths: Dict[str, Path] = field(default_factory=dict)
    test_dirs: Dict[str, Path] = field(default_factory=dict)
    metadata_path: Optional[Path] = None


# ------------------------------ helpers ------------------------------
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


def _apply_route_settle_time(switch, ms: int):
    try:
        ms_val = max(0, int(ms))
    except Exception:
        return
    try:
        if hasattr(switch, "set_settle_time"):
            switch.set_settle_time(ms_val)
    except Exception as exc:
        LOG.warning("Failed to apply routing settle time: %s", exc)


def _loop_to_csv_name(base: str, loop_idx: int, voltage: Optional[float]) -> str:
    if voltage is None:
        return f"{base}_{loop_idx:03d}_data.csv"
    tag = f"{voltage:+.3f}V".replace("+", "p").replace("-", "m").replace(".", "p")
    return f"{base}_{loop_idx:03d}_{tag}.csv"


# ------------------------------ worker ------------------------------
class CharacterizationWorker(QtCore.QObject):
    progressChanged = QtCore.Signal(int, int, str)
    stepsPlanned = QtCore.Signal(object)
    stepProgress = QtCore.Signal(str, int, int)
    statusMessage = QtCore.Signal(str)
    finished = QtCore.Signal(bool, object)
    error = QtCore.Signal(str)

    def __init__(
        self,
        *,
        settings: CharacterizationSettings,
        pixel_indices: Sequence[int],
        sm,
        bias_sm,
        switch,
        output_root: Path,
    ):
        super().__init__()
        self._settings = settings
        self._pixels = list(pixel_indices)
        self._sm = sm
        self._bias_sm = bias_sm
        self._switch = switch
        self._root = output_root
        self._stop = False
        self._captures: Dict[str, ScanResult] = {}
        self._summary = CharacterizationSummary(suite_root=output_root)
        self._total_steps = 1
        self._done_steps = 0
        self._active_scan_worker: Optional[ScanWorker] = None
        self._step_plan: list[tuple[str, str, int]] = []
        self._step_state: dict[str, dict] = {}

    @QtCore.Slot()
    def run(self):
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            self._step_plan = self._build_step_plan()
            self.stepsPlanned.emit(self._step_plan)
            total = self._estimate_total_loops()
            self._total_steps = total
            _apply_route_settle_time(self._switch, self._settings.route_settle_ms)
            self._emit_progress("Starting characterization suite", reset=True)
            self._run_suite()
            self._write_metadata()
            self.finished.emit(not self._stop, self._summary)
        except Exception as exc:
            LOG.exception("Characterization suite failed: %s", exc)
            self.error.emit(str(exc))
            self.finished.emit(False, self._summary)

    def stop(self):
        self._stop = True
        if self._active_scan_worker is not None:
            try:
                self._active_scan_worker.stop()
            except Exception:
                pass

    def _build_step_plan(self) -> list[tuple[str, str, int]]:
        plan: list[tuple[str, str, int]] = []
        res_steps = len(
            _generate_voltage_steps(
                self._settings.resistance_start_v,
                self._settings.resistance_end_v,
                self._settings.resistance_step_v,
            )
        )
        wide_steps = len(
            _apply_zero_centering(
                _generate_voltage_steps(
                    self._settings.jv_dark_start_v,
                    self._settings.jv_dark_end_v,
                    self._settings.jv_dark_step_v,
                ),
                self._settings.jv_dark_zero_center,
            )
        )
        plan.append(("short", "Shorted pixel test", 1))
        plan.append(("dead", "Dead short test", 1))
        plan.append(("resistance", "Resistance JV", max(1, res_steps)))
        plan.append(("dark_operating", "Dark @ operating bias", 1))
        plan.append(("wide_dark", "Wide dark JV", max(1, wide_steps)))
        plan.append(("light_jt", "Light JT", 1))
        plan.append(("plots", "Plot generation", 1))
        return plan

    def _begin_step(self, key: str):
        entry = next((p for p in self._step_plan if p[0] == key), None)
        total = entry[2] if entry else 1
        self._step_state[key] = {"done": 0, "total": max(1, int(total))}
        self.stepProgress.emit(key, 0, max(1, int(total)))

    def _increment_step(self, key: str, inc: int = 1):
        state = self._step_state.get(key)
        if not state:
            return
        state["done"] = min(state["total"], state.get("done", 0) + max(1, int(inc)))
        self.stepProgress.emit(key, int(state["done"]), int(state["total"]))

    def _complete_step(self, key: str):
        state = self._step_state.get(key)
        if not state:
            return
        state["done"] = state["total"]
        self.stepProgress.emit(key, int(state["done"]), int(state["total"]))

    # -------------------------- suite orchestration --------------------------
    def _run_suite(self):
        self._begin_step("short")
        self._run_shorted_pixel_test()
        self._complete_step("short")
        if self._stop:
            return
        self._begin_step("dead")
        dead_found = self._run_dead_short_test()
        self._complete_step("dead")
        if self._stop or (dead_found and self._settings.stop_on_dead_short):
            return
        self._begin_step("resistance")
        self._run_resistance_test()
        self._complete_step("resistance")
        if self._stop:
            return
        self._begin_step("dark_operating")
        self._run_operating_dark_current()
        self._complete_step("dark_operating")
        if self._stop:
            return
        self._begin_step("wide_dark")
        self._run_wide_dark_jv()
        self._complete_step("wide_dark")
        if self._stop:
            return
        self._begin_step("light_jt")
        self._run_light_jt()
        self._complete_step("light_jt")
        if self._stop:
            return
        self._begin_step("plots")
        self._generate_plots()
        self._complete_step("plots")

    def _estimate_total_loops(self) -> int:
        try:
            return max(1, sum(total for _, _, total in self._step_plan))
        except Exception:
            return 10

    # -------------------------- individual tests --------------------------
    def _run_shorted_pixel_test(self):
        name = "shorted_pixels"
        output_dir = self._root / "01_shorted_pixels"
        settings = self._settings
        acquisition = AcquisitionSettings(
            pixels=self._pixels,
            samples_per_pixel=settings.samples_per_pixel,
            nplc=settings.nplc,
            loops=1,
            inter_loop_delay_s=0.0,
            auto_range=settings.auto_range,
            current_range=settings.current_range,
            current_limit=None,
            measurement_mode="time",
            voltage_steps=None,
            voltage_settle_s=0.0,
            voltage_zero_center=False,
            voltage_zero_pause_s=0.0,
            constant_bias_voltage=None,
            use_local_readout=settings.use_local_readout,
            use_local_bias=settings.use_local_bias,
        )
        capture = self._run_scan(name=name, acquisition=acquisition, output_dir=output_dir, step_key="short")
        self._captures[name] = capture
        matrix = capture.loops[0].data if capture.loops else np.full((10, 10), np.nan)
        shorted: Set[int] = set()
        threshold = abs(settings.short_threshold_a)
        pin_map = self._load_pin_map(settings.pin_map_path)
        summaries = []
        for idx, value in enumerate(matrix.flatten(), start=1):
            if np.isnan(value):
                continue
            if abs(value) >= threshold:
                shorted.add(idx)
                entry = {"pixel": idx, "current_a": float(value)}
                if pin_map and idx in pin_map:
                    entry["pin"] = pin_map[idx]
                summaries.append(entry)
        summary_path = output_dir / "shorted_pixels.json"
        summary_payload = {"threshold_a": threshold, "shorted": summaries}
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        self._summary.shorted_pixels = shorted
        self._summary.test_dirs[name] = output_dir
        self._emit_progress("Shorted pixel test complete")

    def _run_dead_short_test(self) -> bool:
        name = "dead_short"
        output_dir = self._root / "02_dead_short"
        settings = self._settings
        acquisition = AcquisitionSettings(
            pixels=self._pixels,
            samples_per_pixel=settings.samples_per_pixel,
            nplc=settings.nplc,
            loops=1,
            inter_loop_delay_s=0.0,
            auto_range=settings.auto_range,
            current_range=settings.current_range,
            current_limit=settings.dead_short_threshold_a,
            measurement_mode="time",
            voltage_steps=None,
            voltage_settle_s=0.0,
            voltage_zero_center=False,
            voltage_zero_pause_s=0.0,
            constant_bias_voltage=settings.dead_short_voltage_v,
            use_local_readout=settings.use_local_readout,
            use_local_bias=settings.use_local_bias,
        )
        capture = self._run_scan(name=name, acquisition=acquisition, output_dir=output_dir, step_key="dead")
        self._captures[name] = capture
        matrix = capture.loops[0].data if capture.loops else np.full((10, 10), np.nan)
        threshold = abs(settings.dead_short_threshold_a)
        dead_pixels: Set[int] = set()
        flagged = []
        for idx, value in enumerate(matrix.flatten(), start=1):
            if np.isnan(value):
                continue
            if abs(value) >= threshold:
                dead_pixels.add(idx)
                flagged.append({"pixel": idx, "current_a": float(value)})
        dead_pixels.update(capture.excluded_pixels)
        for p in capture.excluded_pixels:
            flagged.append({"pixel": p, "current_a": None, "reason": "current_limit"})
        summary_path = output_dir / "dead_shorts.json"
        summary_payload = {
            "threshold_a": threshold,
            "dead_shorted_pixels": sorted(dead_pixels),
            "details": flagged,
            "stop_on_dead_short": settings.stop_on_dead_short,
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        self._summary.dead_short_pixels = dead_pixels
        self._summary.test_dirs[name] = output_dir
        self._emit_progress(
            "Dead short test complete"
            + (" – stopping remaining tests" if dead_pixels and settings.stop_on_dead_short else "")
        )
        return bool(dead_pixels)

    def _run_resistance_test(self):
        name = "resistance_jv"
        output_dir = self._root / "03_resistance_jv"
        settings = self._settings
        voltages = _generate_voltage_steps(
            settings.resistance_start_v,
            settings.resistance_end_v,
            settings.resistance_step_v,
        )
        acquisition = AcquisitionSettings(
            pixels=self._pixels,
            samples_per_pixel=settings.samples_per_pixel,
            nplc=settings.nplc,
            loops=len(voltages),
            inter_loop_delay_s=0.0,
            auto_range=settings.auto_range,
            current_range=settings.current_range,
            current_limit=None,
            measurement_mode="voltage",
            voltage_steps=voltages,
            voltage_settle_s=settings.resistance_settle_s,
            voltage_zero_center=False,
            voltage_zero_pause_s=0.0,
            constant_bias_voltage=None,
            use_local_readout=settings.use_local_readout,
            use_local_bias=settings.use_local_bias,
        )
        capture = self._run_scan(name=name, acquisition=acquisition, output_dir=output_dir, step_key="resistance")
        self._captures[name] = capture
        self._summary.test_dirs[name] = output_dir
        self._emit_progress("Resistance JV sweep complete")

    def _run_operating_dark_current(self):
        name = "dark_current_operating"
        output_dir = self._root / "04_dark_current_operating"
        settings = self._settings
        applied_v = float(settings.operating_field_v_per_cm) * float(settings.device_thickness_cm)
        acquisition = AcquisitionSettings(
            pixels=self._pixels,
            samples_per_pixel=settings.samples_per_pixel,
            nplc=settings.nplc,
            loops=1,
            inter_loop_delay_s=0.0,
            auto_range=settings.auto_range,
            current_range=settings.current_range,
            current_limit=settings.operating_current_limit_a,
            measurement_mode="time",
            voltage_steps=None,
            voltage_settle_s=max(0.0, settings.operating_settle_s),
            voltage_zero_center=False,
            voltage_zero_pause_s=0.0,
            constant_bias_voltage=applied_v,
            use_local_readout=settings.use_local_readout,
            use_local_bias=settings.use_local_bias,
        )
        capture = self._run_scan(name=name, acquisition=acquisition, output_dir=output_dir, step_key="dark_operating")
        self._captures[name] = capture
        self._summary.test_dirs[name] = output_dir
        self._emit_progress("Dark current at operating bias captured")

    def _run_wide_dark_jv(self):
        name = "wide_dark_jv"
        output_dir = self._root / "05_wide_dark_jv"
        settings = self._settings
        voltages = _apply_zero_centering(
            _generate_voltage_steps(
                settings.jv_dark_start_v,
                settings.jv_dark_end_v,
                settings.jv_dark_step_v,
            ),
            settings.jv_dark_zero_center,
        )
        acquisition = AcquisitionSettings(
            pixels=self._pixels,
            samples_per_pixel=settings.samples_per_pixel,
            nplc=settings.nplc,
            loops=len(voltages),
            inter_loop_delay_s=0.0,
            auto_range=settings.auto_range,
            current_range=settings.current_range,
            current_limit=settings.jv_dark_current_limit_a,
            measurement_mode="voltage",
            voltage_steps=voltages,
            voltage_settle_s=settings.jv_dark_settle_s,
            voltage_zero_center=settings.jv_dark_zero_center,
            voltage_zero_pause_s=settings.jv_dark_zero_pause_s,
            constant_bias_voltage=None,
            use_local_readout=settings.use_local_readout,
            use_local_bias=settings.use_local_bias,
        )
        capture = self._run_scan(name=name, acquisition=acquisition, output_dir=output_dir, step_key="wide_dark")
        self._captures[name] = capture
        self._summary.test_dirs[name] = output_dir
        self._summary.wide_jv_over_limit = set(capture.excluded_pixels)
        self._emit_progress("Wide dark JV sweep complete")

    def _run_light_jt(self):
        name = "light_jt"
        output_dir = self._root / "06_light_jt"
        settings = self._settings
        # Avoid re-measuring pixels that already screamed during the wide JV.
        pixels = [p for p in self._pixels if p not in self._summary.wide_jv_over_limit and p not in self._summary.shorted_pixels]
        acquisition = AcquisitionSettings(
            pixels=pixels,
            samples_per_pixel=settings.jt_light_samples_per_pixel,
            nplc=settings.nplc,
            loops=1,
            inter_loop_delay_s=0.0,
            auto_range=settings.auto_range,
            current_range=settings.current_range,
            current_limit=settings.jt_light_current_limit_a,
            measurement_mode="time",
            voltage_steps=None,
            voltage_settle_s=max(0.0, settings.jt_light_settle_s),
            voltage_zero_center=False,
            voltage_zero_pause_s=0.0,
            constant_bias_voltage=settings.jt_light_bias_v,
            use_local_readout=settings.use_local_readout,
            use_local_bias=settings.use_local_bias,
        )
        led_enabled = settings.jt_light_use_led
        if led_enabled:
            try:
                self._switch.set_led(True)
            except Exception as exc:
                LOG.warning("Failed to enable LEDs: %s", exc)
        capture = self._run_scan(name=name, acquisition=acquisition, output_dir=output_dir, step_key="light_jt")
        if led_enabled:
            try:
                self._switch.set_led(False)
            except Exception:
                pass
        self._captures[name] = capture
        # Determine active pixels (JT above threshold or already over limit on wide JV)
        matrix = capture.loops[0].data if capture.loops else np.full((10, 10), np.nan)
        active: Set[int] = set(self._summary.wide_jv_over_limit)
        threshold = abs(settings.analysis_active_threshold_a or settings.jt_light_threshold_a)
        for idx, value in enumerate(matrix.flatten(), start=1):
            if np.isnan(value):
                continue
            if abs(value) >= threshold:
                active.add(idx)
        # Exclude shorted pixels from the active list used downstream.
        active = {p for p in active if p not in self._summary.shorted_pixels}
        self._summary.active_pixels = active
        self._summary.test_dirs[name] = output_dir
        self._emit_progress("Light JT check complete")

    # -------------------------- scan wrapper --------------------------
    def _run_scan(
        self,
        *,
        name: str,
        acquisition: AcquisitionSettings,
        output_dir: Path,
        step_key: Optional[str] = None,
    ) -> ScanResult:
        if self._stop:
            return ScanResult(name=name, loops=[], excluded_pixels=set(), output_dir=output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        LOG.info("Running characterization step '%s' into %s", name, output_dir)
        worker = ScanWorker(
            self._sm,
            self._switch,
            n_samples=acquisition.samples_per_pixel,
            nplc=acquisition.nplc,
            pixel_indices=acquisition.pixels,
            loops=acquisition.loops,
            auto_range=acquisition.auto_range,
            current_range=acquisition.current_range,
            inter_sample_delay_s=0.0,
            inter_loop_delay_s=acquisition.inter_loop_delay_s,
            bias_sm=(
                self._bias_sm
                if acquisition.constant_bias_voltage is not None or acquisition.voltage_steps
                else None
            ),
            voltage_steps=acquisition.voltage_steps,
            voltage_settle_s=acquisition.voltage_settle_s,
            voltage_zero_pause_s=acquisition.voltage_zero_pause_s,
            constant_bias_voltage=acquisition.constant_bias_voltage,
            use_local_readout=acquisition.use_local_readout,
            use_local_bias=acquisition.use_local_bias,
            current_limit=acquisition.current_limit,
        )
        matrices: Dict[int, np.ndarray] = {}
        metadata_map: Dict[int, dict] = {}
        excluded: Set[int] = set()

        def handle_data(loop_idx: int, entries):
            mat = matrices.setdefault(loop_idx, np.full((10, 10), np.nan))
            for idx, i_avg in entries or []:
                try:
                    r, c = divmod(int(idx) - 1, 10)
                except Exception:
                    continue
                if not (0 <= r < 10 and 0 <= c < 10):
                    continue
                try:
                    mat[r, c] = float(i_avg)
                except Exception:
                    mat[r, c] = np.nan

        def handle_finished(loop_idx: int, md: Optional[dict]):
            metadata_map[loop_idx] = md or {}
            self._done_steps = min(self._done_steps + 1, self._total_steps)
            self._emit_progress(f"{name} loop {loop_idx} complete")
            if step_key:
                self._increment_step(step_key)

        def handle_excluded(pixel_idx: int, measured: float, loop_idx: int):
            try:
                excluded.add(int(pixel_idx))
            except Exception:
                return

        worker.loopDataReady.connect(handle_data)
        worker.loopFinished.connect(handle_finished)
        worker.pixelExcluded.connect(handle_excluded)
        worker.deviceError.connect(lambda msg: self.error.emit(msg))
        self._active_scan_worker = worker
        try:
            worker.run()
        finally:
            self._active_scan_worker = None

        loops: List[LoopResult] = []
        for idx in sorted(matrices):
            meta = metadata_map.get(idx, {})
            voltage = meta.get("voltage")
            loops.append(
                LoopResult(
                    index=idx,
                    voltage=float(voltage) if voltage is not None else None,
                    data=matrices.get(idx, np.full((10, 10), np.nan)),
                    metadata=meta,
                )
            )
            self._save_loop(output_dir, name, loops[-1])
        return ScanResult(name=name, loops=loops, excluded_pixels=excluded, output_dir=output_dir)

    # -------------------------- analysis --------------------------
    def _generate_plots(self):
        import matplotlib.pyplot as plt

        plots_dir = self._root / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        active = set(self._summary.active_pixels)
        shorted = set(self._summary.shorted_pixels)
        active_minus_shorted = [p for p in active if p not in shorted]
        current_unit = (self._settings.plot_units_current or "A").strip()
        unit_scale = {"A": 1.0, "mA": 1e3, "µA": 1e6, "uA": 1e6, "nA": 1e9, "pA": 1e12}
        curr_scale = unit_scale.get(current_unit, 1.0)
        vmin_curr = self._settings.plot_vmin_current
        vmax_curr = self._settings.plot_vmax_current
        res_unit = (self._settings.plot_units_resistance or "Ohm").strip()
        res_scale_map = {"Ohm": 1.0, "kOhm": 1e-3, "MOhm": 1e-6, "GOhm": 1e-9}
        res_scale = res_scale_map.get(res_unit, 1.0)
        vmin_res = self._settings.plot_vmin_resistance
        vmax_res = self._settings.plot_vmax_resistance

        # Resistance heatmap + histogram
        resistance_map = np.full((10, 10), np.nan)
        res_capture = self._captures.get("resistance_jv")
        if res_capture:
            for pixel in active_minus_shorted:
                curve = self._extract_curve(res_capture, pixel)
                if not curve:
                    continue
                voltages, currents = zip(*curve)
                try:
                    coef = np.polyfit(voltages, currents, 1)
                    slope = coef[0]
                    if slope > 0:
                        resistance_map[(pixel - 1) // 10, (pixel - 1) % 10] = 1.0 / slope
                except Exception:
                    continue
            self._summary.resistance_map = resistance_map
            scaled_res = resistance_map * res_scale
            fig, ax = plt.subplots()
            im = ax.imshow(
                scaled_res,
                cmap=self._settings.heatmap_cmap_resistance,
                vmin=vmin_res if vmin_res is not None else None,
                vmax=vmax_res if vmax_res is not None else None,
            )
            plt.colorbar(im, ax=ax, label=f"Resistance ({res_unit})")
            ax.set_title("Resistance map (active pixels)")
            res_heatmap_path = plots_dir / "resistance_heatmap.png"
            fig.savefig(res_heatmap_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            self._summary.plot_paths["resistance_heatmap"] = res_heatmap_path

            valid_res = scaled_res[~np.isnan(scaled_res)]
            valid_res = self._sigma_clip(valid_res, self._settings.histogram_sigma_clip)
            if valid_res.size:
                fig, ax = plt.subplots()
                ax.hist(valid_res, bins=max(1, int(self._settings.histogram_bins)))
                ax.set_xlabel(f"Resistance ({res_unit})")
                ax.set_ylabel("Pixel count")
                ax.set_title("Resistance histogram (active pixels)")
                res_hist_path = plots_dir / "resistance_histogram.png"
                fig.savefig(res_hist_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                self._summary.plot_paths["resistance_histogram"] = res_hist_path

        # Dark current at operating bias
        dark_capture = self._captures.get("dark_current_operating")
        if dark_capture and dark_capture.loops:
            dark_map = np.array(dark_capture.loops[0].data, copy=True)
            if active_minus_shorted:
                mask = np.ones_like(dark_map, dtype=bool)
                for p in active_minus_shorted:
                    r, c = divmod(p - 1, 10)
                    mask[r, c] = False
                dark_map = np.where(mask, np.nan, dark_map)
            self._summary.dark_current_map = dark_map
            scaled_map = dark_map * curr_scale
            fig, ax = plt.subplots()
            im = ax.imshow(
                scaled_map,
                cmap=self._settings.heatmap_cmap_dark,
                vmin=vmin_curr if vmin_curr is not None else None,
                vmax=vmax_curr if vmax_curr is not None else None,
            )
            plt.colorbar(im, ax=ax, label=f"Current ({current_unit})")
            ax.set_title("Dark current @ operating bias")
            dark_heatmap_path = plots_dir / "dark_current_heatmap.png"
            fig.savefig(dark_heatmap_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            self._summary.plot_paths["dark_heatmap"] = dark_heatmap_path

            valid_dark = scaled_map[~np.isnan(scaled_map)]
            valid_dark = self._sigma_clip(valid_dark, self._settings.histogram_sigma_clip)
            if valid_dark.size:
                fig, ax = plt.subplots()
                ax.hist(valid_dark, bins=max(1, int(self._settings.histogram_bins)))
                ax.set_xlabel(f"Current ({current_unit})")
                ax.set_ylabel("Pixel count")
                ax.set_title("Dark current @ operating bias")
                dark_hist_path = plots_dir / "dark_current_histogram.png"
                fig.savefig(dark_hist_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                self._summary.plot_paths["dark_histogram"] = dark_hist_path

        # Wide dark JV curves + R^2 distribution
        wide_capture = self._captures.get("wide_dark_jv")
        if wide_capture:
            r2_map: Dict[int, float] = {}
            fig, ax = plt.subplots()
            for pixel in active_minus_shorted:
                curve = self._extract_curve(wide_capture, pixel)
                if len(curve) < 2:
                    continue
                voltages, currents = zip(*curve)
                try:
                    coef = np.polyfit(voltages, currents, 1)
                    fit = np.poly1d(coef)
                    fitted = fit(voltages)
                    ss_res = float(np.sum((currents - fitted) ** 2))
                    ss_tot = float(np.sum((currents - np.mean(currents)) ** 2))
                    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
                    r2_map[pixel] = r2
                    ax.plot(voltages, currents, alpha=0.5)
                except Exception:
                    continue
            ax.set_xlabel("Voltage (V)")
            ax.set_ylabel("Current (A)")
            ax.set_title("Wide dark JV (active pixels)")
            wide_plot_path = plots_dir / "wide_dark_jv_curves.png"
            fig.savefig(wide_plot_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            if r2_map:
                fig, ax = plt.subplots()
                ax.boxplot(list(r2_map.values()))
                ax.set_ylabel("R^2")
                ax.set_title("Wide JV linearity (R^2 distribution)")
                r2_plot_path = plots_dir / "wide_dark_jv_r2_boxplot.png"
                fig.savefig(r2_plot_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                self._summary.plot_paths["wide_jv_r2_boxplot"] = r2_plot_path
            self._summary.plot_paths["wide_jv_curves"] = wide_plot_path
            self._summary.r2_values = r2_map
        self._done_steps = min(self._total_steps, self._done_steps + 1)
        self._emit_progress("Plot generation complete")

    def _extract_curve(self, capture: ScanResult, pixel: int) -> List[Tuple[float, float]]:
        curve: List[Tuple[float, float]] = []
        for loop in capture.loops:
            if loop.voltage is None:
                continue
            r, c = divmod(pixel - 1, 10)
            try:
                val = loop.data[r, c]
            except Exception:
                continue
            if np.isnan(val):
                continue
            curve.append((float(loop.voltage), float(val)))
        curve.sort(key=lambda t: t[0])
        return curve

    @staticmethod
    def _sigma_clip(values: np.ndarray, sigma: float) -> np.ndarray:
        if sigma is None or sigma <= 0 or values.size == 0:
            return values
        mu = float(np.nanmean(values))
        std = float(np.nanstd(values))
        if std <= 0:
            return values
        mask = np.abs(values - mu) <= sigma * std
        return values[mask]

    # -------------------------- saving helpers --------------------------
    def _save_loop(self, output_dir: Path, name: str, loop: LoopResult):
        csv_name = _loop_to_csv_name(name, loop.index, loop.voltage)
        csv_path = output_dir / csv_name
        metadata = dict(loop.metadata or {})
        metadata["loop_index"] = loop.index
        if loop.voltage is not None:
            metadata["voltage"] = float(loop.voltage)
        metadata["timestamp"] = datetime.utcnow().isoformat()
        header = self._format_metadata(metadata)
        save_kwargs = {"delimiter": ",", "fmt": "%.5e"}
        if header:
            save_kwargs["header"] = header
            save_kwargs["comments"] = "# "
        np.savetxt(csv_path, loop.data, **save_kwargs)

    @staticmethod
    def _format_metadata(metadata: dict) -> str:
        try:
            payload = json.dumps(metadata, separators=(",", ":"), sort_keys=True)
            return f"READOUT_METADATA {payload}"
        except Exception:
            return ""

    def _write_metadata(self):
        meta = {
            "created_at": datetime.utcnow().isoformat(),
            "settings": self._settings.__dict__,
            "shorted_pixels": sorted(self._summary.shorted_pixels),
            "dead_short_pixels": sorted(self._summary.dead_short_pixels),
            "wide_jv_over_limit": sorted(self._summary.wide_jv_over_limit),
            "active_pixels": sorted(self._summary.active_pixels),
            "active_pixel_count": len(self._summary.active_pixels),
            "plots": {k: str(v) for k, v in self._summary.plot_paths.items()},
            "test_dirs": {k: str(v) for k, v in self._summary.test_dirs.items()},
        }
        path = self._root / "characterization_metadata.json"
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._summary.metadata_path = path

    def _emit_progress(self, text: str, *, reset: bool = False):
        if reset:
            self._done_steps = 0
        self.progressChanged.emit(self._done_steps, self._total_steps, text)
        self.statusMessage.emit(text)

    def _advance_progress(self, text: str, steps: int = 1):
        self._done_steps = min(self._total_steps, self._done_steps + max(steps, 1))
        self._emit_progress(text)

    # -------------------------- pin map loader --------------------------
    @staticmethod
    def _load_pin_map(path_str: Optional[str]) -> Dict[int, str]:
        if not path_str:
            return {}
        path = Path(path_str)
        if not path.exists():
            return {}
        try:
            if path.suffix.lower() == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                return {int(k): str(v) for k, v in payload.items()}
            mapping: Dict[int, str] = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                parts = [p.strip() for p in line.split(",") if p.strip()]
                if len(parts) < 2:
                    continue
                try:
                    pixel = int(parts[0])
                except ValueError:
                    continue
                mapping[pixel] = parts[1]
            return mapping
        except Exception as exc:
            LOG.warning("Failed to read pin map %s: %s", path, exc)
            return {}
