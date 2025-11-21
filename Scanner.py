import logging
from typing import Iterable, Optional

import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import Signal


DEVICE_LOGGER = logging.getLogger("readoubt.devices")


class ScanWorker(QtCore.QObject):
    loopDataReady = Signal(int, object)
    loopStarted = Signal(int, object)
    loopFinished = Signal(int, object)
    loopProgress = Signal(int, int, int)
    pixelExcluded = Signal(int, float, int)
    deviceError = Signal(str)
    finished = Signal()

    def __init__(
        self,
        sm,
        switch,
        n_samples: int,
        nplc: float,
        pixel_indices: Iterable[int],
        loops: int,
        auto_range: bool,
        current_range: float,
        inter_sample_delay_s: float,
        inter_loop_delay_s: float,
        bias_sm=None,
        voltage_steps: Optional[Iterable[float]] = None,
        voltage_settle_s: float = 0.0,
        constant_bias_voltage: Optional[float] = None,
        use_local_readout: bool = False,
        use_local_bias: bool = False,
        current_limit: Optional[float] = None,
    ):
        super().__init__()
        self._sm = sm
        self._sw = switch
        self._n = max(1, int(n_samples))
        self._nplc = max(0.01, float(nplc))
        self._pixels = list(pixel_indices)
        self._loops = max(1, int(loops))
        self._auto_range = bool(auto_range)
        self._current_range = float(current_range)
        self._delay_s = float(inter_sample_delay_s)
        self._inter_loop_delay_s = float(inter_loop_delay_s)
        self._stop = False
        self._paused = False
        self._bias_sm = bias_sm
        self._voltage_points = list(voltage_steps or [])
        self._voltage_settle_s = max(0.0, float(voltage_settle_s))
        self._constant_bias_voltage = (
            float(constant_bias_voltage)
            if constant_bias_voltage is not None
            else None
        )
        if self._voltage_points:
            self._loops = len(self._voltage_points)
        self._use_local_readout = bool(use_local_readout)
        self._use_local_bias = bool(use_local_bias)
        self._current_limit = float(current_limit) if current_limit is not None else None
        self._excluded_pixels: set[int] = set()
        self._per_loop_total = max(1, len(self._pixels) or 1)

    def _emit_loop_progress(self, loop_idx: int, done: int, total: Optional[int] = None):
        total_count = max(1, int(total if total is not None else self._per_loop_total))
        done_count = max(0, min(int(done), total_count))
        self.loopProgress.emit(loop_idx, done_count, total_count)

    def _check_limit(self, pixel: int, value: float, loop_idx: int) -> bool:
        if self._current_limit is None:
            return False
        try:
            over_limit = abs(float(value)) > self._current_limit
        except Exception:
            return False
        if not over_limit:
            return False
        if pixel not in self._excluded_pixels:
            self._excluded_pixels.add(int(pixel))
            try:
                self.pixelExcluded.emit(int(pixel), float(value), int(loop_idx))
            except Exception:
                pass
        return True

    @QtCore.Slot()
    def run(self):
        if not self._use_local_readout:
            DEVICE_LOGGER.info(
                "Configuring read SMU (NPLC=%s, auto_range=%s, range=%s)",
                self._nplc,
                self._auto_range,
                self._current_range,
            )
            try:
                DEVICE_LOGGER.debug("Resetting read SMU before measurement")
                self._sm.reset()
                DEVICE_LOGGER.debug("Enabling read SMU source output")
                self._sm.enable_source()
                try:
                    self._sm.measure_current(
                        nplc=self._nplc, auto_range=self._auto_range
                    )
                except TypeError:
                    self._sm.measure_current(nplc=self._nplc)
                if not self._auto_range:
                    try:
                        self._sm.current_range = self._current_range
                    except Exception:
                        pass
            except Exception as e:
                self.deviceError.emit(f"Failed to configure Sourcemeter: {e}")
                self.finished.emit()
                return

        sweep_mode = bool(self._voltage_points)
        constant_bias_mode = self._constant_bias_voltage is not None
        bias_mode = sweep_mode or constant_bias_mode
        if bias_mode and not (self._bias_sm or self._use_local_bias):
            self.deviceError.emit("No bias source configured for requested scan")
            self.finished.emit()
            return
        if self._current_limit is not None:
            DEVICE_LOGGER.info(
                "Current limit enabled: %.3e A (absolute value)", self._current_limit
            )

        if sweep_mode:
            loop_plan = [
                (idx + 1, float(v)) for idx, v in enumerate(self._voltage_points)
            ]
        elif constant_bias_mode:
            loop_plan = [
                (idx, float(self._constant_bias_voltage))
                for idx in range(1, self._loops + 1)
            ]
        else:
            loop_plan = [(idx, None) for idx in range(1, self._loops + 1)]

        try:
            for loop_idx, voltage in loop_plan:
                if self._stop:
                    break
                active_pixels = [p for p in self._pixels if p not in self._excluded_pixels]
                if not active_pixels:
                    DEVICE_LOGGER.info("All pixels excluded by current limit; ending scan.")
                    break
                self._per_loop_total = max(1, len(active_pixels))
                requested_voltage = voltage
                applied_voltage = None
                if bias_mode and voltage is not None:
                    try:
                        applied_voltage = self._apply_bias_voltage(voltage)
                    except Exception as e:
                        self.deviceError.emit(
                            f"Failed to set bias voltage {voltage:.6f} V: {e}"
                        )
                        raise
                    if self._voltage_settle_s > 0:
                        self._sleep_with_stop(
                            self._voltage_settle_s, allow_pause=True
                        )
                        if self._stop:
                            break
                metadata = None
                if requested_voltage is not None:
                    metadata = {
                        "voltage": (
                            applied_voltage
                            if applied_voltage is not None
                            else requested_voltage
                        )
                    }
                    if (
                        applied_voltage is not None
                        and abs(applied_voltage - requested_voltage) > 5e-4
                    ):
                        metadata["requested_voltage"] = requested_voltage
                self.loopStarted.emit(loop_idx, metadata)
                runtime_ms = None
                loop_results = []
                if self._use_local_readout:
                    runtime_total_ms = 0.0
                    # Repeat full-board captures to honor samples/pixel for local readout.
                    while self._paused and not self._stop:
                        QtCore.QThread.msleep(100)
                    if self._stop:
                        break
                    try:
                        DEVICE_LOGGER.info(
                            "Switch board local measurement requested (%s samples/pixel)",
                            self._n,
                        )
                        def local_progress_cb(done_count, total_count):
                            try:
                                done_i = int(done_count)
                                total_i = int(total_count)
                            except Exception:
                                return
                            self._emit_loop_progress(loop_idx, done_i, total_i or 1)

                        avg_currents, sample_runtime_ms = self._sw.measure_local(
                            n_samples=self._n,
                            progress_cb=local_progress_cb,
                            omit_indices=self._excluded_pixels,
                        )
                    except Exception as e:
                        logging.warning(f"Local measurement failed: {e}")
                        self.deviceError.emit(f"Local measurement failed: {e}")
                        raise
                    if sample_runtime_ms is not None:
                        runtime_total_ms += float(sample_runtime_ms)
                    if self._stop:
                        break
                    runtime_ms = runtime_total_ms
                    for p in active_pixels:
                        if self._stop:
                            break
                        idx = p - 1
                        if idx < 0 or idx >= len(avg_currents):
                            continue
                        current_a = float(avg_currents[idx]) * 1e-9
                        limit_hit = self._check_limit(p, current_a, loop_idx)
                        loop_results.append((p, np.nan if limit_hit else current_a))
                    active_count = len(active_pixels)
                    self._emit_loop_progress(loop_idx, active_count, active_count or 1)
                else:
                    active_total = len(active_pixels) or 1
                    for p in active_pixels:
                        while self._paused and not self._stop:
                            QtCore.QThread.msleep(100)
                        if self._stop:
                            break
                        try:
                            DEVICE_LOGGER.debug(
                                "Loop %s: routing pixel %s via switch board",
                                loop_idx,
                                p,
                            )
                            self._sw.route(p)  # blocks until ACK
                        except Exception as e:
                            logging.warning(
                                f"Switch route failed for pixel {p}: {e}"
                            )
                            self.deviceError.emit(f"Switch connection lost: {e}")
                            raise
                        vals = []
                        limit_hit = False
                        for _ in range(self._n):
                            if self._stop:
                                break
                            try:
                                val = -1*float(self._sm.current) # We need to invert, as the sourcemeter ammeter is set up to see currents leaving it as positive.
                                limit_hit = self._check_limit(p, val, loop_idx)
                                if limit_hit:
                                    break
                                if self._delay_s > 0:
                                    self._sleep_with_stop(
                                        self._delay_s, allow_pause=True
                                    )
                            except Exception as e:
                                self.deviceError.emit(
                                    f"Sourcemeter connection lost: {e}"
                                )
                                raise
                            vals.append(val)
                        if self._stop:
                            break
                        value = np.nan
                        if vals and not limit_hit:
                            value = float(np.mean(vals))
                        loop_results.append((p, value))
                        self._emit_loop_progress(loop_idx, len(loop_results), active_total)

                if self._stop:
                    break
                self.loopDataReady.emit(loop_idx, loop_results)
                if runtime_ms is not None:
                    if metadata is None:
                        metadata = {}
                    metadata["runtime_ms"] = float(runtime_ms)
                self.loopFinished.emit(loop_idx, metadata)
                if (
                    loop_idx < (loop_plan[-1][0] if loop_plan else 0)
                    and self._inter_loop_delay_s > 0
                ):
                    self._sleep_with_stop(self._inter_loop_delay_s, allow_pause=True)
        except StopIteration:
            pass
        except Exception:
            pass
        finally:
            try:
                self._sm.disable_source()
                DEVICE_LOGGER.debug("Read SMU source disabled after scan")
            except Exception:
                pass
            if bias_mode and self._bias_sm and not self._use_local_bias:
                try:
                    self._bias_sm.source_voltage = 0.0
                    if hasattr(self._bias_sm, "disable_source"):
                        self._bias_sm.disable_source()
                    DEVICE_LOGGER.debug("Bias SMU returned to 0 V and disabled")
                except Exception:
                    pass
            self.finished.emit()

    def _apply_bias_voltage(self, voltage: float) -> float:
        if self._use_local_bias:
            DEVICE_LOGGER.info(
                "Switch board local bias set to %.3f V", float(voltage)
            )
            return float(self._sw.set_local_voltage(voltage))
        DEVICE_LOGGER.info("Bias SMU set to %.3f V", float(voltage))
        self._bias_sm.source_voltage = float(voltage)
        self._bias_sm.enable_source()
        try:
            _ = self._bias_sm.current
        except Exception:
            pass
        return float(voltage)

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        self._stop = True
        self._paused = False

    def _sleep_with_stop(self, seconds: float, allow_pause: bool = False):
        remaining_ms = int(max(0.0, float(seconds)) * 1000)
        while remaining_ms > 0 and not self._stop:
            if allow_pause and self._paused:
                QtCore.QThread.msleep(100)
                continue
            chunk = min(remaining_ms, 100)
            QtCore.QThread.msleep(chunk)
            remaining_ms -= chunk
