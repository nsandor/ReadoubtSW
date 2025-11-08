import logging
from typing import Iterable, Optional

import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import Signal


class ScanWorker(QtCore.QObject):
    loopDataReady = Signal(int, object)
    loopStarted = Signal(int, object)
    loopFinished = Signal(int, object)
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

    @QtCore.Slot()
    def run(self):
        if not self._use_local_readout:
            try:
                self._sm.reset()
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
                    samples_taken = 0
                    runtime_total_ms = 0.0
                    summed_currents = None
                    # Repeat full-board captures to honor samples/pixel for local readout.
                    for sample_idx in range(self._n):
                        while self._paused and not self._stop:
                            QtCore.QThread.msleep(100)
                        if self._stop:
                            break
                        try:
                            currents_na, sample_runtime_ms = self._sw.measure_local()
                        except Exception as e:
                            logging.warning(f"Local measurement failed: {e}")
                            self.deviceError.emit(f"Local measurement failed: {e}")
                            raise
                        arr = np.asarray(currents_na, dtype=float)
                        summed_currents = (
                            arr.copy() if summed_currents is None else summed_currents + arr
                        )
                        samples_taken += 1
                        if sample_runtime_ms is not None:
                            runtime_total_ms += float(sample_runtime_ms)
                        if (
                            sample_idx < self._n - 1
                            and self._delay_s > 0
                            and not self._stop
                        ):
                            self._sleep_with_stop(self._delay_s, allow_pause=True)
                            if self._stop:
                                break
                    if self._stop:
                        break
                    if samples_taken == 0 or summed_currents is None:
                        continue
                    avg_currents = summed_currents / float(samples_taken)
                    runtime_ms = runtime_total_ms if samples_taken else None
                    for p in self._pixels:
                        if self._stop:
                            break
                        idx = p - 1
                        if idx < 0 or idx >= len(avg_currents):
                            continue
                        nanoamps = avg_currents[idx]
                        loop_results.append((p, float(nanoamps) * 1e-9))
                else:
                    for p in self._pixels:
                        while self._paused and not self._stop:
                            QtCore.QThread.msleep(100)
                        if self._stop:
                            break
                        try:
                            self._sw.route(p)  # blocks until ACK
                        except Exception as e:
                            logging.warning(
                                f"Switch route failed for pixel {p}: {e}"
                            )
                            self.deviceError.emit(f"Switch connection lost: {e}")
                            raise
                        vals = []
                        for _ in range(self._n):
                            if self._stop:
                                break
                            try:
                                val = float(self._sm.current)
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
                        loop_results.append((p, float(np.mean(vals))))

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
            except Exception:
                pass
            if bias_mode and self._bias_sm and not self._use_local_bias:
                try:
                    self._bias_sm.source_voltage = 0.0
                    if hasattr(self._bias_sm, "disable_source"):
                        self._bias_sm.disable_source()
                except Exception:
                    pass
            self.finished.emit()

    def _apply_bias_voltage(self, voltage: float) -> float:
        if self._use_local_bias:
            return float(self._sw.set_local_voltage(voltage))
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
