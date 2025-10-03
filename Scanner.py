import logging
from typing import Iterable, Optional

import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import Signal


class ScanWorker(QtCore.QObject):
    pixelDone = Signal(int, float)
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
        if self._voltage_points:
            self._loops = len(self._voltage_points)

    @QtCore.Slot()
    def run(self):
        try:
            self._sm.reset()
            self._sm.enable_source()
            try:
                self._sm.measure_current(nplc=self._nplc, auto_range=self._auto_range)
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

        bias_mode = bool(self._voltage_points)
        if bias_mode and not self._bias_sm:
            self.deviceError.emit("Bias SMU not configured for sweep")
            self.finished.emit()
            return

        loop_plan = (
            [(idx + 1, v) for idx, v in enumerate(self._voltage_points)]
            if bias_mode
            else [(idx, None) for idx in range(1, self._loops + 1)]
        )

        try:
            for loop_idx, voltage in loop_plan:
                if self._stop:
                    break
                metadata = {"voltage": voltage} if voltage is not None else None
                if bias_mode:
                    try:
                        #self._bias_sm.apply_voltage()
                        self._bias_sm.source_voltage = float(voltage)
                        self._bias_sm.enable_source()
                        current = self._bias_sm.current
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
                self.loopStarted.emit(loop_idx, metadata)
                for p in self._pixels:
                    while self._paused and not self._stop:
                        QtCore.QThread.msleep(100)
                    if self._stop:
                        break
                    try:
                        self._sw.route(p)  # blocks until ACK
                    except Exception as e:
                        logging.warning(f"Switch route failed for pixel {p}: {e}")
                        self.deviceError.emit(f"Switch connection lost: {e}")
                        raise
                    vals = []
                    for _ in range(self._n):
                        if self._stop:
                            break
                        try:
                            #current = self._bias_sm.current
                            val = float(self._sm.current)
                            if self._delay_s > 0:
                                self._sleep_with_stop(self._delay_s, allow_pause=True)
                        except Exception as e:
                            self.deviceError.emit(f"Sourcemeter connection lost: {e}")
                            raise
                        vals.append(val)
                    if self._stop:
                        break
                    self.pixelDone.emit(p, float(np.mean(vals)))

                if self._stop:
                    break
                self.loopFinished.emit(loop_idx, metadata)
                if (
                    loop_idx < (loop_plan[-1][0] if loop_plan else 0)
                    and self._inter_loop_delay_s > 0
                ):
                    self._sleep_with_stop(self._inter_loop_delay_s, allow_pause=True)
        except Exception:
            pass
        finally:
            try:
                self._sm.disable_source()
            except Exception:
                pass
            if bias_mode and self._bias_sm:
                try:
                    self._bias_sm.source_voltage = 0.0
                    if hasattr(self._bias_sm, "disable_source"):
                        self._bias_sm.disable_source()
                except Exception:
                    pass
            self.finished.emit()

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
