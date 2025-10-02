import logging
from typing import Iterable

import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import Signal


class ScanWorker(QtCore.QObject):
    pixelDone = Signal(int, float)
    loopStarted = Signal(int)
    loopFinished = Signal(int)
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

        try:
            for loop_idx in range(1, self._loops + 1):
                if self._stop:
                    break
                self.loopStarted.emit(loop_idx)
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
                            val = float(self._sm.current)
                            QtCore.QThread.msleep(int(self._delay_s * 1000))
                        except Exception as e:
                            self.deviceError.emit(f"Sourcemeter connection lost: {e}")
                            raise
                        vals.append(val)
                    if self._stop:
                        break
                    self.pixelDone.emit(p, float(np.mean(vals)))

                if self._stop:
                    break
                self.loopFinished.emit(loop_idx)
                if loop_idx < self._loops and self._inter_loop_delay_s > 0:
                    for _ in range(int(self._inter_loop_delay_s * 10)):
                        if self._stop:
                            break
                        QtCore.QThread.msleep(100)
        except Exception:
            pass
        finally:
            try:
                self._sm.disable_source()
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
