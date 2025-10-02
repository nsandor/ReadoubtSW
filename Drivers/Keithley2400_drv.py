from pymeasure.instruments.keithley import Keithley2400
import numpy as np


class ReadoutSafe2400(Keithley2400 if Keithley2400 else object):
    # Keithley 2400 locked to 0 V source.

    def __init__(self, adapter, **kwargs):
        super().__init__(adapter, **kwargs)
        if Keithley2400:
            self.write("SOUR:FUNC VOLT")
            self.source_voltage = 0
            self.disable_source()

    @property
    def source_voltage(self):
        if not Keithley2400:
            return 0.0
        return float(self.ask(":SOUR:VOLT?").strip())

    @source_voltage.setter
    def source_voltage(self, val):
        if abs(val) > 1e-9:
            raise RuntimeError("Readout SM must remain at 0 V – refusing")
        if Keithley2400:
            super(ReadoutSafe2400, self.__class__).source_voltage.fset(self, 0)

    def enable_source(self):
        if abs(self.source_voltage) > 1e-9:
            raise RuntimeError("Refusing to enable source – voltage ≠ 0 V")
        if Keithley2400:
            super().enable_source()


class DummyKeithley2400:
    def __init__(self):
        self._current = 0.0
        np.random.seed(0)

    def measure_current(self, **_):
        pass

    def enable_source(self):
        pass

    def disable_source(self):
        pass

    def reset(self):
        pass

    @property
    def current(self):
        self._current = 50e-9 * (2 * np.random.rand() - 1)
        return self._current

    def ask(self, _):
        return str(self.current)
