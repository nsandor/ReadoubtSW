import re
from typing import List, Tuple

import serial


class SwitchBoard:
    """USB 100:1 switch with ACK sync."""

    def __init__(self, port: str, baud: int = 115200, timeout: float = 2):
        if serial is None:
            raise RuntimeError("pyserial not available")
        self.ser = serial.Serial(port, baudrate=baud, timeout=timeout)
        self._local_mode = False
        self._last_voltage = None

    def _readline(self) -> str:
        resp = self.ser.readline()
        if not resp:
            raise TimeoutError("Switch board timed out waiting for response")
        return resp.decode("utf-8", errors="ignore").strip()

    def _drain_lines(self, max_lines: int = 5) -> List[str]:
        lines: List[str] = []
        for _ in range(max_lines):
            resp = self.ser.readline()
            if not resp:
                break
            line = resp.decode("utf-8", errors="ignore").strip()
            lines.append(line)
            if not getattr(self.ser, "in_waiting", 0):
                break
        return lines

    def _ensure_external_mode(self):
        if not self._local_mode:
            return
        self.ser.write(b"MEASURE_EXTERNAL\n")
        self._drain_lines(max_lines=6)
        self._local_mode = False

    @staticmethod
    def _extract_float(text: str):
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        return float(match.group(0)) if match else None

    def route(self, idx: int):
        if not 1 <= idx <= 100:
            raise ValueError("Pixel index must be 1-100")
        self._ensure_external_mode()
        self.ser.write(f"{idx}\n".encode())
        resp = self._readline()
        if "ACK" not in resp.upper():
            raise TimeoutError(
                f"Switch did not ACK for pixel {idx}. Response: {resp}"
            )

    def set_led(self, enabled: bool):
        cmd = "LED ON" if enabled else "LED OFF"
        self.ser.write(f"{cmd}\n".encode())
        self._drain_lines(max_lines=2)

    def set_settle_time(self, milliseconds: int) -> int:
        value = max(0, int(milliseconds))
        self.ser.write(f"SETTLE {value}\n".encode())
        try:
            self._readline()
        except Exception:
            pass
        return value

    def set_local_voltage(self, voltage: float) -> float:
        value = float(voltage)
        if not (6.0 <= value <= 87.0):
            raise ValueError("Local bias must be between 6 V and 87 V")
        self.ser.write(f"SETVOLT {value:.3f}\n".encode())
        for _ in range(4):
            try:
                line = self._readline()
            except TimeoutError:
                break
            parsed = self._extract_float(line)
            if parsed is not None:
                self._last_voltage = parsed
                return parsed
        self._last_voltage = value
        return value

    def measure_local(self, n_samples) -> Tuple[List[float], float]:
        self.ser.flush()
        self.ser.write(f"MEASURE_LOCAL {n_samples}\n".encode())
        floats: List[float] = []
        self.ser.timeout = None
        attempts = 0
        while len(floats) < 101:
            line = self._readline()
            try:
                floats.append(float(line))
            except ValueError:
                # Skip informational lines produced by GPIO helpers.
                pass
            attempts += 1
            if attempts > 200:
                raise TimeoutError("Switch board local measurement did not finish")
        self._local_mode = True
        runtime_ms = floats[-1]
        currents_nanoamps = floats[:-1]
        return currents_nanoamps, runtime_ms

    def close(self):
        self.ser.close()


class DummySwitchBoard:
    def __init__(self):
        self._led_enabled = False
        self._local_mode = False
        self._last_voltage = None

    def route(self, *_):
        pass

    def set_led(self, enabled: bool):
        self._led_enabled = bool(enabled)

    def set_settle_time(self, milliseconds: int) -> int:
        value = max(0, int(milliseconds))
        return value

    def set_local_voltage(self, voltage: float) -> float:
        self._last_voltage = float(voltage)
        return self._last_voltage

    def measure_local(self) -> Tuple[List[float], float]:
        import numpy as np

        rng = np.random.default_rng()
        currents = (rng.random(100) - 0.5) * 100  # nanoamps
        runtime_ms = float(rng.integers(100, 500))
        self._local_mode = True
        return currents.tolist(), runtime_ms

    def ensure_external_mode(self):
        self._local_mode = False

    def close(self):
        pass
