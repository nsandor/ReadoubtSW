from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple


@dataclass
class StageLimits:
    """Optional travel limits in millimetres."""

    x_max: float
    y_max: float


class DummyXYStage:
    """Placeholder XY stage driver used until hardware details are known."""

    def __init__(self, *, move_delay_s: float = 0.05, limits: StageLimits | None = None) -> None:
        self._connected = False
        self._position = (0.0, 0.0)
        self._move_delay_s = max(0.0, float(move_delay_s))
        self._limits = limits

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def home(self) -> None:
        self._require_connection()
        self._sleep_delay()
        self._position = (0.0, 0.0)

    def move_to(self, x_mm: float, y_mm: float) -> None:
        self._require_connection()
        target = (float(x_mm), float(y_mm))
        if self._limits:
            if not (0.0 <= target[0] <= self._limits.x_max):
                raise ValueError("X target exceeds configured travel limits.")
            if not (0.0 <= target[1] <= self._limits.y_max):
                raise ValueError("Y target exceeds configured travel limits.")
        self._sleep_delay()
        self._position = target

    def move_by(self, dx_mm: float, dy_mm: float) -> None:
        x, y = self._position
        self.move_to(x + float(dx_mm), y + float(dy_mm))

    def position(self) -> Tuple[float, float]:
        return self._position

    def _sleep_delay(self) -> None:
        if self._move_delay_s <= 0:
            return
        time.sleep(self._move_delay_s)

    def _require_connection(self) -> None:
        if not self._connected:
            raise RuntimeError("Stage is not connected.")


class DummyRotationStage:
    """Simplified rotation stage interface."""

    def __init__(self, *, move_delay_s: float = 0.05) -> None:
        self._connected = False
        self._angle = 0.0
        self._move_delay_s = max(0.0, float(move_delay_s))

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def home(self) -> None:
        self._require_connection()
        self._sleep_delay()
        self._angle = 0.0

    def move_to(self, angle_deg: float) -> None:
        self._require_connection()
        normalized = float(angle_deg) % 360.0
        self._sleep_delay()
        self._angle = normalized

    def angle(self) -> float:
        return self._angle

    def _sleep_delay(self) -> None:
        if self._move_delay_s <= 0:
            return
        time.sleep(self._move_delay_s)

    def _require_connection(self) -> None:
        if not self._connected:
            raise RuntimeError("Rotation stage is not connected.")
