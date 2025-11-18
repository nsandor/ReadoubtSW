from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from newportxps import NewportXPS, XPSException
except ImportError:  # pragma: no cover - fallback when driver is unavailable
    NewportXPS = None  # type: ignore

    class XPSException(Exception):
        pass


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


@dataclass
class StageConnectionSettings:
    host: str = "192.168.0.254"
    port: int = 5001
    username: str = "Administrator"
    password: str = "Administrator"
    timeout_s: float = 10.0


@dataclass
class StageAxisState:
    axis: str
    display_name: str
    units: str
    stage_name: Optional[str] = None
    zero_reference: Optional[float] = None


class StageDriverError(RuntimeError):
    """Raised when the Newport XPS stage encounters an error."""


class NewportXPSStageDriver:
    """Wrapper around the newportxps library with zeroing helpers."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._client: Optional[NewportXPS] = None
        self._settings = StageConnectionSettings()
        self._axes: Dict[str, StageAxisState] = {
            "x": StageAxisState("x", "X axis", "mm"),
            "y": StageAxisState("y", "Y axis", "mm"),
            "theta": StageAxisState("theta", "Rotation", "deg"),
        }

    # ------------------------------------------------------------------ connection helpers
    def library_available(self) -> bool:
        return NewportXPS is not None

    def connection_settings(self) -> StageConnectionSettings:
        return StageConnectionSettings(**vars(self._settings))

    def connect(self, settings: Optional[StageConnectionSettings] = None) -> None:
        if settings is not None:
            self._settings = settings
        if NewportXPS is None:
            raise StageDriverError(
                "The newportxps package is not installed. Install it to control the stage."
            )
        with self._lock:
            self.disconnect()
            try:
                self._client = NewportXPS(
                    host=self._settings.host,
                    port=self._settings.port,
                    username=self._settings.username,
                    password=self._settings.password,
                    timeout=self._settings.timeout_s,
                )
            except Exception as exc:  # pragma: no cover - hardware specific
                self._client = None
                raise StageDriverError(f"Failed to connect to XPS controller: {exc}") from exc

    def disconnect(self) -> None:
        with self._lock:
            if self._client is not None:
                try:
                    self._client.disconnect()
                except Exception:
                    pass
            self._client = None

    def is_connected(self) -> bool:
        with self._lock:
            return self._client is not None

    # ------------------------------------------------------------------ axis/assignment helpers
    def available_stage_names(self) -> List[str]:
        with self._lock:
            if not self._client:
                return []
            return sorted(self._client.stages.keys())

    def describe_stages(self) -> List[Dict[str, str]]:
        with self._lock:
            if not self._client:
                return []
            out: List[Dict[str, str]] = []
            for name, meta in self._client.stages.items():
                out.append(
                    {
                        "name": name,
                        "type": str(meta.get("stagetype", "")),
                        "group": name.split(".")[0] if "." in name else "",
                    }
                )
            return out

    def assign_axis(self, axis: str, stage_name: Optional[str]) -> None:
        state = self._axis_state(axis)
        with self._lock:
            if stage_name:
                if not self._client:
                    raise StageDriverError("Connect to the controller before assigning stages.")
                if stage_name not in self._client.stages:
                    raise StageDriverError(f"Stage '{stage_name}' is not available on the controller.")
            state.stage_name = stage_name or None
            state.zero_reference = None

    def axis_assignment(self, axis: str) -> Optional[str]:
        return self._axis_state(axis).stage_name

    def axis_units(self, axis: str) -> str:
        return self._axis_state(axis).units

    def axis_zeroed(self, axis: str) -> bool:
        return self._axis_state(axis).zero_reference is not None

    def axis_zero_reference(self, axis: str) -> Optional[float]:
        return self._axis_state(axis).zero_reference

    # ------------------------------------------------------------------ motion helpers
    def ready_for_scan(self, *, require_rotation: bool = True) -> Tuple[bool, str]:
        if not self.is_connected():
            return False, "Stage controller is not connected. Use the Stage Controller window to connect."
        for axis in ("x", "y"):
            state = self._axis_state(axis)
            if not state.stage_name:
                return False, f"Assign a stage to the {state.display_name} in the Stage Controller."
            if state.zero_reference is None:
                return False, f"Zero the {state.display_name} before starting a scan."
        if require_rotation:
            state = self._axis_state("theta")
            if not state.stage_name:
                return False, "Assign a rotation stage in the Stage Controller."
            if state.zero_reference is None:
                return False, "Zero the rotation stage before starting a scan."
        return True, ""

    def axis_position(self, axis: str) -> Optional[float]:
        state = self._axis_state(axis)
        with self._lock:
            client = self._client
            stage_name = state.stage_name
            zero = state.zero_reference
        if not client or not stage_name or zero is None:
            return None
        try:
            absolute = client.get_stage_position(stage_name)
        except Exception as exc:
            raise StageDriverError(
                f"Failed to read {state.display_name} position: {exc}"
            ) from exc
        return absolute - zero

    def axis_absolute_position(self, axis: str) -> Optional[float]:
        state = self._axis_state(axis)
        with self._lock:
            client = self._client
            stage_name = state.stage_name
        if not client or not stage_name:
            return None
        try:
            return client.get_stage_position(stage_name)
        except Exception as exc:
            raise StageDriverError(
                f"Failed to read {state.display_name} position: {exc}"
            ) from exc

    def zero_axis(self, axis: str) -> None:
        state = self._axis_state(axis)
        with self._lock:
            client = self._client
            stage_name = state.stage_name
        if not client or not stage_name:
            raise StageDriverError(f"Assign a stage to {state.display_name} before zeroing it.")
        position = client.get_stage_position(stage_name)
        state.zero_reference = position

    def zero_all_axes(self) -> None:
        for axis in ("x", "y", "theta"):
            try:
                self.zero_axis(axis)
            except StageDriverError:
                continue

    def move_axis_to(self, axis: str, position: float, *, wait: bool = True) -> None:
        state = self._axis_state(axis)
        with self._lock:
            client = self._client
            stage_name = state.stage_name
            zero = state.zero_reference
        if not client or not stage_name:
            raise StageDriverError(f"{state.display_name} is not assigned.")
        if zero is None:
            raise StageDriverError(f"Zero the {state.display_name} before moving it.")
        absolute = position + zero
        client.move_stage(stage_name, absolute)
        if wait:
            self._wait_for_stage(stage_name, absolute)

    def jog_axis(self, axis: str, delta: float) -> None:
        current = self.axis_position(axis)
        if current is None:
            raise StageDriverError(f"Zero the {self._axis_state(axis).display_name} before jogging it.")
        self.move_axis_to(axis, current + delta)

    def move_xy_to(self, x: float, y: float) -> None:
        self.move_axis_to("x", x)
        self.move_axis_to("y", y)

    def jog_xy(self, dx: float, dy: float) -> None:
        self.jog_axis("x", dx)
        self.jog_axis("y", dy)

    def home_axis(self, axis: str) -> None:
        state = self._axis_state(axis)
        with self._lock:
            client = self._client
            stage_name = state.stage_name
        if not client or not stage_name:
            raise StageDriverError(f"Assign the {state.display_name} before homing it.")
        group = self._group_for_stage(stage_name)
        if not group:
            raise StageDriverError(f"Unable to infer the group name for stage '{stage_name}'.")
        client.home_group(group=group)
        self._wait_for_group(group)
        state.zero_reference = None

    def home_xy(self) -> None:
        self.home_axis("x")
        self.home_axis("y")

    def home_rotation(self) -> None:
        self.home_axis("theta")

    def axes_ready(self, axes: Sequence[str], *, require_zero: bool = True) -> bool:
        if not self.is_connected():
            return False
        for axis in axes:
            state = self._axis_state(axis)
            if not state.stage_name:
                return False
            if require_zero and state.zero_reference is None:
                return False
        return True

    def status_summary(self) -> str:
        if not self.is_connected():
            return "Controller disconnected"
        entries = []
        for axis in ("x", "y", "theta"):
            state = self._axis_state(axis)
            if not state.stage_name:
                entries.append(f"{state.display_name}: not assigned")
                continue
            suffix = "zeroed" if state.zero_reference is not None else "zero not set"
            entries.append(f"{state.display_name}: {state.stage_name} ({suffix})")
        return "; ".join(entries)

    # ------------------------------------------------------------------ private helpers
    def _axis_state(self, axis: str) -> StageAxisState:
        if axis not in self._axes:
            raise ValueError(f"Unknown axis '{axis}'")
        return self._axes[axis]

    def _group_for_stage(self, stage_name: str) -> Optional[str]:
        if not stage_name:
            return None
        if "." in stage_name:
            return stage_name.split(".")[0]
        return None

    def _wait_for_group(self, group: str, timeout: float = 120.0) -> None:
        start = time.monotonic()
        while True:
            with self._lock:
                client = self._client
            if not client:
                return
            status = client.get_group_status().get(group, "")
            text = (status or "").lower()
            if any(token in text for token in ("ready", "standby", "idle")):
                return
            if "error" in text:
                raise StageDriverError(f"Group '{group}' reported error state: {status}")
            if time.monotonic() - start > timeout:
                raise StageDriverError(f"Timed out waiting for group '{group}' to become ready.")
            time.sleep(0.1)

    def _wait_for_stage(self, stage_name: str, absolute_target: float, timeout: float = 120.0) -> None:
        group = self._group_for_stage(stage_name)
        start = time.monotonic()
        while True:
            with self._lock:
                client = self._client
            if not client:
                return
            position = client.get_stage_position(stage_name)
            if abs(position - absolute_target) <= 1e-4:
                if not group:
                    return
                status = client.get_group_status().get(group, "")
                if any(token in (status or "").lower() for token in ["ready", "idle", "standby"]):
                    return
            if time.monotonic() - start > timeout:
                raise StageDriverError(f"Timed out waiting for stage '{stage_name}' to reach {absolute_target:.4f}.")
            time.sleep(0.1)


class NewportXYStageAdapter:
    """Adapter exposing the Newport driver via the XY stage interface."""

    def __init__(self, driver: NewportXPSStageDriver) -> None:
        self._driver = driver

    def is_connected(self) -> bool:
        return self._driver.axes_ready(("x", "y"))

    def home(self) -> None:
        self._driver.home_xy()

    def move_to(self, x_mm: float, y_mm: float) -> None:
        self._driver.move_xy_to(float(x_mm), float(y_mm))

    def move_by(self, dx_mm: float, dy_mm: float) -> None:
        self._driver.jog_xy(float(dx_mm), float(dy_mm))

    def position(self) -> Tuple[float, float]:
        x = self._driver.axis_position("x")
        y = self._driver.axis_position("y")
        if x is None or y is None:
            return (0.0, 0.0)
        return float(x), float(y)


class NewportRotationStageAdapter:
    """Adapter exposing the Newport driver via the rotation interface."""

    def __init__(self, driver: NewportXPSStageDriver) -> None:
        self._driver = driver

    def is_connected(self) -> bool:
        return self._driver.axes_ready(("theta",))

    def home(self) -> None:
        self._driver.home_rotation()

    def move_to(self, angle_deg: float) -> None:
        self._driver.move_axis_to("theta", float(angle_deg))

    def angle(self) -> float:
        val = self._driver.axis_position("theta")
        return float(val) if val is not None else 0.0
