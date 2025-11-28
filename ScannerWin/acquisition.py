from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AcquisitionSettings:
    pixels: List[int]
    samples_per_pixel: int
    nplc: float
    loops: int
    inter_loop_delay_s: float
    auto_range: bool
    current_range: float
    measurement_mode: str  # "time" or "voltage"
    voltage_steps: Optional[List[float]]
    voltage_settle_s: float
    constant_bias_voltage: Optional[float]
    use_local_readout: bool
    use_local_bias: bool
    voltage_zero_center: bool = False
    voltage_zero_pause_s: float = 0.0
    current_limit: Optional[float] = None
