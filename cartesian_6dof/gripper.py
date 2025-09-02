# gripper.py
from __future__ import annotations
from dataclasses import dataclass
import mujoco
import numpy as np

# Exported convenience constants so callers can keep writing OPEN/CLOSE
OPEN  = -255.0
CLOSE =  255.0

@dataclass
class GripperConfig:
    ctrl_index: int            # index into data.ctrl for the gripper actuator
    open_value: float = OPEN
    close_value: float = CLOSE
    name: str | None = None    # optional actuator name (for logging/debug)

class Gripper:
    """
    Thin wrapper around a single DoF gripper actuator in MuJoCo.

    Usage:
        gr = Gripper(model, data, GripperConfig(ctrl_index=6))
        gr.open()
        gr.close()
        gr.set(123.0)  # raw command passthrough if you want
    """
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, cfg: GripperConfig):
        self.model = model
        self.data = data
        self.cfg = cfg
        self._last_cmd: float | None = None

    @property
    def index(self) -> int:
        return self.cfg.ctrl_index

    def set(self, cmd: float) -> None:
        """Set raw gripper command (e.g., PWM/target position etc.)."""
        # avoid redundant writes if unchanged (nice but optional)
        if self._last_cmd is None or self._last_cmd != cmd:
            self.data.ctrl[self.index] = float(cmd)
            self._last_cmd = float(cmd)

    # Convenience actions
    def open(self) -> None:
        self.set(self.cfg.open_value)

    def close(self) -> None:
        self.set(self.cfg.close_value)

    # Optional helpers
    def is_open_cmd(self, tol: float = 1e-9) -> bool:
        return self._last_cmd is not None and abs(self._last_cmd - self.cfg.open_value) <= tol

    def is_close_cmd(self, tol: float = 1e-9) -> bool:
        return self._last_cmd is not None and abs(self._last_cmd - self.cfg.close_value) <= tol
