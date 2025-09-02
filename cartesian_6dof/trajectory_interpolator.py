"""
polynomial time-scaling for SE(3) trajectories
"""
from __future__ import annotations

from typing import List
import numpy as np
import cartesian_6dof

class PolynomialInterpolator:
    """
    Polynomial blend s(τ) ∈ [0,1] with user-specified endpoint derivatives
    in *normalized time* τ ∈ [0,1].
    """

    def __init__(
        self,
        order: int = 5,
        *,
        s1_0: float = 0.0,
        s1_1: float = 0.0,
        s2_0: float = 0.0,
        s2_1: float = 0.0,
    ) -> None:
        if order not in (1, 3, 5):
            raise ValueError("order must be 1, 3, or 5")
        self.order = order

        if order == 1:
            self.coeff = np.array([0.0, 1.0])
        elif order == 3:
            A = np.array([
                [1, 0, 0, 0],   # s(0)=0
                [1, 1, 1, 1],   # s(1)=1
                [0, 1, 0, 0],   # s'(0)=s1_0
                [0, 1, 2, 3],   # s'(1)=s1_1
            ])
            b = np.array([0.0, 1.0, s1_0, s1_1])
            self.coeff = np.linalg.solve(A, b)
        else:  # quintic
            A = np.array([
                [1, 0, 0, 0, 0, 0],       # s(0)=0
                [1, 1, 1, 1, 1, 1],       # s(1)=1
                [0, 1, 0, 0, 0, 0],       # s'(0)=s1_0
                [0, 1, 2, 3, 4, 5],       # s'(1)=s1_1
                [0, 0, 2, 0, 0, 0],       # s''(0)=s2_0
                [0, 0, 2, 6, 12, 20],     # s''(1)=s2_1
            ])
            b = np.array([0.0, 1.0, s1_0, s1_1, s2_0, s2_1])
            self.coeff = np.linalg.solve(A, b)

    def _eval(self, tau: np.ndarray) -> np.ndarray:
        return np.polyval(self.coeff[::-1], tau)

    def __call__(self, start_T, goal_T, *, n_steps=10):
        tau = np.linspace(0.0, 1.0, n_steps)
        s = np.clip(self._eval(tau), 0.0, 1.0)
        return [start_T.interpolate(goal_T, float(a)) for a in s]
