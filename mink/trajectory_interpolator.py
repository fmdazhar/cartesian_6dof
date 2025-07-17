"""
polynomial time-scaling for SE(3) trajectories
"""
from __future__ import annotations

from typing import List
import numpy as np
import mink


class PolynomialInterpolator:
    """
    Generate poses along a polynomial blend curve that guarantees Cⁿ
    continuity at the end-points (n = 0, 1, 2 for orders 1, 3, 5).

    Parameters
    ----------
    order : {1, 3, 5}
        1  → linear          (position only,  C⁰)
        3  → cubic “smooth-step” (pos+vel,   C¹)
        5  → quintic          (pos+vel+acc, C²)
    """

    def __init__(self, order: int = 5) -> None:
        if order not in (1, 3, 5):
            raise ValueError("order must be 1, 3, or 5")
        self.order = order

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        start_T: mink.SE3,
        goal_T:  mink.SE3,
        *,
        n_steps: int = 100,
    ) -> List[mink.SE3]:
        """
        Parameters
        ----------
        start_T, goal_T : mink.SE3    Start/goal poses in world frame
        n_steps          : int        Number of discrete samples

        Returns
        -------
        list[mink.SE3] – evenly-spaced poses
        """
        tau = np.linspace(0.0, 1.0, n_steps)     # normalized time
        s   = self._blend(tau)                   # polynomial blend
        return [start_T.interpolate(goal_T, alpha=float(a)) for a in s]

    # ------------------------------------------------------------------ #
    # private helpers
    # ------------------------------------------------------------------ #
    def _blend(self, tau: np.ndarray) -> np.ndarray:
        """Return the scalar blend function s(τ) ∈ [0,1]."""
        if self.order == 1:                       # linear
            return tau
        if self.order == 3:                       # cubic C¹
            return 3*tau**2 - 2*tau**3
        # quintic C²
        return 6*tau**5 - 15*tau**4 + 10*tau**3
