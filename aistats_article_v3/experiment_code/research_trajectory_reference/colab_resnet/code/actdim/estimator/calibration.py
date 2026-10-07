"""The monotone map from an estimate to a dimension, fitted once and then frozen.

The estimate is not the active dimension. It saturates: on the article's twenty-direction
range it is a concave increasing function of the truth, crossing the identity somewhere near
twelve directions. A calibration absorbs that curve, so what is reported afterwards is a
recovery and not a raw statistic -- and because the map is monotone it cannot change the rank
correlation, which is why the article reports the correlation on the raw estimate.

The split the map is fitted on must be disjoint in **both** the rank and the seed. In the
archived tree's predecessor, "held out" meant held-out seeds while the frequency geometry was
identical across seeds, so the very thing the estimate responds to was never held out at all
and the reported error was a training error. That is the failure this class exists to make
visible: a calibration used before it is fitted raises, and the number of points it was fitted
on is recorded on the object.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np

KINDS = ("isotonic", "affine", "identity")


class Calibration:
    """A monotone map from an estimate to a dimension.

    ``isotonic`` fits a non-decreasing step function and interpolates between its knots;
    ``affine`` fits a line, which is what the twenty-direction pipeline used; ``identity``
    is the null, kept so that "uncalibrated" is a choice a caller states rather than a branch
    it takes.
    """

    def __init__(self, kind: str = "isotonic"):
        if kind not in KINDS:
            raise ValueError(f"unknown calibration {kind!r}. Expected one of {KINDS}")
        self.kind = kind
        self.fitted = False
        self.n_points = 0
        self.coef: Optional[Any] = None
        self._predict = None

    # -- fitting ---------------------------------------------------------------

    def fit(self, estimates: Sequence[float], truth: Sequence[float]) -> "Calibration":
        """Fit on paired estimates and known dimensions, dropping non-finite pairs."""
        est = np.asarray(estimates, dtype=np.float64)
        true = np.asarray(truth, dtype=np.float64)
        keep = np.isfinite(est) & np.isfinite(true)
        est, true = est[keep], true[keep]
        if len(est) < 3:
            raise ValueError("a calibration needs at least three finite pairs")

        if self.kind == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            fitted = IsotonicRegression(out_of_bounds="clip").fit(est, true)
            self._predict = fitted.predict
        elif self.kind == "affine":
            slope, intercept = np.polyfit(est, true, 1)
            self.coef = (float(slope), float(intercept))
            self._predict = lambda z: self.coef[0] * z + self.coef[1]
        else:
            self._predict = lambda z: z

        self.fitted = True
        self.n_points = int(len(est))
        return self

    @classmethod
    def from_points(cls, estimates: Sequence[float], truth: Sequence[float]) -> "Calibration":
        """Rebuild a stored isotonic map from its knots, without refitting.

        The frozen configuration files carry the knots of the map that was fitted at
        selection time. Rebuilding from them, rather than refitting on whatever data is to
        hand, is what makes "frozen" mean anything.
        """
        cal = cls("isotonic")
        knots_x = np.asarray(estimates, dtype=np.float64)
        knots_y = np.asarray(truth, dtype=np.float64)
        if len(knots_x) < 2 or len(knots_x) != len(knots_y):
            raise ValueError("a stored map needs at least two matching knots")
        order = np.argsort(knots_x)
        knots_x, knots_y = knots_x[order], knots_y[order]
        # Linear between knots and clipped outside them, which is what the fitted map does.
        cal._predict = lambda z: np.interp(z, knots_x, knots_y,
                                           left=knots_y[0], right=knots_y[-1])
        cal.fitted = True
        cal.n_points = int(len(knots_x))
        return cal

    # -- use -------------------------------------------------------------------

    def predict(self, estimates: Union[float, Sequence[float]]) -> Union[float, np.ndarray]:
        """Map estimates through the calibration. NaN in, NaN out."""
        if not self.fitted:
            raise RuntimeError("this calibration has not been fitted")
        values = np.atleast_1d(np.asarray(estimates, dtype=np.float64))
        out = np.full(len(values), np.nan)
        finite = np.isfinite(values)
        if finite.any():
            out[finite] = self._predict(values[finite])
        return out if np.ndim(estimates) else float(out[0])
