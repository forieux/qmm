#!/usr/bin/env python3

"""This module contains the definition of potentials."""

import abc
from typing import Callable

import numpy as np


class Criterion:
    """A criterion μ sum_c φ(v_c^t·x - ω)"""

    def __init__(
        self,
        operator: Callable,
        adjoint: Callable,
        potential: Callable,
        potential_grad: Callable,
        hyper: float = 1,
        mean: np.ndarray = 0,
    ):
        """A criterion μ sum_c φ(v_c^t·x - ω)

        Parameters
        ----------
        operator: callable
          V^t·x
        adjoing: callable
          V·e
        potential: callable
          φ
        potential: callable
          φ'
        hyper: float
          μ
        mean: ndarray (optionnal)
          ω
        """
        self.operator = operator
        self.adjoint = adjoint
        self.potential = potential
        self.potential_grad = potential_grad
        self.hyper = hyper
        self.mean = mean

    def value(self, point):
        # λ·φ(Dx - ω)
        return self.hyper * np.sum(self.potential(self.operator(point) - self.mean))

    def gradient(self, point):
        return self.hyper * self.adjoint(
            self.potential_grad(self.operator(point) - self.mean)
        )

    def norm_mat_major(self, vecs: np.ndarray, point: np.ndarray):
        """Return the normal matrix of the major function

        Given vecs D = V·S, return D^T·diag(b)·D

        where S are the vectors defining the subspace
        """
        return vecs.T @ (self.gr_coeffs(point).reshape((-1, 1)) * vecs)

    def gr_coeffs(self, point):
        """Return φ'(x) / x"""
        obj = self.operator(point) - self.mean
        return self.potential.gr_coeffs(obj)

    def __call__(self, point):
        return self.value(point)


class QuadCriterion:
    """A quadratic criterion μ||H·x - ω||^2"""

    def __init__(
        self,
        operator: Callable,
        adjoint: Callable,
        normal: Callable,
        hyper: float = 1,
        mean: np.ndarray = None,
    ):
        """A quadratic criterion μ||H·x - ω||^2

        Parameters
        ----------
        operator: Callable
          H·x
        adjoing: Callable
          H^t·e
        normal: Callable
          H^tH·x
        hyper: float
          μ
        mean: ndarray (optionnal)
          ω
        """
        self.operator = operator
        self.normal = normal
        self.hyper = hyper
        if mean is not None:
            self.mean = mean
            self.mean_t = adjoint(mean)
        else:
            self.mean = 0
            self.mean_t = 0

    def value(self, point):
        # ||H·x - ω||_2^2
        return self.hyper * np.sum((self.operator(point) - self.mean) ** 2) / 2

    def norm_mat_major(self, vecs: np.ndarray, point: np.ndarray):
        """Return the normal matrix of the major function

        Given V = H·D, return V^T·diag(b)·V
        """
        return vecs.T @ vecs

    def gradient(self, point):
        return self.hyper * (self.normal(point) - self.mean_t)

    def __call__(self, point):
        return self.value(point)


class Potential(abc.ABC):
    """An abstract base class for the potentials."""

    def __init__(self):
        self.inf = 1

    @abc.abstractmethod
    def value(self, point: np.ndarray):
        """The value at current point"""
        return NotImplemented

    @abc.abstractmethod
    def gradient(self, point: np.ndarray):
        """The gadient at current point"""
        return NotImplemented

    def gr_coeffs(self, point: np.ndarray):
        """The GR coefficients at current point"""
        aux = self.inf * np.ones_like(point)
        idx = point != 0
        aux[idx] = self.gradient(point[idx]) / point[idx]
        return aux

    def __call__(self, point: np.ndarray):
        """The value at current point"""
        return self.value(point)


class VminProj(Potential):
    """The projection criterion

    D(u) = ½ ||P_[m, +∞[(u) - m||^2
    """

    def __init__(self, vmin):
        super().__init__()
        self.vmin = vmin
        self.convex = True
        self.coercive = True

    def value(self, point):
        return np.sum((point[point < self.vmin] - self.vmin) ** 2 / 2)

    def gradient(self, point):
        return np.where(point > self.vmin, 0, point - self.vmin)


class VmaxProj(Potential):
    """The projection criterion

    D(u) = ½ ||P_]-∞, M](u) - M||^2
    """

    def __init__(self, vmax):
        super().__init__()
        self.vmax = vmax
        self.convex = True
        self.coercive = True

    def value(self, point):
        return np.sum((point[point > self.vmax] - self.vmax) ** 2 / 2)

    def gradient(self, point):
        return np.where(point < self.vmax, 0, point - self.vmax)


class Square(Potential):
    """The square convex coercive function

    phi(u) = ½ u^2

    """

    def __init__(self):
        super().__init__()
        self.convex = True
        self.coercive = True

    def value(self, point):
        return point ** 2 / 2

    def gradient(self, point):
        return point

    def __repr__(self):
        return """

φ(u) = ½ u²

Convex and coercive
"""


class Hyperbolic(Potential):
    """The convex coercive hyperbolic function

    phi(u) = sqrt(1 + u^2/delta^2)-1

    """

    def __init__(self, delta):
        super().__init__()
        self.inf = 1 / (2 * delta)  # To check
        self.delta = delta
        self.convex = True
        self.coercive = True

    def value(self, point):
        return np.sqrt(1 + (point ** 2) / (self.delta ** 2)) - 1

    def gradient(self, point):
        return (point / (self.delta ** 2)) * np.sqrt(1 + (point ** 2) / self.delta ** 2)

    def __repr__(self):
        return """
           _______
          ╱ u²
φ(u) =   ╱  ── + 1 - 1
       ╲╱   δ²

Convex and coercive
"""


class Huber(Potential):
    """The convex coercive Huber function

    phi(u) = u^2

    if |u| < δ, and

    phi(u) = δ|u| - δ^2/2

    otherwise.
    """

    def __init__(self, delta):
        super().__init__()
        self.delta = delta
        self.convex = True
        self.coercive = True

    def value(self, point):
        return np.where(np.abs(point) <= self.delta, point ** 2, np.abs(point))

    def gradient(self, point):
        return np.where(
            np.abs(point) <= self.delta, 2 * point, 2 * self.delta * np.sign(point)
        )

    def __repr__(self):
        return """
       ⎛
       ⎜  u², if |u| < δ
φ(u) = ⎜
       ⎜ |u|, otherwise
       ⎝

Convex and coercive.
"""


class GemanMcClure(Potential):
    """The Geman & McClure non-convex non-coervice function

    phi(u) = u^2 / (2 δ^2 + u^2)

    """

    def __init__(self, delta):
        super().__init__()
        self.delta = delta
        self.convex = False
        self.coercive = False

    def value(self, point):
        return point ** 2 / (2 * self.delta ** 2 + point ** 2)

    def gradient(self, point):
        return 4 * point * self.delta ** 2 / (2 * self.delta ** 2 + point ** 2) ** 2

    def __repr__(self):
        return """

          u²
φ(u) = ─────────
       u² + 2⋅δ²

Non-convex and non-coercive
"""


class SquareTruncApprox(Potential):
    """The non-convex non-coercive, truncated square approximation

    phi(u) = 1 - exp(- u^2/(2 δ^2) );

    """

    def __init__(self, delta):
        super().__init__()
        self.delta = delta
        self.convex = False
        self.coercive = False

    def value(self, point):
        return 1 - np.exp(-(point ** 2) / (2 * self.delta ** 2))

    def gradient(self, point):
        return point / (self.delta ** 2) * np.exp(-(point ** 2) / (2 * self.delta ** 2))

    def __repr__(self):
        return """

               u²
            - ────
              2⋅δ²
φ(u) = 1 - e

Non-convex and non-coercive
"""


class HerbertLeahy(Potential):
    """The Herbert & Leahy non-convex coercive function

    phi(u) = log(1 + u^2 / δ^2)

    """

    def __init__(self, delta):
        super().__init__()
        self.delta = delta
        self.convex = False
        self.coercive = True

    def value(self, point):
        return np.log(1 + point ** 2 / self.delta ** 2)

    def gradient(self, point):
        return 2 * point / (self.delta ** 2 + point ** 2)

    def __repr__(self):
        return """

          ⎛    u²⎞
φ(u) = log⎜1 + ──⎟
          ⎝    δ²⎠

Non-convex and coercive
"""
