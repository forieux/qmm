# Q-MM: A Python Quadratic Majorization Minimization toolbox
# Copyright (C) 2021 François Orieux <francois.orieux@universite-paris-saclay.fr>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""The ``qmm`` module
==================

This module implements Majorize-Minimize Quadratic optimization algorithms

The main points of interest are the ``mmmg`` and ``mmcg`` functions.
``Criterion`` is used to easily build criterion optimized by these algorithms.
The ``Potential`` classes are used by ``Criterion``.

"""

# pylint: disable=bad-continuation

import abc
import functools
from typing import Callable, List, Tuple, Union

import numpy as np  # type: ignore
import numpy.linalg as la  # type: ignore
from numpy import ndarray as array

ArrOrList = Union[array, List[array]]


def mmmg(
    crit_list: List["BaseCrit"],
    init: array,
    tol: float = 1e-4,
    max_iter: int = 500,
) -> Tuple[array, List[float]]:
    r"""The Majorize-Minimize Memory Gradient (`3mg`) algorithm.

    The `mmmg` (`3mg`) algorithm is a subspace memory-gradient optimization
    algorithm with an explicit step formula based on Majorize-Minimize Quadratic
    approach [2]_.

    Parameters
    ----------
    crit_list : list of `BaseCrit`
        A list of :class:`BaseCrit` objects that each represent a `μ ψ(V·x - ω)`.
        The criteria are implicitly summed.
    init : array
        The initial point. The `init` array is updated in place to return the
        output. The user must make a copy before calling `mmmg` if this is not
        the desired behavior.
    tol : float, optional
        The stopping tolerance. The algorithm is stopped when the gradient norm
        is inferior to `init.size * tol`.
    max_iter : int, optional
        The maximum number of iterations.

    Returns
    -------
    minimizer : array
        The minimizer of the criterion with same shape than `init`.
    norm_grad : list of float
        The norm of the gradient during iterations.

    Notes
    -----
    The output of :meth:`BaseCrit.operator`, and the `init` value, are
    automatically vectorized internally. However, the output is reshaped as the
    `init` array.

    References
    ----------
    .. [2] E. Chouzenoux, J. Idier, and S. Moussaoui, “A Majorize-Minimize
       Strategy for Subspace Optimization Applied to Image Restoration,” IEEE
       Trans. on Image Process., vol. 20, no. 6, pp. 1517–1528, Jun. 2011, doi:
       10.1109/TIP.2010.2103083.

    """
    point = init.reshape((-1, 1))
    norm_grad = []

    # The first previous moves are initialized with 0 array. Consequently, the
    # first iterations implementation can be improved, at the cost of if
    # statement.
    move = np.zeros_like(point)
    op_directions = [
        np.tile(_vect(crit.operator, move, init.shape), 2) for crit in crit_list
    ]
    step = np.ones((2, 1))

    for _ in range(max_iter):
        # Vectorized gradient
        grad = _gradient(crit_list, point, init.shape)
        norm_grad.append(la.norm(grad))

        # Stopping test
        if norm_grad[-1] < point.size * tol:
            break

        # Memory gradient directions
        directions = np.c_[-grad, move]

        # Step by Majorize-Minimize
        op_directions = [
            np.c_[_vect(crit.operator, grad, init.shape), i_op_dir @ step]
            for crit, i_op_dir in zip(crit_list, op_directions)
        ]
        step = -la.pinv(
            sum(
                crit.norm_mat_major(i_op_dir, point.reshape(init.shape))
                for crit, i_op_dir in zip(crit_list, op_directions)
            )
        ) @ (directions.T @ grad)
        move = directions @ step

        # update
        point += move

    return np.reshape(point, init.shape), norm_grad


def mmcg(
    crit_list: List["BaseCrit"],
    init: array,
    precond: Callable[[array], array] = None,
    tol: float = 1e-4,
    max_iter: int = 500,
) -> Tuple[array, List[float]]:
    """The Majorize-Minimize Conjugate Gradient (MM-CG) algorithm.

    The MM-CG is a nonlinear conjugate gradient (NL-CG) optimization algorithm
    with an explicit step formula based on Majorize-Minimize Quadratic approach
    [1]_.

    Parameters
    ----------
    crit_list : list of `BaseCrit`
        A list of :class:`BaseCrit` objects that each represent a `μ ψ(V·x - ω)`.
        The criteria are implicitly summed.
    init : ndarray
        The initial point. The `init` array is updated in place to return the
        output. The user must make a copy before calling `mmmg` if this is not
        the desired behavior.
    precond : callable, optional
        A callable that must implement a preconditioner, that is `M⁻¹·x`. Must
        be a callable with a unique input parameter `x` and unique output.
    tol : float, optional
        The stopping tolerance. The algorithm is stopped when the gradient norm
        is inferior to `init.size * tol`.
    max_iter : int, optional
        The maximum number of iterations.

    Returns
    -------
    minimizer : array
        The minimizer of the criterion with same shape than `init`.
    norm_grad : list of float
        The norm of the gradient during iterations.

    Notes
    -----
    The output of :meth:`BaseCrit.operator`, and the `init` value, are
    automatically vectorized internally. However, the output is reshaped as the
    `init` array.

    References
    ----------
    .. [1] C. Labat and J. Idier, “Convergence of Conjugate Gradient Methods
       with a Closed-Form Stepsize Formula,” J Optim Theory Appl, p. 18, 2008.

    """
    if precond is None:
        precond = lambda x: x

    point = init.reshape((-1, 1))

    residual = -_gradient(crit_list, point, init.shape)
    sec = precond(residual)
    direction = sec
    delta = residual.T @ direction
    norm_res: List[float] = [la.norm(residual)]

    for _ in range(max_iter):
        # update
        op_direction = [
            _vect(crit.operator, direction, init.shape) for crit in crit_list
        ]

        step = direction.T @ residual
        step = step / sum(
            crit.norm_mat_major(i_op_dir, point.reshape(init.shape))
            for crit, i_op_dir in zip(crit_list, op_direction)
        )

        point += step * direction

        # Gradient
        residual = -_gradient(crit_list, point, init.shape)

        # Stop test
        norm_res.append(la.norm(residual))
        if norm_res[-1] < point.size * tol:
            break

        # Conjugate direction. No reset is done, see Shewchuck.
        delta_old = delta
        delta_mid = residual.T @ sec
        sec = precond(residual)
        delta = residual.T @ sec
        if (delta - delta_mid) / delta_old >= 0:
            direction = sec + (delta - delta_mid) / delta_old * direction
        else:
            direction = sec

    return np.reshape(point, init.shape), norm_res


# Vectorized call
def _vect(func: Callable[[array], array], point: array, shape: Tuple) -> array:
    """Call func with point reshaped as shape and return vectorized output"""
    return np.reshape(func(np.reshape(point, shape)), (-1, 1))


# Vectorized gradient
def _gradient(crit_list: List["Criterion"], point: array, shape: Tuple) -> array:
    """Compute sum of gradient with vectorized parameters and return"""
    return sum(_vect(crit.gradient, point, shape) for crit in crit_list)


class BaseCrit(abc.ABC):
    r"""An abstract base class for criterion

    .. math::
        J(x) = \mu \Psi \left(V x - \omega \right)

    with :math:`\Psi(u) = \sum_i \phi(u_i)`.
    """

    @abc.abstractmethod
    def operator(self, point: array) -> array:
        """Compute the output of `V·x`."""
        return NotImplemented

    @abc.abstractmethod
    def gradient(self, point: array) -> array:
        """Compute the gradient at current point."""
        return NotImplemented

    @abc.abstractmethod
    def norm_mat_major(self, vecs: array, point: array) -> array:
        """Return the normal matrix of the quadratic major function.

        Given vectors `W = V·S`, return `Wᵗ·diag(b)·W`

        where S are the vectors defining a subspace and `b` are Geman &
        Reynolds coefficients at given `point`.

        Parameters
        ----------
        vecs : array
            The `W` vectors.
        point : array
            The given point where to compute Geman & Reynolds coefficients `b`.

        Returns
        -------
        out : array
            The normal matrix
        """
        return NotImplemented


class Criterion(BaseCrit):
    r"""A criterion defined as

    .. math::
        J(x) = \mu \Psi \left(V x - \omega \right)

    with :math:`\Psi(u) = \sum_i \phi(u_i)`.

    data : array
        The `data` array, or the vectorized list of array given at init.
    hyper : float
        The hyperparameter value μ.
    potential : Potential
        The potential φ.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        operator: Callable[[array], ArrOrList],
        adjoint: Callable[[ArrOrList], array],
        potential: "Potential",
        data: ArrOrList = 0,
        hyper: float = 1,
    ):
        """A criterion μ ψ(V·x - ω).

        Parameters
        ----------
        operator: callable
            A callable that compute the output V·x.
        adjoint: callable
            A callable that compute Vᵗ·e.
        potential: Potential
            The potential φ
        data: array or list of array, optional
            The data vector ω.
        hyper: float, optional
            The hyperparameter μ.

        Notes
        -----
        For implementation issue, `operator` and `adjoint` are wrapped by
        methods of same name.

        If `data` is a list of array, `operator` must return a similar list with
        arrays of same shape, and `adjoint` must accept a similar list also.

        In that case, however, and for algorithm purpose, everything is
        internally stacked as a column vector and values are therefore copied.
        This is not efficient but flexible. Users are encouraged to do the
        vectorization themselves and not use the list of array feature.
        """
        self._operator = operator
        self._adjoint = adjoint

        self._data = data
        if isinstance(data, list):
            self._stacked = True
            self._shape = [arr.shape for arr in data]
            self._idx = np.cumsum([0] + [arr.size for arr in data])
            self.data = self._list2vec(data)
        else:
            self._stacked = False
            self.data = data

        self.hyper = hyper
        self.potential = potential

    def _list2vec(self, arr_list: List[array]) -> array:  #  pylint: disable=no-self-use
        """Vectorize a list of array."""
        return np.vstack([arr.reshape((-1, 1)) for arr in arr_list])

    def _vec2list(self, arr: array) -> List[array]:
        """De-vectorize to a list of array."""
        return [
            np.reshape(arr[self._idx[i] : self._idx[i + 1]], shape)
            for i, shape in enumerate(self._shape)
        ]

    def operator(self, point: array) -> array:
        """Return V·x"""
        if self._stacked:
            out = self._list2vec(self._operator(point))
        else:
            out = self._operator(point)
        return out

    def adjoint(self, point: array) -> array:
        """Return Vᵗ·x"""
        if self._stacked:
            out = self._adjoint(self._vec2list(point))
        else:
            out = self._adjoint(point)
        return out

    def value(self, point: array) -> float:
        """The value of the criterion at given point

        Return μ ψ(V·x - ω)
        """
        return self.hyper * np.sum(self.potential(self.operator(point) - self.data))

    def gradient(self, point: array) -> array:
        """The gradient of the criterion at given point

        Return μ Vᵗ·ψ'(V·x - ω)
        """
        return self.hyper * self.adjoint(
            self.potential.gradient(self.operator(point) - self.data)
        )

    def norm_mat_major(self, vecs: array, point: array) -> array:
        matrix = vecs.T @ (self.gr_coeffs(point).reshape((-1, 1)) * vecs)
        if matrix.size == 1:
            matrix = float(matrix)
        return matrix

    def gr_coeffs(self, point: array) -> array:
        """The Geman & Reynolds coefficients at given point

        Given `x` return `φ'(V·x - ω) / (V·x - ω)`
        """
        obj = self.operator(point) - self.data
        return self.potential.gr_coeffs(obj)

    def __call__(self, point: array) -> float:
        return self.value(point)


class QuadCriterion(Criterion):
    r"""A quadratic criterion

    .. math::
        J(x) = \frac{1}{2} \mu \|V x - \omega\|_2^2.

    data : array
        The `data` array, or the vectorized list of array given at init.
    hyper : float
        The hyperparameter value μ.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        operator: Callable[[array], ArrOrList],
        adjoint: Callable[[ArrOrList], array],
        normal: Callable[[array], array] = None,
        data: array = 0,
        hyper: float = 1,
    ):
        """A quadratic criterion ½ μ ||V·x - ω||²

        Parameters
        ----------
        operator: callable
            A callable that compute the output V·x.
        adjoint: callable
            A callable that compute Vᵗ·e.
        normal: callable, optional
            A callable that compute Q·x as Q·x = VᵗV·x
        data: array or list of array, optional
            The data vector ω.
        hyper: float, optional
            The hyperparameter μ.

        Notes
        -----
        The `normal` (`Q`) callable is used for gradient computation as `∇ = μ
        (Q·x - b)` where `b = Vᵗ·ω` instead of `∇ = μ Vᵗ·(V·x - ω)`. This is
        optional and in some case this is more efficient.

        The variable `b = Vᵗ·ω` is computed at object creation.
        """
        super().__init__(operator, adjoint, Square(), data=data, hyper=hyper)
        if normal is not None:
            self._normal = normal
        else:
            self._normal = lambda x: adjoint(operator(x))
        self._data_t = self.adjoint(self.data)

    def value(self, point: array) -> float:
        """The value of the criterion at given point

        Return ½ μ ||V·x - ω||².
        """
        return self.hyper * np.sum((self.operator(point) - self.data) ** 2) / 2

    def gradient(self, point: array) -> array:
        """The gradient at given point

        Return `∇ = μ (Q·x - b) = μ Vᵗ·(V·x - ω)`.
        """
        return self.hyper * (self._normal(point) - self._data_t)

    def norm_mat_major(self, vecs: array, point: array) -> array:
        return vecs.T @ vecs

    def gr_coeffs(self, point: array) -> array:
        """Return 1."""
        return 1

    def __call__(self, point: array) -> float:
        return self.value(point)


class Vmin(BaseCrit):
    r"""A minimum value criterion

    .. math::

        J(u) = \frac{1}{2} \mu \|P_{]-\infty, m]}(u) - m\|^2.

    vmin : float
        The minimum value `m`.
    hyper : float
        The hyperparameter value μ.
    """

    def __init__(self, vmin: float, hyper: float):
        """A minimum value criterion

        J(u) = ½ μ ||P_[m, +∞[(u) - m||².

        Parameters
        ----------
        vmin : float
            The minimum value `m`.
        hyper : float
            The hyperparameter value μ.
        """
        self.vmin = vmin
        self.hyper = hyper

    def operator(self, point):
        return point[point <= self.vmin]

    def value(self, point: array) -> array:
        """Return the value at current point."""
        return self.hyper * np.sum((point[point <= self.vmin] - self.vmin) ** 2 / 2)

    def gradient(self, point: array) -> array:
        return self.hyper * np.where(point <= self.vmin, point - self.vmin, 0)

    def norm_mat_major(self, vecs: array, point: array) -> array:
        return vecs.T @ vecs


class Vmax(BaseCrit):
    r"""A maximum value criterion

    .. math::

        J(u) = \frac{1}{2} \mu \|P_{[M, +\infty[}(u) - m\|^2.

    vmax : float
        The maximum value `M`.
    hyper : float
        The hyperparameter value μ.
    """

    def __init__(self, vmax: float, hyper: float):
        """A maximum value criterion

        Return J(u) = ½ μ ||P_[M, +∞[(u) - M||².

        Parameters
        ----------
        vmax : float
            The maximum value `M`.
        hyper : float
            The hyperparameter value μ.
        """
        self.vmax = vmax
        self.hyper = hyper

    def operator(self, point):
        return point[point >= self.vmax]

    def value(self, point: array) -> array:
        """Return the value at current point."""
        return self.hyper * np.sum((point[point >= self.vmax] - self.vmax) ** 2 / 2)

    def gradient(self, point: array) -> array:
        return self.hyper * np.where(point >= self.vmax, point - self.vmax, 0)

    def norm_mat_major(self, vecs: array, point: array) -> array:
        return vecs.T @ vecs


class Potential(abc.ABC):
    """An abstract base class for potential φ.

    The class has the following attributes.

    inf : float
      The value of lim_{u→0} φ'(u) / u.
    convex : boolean
      A flag indicating if the potential is convex.
    coercive : boolean
      A flag indicating if the potential is coercive.
    """

    def __init__(self, inf: float, convex: bool = False, coercive: bool = False):
        """The potential φ

        Parameters
        ----------
        inf : float
          The value of lim_{u→0} φ'(u) / u
        convex : boolean
          A flag indicating if the potential is convex.
        coercive : boolean
          A flag indicating if the potential is coercive.
        """
        self.inf = inf
        self.convex = convex
        self.coercive = coercive

    @abc.abstractmethod
    def value(self, point: array) -> array:
        """The value φ(·) at given point."""
        return NotImplemented

    @abc.abstractmethod
    def gradient(self, point: array) -> array:
        """The gradient φ'(·) at given point."""
        return NotImplemented

    def gr_coeffs(self, point: array) -> array:
        """The Geman & Reynolds φ'(·)/· coefficients at given point."""
        aux = self.inf * np.ones_like(point)
        idx = point != 0
        aux[idx] = self.gradient(point[idx]) / point[idx]
        return aux

    def __call__(self, point: array) -> array:
        """The value at given point."""
        return self.value(point)


class Square(Potential):
    r"""The Square function

    .. math::

       \phi(u) = \frac{1}{2} u^2.
    """

    def __init__(self):
        """The Square function

        φ(u) = ½ u².
        """
        super().__init__(inf=1, convex=True, coercive=True)

    def value(self, point: array) -> array:
        return point ** 2 / 2

    def gradient(self, point: array) -> array:
        return point

    def __repr__(self):
        return """φ(u) = ½ u²
"""


class Huber(Potential):
    r"""The convex coercive Huber function

    .. math::

       \phi(u) =
       \begin{cases}
          \frac{1}{2} u^2 & \text{, if } u \leq \delta, \\
          \delta |u| - \frac{\delta^2}{2} & \text{, otherwise.}
       \end{cases}

    """

    def __init__(self, delta: float):
        """The Huber function."""
        super().__init__(inf=1, convex=True, coercive=True)
        self.delta = delta

    def value(self, point: array) -> array:
        return np.where(
            np.abs(point) <= self.delta,
            point ** 2 / 2,
            self.delta * (np.abs(point) - self.delta / 2),
        )

    def gradient(self, point: array) -> array:
        return np.where(np.abs(point) <= self.delta, point, self.delta * np.sign(point))

    def __repr__(self):
        return f"""{type(self)}

       ⎛
       ⎜ ½ u²        , if |u| < δ
φ(u) = ⎜
       ⎜ δ|u| - δ²/2 , otherwise.
       ⎝

with δ = {self.delta}
"""


class Hyperbolic(Potential):
    r"""The convex coercive hyperbolic function

    .. math::

       \phi(u) = \delta^2 \left( \sqrt{1 + \frac{u^2}{\delta^2}} -1 \right)

    This is sometimes called Pseudo-Huber.
    """

    def __init__(self, delta: float):
        """The hyperbolic function."""
        super().__init__(inf=1, convex=True, coercive=True)
        self.delta = delta

    def value(self, point: array) -> array:
        return self.delta ** 2 * (np.sqrt(1 + (point ** 2) / (self.delta ** 2)) - 1)

    def gradient(self, point: array) -> array:
        return point / np.sqrt(1 + (point ** 2) / self.delta ** 2)

    def __repr__(self):
        return f"""{type(self)}
               _______
          ⎛   ╱     u²     ⎞
φ(u) = δ²⋅⎜  ╱  1 + ──  - 1⎟
          ⎝╲╱       δ²     ⎠


with δ = {self.delta}
"""


class HerbertLeahy(Potential):
    r"""The non-convex coercive function Herbert & Leahy

    .. math::

       \phi(u) = \log \left(1 + \frac{u^2}{\delta^2} \right)

    """

    def __init__(self, delta: float):
        """The Herbert & Leahy function."""
        super().__init__(inf=2 / delta ** 2, convex=False, coercive=True)
        self.delta = delta

    def value(self, point: array) -> array:
        return np.log(1 + point ** 2 / self.delta ** 2)

    def gradient(self, point: array) -> array:
        return 2 * point / (self.delta ** 2 + point ** 2)

    def __repr__(self):
        return f"""{type(self)}

          ⎛    u²⎞
φ(u) = log⎜1 + ──⎟
          ⎝    δ²⎠

with δ = {self.delta}
"""


class GemanMcClure(Potential):
    r"""The non-convex non-coervice Geman & McClure function

    .. math::

       \phi(u) = \frac{u^2}{2\delta^2 + u^2}

    """

    def __init__(self, delta: float):
        r"""The Geman & Mc Clure function."""
        super().__init__(1 / (delta ** 2), convex=False, coercive=False)
        self.delta = delta

    def value(self, point: array) -> array:
        return point ** 2 / (2 * self.delta ** 2 + point ** 2)

    def gradient(self, point: array) -> array:
        return 4 * point * self.delta ** 2 / (2 * self.delta ** 2 + point ** 2) ** 2

    def __repr__(self):
        return f"""{type(self)}

          u²
φ(u) = ─────────
       u² + 2⋅δ²

with δ = {self.delta}
"""


class TruncSquareApprox(Potential):
    r"""The non-convex non-coercive truncated square approximation

    .. math::

       \phi(u) = 1 - \exp \left(- \frac{u^2}{2\delta^2} \right)

    """

    def __init__(self, delta: array):
        """The truncated square approximation."""
        super().__init__(inf=1 / (delta ** 2), convex=False, coercive=False)
        self.delta = delta

    def value(self, point: array) -> array:
        return 1 - np.exp(-(point ** 2) / (2 * self.delta ** 2))

    def gradient(self, point: array) -> array:
        return point / (self.delta ** 2) * np.exp(-(point ** 2) / (2 * self.delta ** 2))

    def __repr__(self):
        return f"""{type(self)}

               u²
            - ────
              2⋅δ²
φ(u) = 1 - e

with δ = {self.delta}
"""


# Not used finally
def _vectorize(
    func: Callable[[array], array], input_shape: Tuple
) -> Callable[[array], array]:
    """Vectorize a callable.

    Wrap a function to accept a vectorized version of the input and to produce
    vectorized version of the ouput


    Parameters
    ----------
    func : callable
        The function to wrap. Must be a single ndarray parameter callable that
        produce a single ndarray output.
    input_shape : tuple
        The shape that `func` ask as input.

    Returns
    -------
    out : callable
      The wrapped callable.

    """

    @functools.wraps(func)
    def wrapper(arr):
        out = func(np.reshape(arr, input_shape))
        return out.reshape((-1, 1))

    return wrapper
