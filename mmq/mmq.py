"""
The MMQ module
==============


This module implements Majorize-Minimize Quadratic optimization algorithms

The main points of interest are the `mmmg` and `mm_cg` functions. `Criterion` is
use to easily build criterion optimized by these function.

Comments on `Criterion`
-----------------------

The module provides a `Criterion` object for convenience. This object has be
made to help the implementation. However, thanks to dynamic nature of python,
the algorithms need in practive any object with three specific methods.

- `operator` : a callable with a point `x` as unique parameter, that must return
  the application of `V` (that is `V·x`).
- `gradient` : a callable with a point `x` as unique parameter, that must return
  the gradient of the criterion (that is `Vᵗ·φ'(V·x - ω)`).
- `norm_mat_major` : a callable with two parameters. The first one is the result
  of the operator applied on the subspace vectors. The second is the point `x`,
  where the normal matrix of the quadratic major function must be returned.

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
    crit_list: List["Criterion"],
    init: array,
    tol: float = 1e-4,
    max_iter: int = 500,
) -> Tuple[array, List[float]]:
    r"""The Majorize-Minimize Memory Gradient (`3mg`) algorithm

    The `mmmg`, or `3mg`, algorithm is a subspace memory-gradient optimization
    algorithm with an explicit step formula based on Majorize-Minimize Quadratic
    approach.

    Parameters
    ----------
    crit_list : list of Criterion
        A list of `Criterion` object that represent `φ(V·x - ω)`.
        The use of this list is necessary to allow efficient implementation and
        reuse of calculations. See notes section for details.

    init : array
        The initial point. The init is update inplace. The user must make a copy
        before calling mmmg if this is not the desired behaviour.

    tol : float, optional
        The stopping tolerance. The algorithm is stopped when the gradient norm
        is inferior to `init.size * tol`.

    max_iter : int, optional
        The maximum number of iteration.

    Returns
    -------
    minimiser : array
        The minimiser of the criterion with same shape than `init`.

    norm_grad : list of float
        The norm of the gradient during iterations.

    Notes
    -----

    The explicit step formula ensures fast convergence of the algorithm to a
    minimizer of the criterion without line search for the step and without
    tuning parameters. On the contrary, the criterion must meet conditions. In
    particular, the criterion must be like

    .. math::
       J(x) = \sum_k \phi_k(V_k x - \omega_k)

    where `x` is the unkown of size `N`, `V` a matrix, and `ω` of size `M`. In
    addition, among other conditions, `φ` must be differentiable (see
    documentation and [2]_ for details).

    The output of callable (e. g. operator in Criterion), and the `init` value,
    are vectorized internally. The output `minimiser` is reshaped as the `init`
    array.

    The algorithm use `Criterion` data structure. Thanks to dynamic nature of
    python, this is not required and user can provide it's own structure, see
    documentation. `Criterion` however comes with boilerplate.

    References
    ----------
    .. [2] E. Chouzenoux, J. Idier, and S. Moussaoui, “A Majorize–Minimize
       Strategy for Subspace Optimization Applied to Image Restoration,” IEEE
       Trans. on Image Process., vol. 20, no. 6, pp. 1517–1528, Jun. 2011, doi:
       10.1109/TIP.2010.2103083.

    """
    point = init.reshape((-1, 1))
    norm_grad = []

    # The first previous moves are initialized with 0 array. Consquently, the
    # first iterations implementation can be improved, at the cost of if
    # statement.
    move = np.zeros_like(point)
    op_directions = [
        np.tile(vect(crit.operator, move, init.shape), 2) for crit in crit_list
    ]
    step = np.ones((2, 1))

    for _ in range(max_iter):
        # Vectorized gradient
        grad = gradient(crit_list, point, init.shape)
        norm_grad.append(la.norm(grad))

        # Stopping test
        if norm_grad[-1] < point.size * tol:
            break

        # Memory gradient directions
        directions = np.c_[-grad, move]

        # Step by Majorize-Minimize
        op_directions = [
            np.c_[vect(crit.operator, grad, init.shape), i_op_dir @ step]
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
    crit_list: List["Criterion"],
    init: array,
    precond: Callable[[array], array] = None,
    tol: float = 1e-4,
    max_iter: int = 500,
) -> Tuple[array, List[float]]:
    """The Majorize-Minimize Conjugate Gradient (MM-CG) algorithm

    The MM-CG is a nonlinear conjugate gradient (NL-CG) (NL-CG) (NL-CG) (NL-CG)
    (NL-CG) (NL-CG) (NL-CG) (NL-CG) (NL-CG) optimization algorithm with an
    explicit step formula based on Majorize-Minimize Quadratic approach. This
    ensures quick convergence of the algorithm to a minimizer of the criterion
    without line search for the step and without tuning parameters. On the
    contrary, the criterion must meet conditions. In particular, the criterion
    must be like

       `J(x) = ∑ₖ φₖ(Vₖ·x - ωₖ)`

    where `x` is the unkown of size `N`, `V` a matrix of size `M × N` and `ω` of
    size `M`. In addition, among other conditions, `φ` must be differentiable
    (see documentation and [1]_ for details).

    Parameters
    ----------
    crit_list : list of Criterion
        A list of `Criterion` object that represent `φ(V·x - ω)`.
        The use of this list is necessary to allow efficient implementation and
        reuse of calculations. See notes section for details.

    init : ndarray
        The initial point. The init is update inplace. The user must make a copy
        before calling mmmg if this is not the desired behaviour.

    precond : callable, optional
        A callable that must implement a preconditioner, that is `M⁻¹·x`. Must
        be a callable with a unique input parameter `x` and unique output.

    tol : float, optional
        The stopping tolerance. The algorithm is stopped when the gradient norm
        is inferior to `init.size * tol`.

    max_iter : int, optional
        The maximum number of iteration.

    Returns
    -------
    minimiser : ndarray
        The minimiser of the criterion with same shape than `init`.

    norm_grad : list of float
        The norm of the gradient during iterations.

    Notes
    -----
    The output of callable (e. g. operator in Criterion), and the `init` value,
    are automatically vectorized internally. However, the output is reshaped as
    the `init` array.

    The algorithm use `Criterion` data structure. Thanks to dynamic nature of
    python, this is not required and user can provide it's own structure, see
    documentation. `Criterion` however comes with boilerplate.

    References
    ----------
    .. [1] C. Labat and J. Idier, “Convergence of Conjugate Gradient Methods
       with a Closed-Form Stepsize Formula,” J Optim Theory Appl, p. 18, 2008.

    """
    if precond is None:
        precond = lambda x: x

    point = init.reshape((-1, 1))

    residual = -gradient(crit_list, point, init.shape)
    sec = precond(residual)
    direction = sec
    delta = residual.T @ direction
    norm_res: List[float] = [la.norm(residual)]

    for _ in range(max_iter):
        # update
        op_direction = [
            vect(crit.operator, direction, init.shape) for crit in crit_list
        ]

        step = direction.T @ residual
        step = step / sum(
            crit.norm_mat_major(i_op_dir, point.reshape(init.shape))
            for crit, i_op_dir in zip(crit_list, op_direction)
        )

        point += step * direction

        # Gradient
        residual = -gradient(crit_list, point, init.shape)

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
def vect(func: Callable[[array], array], point: array, shape: Tuple) -> array:
    """Call func with point reshaped as shape and return vectorized output"""
    return np.reshape(func(np.reshape(point, shape)), (-1, 1))


# Vectorized gradient
def gradient(crit_list: List["Criterion"], point: array, shape: Tuple) -> array:
    """Compute sum of gradient with vectorized parameters and return"""
    return sum(vect(crit.gradient, point, shape) for crit in crit_list)


class Criterion:
    """A criterion μ ∑ φ(V·x - ω)"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        operator: Callable[[array], ArrOrList],
        adjoint: Callable[[ArrOrList], array],
        potential: "Potential",
        hyper: float = 1,
        data: ArrOrList = 0,
    ):
        """A criterion μ ∑ φ(V·x - ω)

        Parameters
        ----------
        operator: callable
          A callable that compute the output `V·x` given `x`.
        adjoint: callable
          Vᵗ·e
        potential: Potential
          φ and φ'
        hyper: float
          μ
        data: array or list of array (optional)
          ω

        Notes
        -----
        If `data` is a list of array, `operator` must return a similar list with
        array of same shape, and `adjoint` must accept a similar list also.

        In that case, however and for algorithm purpose, everything is
        internally stacked as a column vector and values are therefor copied.
        This is not efficient but flexible. Users are encouraged to do the
        vectorization themselves.

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
        return np.vstack([arr.reshape((-1, 1)) for arr in arr_list])

    def _vec2list(self, arr: array) -> List[array]:
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

        Return μ·φ(V·x - ω)
        """
        return self.hyper * np.sum(self.potential(self.operator(point) - self.data))

    def gradient(self, point: array) -> array:
        """The gradient of the criterion at given point

        Return μ·Vᵗ·φ'(V·x - ω)
        """
        return self.hyper * self.adjoint(
            self.potential.gradient(self.operator(point) - self.data)
        )

    def norm_mat_major(self, vecs: array, point: array) -> array:
        """Return the normal matrix of the major function

        Given vecs `W = V·S`, return `Wᵗ·diag(b)·W`

        where S are the vectors defining the subspace and `b` are GR
        coefficients at given point.

        Parameters
        ----------
        vecs : array
            The `W` vectors

        point : array
            The given point where to compute GR coefficients `b`

        """
        matrix = vecs.T @ (self.gr_coeffs(point).reshape((-1, 1)) * vecs)
        if matrix.size == 1:
            matrix = float(matrix)
        return matrix

    def gr_coeffs(self, point: array) -> array:
        """Return the GR coefficients at given point

        φ'(V·x - ω) / (V·x - ω)

        """
        obj = self.operator(point) - self.data
        return self.potential.gr_coeffs(obj)

    def __call__(self, point: array) -> float:
        return self.value(point)


class QuadCriterion(Criterion):
    """A quadratic criterion ½ μ ||V·x - ω||²"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        operator: Callable[[array], ArrOrList],
        adjoint: Callable[[ArrOrList], array],
        normal: Callable[[array], array],
        hyper: float = 1,
        data: array = 0,
    ):
        """A quadratic criterion μ||V·x - ω||²

        Parameters
        ----------
        operator: callable
          V·x
        adjoing: callable
          Vᵗ·e
        normal: callable
          VᵗV·x
        hyper: float, optionnal
          μ
        data: ndarray, optionnal
          ω
        """
        super().__init__(operator, adjoint, Square(), hyper, data)
        self.normal = normal
        self.data_t = self.adjoint(self.data)

    def value(self, point: array) -> float:
        """The value of the criterion at given point

        Return `½ μ ||V·x - ω||²`
        """
        return self.hyper * np.sum((self.operator(point) - self.data) ** 2) / 2

    def gradient(self, point: array) -> array:
        """The gradient of the criterion at given point

        Return `μ Vᵗ·(V·x - ω)`

        Notes
        -----
        Use `normal` `Q` callable internally for potential better efficiency,
        with computation of `μ (Q·x - b)` where `Q = Vᵗ·V` and `b = Vᵗ·ω` is
        precomputed

        """
        return self.hyper * (self.normal(point) - self.data_t)

    def norm_mat_major(self, vecs: array, point: array) -> array:
        """Return the normal matrix of the major function

        Given W = V·D, return Wᵗ·W.

        Parameters
        ----------
        vecs : array
            The `W` vectors.

        point : array
            The given point where to compute GR coefficients.
        """
        return vecs.T @ vecs

    def gr_coeffs(self, point: array) -> array:
        """Return the GR coefficients at given point

        Always return the scalar 1.

        Notes
        -----
        Not use internally and present for consistency.

        """
        return 1

    def __call__(self, point: array) -> float:
        return self.value(point)


class Potential(abc.ABC):
    """An abstract base class for the potentials φ.

    Attributs
    ---------
    inf : float
      The value of lim_{u→0} φ'(u) / u.

    convex : boolean
      A flag indicating if the potential is convex.

    coercive : boolean
      A flag indicating if the potential is coercive.
    """

    def __init__(self, inf: float, convex: bool = False, coercive: bool = False):
        """The potentials φ

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
        """The Geman \& Reynolds φ'(·)/· coefficients at given point."""
        aux = self.inf * np.ones_like(point)
        idx = point != 0
        aux[idx] = self.gradient(point[idx]) / point[idx]
        return aux

    def __call__(self, point: array) -> array:
        """The value at given point."""
        return self.value(point)


class VminProj(Potential):
    """The projection criterion

    D(u) = ½ ||P_[m, +∞[(u) - m||²
    """

    def __init__(self, vmin: float):
        """The projection criterion

        D(u) = ½ ||P_[m, +∞[(u) - m||²
        """
        super().__init__(inf=1, convex=True, coercive=True)
        self.vmin = vmin

    def value(self, point: array) -> array:
        return np.sum((point[point < self.vmin] - self.vmin) ** 2 / 2)

    def gradient(self, point: array) -> array:
        return np.where(point > self.vmin, 0, point - self.vmin)


class VmaxProj(Potential):
    """The projection criterion

    D(u) = ½ ||P_]-∞, M](u) - M||²
    """

    def __init__(self, vmax: float, convex=True, coercive=True):
        super().__init__(inf=1)
        self.vmax = vmax

    def value(self, point: array) -> array:
        return np.sum((point[point > self.vmax] - self.vmax) ** 2 / 2)

    def gradient(self, point: array) -> array:
        return np.where(point < self.vmax, 0, point - self.vmax)


class Square(Potential):
    r"""The square function

    .. math::

       \phi(u) = \frac{1}{2} u^2

    """

    def __init__(self):
        super().__init__(inf=1, convex=True, coercive=True)

    def value(self, point: array) -> array:
        return point ** 2 / 2

    def gradient(self, point: array) -> array:
        return point

    def __repr__(self):
        return """
φ(u) = ½ u²

Convex and coercive
"""


class Hyperbolic(Potential):
    r"""The convex coercive hyperbolic function

    .. math::

       \phi(u) = \delta^2 \left( \sqrt{1 + \frac{u^2}{\delta^2}} -1 \right)

    This is called sometimes Pseudo-Huber.
    """

    def __init__(self, delta: float):
        super().__init__(inf=1 / (delta ** 2), convex=True, coercive=True)
        self.inf = 1 / (2 * delta)  # To check
        self.delta = delta

    def value(self, point: array) -> array:
        return self.delta ** 2 * np.sqrt(1 + (point ** 2) / (self.delta ** 2)) - 1

    def gradient(self, point: array) -> array:
        return point / np.sqrt(1 + (point ** 2) / self.delta ** 2)

    def __repr__(self):
        return """
          ⎛    _______     ⎞
          ⎜   ╱     x²     ⎟
φ(u) = δ²⋅⎜  ╱  1 + ──  - 1⎟
          ⎝╲╱       δ²     ⎠

Convex and coercive
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
        return """
       ⎛
       ⎜ ½ u²        , if |u| < δ
φ(u) = ⎜
       ⎜ δ|u| - δ²/2 , otherwise.
       ⎝

Convex and coercive.
"""


class GemanMcClure(Potential):
    r"""The Geman & McClure non-convex non-coervice function

    .. math::

       \phi(u) = \frac{u^2}{2\delta^2 + u^2}

    """

    def __init__(self, delta: float):
        super().__init__(1 / (delta ** 2), convex=False, coercive=False)
        self.delta = delta

    def value(self, point: array) -> array:
        return point ** 2 / (2 * self.delta ** 2 + point ** 2)

    def gradient(self, point: array) -> array:
        return 4 * point * self.delta ** 2 / (2 * self.delta ** 2 + point ** 2) ** 2

    def __repr__(self):
        return """
          u²
φ(u) = ─────────
       u² + 2⋅δ²

Non-convex and non-coercive
"""


class SquareTruncApprox(Potential):
    r"""The non-convex non-coercive truncated square approximation

    .. math::

       \phi(u) = 1 - \exp \left(- \frac{u^2}{2\delta^2} \right)

    """

    def __init__(self, delta: array):
        super().__init__(inf=1 / (delta ** 2), convex=False, coercive=False)
        self.delta = delta

    def value(self, point: array) -> array:
        return 1 - np.exp(-(point ** 2) / (2 * self.delta ** 2))

    def gradient(self, point: array) -> array:
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
    r"""The Herbert & Leahy non-convex coercive function

    .. math::

       \phi(u) = \log \left(1 + \frac{u^2}{\delta^2} \right)

    """

    def __init__(self, delta: float):
        super().__init__(inf=np.inf, convex=False, coercive=True)
        self.delta = delta

    def value(self, point: array) -> array:
        return np.log(1 + point ** 2 / self.delta ** 2)

    def gradient(self, point: array) -> array:
        return 2 * point / (self.delta ** 2 + point ** 2)

    def __repr__(self):
        return """
          ⎛    u²⎞
φ(u) = log⎜1 + ──⎟
          ⎝    δ²⎠

Non-convex and coercive
"""


# Not used finally
def vectorize(
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
