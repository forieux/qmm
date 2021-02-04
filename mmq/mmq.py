"""This module implements Majorize-Minimize Quadratic algorithms"""

import functools
import abc
from typing import Callable, Tuple, List

import numpy as np
import numpy.linalg as la


class BaseCriterion(abc.ABC):
    """An abstract base class for criteria like μ sum_c φ(v_c^t·x - ω)"""

    def __init__(
        self,
        operator: Callable,
        adjoint: Callable,
        hyper: float = 1,
        mean: np.ndarray = 0,
    ):
        self.operator = operator
        self.adjoint = adjoint
        self.hyper = hyper
        self.mean = mean

    @abc.abstractmethod
    def value(self, point):
        """The value at current point"""
        return NotImplemented

    @abc.abstractmethod
    def gradient(self, point):
        """The gadient at current point"""
        return NotImplemented

    @abc.abstractmethod
    def norm_mat_major(self, vecs: np.ndarray, point: np.ndarray):
        """Return the normal matrix of the major function

        Given vecs D = V·S, return D^T·diag(b)·D

        where S are the vectors defining the subspace
        """
        return NotImplemented

    def __call__(self, point):
        """The value at current point"""
        return self.value(point)


class Criterion(BaseCriterion):
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
        super().__init__(operator, adjoint, hyper, mean)
        self.potential = potential
        self.potential_grad = potential_grad

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
        """Return φ'(V·x - ω) / (V·x - ω)"""
        obj = self.operator(point) - self.mean
        return self.potential.gr_coeffs(obj)

    def __call__(self, point):
        return self.value(point)


class QuadCriterion(BaseCriterion):
    """A quadratic criterion μ||H·x - ω||^2"""

    def __init__(
        self,
        operator: Callable,
        adjoint: Callable,
        normal: Callable,
        hyper: float = 1,
        mean: np.ndarray = 0,
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
        hyper: float (optionnal)
          μ
        mean: ndarray (optionnal)
          ω
        """
        super().__init__(operator, adjoint, hyper, mean)
        self.normal = normal
        self.mean_t = self.adjoint(mean)

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

    def __init__(self, inf):
        """
        Parameters
        ----------
        inf : float
          The value of lim_{u→0} φ'(u) / 2 u
        """
        self.inf = inf

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
        super().__init__(inf=1)
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
        super().__init__(inf=1)
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
        super().__init__(inf=1)
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
        super().__init__(inf=1 / (delta ** 2))
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
        super().__init__(inf=1)
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
        super().__init__(1 / (delta ** 2))
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
        super().__init__(inf=1 / (delta ** 2))
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
        super().__init__(inf=np.inf)
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


def mmmg(
    crit_list: List[Criterion],
    init: np.ndarray,
    tol: float = 1e-4,
    max_iter: int = 500,
):
    """The Majorize-Minimize Memory Gradient (3mg) algorithm

    The 3mg (see [2] and [1]) is a subspace memory-gradient optimization
    algorithm with an explicit step formula based on Majorize-Minimize Quadratic
    approach. This ensures quick convergence of the algorithm to a minimizer of
    the criterion without line search for the step and without tuning
    parameters. On the contrary, the criterion must meet conditions. In
    particular, the criterion must be like

    .. math::
       J(x) = \sum_k \phi(V x - \omega)

    where :math:`x` is the unkown of size :math:`N`, :math:`V` a matrix of size
    :math:`M \times N` and :math:`\omega` of size :math:`M`. In addition, among
    other conditions, :math:`\phi` must be differentiable (see documentation,
    [1], and [2] for details).

    Parameters
    ----------
    crit_list : list of Criterion
        A list of `Criterion` object that represent :math:`\phi(V x - \omega)`.
        The use of this list is necessary to allow efficient implementation and
        reuse of calculations. See notes section for details.

    init : ndarray
        The initial point

    tol : float, optional
        The stopping tolerance. The algorithm is stopped with the gradient norm
        is inferior to `init.size * tol`.

    max_iter : int, optional
        The maximum number of iteration

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

    .. [2] E. Chouzenoux, J. Idier, and S. Moussaoui, “A Majorize–Minimize
       Strategy for Subspace Optimization Applied to Image Restoration,” IEEE
       Trans. on Image Process., vol. 20, no. 6, pp. 1517–1528, Jun. 2011, doi:
       10.1109/TIP.2010.2103083.

    """
    shape = init.shape

    point = init.reshape((-1, 1))
    norm_grad = []

    # Vectorized call
    def vect(func, point):
        return np.reshape(func(np.reshape(point, shape)), (-1, 1))

    # Vectorized gradient
    def gradient(point):
        return sum(vect(crit.gradient, point) for crit in crit_list)

    # The first previous moves are initialized with 0 array. Consquently, the
    # first iterations implementation can be improved, at the cost of if
    # statement.
    move = np.zeros_like(point)
    op_directions = [np.tile(vect(crit.operator, move), 2) for crit in crit_list]
    step = np.ones((2, 1))
    step_list = [step]

    for _ in range(max_iter):
        # Vectorized gradient
        grad = gradient(point)
        norm_grad.append(la.norm(grad))

        # Stopping test
        if norm_grad[-1] < point.size * tol:
            break

        # Memory gradient directions
        directions = np.c_[-grad, move]

        # Step by Majorize-Minimize
        op_directions = [
            np.c_[vect(crit.operator, grad), i_op_dir @ step]
            for crit, i_op_dir in zip(crit_list, op_directions)
        ]
        norm_mat_major = sum(
            crit.norm_mat_major(i_op_dir, point.reshape(shape))
            for crit, i_op_dir in zip(crit_list, op_directions)
        )
        step = -la.pinv(norm_mat_major) @ (directions.T @ grad)
        move = directions @ step

        # update
        point += move

        step_list.append(step)

    return np.reshape(point, shape), norm_grad


# Not used finally
def vectorize(func: Callable, in_shape: Tuple):
    """Vectorize a function

    Wrap a function to accept a vectorized version of the input and to produce
    vectorized version of the ouput


    Parameters
    ----------
    func : Callable
        The function to wrap. Must be a single ndarray parameter callable that
        produce and a single ndarray output

    in_shape : tuple
        The shape that func ask as input

    Returns
    -------
    The wrapped callable.

    """

    @functools.wraps(func)
    def wrapper(arr):
        out = func(np.reshape(arr, in_shape))
        return out.reshape((-1, 1))

    return wrapper
