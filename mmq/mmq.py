"""This module implements Majorize-Minimize Quadratic optimization algorithms

The main point of interest in the module is the `mmmg` function. `Criterion` is
use to easily build criterion.

Comments on `Criterion`
-----------------------

The module provides a `Criterion` object for convenience. This object has be
made to help the implementation. However, thanks to dynamic nature of python,
the `mmmg` function needs in practive any object with three specific methods.

- `operator` : a callable with a point `x` as unique parameter, that must return
  the application of `V` (that is `V·x`).
- `gradient` : a callable with a point `x` as unique parameter, that must return
  the gradient of the criterion (that is `V^t φ'(V·x - ω)`).
- `norm_mat_major` : a callable with two parameters. The first one is the result
  of the operator applied on the subspace vectors. The second is the point `x`,
  where the normal matrix of the quadratic major function must be returned.

"""

import functools
import abc
from typing import Callable, Tuple, List

import numpy as np
import numpy.linalg as la


def mmmg(
    crit_list: List["Criterion"],
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

       `J(x) = ∑ₖ φₖ(Vₖ·x - ωₖ)`

    where `x` is the unkown of size `N`, `V` a matrix of size `M × N` and `ω` of
    size `M`. In addition, among other conditions, `φ` must be differentiable
    (see documentation, [1], and [2] for details).

    Parameters
    ----------
    crit_list : list of Criterion
        A list of `Criterion` object that represent `φ(V·x - ω)`.
        The use of this list is necessary to allow efficient implementation and
        reuse of calculations. See notes section for details.

    init : ndarray
        The initial point

    tol : float, optional
        The stopping tolerance. The algorithm is stopped when the gradient norm
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


class Criterion:
    """A criterion μ ∑ φ(V·x - ω)"""

    def __init__(
        self,
        operator: Callable,
        adjoint: Callable,
        potential: Callable,
        potential_grad: Callable,
        hyper: float = 1,
        mean: np.ndarray = 0,
    ):
        """A criterion μ ∑ φ(V·x - ω)

        Parameters
        ----------
        operator: callable
          V·x
        adjoing: callable
          Vᵗ·e
        potential: callable
          φ
        potential_grad: callable
          φ'
        hyper: float
          μ
        mean: ndarray (optionnal)
          ω
        """
        self.operator = operator
        self.adjoint = adjoint
        self.hyper = hyper
        self.mean = mean
        self.potential = potential
        self.potential_grad = potential_grad

    def value(self, point: np.ndarray):
        """The value of the criterion at given point

        Return μ·φ(V·x - ω)
        """
        return self.hyper * np.sum(self.potential(self.operator(point) - self.mean))

    def gradient(self, point: np.ndarray):
        """The gradient of the criterion at given point

        Return μ·Vᵗ·φ'(V·x - ω)
        """
        return self.hyper * self.adjoint(
            self.potential_grad(self.operator(point) - self.mean)
        )

    def norm_mat_major(self, vecs: np.ndarray, point: np.ndarray):
        """Return the normal matrix of the major function

        Given vecs `W = V·S`, return `Wᵗ·diag(b)·W`

        where S are the vectors defining the subspace and `b` are GR
        coefficients at given point.

        Parameters
        ----------
        vecs : np.ndarray
            The `W` vectors

        point : np.ndarray
            The given point where to compute GR coefficients `b`

        """
        return vecs.T @ (self.gr_coeffs(point).reshape((-1, 1)) * vecs)

    def gr_coeffs(self, point: np.ndarray):
        """Return the GR coefficients at given point

        φ'(V·x - ω) / (V·x - ω)

        """
        obj = self.operator(point) - self.mean
        return self.potential.gr_coeffs(obj)

    def __call__(self, point: np.ndarray):
        return self.value(point)


class QuadCriterion(Criterion):
    """A quadratic criterion μ||V·x - ω||^2"""

    def __init__(
        self,
        operator: Callable,
        adjoint: Callable,
        normal: Callable,
        hyper: float = 1,
        mean: np.ndarray = 0,
    ):
        """A quadratic criterion μ||V·x - ω||^2

        Parameters
        ----------
        operator: Callable
          V·x
        adjoing: Callable
          Vᵗ·e
        normal: Callable
          VᵗV·x
        hyper: float (optionnal)
          μ
        mean: ndarray (optionnal)
          ω
        """
        square = Square()
        super().__init__(operator, adjoint, square, square.gradient, hyper, mean)
        self.normal = normal
        self.mean_t = self.adjoint(mean)

    def value(self, point):
        """The value of the criterion at given point

        Return `μ·||V·x - ω||^2`
        """
        return self.hyper * np.sum((self.operator(point) - self.mean) ** 2) / 2

    def gradient(self, point: np.ndarray):
        """The gradient of the criterion at given point

        Return `μ·Vᵗ(V·x - ω)`

        Notes
        -----
        Use `normal` `Q` callable internally for potential better efficiency,
        with computation of `μ·(Q.x - b)` where `Q = Vᵗ·V` and `b = Vᵗ·ω` is
        precomputed

        """
        return self.hyper * (self.normal(point) - self.mean_t)

    def norm_mat_major(self, vecs: np.ndarray, point: np.ndarray):
        """Return the normal matrix of the major function

        Given W = V·D, return Wᵗ·W.

        Parameters
        ----------
        vecs : np.ndarray
            The `W` vectors

        point : np.ndarray
            The given point where to compute GR coefficients
        """
        return vecs.T @ vecs

    def gr_coeffs(self, point: np.ndarray):
        """Return the GR coefficients at given point

        Always return the scalar 1.

        Notes
        -----
        Not use internally and present for consistency.

        """
        return 1

    def __call__(self, point):
        return self.value(point)


class Potential(abc.ABC):
    """An abstract base class for the potentials φ.

    Attributs
    ---------
    inf : float
        The value of lim_{u→0} φ'(u) / u
    """

    def __init__(self, inf):
        """An abstract base class for the potentials φ

        Parameters
        ----------
        inf : float
          The value of lim_{u→0} φ'(u) / u
        """
        self.inf = inf

    @abc.abstractmethod
    def value(self, point: np.ndarray):
        """The value at given point"""
        return NotImplemented

    @abc.abstractmethod
    def gradient(self, point: np.ndarray):
        """The gradient at given point"""
        return NotImplemented

    def gr_coeffs(self, point: np.ndarray):
        """The GR coefficients at given point"""
        aux = self.inf * np.ones_like(point)
        idx = point != 0
        aux[idx] = self.gradient(point[idx]) / point[idx]
        return aux

    def __call__(self, point: np.ndarray):
        """The value at given point"""
        return self.value(point)


class VminProj(Potential):
    """The projection criterion

    D(u) = ½ ||P_[m, +∞[(u) - m||^2
    """

    def __init__(self, vmin):
        """The projection criterion

        D(u) = ½ ||P_[m, +∞[(u) - m||^2
        """
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

    φ(u) = ½ u²

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

    φ(u) = √(1 + u²/delta²) - 1

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

    φ(u) = u², if |u| < δ, δ|u| - δ²/2, otherwise.

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

    φ(u) = u² / (2 δ² + u²)

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

    φ(u) = 1 - exp(- u²/(2 δ²)

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

    φ(u) = log(1 + u² / δ²)

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
