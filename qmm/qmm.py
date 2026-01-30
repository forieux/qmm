# Q-MM: A Python Quadratic Majorization Minimization toolbox
# Copyright (C) 2021-2024 François Orieux <francois.orieux@universite-paris-saclay.fr>

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

This module implements Quadratic Majorize-Minimize optimization algorithms.

"""

import abc
import collections.abc
import time
from functools import reduce, wraps
from operator import iadd
from typing import Callable, MutableSequence, Optional, Sequence, Union

import numpy as np  # type: ignore
import numpy.linalg as la  # type: ignore
from numpy import ndarray as array

__author__ = "François Orieux"
__copyright__ = "2021, François Orieux <francois.orieux@universite-paris-saclay.fr>"
__credits__ = ["François Orieux"]
__license__ = "GPL-3.0-or-later"
__version__ = "0.19.0"
__maintainer__ = "François Orieux"
__email__ = "francois.orieux@universite-paris-saclay.fr"
__status__ = "beta"
__url__ = "https://github.com/forieux/qmm/"


ArrOrSeq = Union[array, Sequence[array]]

__all__ = [
    "OptimizeResult",
    "mmmg",
    "mmcg",
    "lcg",
    "pcg",
    "BaseObjective",
    "MixedObjective",
    "Objective",
    "QuadObjective",
    "Vmin",
    "Vmax",
    "Loss",
    "Square",
    "Huber",
    "Hyperbolic",
    "HebertLeahy",
    "GemanMcClure",
    "TruncSquareApprox",
    "vectorize",
]


class OptimizeResult(dict):  # pylint: disable=too-many-instance-attributes
    """Represents the optimization result.

    x: array
        The solution of the optimization, with same shape than `x0`.
    success: bool
        Whether or not the optimizer exited successfully.
    status: int
        Termination status of the optimizer. Its value depends on the underlying
        solver. Refer to message for details.
    message: str
        Description of the cause of the termination.
    nit: int
        Number of iterations performed by the optimizer.
    diff: list of float
        The value of ||x_{k+1} - x_{k}||² at each iteration
    time: list of float
        The time at each iteration, starting at 0, in seconds.
    fun: float
        The value of the objective function.
    objv_val: list of float
        The objective value at each iteration
    jac: array
        The gradient of the objective function.
    grad_norm: list of float
        The gradient norm at each iteration

    Notes
    -----
    :class:`OptimizeResult` mimes `OptimizeResult` of scipy for compatibility.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.jac = None
        self.success = False
        self.status = 99
        self.message = "NA"
        self.njev = 0
        self.nit = 0
        self.grad_norm = []
        self.diff = []
        self.time = []
        self.objv_val = []
        self.x = None  # pylint: disable=invalid-name

    @property
    def fun(self):
        """Last objective value"""
        if len(self.objv_val) != 0:
            return self.objv_val[-1]
        return None

    @fun.setter
    def fun(self, value):
        self.objv_val.append(value)

    def __getattr__(self, name):
        if name == "fun":
            return self["objv_val"][-1]
        if name in self:
            return self[name]
        raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        if name == "fun":
            self["objv_val"].append(value)
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __repr__(self):
        if self.keys():
            rjust = max(map(len, list(self.keys()))) + 1
            return "\n".join(
                [k.rjust(rjust) + ": " + str(v) for k, v in sorted(self.items())]
            )
        return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


def mmmg(  # pylint: disable=too-many-locals
    objv_list: Sequence["BaseObjective"],
    x0: array,  # pylint: disable=invalid-name
    tol: float = 1e-4,
    max_iter: int = 500,
    min_iter: int = 0,
    precond: Optional[Callable[[array], array]] = None,
    callback: Optional[Callable[[OptimizeResult], None]] = None,
    calc_objv: bool = False,
) -> OptimizeResult:
    r"""The Majorize-Minimize Memory Gradient (`3MG`) algorithm.

    The `mmmg` (`3MG`) algorithm is a subspace memory-gradient optimization
    algorithm with an explicit step formula based on Majorize-Minimize Quadratic
    approach [2]_.

    Parameters
    ----------
    objv_list : list of `BaseObjective`
        A list of :class:`BaseObjective` objects that each represents
        a `μ ψ(Vx - ω)`. The objectives are summed.
    x0 : array
        The initial point.
    tol : float, optional
        The stopping tolerance. The algorithm is stopped when the gradient norm
        is inferior to `x0.size * tol`.
    max_iter : int, optional
        The maximum number of iterations.
    min_iter : int, optional
        The minimum number of iterations.
    precond : callable, optional
        A callable that must implement a preconditioner, that is `Px`. Must be a
        callable with a unique input parameter `x` and unique output like `x`.
    callback : callable, optional
        A function that receive the `OptimizeResult` at the end of each
        iteration.
    calc_objv: boolean, optional
        If True, objective function is computed at each iteration with low
        overhead. False by default. Not used by the algorithm.

    Returns
    -------
    result : OptimizeResult

    References
    ----------
    .. [2] E. Chouzenoux, J. Idier, and S. Moussaoui, “A Majorize-Minimize
       Strategy for Subspace Optimization Applied to Image Restoration,” IEEE
       Trans. on Image Process., vol. 20, no. 6, pp. 1517–1528, Jun. 2011, doi:
       10.1109/TIP.2010.2103083.

    """
    if precond is None:
        precond = lambda x: x
    res = OptimizeResult()
    previous_flag = []
    for objv in objv_list:
        previous_flag.append(objv.calc_objv)
        objv.calc_objv = calc_objv

    res.x = x0.copy().reshape((-1, 1))

    # The first previous moves are initialized with 0 array. Consequently, the
    # first iterations implementation can be improved, at the cost of if
    # statement.
    move = np.zeros_like(res.x)
    op_directions = [
        np.tile(vect_call(objv.operator, move, x0.shape), 2) for objv in objv_list
    ]
    step = np.ones((2, 1))

    res.time.append(time.time())

    for iteration in range(max_iter):
        # Vectorized gradient
        grad = vectgradient(objv_list, res.x, x0.shape)
        res.grad_norm.append(la.norm(grad))
        res.jac = grad.reshape(x0.shape)
        res.fun = lastgv(objv_list)

        # Stopping test
        if res.grad_norm[-1] < x0.size * tol and iteration >= min_iter:
            res.success = True
            res.status = 0
            break

        # Memory gradient directions
        new_dir = vect_call(precond, grad, x0.shape)
        directions = np.c_[-new_dir, move]

        # Step by Majorize-Minimize
        op_directions = [
            np.c_[vect_call(objv.operator, new_dir, x0.shape), i_op_dir @ step]
            for objv, i_op_dir in zip(objv_list, op_directions)
        ]
        step = -la.pinv(
            sum(
                objv.norm_mat_major(i_op_dir, res.x.reshape(x0.shape))
                for objv, i_op_dir in zip(objv_list, op_directions)
            )
        ) @ (directions.T @ grad)
        move = directions @ step

        # update
        res.x += move

        res.diff.append(np.sum(move) ** 2)
        res.time.append(time.time())

        if callback is not None:
            callback(res)

    if res.status == 0:
        res.message = "Stopping conditions reached."
    else:
        res.success = False
        res.status = 1
        res.message = "Maximum number of iterations has been exceeded."
        del res.time[-1]
    res.x = np.reshape(res.x, x0.shape)
    res.njev = iteration + 1
    res.nit = iteration + 1
    res.time = list(np.asarray(res.time) - res.time[0])

    for objv, flag in zip(objv_list, previous_flag):
        objv.calc_objv = flag

    return res


def mmcg(  # pylint: disable=too-many-locals
    objv_list: Sequence["BaseObjective"],
    x0: array,  # pylint: disable=invalid-name
    tol: float = 1e-4,
    max_iter: int = 500,
    min_iter: int = 0,
    precond: Optional[Callable[[array], array]] = None,
    callback: Optional[Callable[[OptimizeResult], None]] = None,
    calc_objv: bool = False,
) -> OptimizeResult:
    """The Majorize-Minimize Conjugate Gradient (MM-CG) algorithm.

    The MM-CG is a nonlinear conjugate gradient (NL-CG) optimization algorithm
    with an explicit step formula based on Majorize-Minimize Quadratic approach
    [1]_.

    Parameters
    ----------
    objv_list : list of `BaseObjective`
        A list of :class:`BaseObjective` objects that each represents
        a `μ ψ(Vx - ω)`. The objectives are summed.
    x0 : ndarray
        The initial point.
    tol : float, optional
        The stopping tolerance. The algorithm is stopped when the gradient norm
        is inferior to `x0.size * tol`.
    max_iter : int, optional
        The maximum number of iterations.
    min_iter : int, optional
        The minimum number of iterations.
    precond : callable, optional
        A callable that must implement a preconditioner, that is `Px`. Must be a
        callable with a unique input parameter `x` and unique output like `x`.
    callback : callable, optional
        A function that receive the `OptimizeResult` at the end of each
        iteration.
    calc_objv: boolean, optional
        If True, objective function is computed at each iteration with low
        overhead. False by default. Not used by the algorithm.

    Returns
    -------
    result : OptimizeResult

    References
    ----------
    .. [1] C. Labat and J. Idier, “Convergence of Conjugate Gradient Methods
       with a Closed-Form Stepsize Formula,” J Optim Theory Appl, p. 18, 2008.
    """
    if precond is None:
        precond = lambda x: x
    res = OptimizeResult()
    previous_flag = []
    for objv in objv_list:
        previous_flag.append(objv.calc_objv)
        objv.calc_objv = calc_objv

    res.x = x0.copy().reshape((-1, 1))

    # pylint: disable=invalid-unary-operand-type
    residual = -vectgradient(objv_list, res.x, x0.shape)
    sec = vect_call(precond, residual, x0.shape)
    direction = sec
    delta = residual.T @ direction

    res.time.append(time.time())

    for iteration in range(max_iter):
        # Stop test
        res.grad_norm.append(la.norm(residual))
        if res.grad_norm[-1] < x0.size * tol and iteration >= min_iter:
            break

        # update
        op_direction = [
            vect_call(objv.operator, direction, x0.shape) for objv in objv_list
        ]

        step = direction.T @ residual
        step = step / sum(
            objv.norm_mat_major(i_op_dir, res.x.reshape(x0.shape))
            for objv, i_op_dir in zip(objv_list, op_direction)
        )

        res.x += step * direction

        res.diff.append(np.sum(step * direction) ** 2)
        res.time.append(time.time())

        # Gradient
        residual = -vectgradient(objv_list, res.x, x0.shape)
        res.jac = -residual.reshape(x0.shape)
        res.fun = lastgv(objv_list)

        # Conjugate direction. No reset is done, see Shewchuck.
        delta_old = delta
        delta_mid = residual.T @ sec
        sec = vect_call(precond, residual, x0.shape)
        delta = residual.T @ sec
        if (delta - delta_mid) / delta_old >= 0:
            direction = sec + (delta - delta_mid) / delta_old * direction
        else:
            direction = sec

        if callback is not None:
            callback(res)

    if res.status == 0:
        res.message = "Stopping conditions reached."
    else:
        res.success = False
        res.status = 1
        res.message = "Maximum number of iterations has been exceeded."
    res.x = np.reshape(res.x, x0.shape)
    res.njev = iteration + 1
    res.nit = iteration + 1
    res.time = list(np.asarray(res.time) - res.time[0])

    for objv, flag in zip(objv_list, previous_flag):
        objv.calc_objv = flag

    return res


def lcg(  # pylint: disable=too-many-locals
    objv_list: Sequence["QuadObjective"],
    x0: array,  # pylint: disable=invalid-name
    tol: float = 1e-4,
    max_iter: int = 500,
    min_iter: int = 0,
    precond: Optional[Callable[[array], array]] = None,
    callback: Optional[Callable[[OptimizeResult], None]] = None,
    calc_objv: bool = False,
    hdot_shape: tuple[int] = None,
) -> OptimizeResult:
    """Linear Conjugate Gradient (CG) algorithm.

    Linear Conjugate Gradient optimization algorithm for quadratic objective.

    Parameters
    ----------
    objv_list : list of `QuadObjective`
        A list of :class:`QuadObjective` objects that each represents
        a `½ μ ||Vx - ω||²_B`. The objectives are summed.
    x0 : ndarray
        The initial point.
    precond : callable, optional
        A callable that must implement a preconditioner, that is `Px`. Must be a
        callable with a unique input parameter `x` and unique output like `x`.
    tol : float, optional
        The stopping tolerance. The algorithm is stopped when the gradient norm
        is inferior to `x0.size * tol`.
    max_iter : int, optional
        The maximum number of iterations.
    min_iter : int, optional
        The minimum number of iterations.
    callback : callable, optional
        A function that receive the `OptimizeResult` at the end of each
        iteration.
    calc_objv: boolean, optional
        If True, objective function is computed at each iteration with low
        overhead. False by default. Not used by the algorithm.

    Returns
    -------
    result : OptimizeResult

    """

    if precond is None:
        precond = lambda x: x
    res = OptimizeResult()

    res.x = x0.copy().reshape((-1, 1))

    # A more efficient solution will be to use np.add with `out` argument.
    # Previous implementation used iadd but there was a bug where the second
    # member of the first criterion where updated to the sum of all the second
    # member, that was the same array.
    second_member = np.reshape(sum(qobjv.VtB_data for qobjv in objv_list), (-1, 1))

    constant = sum(c.constant for c in objv_list)

    def hessian(arr):
        return reduce(iadd, (vect_call(c.hessp, arr, x0.shape) for c in objv_list))

    def value_residual(arr, residual):
        arr = np.reshape(arr, x0.shape)
        residual = np.reshape(residual, x0.shape)
        if hdot_shape is None:
            return (
                np.sum(np.conj(arr) * (-second_member.reshape(x0.shape) - residual))
                + constant
            ) / 2
        return hdot(arr, -second_member.reshape(x0.shape) - residual, hdot_shape) / 2

    # Gradient at current x0
    residual = second_member - hessian(res.x)
    direction = vect_call(precond, residual, x0.shape)

    res.grad_norm.append(np.sum(np.real(np.conj(residual) * direction)))
    res.time.append(time.time())

    for iteration in range(max_iter):
        hessp = hessian(direction)
        if calc_objv:
            res.fun = value_residual(res.x, residual)

        # s = rᵀr / dᵀAd
        # Optimal step
        step = res.grad_norm[-1] / np.sum(np.real(np.conj(direction) * hessp))

        # Descent x^(i+1) = x^(i) + s*d
        res.x += step * direction

        # r^(i+1) = r^(i) - s * A·d
        if iteration % 50 == 0:
            residual = second_member - hessian(res.x)
        else:
            residual -= step * hessp
        res.jac = -residual.reshape(x0.shape)

        # Conjugate direction with preconditionner
        secant = vect_call(precond, residual, x0.shape)
        res.grad_norm.append(np.sum(np.real(np.conj(residual) * secant)))
        direction = secant + (res.grad_norm[-1] / res.grad_norm[-2]) * direction

        res.diff.append(np.sum(step * direction) ** 2)
        res.time.append(time.time())

        # Stopping condition
        if np.sqrt(res.grad_norm[-1]) < x0.size * tol and iteration >= min_iter:
            res.success = True
            res.status = 0
            break

        if callback is not None:
            callback(res)

    if res.status == 0:
        res.message = "Stopping conditions reached."
    else:
        res.success = False
        res.status = 1
        res.message = "Maximum number of iterations has been exceeded."
    res.x = np.reshape(res.x, x0.shape)
    res.njev = iteration + 1
    res.nit = iteration + 1
    res.grad_norm = list(np.sqrt(res.grad_norm))
    res.time = list(np.asarray(res.time) - res.time[0])

    return res


def pcg(  # pylint: disable=too-many-locals
    normalp: Callable[[array], array],
    secondm: array,
    x0: array,  # pylint: disable=invalid-name
    tol: float = 1e-4,
    max_iter: int = 500,
    min_iter: int = 0,
    precond: Optional[Callable[[array], array]] = None,
    callback: Optional[Callable[[OptimizeResult], None]] = None,
    hdot_shape: tuple[int] = None,
) -> OptimizeResult:
    """Preconditionnate Conjugate Gradient (CG) algorithm.

    Preconditionnate Conjugate Gradient for solving SDP linear system

    .. math::
        A x = b

    where :math:`A` is a SDP matrix and :math:`b` a vector. It comes from the
    gradient cancelling of quadratic criterion

    .. math::
        J(x) = \frac{1}{2} x^T A x - 2 x^T b + c.

    Parameters
    ----------
    normalp : callable
        A callable that apply the matrix `A` on a vector `x`. Must accept an
        array and return an array, both of any shape.
    secondm : array
        The second member `b`. Can be of any shape.
    x0 : ndarray
        The initial point.
    precond : callable, optional
        A callable that must implement a preconditioner, that is `Px`. Must be a
        callable with a unique input parameter `x` and unique output like `x`.
    tol : float, optional
        The stopping tolerance. The algorithm is stopped when the gradient norm
        is inferior to `x0.size * tol`.
    max_iter : int, optional
        The maximum number of iterations.
    min_iter : int, optional
        The minimum number of iterations.
    callback : callable, optional
        A function that receive the `OptimizeResult` at the end of each
        iteration.
    hdot_shape : shape, optional
        If x0 is in Fourier space with rfftn, set to shape of x0 in real space
        for correct computation of dot product.

    Returns
    -------
    result : OptimizeResult

    Notes
    -----
    This is the same algorithm than `lcg` with a different API.

    """

    if precond is None:
        precond = lambda x: x
    res = OptimizeResult()

    res.x = x0.copy().reshape((-1, 1))

    second_member = np.reshape(secondm, (-1, 1))

    def hessian(arr):
        return vect_call(normalp, arr, x0.shape)

    # Gradient at current x0
    residual = second_member - hessian(res.x)
    direction = vect_call(precond, residual, x0.shape)

    def value_residual(arr, residual):
        arr = np.reshape(arr, x0.shape)
        residual = np.reshape(residual, x0.shape)
        if hdot_shape is None:
            return (
                np.sum(np.conj(arr) * (-second_member.reshape(x0.shape) - residual)) / 2
            )
        return hdot(arr, -second_member.reshape(x0.shape) - residual, hdot_shape) / 2

    res.grad_norm.append(np.sum(np.real(np.conj(residual) * direction)))
    res.time.append(time.time())

    for iteration in range(max_iter):
        hessp = hessian(direction)
        res.fun = value_residual(res.x, residual)

        # s = rᵀr / dᵀAd
        # Optimal step
        step = res.grad_norm[-1] / np.sum(np.real(np.conj(direction) * hessp))

        # Descent x^(i+1) = x^(i) + s*d
        res.x += step * direction

        # r^(i+1) = r^(i) - s * A·d
        if iteration % 50 == 0:
            residual = second_member - hessian(res.x)
        else:
            residual -= step * hessp
        res.jac = -residual.reshape(x0.shape)

        # Conjugate direction with preconditionner
        secant = vect_call(precond, residual, x0.shape)
        res.grad_norm.append(np.sum(np.real(np.conj(residual) * secant)))
        direction = secant + (res.grad_norm[-1] / res.grad_norm[-2]) * direction

        res.diff.append(np.sum(step * direction) ** 2)
        res.time.append(time.time())

        # Stopping condition
        if np.sqrt(res.grad_norm[-1]) < x0.size * tol and iteration >= min_iter:
            res.success = True
            res.status = 0
            break

        if callback is not None:
            callback(res)

    if res.status == 0:
        res.message = "Stopping conditions reached."
    else:
        res.success = False
        res.status = 1
        res.message = "Maximum number of iterations has been exceeded."
    res.x = np.reshape(res.x, x0.shape)
    res.njev = iteration + 1
    res.nit = iteration + 1
    res.grad_norm = list(np.sqrt(res.grad_norm))
    res.time = list(np.asarray(res.time) - res.time[0])

    return res


#%% \
# Utilities


def hdot(arr1: np.ndarray, arr2: np.ndarray, inshape: tuple[int, ...]) -> float:
    """Hermitian dot product xᴴc with x and c ∈ ℂ

    If x and c are Fourier transform of vector, compute the complexe dot product
    with hermitian property.

    """
    axis = tuple(range(-len(inshape), 0))
    axis2 = tuple(range(-(len(inshape) - 1), 0))

    dotprod = 2 * np.sum(np.conj(arr1) * arr2, axis=axis) - np.sum(
        np.conj(arr1[..., 0]) * arr2[..., 0], axis=axis2
    )

    if inshape[-1] % 2 == 0:
        dotprod -= np.sum(np.conj(arr1[..., -1]) * arr2[..., -1], axis=axis2)

    return np.real(dotprod)


# Vectorized call
def vect_call(func: Callable[[array], array], point: array, shape: tuple) -> array:
    """Call func with point reshaped as shape and return vectorized output"""
    return np.reshape(func(np.reshape(point, shape)), (-1, 1))


# Not used in the module, only provided for user convenience
def vectorize(shape: tuple) -> Callable:
    """Return a decorator to vectorize input and output given `shape`.

    Suppose you have a function (a gradient typically) that demands an array
    with a specific shape, say (N, M) and return an array with another shape,
    say (P, Q). But you use an algorithm (optimize from scipy for instance) that
    demands vector only of shape (R, ) and (T,). This decorator factory return a
    decorator that do the job conversion

    """

    def decorator(func: Callable[[array], array]) -> Callable[[array], array]:
        """A decorator to vectorize input and output"""

        @wraps(func)
        def wrapper(arr: array) -> array:
            return np.reshape(func(np.reshape(arr, shape)), (-1,))

        return wrapper

    return decorator


# Vectorized gradient
def vectgradient(
    objv_list: Sequence["BaseObjective"], point: array, shape: tuple
) -> array:
    """Compute sum of gradient with vectorized parameters and return."""
    # The use of reduce and iadd do a more efficient numpy inplace sum
    return reduce(iadd, (vect_call(objv.gradient, point, shape) for objv in objv_list))


def lastgv(objv_list: Sequence["BaseObjective"]) -> float:
    """Return the value of objective computed after gradient evaluation"""
    return sum(objv.lastgv for objv in objv_list)


class Stacked:
    """A wrapper for list of array feature.

    Interface for operator callable like

         |V1|   y1
    Vx = |: | = :
         |Vq|   yq

    associated with a list of output y = [y1, ..., yq].

    If output shapes of V_i are shape_i, the `Stacked` operator return a
    vectorized array of shape (M, 1) = (sum(prod(shape_i) for all i), 1).

    Two methods are implemented.

    - `operator` takes an array `x` in shape (N, ..., P), apply the callable
      `V`, and return a the concatenation of [y1, ..., yq] in a array `y` of
      shape (M, 1).

    - `adjoint` takes an array `y` in shape (M, 1), split in a list `[y1, ...
      yq]` and apply the callable `Vᵀ`.

    Attributes
    ----------
    - shapes : list of tuple
      The list of output shape
    - data: array
      The vectorized data.

    """

    def __init__(
        self,
        operator: Callable[[array], Sequence[array]],
        adjoint: Callable[[Sequence[array]], array],
        shapes: list[tuple[int]],
    ):
        """A wrapper for list of array feature.

        Parameters
        ----------
        - operator: callable
          Apply V
        - adjoint: callable
          Apply Vᵀ
        - shapes: list of shape
          Shape of each output of Vx.
        """
        self.shapes = shapes
        self._operator = operator
        self._adjoint = adjoint
        self._idx = np.cumsum([0] + [np.prod(shape) for shape in shapes])

    @staticmethod
    def list2vec(arr_list: Sequence[array]) -> array:
        """Vectorize a list of array."""
        return np.vstack([arr.reshape((-1, 1)) for arr in arr_list])

    def vec2list(self, arr: array) -> list[array]:
        """De-vectorize to a list of array."""
        return [
            np.reshape(arr[self._idx[i] : self._idx[i + 1]], shape)
            for i, shape in enumerate(self.shapes)
        ]

    def operator(self, point: array) -> array:
        """Return `Vx` as array."""
        return self.list2vec(self._operator(point))

    def adjoint(self, point: array) -> array:
        """Return `Vᵀx` from array."""
        return self._adjoint(self.vec2list(point))


#%% \
# Objectives


class BaseObjective(abc.ABC):
    r"""An abstract base class for objective function

    .. math::
        J(x) = \mu \Psi \left(V x - \omega \right)

    with :math:`\Psi(u) = \sum_i \varphi(u_i)`.

    Attributes
    ----------
    calc_objv: boolean
        If true, compute the objective value when gradient is computed and store
        in `lastgv` attribute (False by default).
    name: str
        The name of the objective.
    hyper: float
        The hyperparameter value `μ`.
    lastv: float
        The last evaluated value of the objective (0 by default).
    lastgv: float
        The value of objective obtained during gradient computation (0 by default).
    """

    def __init__(self, hyper=1, name=""):
        self.lastgv = 0
        self.lastv = 0
        self.calc_objv = False
        self.hyper = hyper
        self.name = name

    @abc.abstractmethod
    def operator(self, point: array) -> array:
        """Compute the output of `Vx`."""
        return NotImplemented

    @abc.abstractmethod
    def value(self, point: array) -> float:
        """Compute the value at current point."""
        return NotImplemented

    @abc.abstractmethod
    def gradient(self, point: array) -> array:
        """Compute the gradient at current point."""
        return NotImplemented

    def val_grad(self, point: array) -> array:
        """Compute and return the value and gradient at current point."""
        grad = self.gradient(point)
        val = self.value(point)
        return val, grad

    @abc.abstractmethod
    def norm_mat_major(self, vecs: array, point: array) -> array:
        """Return the normal matrix of the quadratic major function.

        Given vectors `W = V·S`, return `Wᵀ·diag(b)·W`

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

    def __call__(self, point: array) -> float:
        return self.value(point)

    def __add__(self, objv: "BaseObjective"):
        return MixedObjective([self, objv])


class MixedObjective(collections.abc.MutableSequence):
    r"""Represents a mixed objective function

    .. math::
        J(x) = \sum_k \mu_k \Psi_k \left(V_k x - \omega_k \right)

    This is a `Sequence` (or list-like) and instance of this class can be used
    in optimization algorithms.

    """

    def __init__(self, objv_list: MutableSequence[BaseObjective]):
        r"""A mixed objective function

        `J(x) = ∑ₖ μₖ ψₖ(Vₖ·x - ωₖ)`.

        Parameters
        ----------
        objv_list: list of `BaseObjective`

        """
        self._objv_list = objv_list

    @property
    def lastv(self):
        """Return the value of objectives obtained during gradient computation."""
        return sum(o.lastv for o in self._objv_list)

    def __getitem__(self, key):
        return self._objv_list.__getitem__(key)

    def __setitem__(self, key, value):
        return self._objv_list.__setitem__(key, value)

    def __delitem__(self, key):
        return self._objv_list.__delitem__(key)

    def __len__(self):
        return len(self._objv_list)

    def insert(self, index, value):
        return self._objv_list.insert(index, value)

    def value(self, point: array) -> float:
        """The value J(x)"""
        return sum(o.value(point) for o in self._objv_list)

    def gradient(self, point: array) -> array:
        """The gradient ∇J(x)"""
        return reduce(iadd, (o.gradient(point) for o in self._objv_list))

    def val_grad(self, point: array) -> array:
        """Compute and return the value and gradient at current point."""
        grad = self.gradient(point)
        val = self.value(point)
        return val, grad

    def __call__(self, point: array) -> float:
        return self.value(point)

    def __add__(self, objv: Union["BaseObjective", "MixedObjective"]):
        if isinstance(objv, BaseObjective):
            self._objv_list.append(objv)
        elif isinstance(objv, MixedObjective):
            self._objv_list.extend(objv._objv_list)
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: 'MixedObjective' and {type(objv)}"
            )
        return self

    def __radd__(self, objv: "BaseObjective"):
        if isinstance(objv, BaseObjective):
            self._objv_list.append(objv)
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: 'MixedObjective' and {type(objv)}"
            )
        return self


class Objective(BaseObjective):
    r"""An objective function defined as

    .. math::
        J(x) = \mu \Psi \left(V x - \omega \right)

    with :math:`\Psi(u) = \sum_i \varphi(u_i)`.

    The instance attributs are:

    data : array
        The `data` array, or the vectorized list of array given at init.
    hyper : float
        The hyperparameter value `μ`.
    loss : Loss
        The loss `φ`.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        operator: Callable[[array], ArrOrSeq],
        adjoint: Callable[[ArrOrSeq], array],
        loss: "Loss",
        data: ArrOrSeq = None,
        hyper: float = 1,
        name: str = "",
    ):
        """A objective function `μ ψ(Vx - ω)`.

        Parameters
        ----------
        operator: callable
            A callable that compute the output `Vx`.
        adjoint: callable
            A callable that compute `Vᵀe`.
        loss: Loss
            The loss `φ`.
        data: array or list of array, optional
            The data vector `ω`.
        hyper: float, optional
            The hyperparameter `μ`.
        name: str, optional
            The name of the objective.

        Notes
        -----
        If `data` is a list of array, `operator` must return a similar list with
        arrays of same shape, and `adjoint` must accept a similar list also.

        In that case, however, and for algorithm purpose, everything is
        internally stacked as a column vector and values are therefore copied,
        by using a `Stacked` object. This is not efficient but flexible. Users
        are encouraged to do the vectorization themselves and use this "list of
        array" feature.

        """
        super().__init__(hyper=hyper, name=name)
        if isinstance(data, list):
            stacked = Stacked(operator, adjoint, [dat.shape for dat in data])
            self._op: Callable[[array], array] = stacked.operator
            self._adj: Callable[[array], array] = stacked.adjoint
            self.data = stacked.list2vec(data)
        else:
            self._op = operator
            self._adj = adjoint
            self.data = 0 if data is None else data

        self.loss = loss

    def operator(self, point: array) -> array:
        return self._op(point)

    def value(self, point: array) -> float:
        """The value of the objective function at given point

        Return `μ ψ(Vx - ω)`.
        """
        self.lastv = self.hyper * np.sum(self.loss(self.operator(point) - self.data))
        return self.lastv

    def gradient(self, point: array) -> array:
        """The gradient and value at given point

        Return `μ Vᵀφ'(Vx - ω)`.
        """
        residual = self.operator(point) - self.data
        if self.calc_objv:
            self.lastgv = self.hyper * np.sum(self.loss(residual))
        return self.hyper * self._adj(self.loss.gradient(residual))

    def norm_mat_major(self, vecs: array, point: array) -> array:
        matrix = np.real(
            np.conj(vecs.T) @ (self.gr_coeffs(point).reshape((-1, 1)) * vecs)
        )
        return float(matrix) if matrix.size == 1 else matrix

    def gr_coeffs(self, point: array) -> array:
        """The Geman & Reynolds coefficients at given point

        Given `x` return `φ'(Vx - ω) / (Vx - ω)`.
        """
        obj = self.operator(point) - self.data
        return self.loss.gr_coeffs(obj)

    def gy_coeffs(self, point: array) -> array:
        """The Geman & Yang coefficients at given point

        Given `x` return `Vx - φ'(Vx - ω)`.
        """
        aux = self.operator(point)
        return aux - self.loss.gradient(aux - self.data)

    def __repr__(self):
        return f"{self.name} ({type(self).__name__}, loss: {type(self.loss).__name__}), hyper {self.hyper}, last eval {self.lastv}"


class QuadObjective(BaseObjective):
    r"""A quadratic objective function

    .. math::
        :nowrap:

        \begin{equation}
        \begin{aligned}
        J(x) & = \frac{1}{2} \mu \|V x - \omega\|_B^2 \\
             & = \frac{1}{2} \mu (V x - \omega)^tB(V x - \omega)
        \end{aligned}
        \end{equation}

    The instance attributes are:

    hyper : float
        The hyperparameter value `μ`.
    ht_data : array
        The retroprojected data `μ VᵀBω`.
    constant : float
        The constant value `μ ωᵀBω`.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        operator: Callable[[array], ArrOrSeq],
        adjoint: Callable[[ArrOrSeq], array],
        hessp: Callable[[array], array] = None,
        data: array = None,
        hyper: float = 1,
        invcovp: Callable[[array], array] = None,
        name: str = "",
    ):
        """A quadratic objective `½ μ ||Vx - ω||²_B`

        Parameters
        ----------
        operator: callable
            A callable that compute the output `Vx`.
        adjoint: callable
            A callable that compute `Vᵀe`.
        hessp: callable, optional
            A callable that compute `Qx` as `Qx = VᵀBVx`. Must take a parameter
            like `operator` and return a parameter like `adjoint` (`x`-like in
            both case).
        data: array or list of array, optional
            The data vector `ω`.
        hyper: float, optional
            The hyperparameter `μ`.
        invcovp: callable, optional
            A callable, that take a parameter like `adjoint` and return like
            `operator` (`ω`-like in both case), that apply the inverse
            covariance, or metric, `B=Σ⁻¹`. Equivalent to Identity if `None`.
        name: str, optional
            The name of the objective.

        Notes
        -----
        The `hessp` (`Q`) callable is used for gradient computation as `∇ = μ
        (Qx - b)` where `b = VᵀBω` instead of `∇ = μ VᵀB(Vx - ω)`. This is
        optional and in some case can be more efficient. Use it only in that case.

        The variable `b = VᵀBω` is computed at object initialisation.

        """
        super().__init__(hyper=hyper, name=name)

        self._unvec_op = operator
        if isinstance(data, list):
            self._op: Callable = Stacked(
                operator, adjoint, [dat.shape for dat in data]
            ).operator
        else:
            self._op = operator
        self._adj = adjoint
        self._data = data

        if invcovp is None:
            self.invcovp = lambda x: x  # Identity
        elif invcovp is not None:
            self.invcovp = invcovp

        if hessp is not None:
            self.hessp = lambda x: hyper * hessp(x)
        else:
            self.hessp = lambda x: hyper * adjoint(self.invcovp(operator(x)))

        if data is None:
            # b = μ VᵀBω
            self._vtb_data = 0
            # c = μ ωᵀBω
            self._cst = 0
        else:
            # Lazy computation of second term b and constant c
            self._vtb_data = None
            self._cst = None

    @property
    def VtB_data(self):  # pylint: disable=invalid-name
        """The second term b = μ ∑ᵢ VᵢᵀBᵢωᵢ"""
        if self._vtb_data is None:
            self._vtb_data = self.hyper * self._adj(self.invcovp(self._data))
        return self._vtb_data

    @property
    def constant(self):
        """The constant c = μ ∑_i ωᵢᵀBᵢωᵢ"""
        if self._cst is None:
            if isinstance(self._data, list):
                self._cst = self.hyper * sum(
                    np.real(np.sum(y * By))
                    for y, By in zip(self._data, self.invcovp(self._data))
                )
            else:
                self._cst = self.hyper * np.real(
                    np.sum(self._data * self.invcovp(self._data))
                )
        return self._cst

    def operator(self, point: array) -> array:
        return self._op(point)

    def value(self, point: array) -> float:
        """The value of the objective function at given point

        Return `½ μ ||Vx - ω||²_B`.
        """
        # Compute J(x) = ½ (μ xVᵀBVx - 2 xᵀb + c), with b = μVᵀBω and c = μωᵀBω
        Vx = self._unvec_op(point)  # pylint: disable=invalid-name
        # xVᵀBVx
        if isinstance(Vx, list):
            self.lastv = self.hyper * sum(
                np.sum(Vx_i * BVx_i) for Vx_i, BVx_i in zip(Vx, self.invcovp(Vx))
            )
        else:
            self.lastv = self.hyper * np.sum(Vx * self.invcovp(Vx))
        self.lastv = (
            self.lastv - 2 * np.sum(point * self.VtB_data) + self.constant
        ) / 2
        return self.lastv

    def gradient(self, point: array) -> array:
        """The gradient and value at given point

        Return `∇ = μ (Qx - b) = μ VᵀB(Vx - ω)`.

        Notes
        -----
        Objective value is always computed with low overhead thanks to the
        relation

        `J(x) = ½ (xᵀ∇ - xᵀb + μ ωᵀBω)`.
        """
        hessp = self.hessp(point)
        self.lastgv = self.value_hessp(point, hessp)
        return self.hessp(point) - self.VtB_data

    def value_hessp(self, point, hessp):
        """Return `J(x)` value at low cost given `x` and `q = Qx`

        thanks to the relation

        `J(x) =  ½ (xᵀq - 2 xᵀb + μ ωᵀBω)`."""
        return (
            np.sum(np.reshape(point, (-1)) * np.reshape(hessp, (-1)))
            - 2 * np.sum(np.reshape(point, (-1)) * np.reshape(self.VtB_data, (-1)))
            + self.constant
        ) / 2

    def value_residual(self, point, residual):
        """Return `J(x)` value at low cost given `x` and `r = b - Qx`

        thanks to the relation

        `J(x) =  ½ (xᵀ(-b - r) + μ ωᵀBω)`."""
        return (
            np.sum(np.conj(point) * (-self.VtB_data - residual)) + self.constant
        ) / 2

    def norm_mat_major(self, vecs: array, point: array) -> array:
        return np.real(np.conj(vecs.T) @ vecs)


class Vmin(BaseObjective):
    r"""A minimum value objective function

    .. math::

        J(x) = \frac{1}{2} \mu \|P_{]-\infty, m]}(x) - m\|_2^2.

    vmin : float
        The minimum value `m`.
    hyper : float
        The hyperparameter value `μ`.
    """

    def __init__(self, vmin: float, hyper: float, name: str = ""):
        """A minimum value objective function

        `J(x) = ½ μ ||P_[m, +∞[(x) - m||²`.

        Parameters
        ----------
        vmin : float
            The minimum value `m`.
        hyper : float
            The hyperparameter value `μ`.
        name: str
            The name of the objective.
        """
        super().__init__(hyper=hyper, name=name)
        self.vmin = vmin

    def operator(self, point):
        return point[point <= self.vmin]

    def value(self, point: array) -> float:
        """Return the value at current point."""
        return self.hyper * np.sum((point[point <= self.vmin] - self.vmin) ** 2) / 2

    def gradient(self, point: array) -> array:
        idx = point <= self.vmin
        if self.calc_objv:
            self.lastgv = self.hyper * np.sum((point[idx] - self.vmin) ** 2) / 2
        return self.hyper * np.where(idx, point - self.vmin, 0)

    def norm_mat_major(self, vecs: array, point: array) -> array:
        return np.real(np.conj(vecs.T) @ vecs)


class Vmax(BaseObjective):
    r"""A maximum value objective function

    .. math::

        J(x) = \frac{1}{2} \mu \|P_{[M, +\infty[}(x) - M\|_2^2.

    vmax : float
        The maximum value `M`.
    hyper : float
        The hyperparameter value `μ`.
    """

    def __init__(self, vmax: float, hyper: float, name: str = ""):
        """A maximum value objective function

        Return `J(x) = ½ μ ||P_[M, +∞[(x) - M||²`.

        Parameters
        ----------
        vmax : float
            The maximum value `M`.
        hyper : float
            The hyperparameter value `μ`.
        name: str
            The name of the objective.
        """
        super().__init__(hyper=hyper, name=name)
        self.vmax = vmax

    def operator(self, point):
        return point[point >= self.vmax]

    def value(self, point: array) -> float:
        """Return the value at current point."""
        return self.hyper * np.sum((point[point >= self.vmax] - self.vmax) ** 2) / 2

    def gradient(self, point: array) -> array:
        idx = point >= self.vmax
        if self.calc_objv:
            self.lastgv = self.hyper * np.sum((point[idx] - self.vmax) ** 2) / 2
        return self.hyper * np.where(idx, point - self.vmax, 0)

    def norm_mat_major(self, vecs: array, point: array) -> array:
        return np.real(np.conj(vecs.T) @ vecs)


#%% \
# Objectives


class Loss(abc.ABC):
    """An abstract base class for loss `φ`.

    The class has the following attributes.

    inf : float
      The value of `lim_{u→0} φ'(u) / u`.
    convex : boolean
      A flag indicating if the loss is convex (not used).
    coercive : boolean
      A flag indicating if the loss is coercive (not used).
    """

    def __init__(self, inf: float, convex: bool = False, coercive: bool = False):
        """The loss φ

        Parameters
        ----------
        inf : float
          The value of `lim_{u→0} φ'(u) / u`.
        convex : boolean
          A flag indicating if the loss is convex.
        coercive : boolean
          A flag indicating if the loss is coercive.
        """
        self.inf = inf
        self.convex = convex
        self.coercive = coercive

    @abc.abstractmethod
    def value(self, point: array) -> array:
        """The value `φ(·)` at given point."""
        return NotImplemented

    @abc.abstractmethod
    def gradient(self, point: array) -> array:
        """The gradient `φ'(·)` at given point."""
        return NotImplemented

    def gr_coeffs(self, point: array) -> array:
        """The Geman & Reynolds `φ'(·)/·` coefficients at given point."""
        aux = self.inf * np.ones_like(point)
        idx = point != 0
        aux[idx] = self.gradient(point[idx]) / point[idx]
        return aux

    def gy_coeffs(self, point: array) -> array:
        """The Geman & Yang `· - φ'(·)` coefficients at given point."""
        return point - self.gradient(point)

    def gr_sup(self, point: array) -> array:
        """The Geman & Reynolds `ψ(u)` sup.

        ψ(u) = sup_x φ(x) - ux^2
        """
        return NotImplemented

    def gy_sup(self, point: array) -> array:
        """The Geman & Yang `ξ(u)` sup.

        ξ(u) = sup_x φ(x) - (x - u)^2 / 2
        """
        return NotImplemented

    def __call__(self, point: array) -> array:
        """The value at given point."""
        return self.value(point)


class Square(Loss):
    r"""The Square loss

    .. math::

       \varphi(u) = \frac{1}{2} u^2.
    """

    def __init__(self):
        """The Square loss `φ(u) = ½ u²`."""
        super().__init__(inf=1, convex=True, coercive=True)

    def value(self, point: array) -> array:
        return point ** 2 / 2

    def gradient(self, point: array) -> array:
        return point

    def __repr__(self):
        return """φ(u) = ½ u²
"""


class Huber(Loss):
    r"""The convex coercive Huber loss

    .. math::

       \varphi(u) =
       \begin{cases}
          \frac{1}{2} u^2, & \text{if } u \leq \delta, \\
          \delta |u| - \frac{\delta^2}{2}, & \text{otherwise.}
       \end{cases}

    """

    def __init__(self, delta: float):
        """The Huber loss."""
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

    def gy_sup(self, point: array) -> array:
        return self.delta * np.abs(point)

    def gr_sup(self, point: array) -> array:
        return self.delta ** 2 / point - self.delta ** 2

    def __repr__(self):
        return f"""{type(self)}

       ⎛
       ⎜ ½ u²,         if |u| < δ
φ(u) = ⎜
       ⎜ δ|u| - δ²/2,  otherwise.
       ⎝

with δ = {self.delta}
"""


class Hyperbolic(Loss):
    r"""The convex coercive hyperbolic loss

    .. math::

       \varphi(u) = \delta^2 \left( \sqrt{1 + \frac{u^2}{\delta^2}} -1 \right)

    This is sometimes called Pseudo-Huber.
    """

    def __init__(self, delta: float):
        """The hyperbolic loss."""
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


class HebertLeahy(Loss):
    r"""The non-convex coercive Hebert & Leahy loss

    .. math::

       \varphi(u) = \log \left(1 + \frac{u^2}{\delta^2} \right)

    """

    def __init__(self, delta: float):
        """The Hebert & Leahy loss."""
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


class GemanMcClure(Loss):
    r"""The non-convex non-coervice Geman & Mc Clure loss

    .. math::

       \varphi(u) = \frac{u^2}{2\delta^2 + u^2}

    """

    def __init__(self, delta: float):
        r"""The Geman & Mc Clure loss."""
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


class TruncSquareApprox(Loss):
    r"""The non-convex non-coercive truncated square approximation

    .. math::

       \varphi(u) = 1 - \exp \left(- \frac{u^2}{2\delta^2} \right)

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


### Local Variables:
### ispell-local-dictionary: "english"
### End:
