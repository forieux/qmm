"""This module contains the 3mg algorithm"""

import functools
from typing import Callable, Tuple, List

import numpy as np
import numpy.linalg as la


def mmcg():
    pass


def mmmg(
    crit_list: List,
    init: np.ndarray,
    tol: float = 1e-4,
    max_iter: int = 500,
):
    """The 3mg optimisation algorithm

    Parameters
    ----------
    crit_list : list of Criterion
        A list of
        A callable that take the current point as a unique argument and return
        the gradient at that point.

    major_curve : callable
        A callable that take the current point as a unique argument and return
        the major_curve of the quadratic majoring function at that point.

    init : ndarray
        The initial point

    tol : float, optional
        The stopping tolerance

    max_it : int, optional
        The maximum number of iteration

    Returns
    -------
    minimiser : ndarray
        The minimiser of the criterion with same shape than `init`.

    norm_grad : list of float
        The successive gradient norm

    Notes
    -----
    The output of callable, and the `init` value, are automatically vectorized
    internally. However, the output is reshaped as the `init` array.

    References
    ----------
    .. [1] C. Labat and J. Idier, “Convergence of Conjugate Gradient Methods
           with a Closed-Form Stepsize Formula,” J Optim Theory Appl, p.
           18, 2008.
    .. [2] E. Chouzenoux, J. Idier, and S. Moussaoui, “A Majorize–Minimize
           Strategy for Subspace Optimization Applied to Image Restoration,”
           IEEE Trans. on Image Process., vol. 20, no. 6, pp. 1517–1528, Jun.
           2011, doi: 10.1109/TIP.2010.2103083.
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
        The shape of the input parameter

    Returns
    -------
    The wrapped callable.

    """

    @functools.wraps(func)
    def wrapper(arr):
        out = func(np.reshape(arr, in_shape))
        return out.reshape((-1, 1))

    return wrapper


# # Decorator version
# def vectorize(in_shape):
#     def decorator(func):
#         @functools.wraps(func)
#         def wrapper(arr, in_shape):
#             out = func(np.reshape(arr, in_shape)
#             return out.reshape(-1, 1)

#         return wrapper
#     return decorator
