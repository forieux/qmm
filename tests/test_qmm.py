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

"""Tests of the `qmm` module. NumPy and pytest are the only requirements."""

# pylint: disable=invalid-name,redefined-outer-name,missing-function-docstring
# pylint: disable=unused-argument,too-few-public-methods

import copy

import numpy as np
import pytest

import qmm
from qmm import operators

N = 50


def identity(point):
    return point


def diff(point):
    return np.diff(point)


def diff_t(point):
    return np.r_[-point[0], -np.diff(point), point[-1]]


def solve_pcg(objv_list, x0, **kwargs):
    """Call `pcg` like the other solvers."""
    kwargs.pop("calc_objv", None)
    return qmm.pcg(
        lambda x: sum(objv.hessp(x) for objv in objv_list),
        sum(objv.VtB_data for objv in objv_list),
        x0,
        **kwargs,
    )


MM_SOLVERS = [qmm.mmmg, qmm.mmcg]
SOLVERS = [qmm.mmmg, qmm.mmcg, qmm.lcg, solve_pcg]
LOSSES = [
    qmm.Square(),
    qmm.Huber(0.5),
    qmm.Hyperbolic(0.5),
    qmm.HebertLeahy(0.5),
    qmm.GemanMcClure(0.5),
    qmm.TruncSquareApprox(0.5),
]


def name_of(param):
    """Return the name of a solver or of a loss for the test identifier."""
    return getattr(param, "__name__", type(param).__name__)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def data(rng):
    return rng.standard_normal(N)


@pytest.fixture
def scaled(data):
    """A quadratic problem that needs several iterations and its solution."""
    coef = np.linspace(1, 5, N)

    def make():
        return [
            qmm.QuadObjective(
                lambda x: np.sqrt(coef) * x, lambda e: np.sqrt(coef) * e, data=data
            )
        ]

    return make, np.sqrt(coef) * data / coef


class Lists:
    """An operator with two outputs of different shapes and the normal equations."""

    def __init__(self, rng):
        self.mat = [rng.standard_normal((30, N)), rng.standard_normal((20, N))]
        self.data = [mat @ np.ones(N) + 1 for mat in self.mat]
        self.weights = [0.5, 4.0]

    def operator(self, point):
        return [mat @ point for mat in self.mat]

    def adjoint(self, point):
        return sum(mat.T @ out for mat, out in zip(self.mat, point))

    def invcovp(self, point):
        return [weight * out for weight, out in zip(self.weights, point)]

    def solution(self, weights=(1, 1)):
        lhs = sum(w * m.T @ m for w, m in zip(weights, self.mat))
        rhs = sum(w * m.T @ d for w, m, d in zip(weights, self.mat, self.data))
        return np.linalg.solve(lhs, rhs)


@pytest.fixture
def lists(rng):
    return Lists(rng)


class Custom(qmm.BaseObjective):
    """The minimal user objective `½ μ ||x||²`."""

    def operator(self, point):
        return point

    def value(self, point):
        return self.hyper * np.sum(point**2) / 2

    def gradient(self, point):
        return self.hyper * point

    def norm_mat_major(self, vecs, point):
        return self.hyper * vecs.T @ vecs


class Linear(qmm.BaseObjective):
    """The objective `cᵀx`, with a gradient that always returns the same array."""

    def __init__(self, coef):
        super().__init__()
        self.coef = coef

    def operator(self, point):
        return np.zeros(1)

    def value(self, point):
        return float(np.real(np.sum(np.conj(self.coef) * point)))

    def gradient(self, point):
        return self.coef

    def norm_mat_major(self, vecs, point):
        return np.zeros((vecs.shape[1], vecs.shape[1]))


# %% Losses


@pytest.mark.parametrize("loss", LOSSES, ids=name_of)
def test_loss_gradient_is_derivative_of_value(loss):
    point = np.linspace(-3, 3, 14)
    eps = 1e-6
    numeric = (loss.value(point + eps) - loss.value(point - eps)) / (2 * eps)
    assert np.allclose(loss.gradient(point), numeric, atol=1e-6)


@pytest.mark.parametrize("loss", LOSSES, ids=name_of)
def test_loss_gr_coeffs(loss):
    point = np.array([-3.0, -0.5, 0.0, 0.25, 2.0])
    expected = np.array([loss.gradient(u) / u if u else loss.inf for u in point])
    assert np.allclose(loss.gr_coeffs(point), expected)
    assert loss.gradient(1e-7) / 1e-7 == pytest.approx(loss.inf, rel=1e-5)


def test_loss_gr_coeffs_with_integers():
    assert np.allclose(
        qmm.Huber(1.0).gr_coeffs(np.array([0, 1, 2, 4])), [1, 1, 0.5, 0.25]
    )


@pytest.mark.parametrize("delta", [0.5, 1.0, 3.0])
@pytest.mark.parametrize("aux", [0.1, 0.5, 1.0, 3.0])
def test_huber_gr_sup(delta, aux):
    grid = np.linspace(-100, 100, 400001)
    loss = qmm.Huber(delta)
    numeric = np.max(loss.value(grid) - aux * grid**2 / 2)
    assert loss.gr_sup(aux) == pytest.approx(numeric, abs=1e-6)


@pytest.mark.parametrize("delta", [0.5, 1.0, 3.0])
@pytest.mark.parametrize("aux", [-2.0, 0.1, 0.5, 3.0])
def test_huber_gy_sup(delta, aux):
    grid = np.linspace(-100, 100, 400001)
    loss = qmm.Huber(delta)
    numeric = np.max(loss.value(grid) - (grid - aux) ** 2 / 2)
    assert loss.gy_sup(aux) == pytest.approx(numeric, abs=1e-6)


def test_sup_not_implemented():
    with pytest.raises(NotImplementedError):
        qmm.Hyperbolic(1.0).gr_sup(0.5)
    with pytest.raises(NotImplementedError):
        qmm.Square().gy_sup(0.5)


# %% Objectives


def directional_error(objv, point, rng):
    """Return the error between the gradient and the finite differences."""
    direction = rng.standard_normal(point.shape)
    eps = 1e-6
    numeric = objv.value(point + eps * direction) - objv.value(point - eps * direction)
    numeric = numeric / (2 * eps)
    return abs(np.sum(objv.gradient(point) * direction) - numeric) / max(
        abs(numeric), 1
    )


@pytest.mark.parametrize(
    "make",
    [
        lambda y: qmm.QuadObjective(diff, diff_t, hyper=3.0),
        lambda y: qmm.QuadObjective(identity, identity, data=y, hyper=3.0),
        lambda y: qmm.QuadObjective(
            identity, identity, data=y, hyper=3.0, invcovp=lambda e: 2 * e
        ),
        lambda y: qmm.Objective(identity, identity, qmm.Huber(0.5), data=y, hyper=3.0),
        lambda y: qmm.Objective(diff, diff_t, qmm.GemanMcClure(0.5), hyper=3.0),
        lambda y: qmm.Vmin(0.0, hyper=3.0),
        lambda y: qmm.Vmax(0.0, hyper=3.0),
        lambda y: qmm.QuadObjective(identity, identity, data=y) + qmm.Vmin(0.0, 3.0),
    ],
    ids=["quad", "quad_data", "quad_invcovp", "huber", "gm", "vmin", "vmax", "mixed"],
)
def test_gradient_is_derivative_of_value(make, data, rng):
    objv = make(data)
    assert directional_error(objv, rng.standard_normal(N), rng) < 1e-6
    value, grad = objv.val_grad(data + 1)
    assert value == pytest.approx(objv.value(data + 1))
    assert np.array_equal(grad, objv.gradient(data + 1))
    assert objv(data + 1) == pytest.approx(value)


def test_quad_value(data, rng):
    point = rng.standard_normal(N)
    objv = qmm.QuadObjective(identity, identity, data=data, hyper=3.0)
    assert objv.value(point) == pytest.approx(1.5 * np.sum((point - data) ** 2))
    objv = qmm.QuadObjective(
        identity, identity, data=data, hyper=3.0, invcovp=lambda e: 2 * e
    )
    assert objv.value(point) == pytest.approx(3 * np.sum((point - data) ** 2))
    objv.calc_objv = True
    objv.gradient(point)
    assert objv.lastgv == pytest.approx(objv.value(point))


def test_quad_value_shortcuts(data, rng):
    objv = qmm.QuadObjective(diff_t, diff, data=data, hyper=3.0)
    point = rng.standard_normal(N - 1)
    hessp = objv.hessp(point)
    assert objv.value_hessp(point, hessp) == pytest.approx(objv.value(point))
    residual = objv.VtB_data - hessp
    assert objv.value_residual(point, residual) == pytest.approx(objv.value(point))


def test_objective_gr_and_gy_coeffs(data, rng):
    loss = qmm.Huber(0.5)
    objv = qmm.Objective(identity, identity, loss, data=data)
    point = rng.standard_normal(N)
    residual = point - data
    assert np.allclose(objv.gr_coeffs(point), loss.gr_coeffs(residual))
    assert np.allclose(objv.gy_coeffs(point), point - loss.gradient(residual))


def test_quad_value_with_list_of_different_shapes(lists, rng):
    point = rng.standard_normal(N)
    objv = qmm.QuadObjective(lists.operator, lists.adjoint, data=lists.data, hyper=3.0)
    expected = sum(
        1.5 * np.sum((mat @ point - dat) ** 2)
        for mat, dat in zip(lists.mat, lists.data)
    )
    assert objv.value(point) == pytest.approx(expected)
    assert directional_error(objv, point, rng) < 1e-6


def test_stacked_roundtrip():
    stacked = qmm.qmm.Stacked(identity, identity, [(2, 3), (4,)])
    arrays = [np.arange(6.0).reshape((2, 3)), np.arange(4.0)]
    vect = stacked.list2vec(arrays)
    assert vect.shape == (10, 1)
    for out, expected in zip(stacked.vec2list(vect), arrays):
        assert np.array_equal(out, expected)


@pytest.mark.parametrize(
    "objv",
    [
        qmm.QuadObjective(identity, identity, data=np.ones(N)),
        qmm.Objective(identity, identity, qmm.Huber(1.0)),
        qmm.Vmin(0.0, hyper=1.0),
        qmm.Vmax(0.0, hyper=1.0),
        Custom(hyper=3.0),
    ],
    ids=name_of,
)
def test_hyper_is_read_only(objv):
    before = objv.hyper
    with pytest.raises(AttributeError):
        objv.hyper = 2.0
    assert objv.hyper == before
    assert copy.deepcopy(objv).hyper == before


# %% Majorant


@pytest.mark.parametrize("hyper", [1.0, 7.0])
@pytest.mark.parametrize("metric", [False, True], ids=["no_invcovp", "invcovp"])
@pytest.mark.parametrize("kind", ["none", "array", "list", "list_same_shape"])
def test_quad_majorant_is_the_hessian(kind, metric, hyper, lists, rng):
    op, adj, invcovp = diff, diff_t, (lambda e: 3 * e) if metric else None
    out_data = {"none": None, "array": np.ones(N - 1)}
    if kind == "list":
        op, adj, out_data[kind] = lists.operator, lists.adjoint, lists.data
        invcovp = lists.invcovp if metric else None
    elif kind == "list_same_shape":
        op, adj = (lambda x: [x, 2 * x]), (lambda e: e[0] + 2 * e[1])
        out_data[kind] = [np.ones(N), np.ones(N)]
        invcovp = (lambda e: [0.5 * e[0], 4 * e[1]]) if metric else None
    objv = qmm.QuadObjective(op, adj, data=out_data[kind], hyper=hyper, invcovp=invcovp)

    subspace = rng.standard_normal((N, 2))
    vecs = np.c_[[np.reshape(objv.operator(col), -1) for col in subspace.T]].T
    expected = subspace.T @ np.c_[[objv.hessp(col) for col in subspace.T]].T
    assert np.allclose(objv.norm_mat_major(vecs, rng.standard_normal(N)), expected)


@pytest.mark.parametrize("loss", LOSSES, ids=name_of)
def test_objective_majorant_is_above_value(loss, rng):
    assert_is_majorant(qmm.Objective(diff, diff_t, loss, data=1.0, hyper=3.0), rng)


@pytest.mark.parametrize(
    "objv", [qmm.Vmin(0.2, hyper=3.0), qmm.Vmax(-0.2, hyper=3.0)], ids=name_of
)
def test_bound_majorant_is_above_value(objv, rng):
    assert_is_majorant(objv, rng)


def assert_is_majorant(objv, rng):
    """Check `J(x + Wa) ≤ J(x) + aᵀWᵀ∇J(x) + ½ aᵀMa`, tight when `a = 0`."""
    point = rng.standard_normal(N)
    subspace = rng.standard_normal((N, 2))
    vecs = np.c_[[np.reshape(objv.operator(col), -1) for col in subspace.T]].T
    matrix = objv.norm_mat_major(vecs, point)
    value, grad = objv.value(point), subspace.T @ objv.gradient(point)
    for scale in (1e-3, 0.1, 1.0, 10.0):
        for coef in scale * rng.standard_normal((100, 2)):
            major = value + coef @ grad + coef @ matrix @ coef / 2
            assert objv.value(point + subspace @ coef) <= major + 1e-9


# %% Solvers


@pytest.mark.parametrize("solver", MM_SOLVERS, ids=name_of)
def test_mm_step_is_exact_with_hyper_and_invcovp(solver, data):
    """With a Hessian `21 I` the first step of the exact majorant reaches `x*`."""
    objv = qmm.QuadObjective(
        identity, identity, data=data, hyper=7.0, invcovp=lambda e: 3 * e
    )
    res = solver([objv], np.zeros(N), tol=1e-9, calc_objv=True)
    assert res.success
    assert res.nit <= 2
    assert np.allclose(res.x, data)


@pytest.mark.parametrize("solver", SOLVERS, ids=name_of)
def test_solvers_reach_the_solution(solver, data):
    objv_list = [
        qmm.QuadObjective(identity, identity, data=data),
        qmm.QuadObjective(diff, diff_t, hyper=5.0, invcovp=lambda e: 3 * e),
    ]
    lhs = np.eye(N) + 15 * (
        np.diag(np.r_[1, 2 * np.ones(N - 2), 1])
        - np.diag(np.ones(N - 1), 1)
        - np.diag(np.ones(N - 1), -1)
    )
    res = solver(objv_list, np.zeros(N), tol=1e-9, max_iter=3000)
    assert res.success
    assert res.status == 0
    assert np.allclose(res.x, np.linalg.solve(lhs, data), atol=1e-6)


@pytest.mark.parametrize("solver", SOLVERS, ids=name_of)
def test_solvers_with_preconditioner(solver, scaled):
    make, solution = scaled
    coef = np.linspace(1, 5, N)
    res = solver(make(), np.zeros(N), tol=1e-9, precond=lambda g: g / coef)
    assert res.success
    assert np.allclose(res.x, solution, atol=1e-6)


@pytest.mark.parametrize("solver", MM_SOLVERS, ids=name_of)
@pytest.mark.parametrize("loss", LOSSES, ids=name_of)
def test_mm_decreases_objective(solver, loss, data):
    objv_list = [
        qmm.QuadObjective(identity, identity, data=data, invcovp=lambda e: 2 * e),
        qmm.Objective(diff, diff_t, loss, hyper=3.0),
    ]
    res = solver(objv_list, np.zeros(N), tol=1e-8, max_iter=300, calc_objv=True)
    assert np.all(np.diff(res.objv_val) < 1e-9)
    assert res.fun == pytest.approx(sum(objv.value(res.x) for objv in objv_list))


def test_mmcg_reports_convergence(data):
    objv_list = [
        qmm.QuadObjective(identity, identity, data=data),
        qmm.Objective(diff, diff_t, qmm.Huber(1.0), hyper=0.5),
    ]
    res = qmm.mmcg(objv_list, np.zeros(N), tol=1e-6, max_iter=500)
    assert res.success
    assert res.status == 0
    assert res.nit < 500
    assert res.message == "Stopping conditions reached."


def test_mmmg_matches_lcg_on_quadratic(data):
    """With a quadratic objective `3MG` and the linear CG have the same iterates."""

    def make():
        return [
            qmm.QuadObjective(identity, identity, data=data),
            qmm.QuadObjective(diff, diff_t, hyper=20.0),
        ]

    mmmg = qmm.mmmg(make(), np.zeros(N), tol=0, max_iter=15, calc_objv=True)
    lcg = qmm.lcg(make(), np.zeros(N), tol=0, max_iter=15, calc_objv=True)
    assert np.allclose(mmmg.objv_val, lcg.objv_val, rtol=1e-6)


@pytest.mark.parametrize("solver", MM_SOLVERS, ids=name_of)
@pytest.mark.parametrize(
    "bound, solution",
    [
        (qmm.Vmin(0.0, hyper=9.0), lambda y: np.where(y < 0, y / 10, y)),
        (qmm.Vmax(0.0, hyper=9.0), lambda y: np.where(y > 0, y / 10, y)),
    ],
    ids=["vmin", "vmax"],
)
def test_bounds_solution(solver, bound, solution, data):
    fidelity = qmm.QuadObjective(identity, identity, data=data)
    objv = fidelity + bound
    res = solver(objv, np.zeros(N), tol=1e-9, max_iter=500, calc_objv=True)
    assert res.success
    assert np.allclose(res.x, solution(data), atol=1e-6)
    assert res.fun == pytest.approx(objv.value(res.x))


def test_lcg_only_with_quad_objective(data):
    with pytest.raises(TypeError, match="QuadObjective"):
        qmm.lcg([qmm.Objective(identity, identity, qmm.Square(), data=data)], data)


# %% Edge cases of solvers


@pytest.mark.parametrize("solver", SOLVERS, ids=name_of)
def test_max_iter_zero(solver, scaled):
    make, _ = scaled
    x0 = np.ones(N)
    res = solver(make(), x0, max_iter=0, tol=0)
    assert res.nit == 0
    assert not res.success
    assert res.status == 1
    assert np.array_equal(res.x, x0)
    assert len(res.grad_norm) == 1


@pytest.mark.parametrize("solver", SOLVERS, ids=name_of)
def test_integer_x0(solver, scaled):
    make, solution = scaled
    x0 = np.zeros(N, dtype=int)
    res = solver(make(), x0, tol=1e-8, max_iter=200)
    assert res.success
    assert np.issubdtype(res.x.dtype, np.floating)
    assert np.allclose(res.x, solution, atol=1e-5)
    assert x0.sum() == 0


@pytest.mark.parametrize("solver", SOLVERS, ids=name_of)
@pytest.mark.parametrize("min_iter", [0, 2])
def test_start_on_the_solution(solver, min_iter, data):
    objv = qmm.QuadObjective(identity, identity, data=data)
    with np.errstate(all="raise"):
        res = solver([objv], data.copy(), min_iter=min_iter, max_iter=5)
    assert res.success
    assert res.nit == 0
    assert np.array_equal(res.x, data)


@pytest.mark.parametrize("solver", SOLVERS, ids=name_of)
def test_min_iter(solver, scaled):
    make, _ = scaled
    assert solver(make(), np.zeros(N), tol=1e3, min_iter=4, max_iter=20).nit == 4


@pytest.mark.parametrize("solver", SOLVERS, ids=name_of)
@pytest.mark.parametrize("dtype", ["float32", "float64", "complex128"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_x0_shape_and_type(solver, dtype, order, rng):
    target = rng.standard_normal((6, 4))
    x0 = np.zeros_like(target, dtype=dtype, order=order)
    saved = x0.copy()
    objv = qmm.QuadObjective(identity, identity, data=target)
    res = solver([objv], x0, tol=1e-6)
    assert res.x.shape == x0.shape
    assert res.x.dtype == x0.dtype
    assert np.allclose(res.x, target, atol=1e-5)
    assert np.array_equal(x0, saved)


# %% OptimizeResult


@pytest.mark.parametrize("solver", SOLVERS, ids=name_of)
@pytest.mark.parametrize(
    "kwargs",
    [{"max_iter": 3, "tol": 0}, {"max_iter": 500, "tol": 1e-6}],
    ids=["max_iter", "converged"],
)
def test_result_lengths(solver, kwargs, scaled):
    make, _ = scaled
    calls = []
    res = solver(
        make(),
        np.zeros(N),
        callback=lambda r: calls.append(1),
        calc_objv=True,
        **kwargs,
    )
    assert res.nit == res.njev == len(res.diff) == len(calls)
    assert len(res.grad_norm) == len(res.time) == res.nit + 1
    assert len(res.objv_val) == res.nit + 1
    assert res.time[0] == 0
    assert res.success == (kwargs["tol"] > 0)


@pytest.mark.parametrize("solver", SOLVERS, ids=name_of)
def test_result_values_at_final_point(solver, scaled):
    make, _ = scaled
    objv = make()[0]
    points = [np.zeros(N)]
    res = solver(
        make(),
        np.zeros(N),
        max_iter=4,
        tol=0,
        calc_objv=True,
        callback=lambda r: points.append(np.array(r.x)),
    )
    expected = [
        np.sum((after - before) ** 2) for before, after in zip(points, points[1:])
    ]
    assert np.allclose(res.diff, expected)
    assert np.allclose(res.jac, objv.gradient(res.x))
    assert res.grad_norm[-1] == pytest.approx(np.linalg.norm(res.jac))
    constant = objv.constant if solver is solve_pcg else 0
    values = [objv.value(point) - constant / 2 for point in points]
    assert np.allclose(res.objv_val, values)
    assert res.fun == pytest.approx(values[-1])


@pytest.mark.parametrize("solver", SOLVERS, ids=name_of)
def test_callback_sees_an_up_to_date_result(solver, rng):
    target = rng.standard_normal((6, 4))
    objv = qmm.QuadObjective(lambda x: 2 * x, lambda e: 2 * e, data=target)
    seen = []

    def callback(res):
        seen.append(
            (
                res.x.shape,
                res.jac.shape,
                np.allclose(res.jac, objv.gradient(res.x)),
                len(res.grad_norm) == len(res.diff) + 1,
            )
        )

    solver([objv], np.zeros_like(target), max_iter=2, tol=0, callback=callback)
    assert seen
    assert all(item == (target.shape, target.shape, True, True) for item in seen)


@pytest.mark.parametrize("solver", [qmm.mmmg, qmm.mmcg, qmm.lcg], ids=name_of)
def test_no_objective_value_without_calc_objv(solver, scaled):
    make, _ = scaled
    res = solver(make(), np.zeros(N), max_iter=2, tol=0)
    assert res.objv_val == []
    assert res.fun is None


def test_pcg_always_has_objective_value(scaled):
    make, _ = scaled
    res = solve_pcg(make(), np.zeros(N), max_iter=2, tol=0)
    assert len(res.objv_val) == 3
    assert res.fun is not None


def test_convergence_at_last_iteration(scaled):
    make, _ = scaled
    nit = qmm.mmmg(make(), np.zeros(N), tol=1e-6).nit
    assert qmm.mmmg(make(), np.zeros(N), tol=1e-6, max_iter=nit).success
    assert not qmm.mmmg(make(), np.zeros(N), tol=1e-6, max_iter=nit - 1).success


def test_optimize_result_fun_appends():
    res = qmm.OptimizeResult()
    res.fun = 2.0
    res.fun = 1.0
    assert res.objv_val == [2.0, 1.0]
    assert res.fun == 1.0
    assert res["objv_val"] == [2.0, 1.0]
    with pytest.raises(AttributeError):
        _ = res.not_an_attribute


# %% calc_objv flag


@pytest.mark.parametrize("solver", MM_SOLVERS, ids=name_of)
@pytest.mark.parametrize("flag", [True, False])
def test_calc_objv_is_restored(solver, flag, data):
    objv = qmm.QuadObjective(identity, identity, data=data)
    objv.calc_objv = flag
    solver([objv], np.zeros(N), max_iter=2, calc_objv=not flag)
    assert objv.calc_objv is flag


@pytest.mark.parametrize("solver", MM_SOLVERS, ids=name_of)
def test_calc_objv_is_restored_on_exception(solver, data):
    class Broken(qmm.QuadObjective):
        """An objective that fails."""

        def gradient(self, point):
            raise RuntimeError("Broken")

    objv = Broken(identity, identity, data=data)
    with pytest.raises(RuntimeError, match="Broken"):
        solver([objv], np.zeros(N), calc_objv=True)
    assert objv.calc_objv is False


# %% Lists of data


@pytest.mark.parametrize("solver", [qmm.mmmg, qmm.mmcg, qmm.lcg], ids=name_of)
@pytest.mark.parametrize("metric", [False, True], ids=["no_invcovp", "invcovp"])
def test_list_data_of_different_shapes(solver, metric, lists):
    objv = qmm.QuadObjective(
        lists.operator,
        lists.adjoint,
        data=lists.data,
        invcovp=lists.invcovp if metric else None,
    )
    res = solver([objv], np.zeros(N), tol=1e-9, max_iter=3000)
    solution = lists.solution(lists.weights if metric else (1, 1))
    assert res.success
    assert np.allclose(res.x, solution, atol=1e-5)


@pytest.mark.parametrize("solver", MM_SOLVERS, ids=name_of)
def test_objective_with_list_data(solver, lists):
    objv = qmm.Objective(lists.operator, lists.adjoint, qmm.Square(), data=lists.data)
    res = solver([objv], np.zeros(N), tol=1e-9, max_iter=3000)
    assert np.allclose(res.x, lists.solution(), atol=1e-5)


@pytest.mark.parametrize("solver", MM_SOLVERS, ids=name_of)
@pytest.mark.parametrize(
    "make",
    [
        lambda op, adj: qmm.QuadObjective(op, adj),
        lambda op, adj: qmm.Objective(op, adj, qmm.Square()),
    ],
    ids=["quad", "objective"],
)
def test_list_operator_without_list_data_is_an_error(solver, make, lists, data):
    objv_list = [
        qmm.QuadObjective(identity, identity, data=data),
        make(lists.operator, lists.adjoint),
    ]
    with pytest.raises(TypeError, match="requires `data` as a list"):
        solver(objv_list, np.zeros(N))


def test_lcg_accepts_list_operator_without_data(lists, data):
    objv_list = [
        qmm.QuadObjective(identity, identity, data=data),
        qmm.QuadObjective(lists.operator, lists.adjoint, hyper=0.2),
    ]
    lhs = np.eye(N) + 0.2 * sum(mat.T @ mat for mat in lists.mat)
    res = qmm.lcg(objv_list, np.zeros(N), tol=1e-9)
    assert np.allclose(res.x, np.linalg.solve(lhs, data), atol=1e-6)


# %% Sum of objectives


@pytest.fixture
def terms(data):
    return (
        qmm.QuadObjective(identity, identity, data=data),
        qmm.Objective(diff, diff_t, qmm.Huber(1.0), hyper=0.2),
        qmm.Objective(diff, diff_t, qmm.Square(), hyper=0.1),
    )


def test_add_is_flat(terms):
    a, b, c = terms
    for mixed in (a + b + c, a + (b + c), sum([a, b, c]), sum([a + b, c])):
        assert isinstance(mixed, qmm.MixedObjective)
        assert list(mixed) == [a, b, c]


def test_add_keeps_operands(terms):
    a, b, c = terms
    base = a + b
    total = base + c
    assert total is not base
    assert list(base) == [a, b]
    assert list(total) == [a, b, c]
    other = b + c
    assert list(base + other) == [a, b, b, c]
    assert list(other) == [b, c]


def test_iadd_is_in_place(terms):
    a, b, c = terms
    mixed = same = a + b
    mixed += c
    assert mixed is same
    assert list(mixed) == [a, b, c]
    mixed += b + c
    assert mixed is same
    assert list(mixed) == [a, b, c, b, c]


def test_sum_of_one_objective_returns_a_new_mixed(terms):
    a, b, _ = terms
    assert list(sum([a])) == [a]
    base = a + b
    out = sum([base])
    assert out is not base
    assert list(out) == [a, b]


@pytest.mark.parametrize("left", [3, 1.0, "a"])
def test_add_of_anything_else_is_an_error(left, terms):
    a, b, _ = terms
    with pytest.raises(TypeError):
        _ = left + a
    with pytest.raises(TypeError):
        _ = (a + b) + left


@pytest.mark.parametrize("solver", MM_SOLVERS, ids=name_of)
def test_solvers_accept_all_sums(solver, terms):
    a, b, c = terms
    x0 = np.zeros(N)
    reference = solver([a, b, c], x0, tol=1e-8).x
    for objv in (a + b + c, a + (b + c), sum([a, b, c])):
        assert np.array_equal(solver(objv, x0, tol=1e-8).x, reference)


def test_mixed_value_and_gradient(terms, rng):
    point = rng.standard_normal(N)
    a, b, c = terms
    mixed = a + (b + c)
    assert mixed.value(point) == pytest.approx(
        a.value(point) + b.value(point) + c.value(point)
    )
    assert np.allclose(
        mixed.gradient(point),
        a.gradient(point) + b.gradient(point) + c.gradient(point),
    )


# %% Gradients of objectives are not modified


@pytest.mark.parametrize("solver", MM_SOLVERS, ids=name_of)
def test_solver_does_not_modify_returned_gradient(solver, data):
    coef = np.ones(N)
    objv_list = [Linear(coef), qmm.QuadObjective(identity, identity, data=data)]
    res = solver(objv_list, np.zeros(N), tol=1e-9, max_iter=500)
    assert np.array_equal(coef, np.ones(N))
    assert np.allclose(res.x, data - 1, atol=1e-6)


def test_mixed_gradient_does_not_modify_returned_gradient(data, rng):
    coef = np.ones(N)
    point = rng.standard_normal(N)
    mixed = Linear(coef) + qmm.QuadObjective(identity, identity, data=data)
    assert np.allclose(mixed.gradient(point), 1 + point - data)
    assert np.array_equal(coef, np.ones(N))


def test_sum_of_real_and_complex_gradients(data, rng):
    target = data + 1j * rng.standard_normal(N)
    objv_list = [Linear(np.ones(N)), qmm.QuadObjective(identity, identity, data=target)]
    res = qmm.mmmg(objv_list, np.zeros(N, dtype=complex), tol=1e-9, max_iter=500)
    assert np.allclose(res.x, target - 1, atol=1e-6)


def test_gradient_calls_hessp_once(data):
    calls = []

    def hessp(point):
        calls.append(1)
        return point

    objv = qmm.QuadObjective(identity, identity, data=data, hessp=hessp)
    objv.gradient(data)
    assert len(calls) == 1


# %% Complex variables


@pytest.fixture
def complex_problem(rng):
    mat = rng.standard_normal((80, N)) + 1j * rng.standard_normal((80, N))
    target = rng.standard_normal(80) + 1j * rng.standard_normal(80)

    def make():
        return [
            qmm.QuadObjective(
                lambda x: mat @ x, lambda e: np.conj(mat.T) @ e, data=target
            )
        ]

    return make, np.linalg.lstsq(mat, target, rcond=None)[0]


@pytest.mark.parametrize("solver", SOLVERS, ids=name_of)
def test_complex_solution(solver, complex_problem):
    make, solution = complex_problem
    res = solver(make(), np.zeros(N, dtype=complex), tol=1e-9, max_iter=3000)
    assert res.success
    assert np.allclose(res.x, solution, atol=1e-6)


@pytest.mark.parametrize("solver", [qmm.mmmg, qmm.mmcg, qmm.lcg], ids=name_of)
def test_complex_objective_value(solver, complex_problem):
    make, _ = complex_problem
    res = solver(make(), np.zeros(N, dtype=complex), tol=1e-3, calc_objv=True)
    values = np.array(res.objv_val)
    assert np.isrealobj(values)
    assert np.all(np.diff(values) < 1e-9)
    assert res.fun == pytest.approx(make()[0].value(res.x))


def test_quad_complex_value(complex_problem, rng):
    make, _ = complex_problem
    objv = make()[0]
    point = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    objv.gradient(point)
    assert np.isreal(objv.value(point))
    assert objv.lastgv == pytest.approx(objv.value(point))


# %% Half-spectrum of rfft


@pytest.mark.parametrize("shape", [(8,), (9,), (6, 8), (6, 9)])
def test_hdot(shape, rng):
    left, right = rng.standard_normal((2,) + shape)
    out = qmm.qmm.hdot(
        np.fft.rfftn(left, norm="ortho"), np.fft.rfftn(right, norm="ortho"), shape
    )
    assert out == pytest.approx(np.sum(left * right))


@pytest.mark.parametrize("use_pcg", [False, True], ids=["lcg", "pcg"])
@pytest.mark.parametrize("size", [64, 65])
def test_hdot_shape_objective_value_is_without_constant(use_pcg, size, rng):
    """With `hdot_shape` the reported value is the true `J(x) - c`."""
    freq = 1 + np.abs(np.fft.rfft(rng.standard_normal(size), norm="ortho"))
    real_target = rng.standard_normal(size)
    target = np.fft.rfft(real_target, norm="ortho")

    def true_value(point):
        real = np.fft.irfft(freq * point, n=size, norm="ortho")
        return np.sum((real - real_target) ** 2) / 2

    points = [np.zeros_like(target)]
    kwargs = {
        "max_iter": 5,
        "tol": 0,
        "hdot_shape": (size,),
        "callback": lambda r: points.append(np.array(r.x)),
    }
    if use_pcg:
        res = qmm.pcg(
            lambda x: freq**2 * x, freq * target, np.zeros_like(target), **kwargs
        )
    else:
        objv = qmm.QuadObjective(lambda x: freq * x, lambda e: freq * e, data=target)
        res = qmm.lcg([objv], np.zeros_like(target), calc_objv=True, **kwargs)
    constant = np.sum(real_target**2) / 2
    assert res.objv_val[0] == pytest.approx(0)
    assert np.allclose(res.objv_val, [true_value(p) - constant for p in points])


def test_without_hdot_shape_value_includes_constant(data):
    objv = qmm.QuadObjective(identity, identity, data=data)
    res = qmm.lcg([objv], np.zeros(N), max_iter=0, calc_objv=True)
    assert res.objv_val[0] == pytest.approx(np.sum(data**2) / 2)


# %% Operators


def direct_valid_convolution(imp_resp, image):
    """The "valid" convolution with the definition."""
    rows, cols = imp_resp.shape
    out = np.zeros((image.shape[0] - rows + 1, image.shape[1] - cols + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            window = image[i : i + rows, j : j + cols]
            out[i, j] = np.sum(imp_resp * window[::-1, ::-1])
    return out


@pytest.mark.parametrize("ir_shape", [(3, 3), (3, 4), (4, 3), (2, 2), (5, 2)])
@pytest.mark.parametrize("shape", [(8, 9), (9, 8), (10, 10)])
def test_conv2_is_valid_convolution(ir_shape, shape, rng):
    imp_resp = rng.standard_normal(ir_shape)
    image = rng.standard_normal(shape)
    conv = operators.Conv2(imp_resp, shape)
    out = conv.forward(image)
    assert np.allclose(out, direct_valid_convolution(imp_resp, image))
    assert np.allclose(conv(image), out)


@pytest.mark.parametrize("ir_shape", [(3, 3), (3, 4), (2, 2)])
def test_conv2_adjoint(ir_shape, rng):
    shape = (8, 9)
    conv = operators.Conv2(rng.standard_normal(ir_shape), shape)
    image = rng.standard_normal(shape)
    other = rng.standard_normal(conv.forward(image).shape)
    assert np.sum(conv.forward(image) * other) == pytest.approx(
        np.sum(image * conv.adjoint(other))
    )
    assert np.allclose(conv.T(other), conv.adjoint(other))
    assert np.allclose(conv.fwback(image), conv.adjoint(conv.forward(image)))


@pytest.mark.parametrize("axis", [0, 1])
def test_diff_adjoint(axis, rng):
    oper = operators.Diff(axis)
    image = rng.standard_normal((8, 9))
    other = rng.standard_normal(oper.forward(image).shape)
    assert np.sum(oper.forward(image) * other) == pytest.approx(
        np.sum(image * oper.adjoint(other))
    )


def test_ir2fr_accepts_list_and_tuple(rng):
    imp_resp = rng.standard_normal((3, 3))
    assert np.array_equal(
        operators.ir2fr(imp_resp, [8, 9]), operators.ir2fr(imp_resp, (8, 9))
    )


@pytest.mark.parametrize("solver", MM_SOLVERS, ids=name_of)
def test_image_deconvolution(solver, rng):
    """A small problem like in `demo.py`."""
    shape = (24, 24)
    conv = operators.Conv2(np.ones((3, 3)) / 9, shape)
    rows, cols = operators.Diff(0), operators.Diff(1)
    truth = np.zeros(shape)
    truth[6:18, 6:18] = 1
    target = conv.forward(truth) + 0.01 * rng.standard_normal((22, 22))
    objv_list = [
        qmm.QuadObjective(conv.forward, conv.adjoint, conv.fwback, data=target),
        qmm.Objective(rows.forward, rows.adjoint, qmm.Huber(0.1), hyper=0.01),
        qmm.Objective(cols.forward, cols.adjoint, qmm.Huber(0.1), hyper=0.01),
    ]
    res = solver(objv_list, np.zeros(shape), max_iter=200, calc_objv=True)
    assert np.all(np.diff(res.objv_val) < 1e-9)
    assert np.mean((res.x - truth) ** 2) < 0.02


# %% Helpers


def test_vectorize():
    @qmm.vectorize((2, 3))
    def func(arr):
        assert arr.shape == (2, 3)
        return 2 * arr

    assert np.array_equal(func(np.ones(6)), 2 * np.ones(6))
