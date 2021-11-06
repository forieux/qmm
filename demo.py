#!/usr/bin/env python3

# pylint: disable=invalid-name

import time

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import numpy.linalg as la
import scipy.misc  # type: ignore
import scipy.optimize
from scipy.signal import convolve2d

from qmm import operators, qmm  # type: ignore

imag = scipy.misc.ascent()
shape = imag.shape

#%% Forward model
ir = np.ones((5, 5)) / 25
obs = operators.Conv2(ir, shape)

#%% Simulated data
data = convolve2d(imag, ir, mode="valid")
data += np.random.standard_normal(data.shape)
init = obs.adjoint(data)

#%% Regularization
diffr = operators.Diff(axis=0)
diffc = operators.Diff(axis=1)
loss = qmm.HebertLeahy(delta=10)

#%% Objectives definition
data_adeq = qmm.QuadObjective(obs.forward, obs.adjoint, obs.fwback, data=data)
priorr_adeq = qmm.Objective(diffr.forward, diffr.adjoint, loss, hyper=0.5)
priorc_adeq = qmm.Objective(diffc.forward, diffc.adjoint, loss, hyper=0.5)

#%% Optimization
crit = data_adeq + priorr_adeq + priorc_adeq
res = qmm.mmmg(crit, init, max_iter=300, tol=1e-7)

#%% Scipy
critv = qmm.vectorize(init.shape)(crit)
jacv = qmm.vectorize(init.shape)(crit.gradient)

# def critv(arr):
#     return np.reshape(crit(arr.reshape(init.shape)), (-1,))
#
# def jacv(arr):
#     return np.reshape(crit.gradient(arr.reshape(init.shape)), (-1,))


def callback(xk):
    scipy_res_norm.append(la.norm(jacv(xk)))
    scipy_time.append(time.time())


scipy_res_norm = [la.norm(jacv(init))]
scipy_time = [time.time()]
scipy_res = scipy.optimize.minimize(
    critv,
    x0=init,
    method="CG",
    jac=jacv,
    options={"maxiter": 300, "gtol": 5e-5},
    callback=callback,
)

scipy_time = np.asarray(scipy_time) - scipy_time[0]

#%% Plot
plt.figure(1)
plt.clf()
plt.subplot(2, 2, 1)
plt.imshow(imag, vmin=imag.min(), vmax=imag.max())
plt.axis([50, 250, 450, 250])
plt.axis("off")
plt.title("Original")
plt.subplot(2, 2, 2)
plt.imshow(data, vmin=imag.min(), vmax=imag.max())
plt.axis([50, 250, 450, 250])
plt.axis("off")
plt.title("Data")
plt.subplot(2, 2, 3)
plt.imshow(res.x, vmin=imag.min(), vmax=imag.max())
plt.axis([50, 250, 450, 250])
plt.axis("off")
plt.title("Restored (Hebert & Leahy)")
plt.subplot(2, 2, 4)
plt.semilogy(res.time, res.grad_norm, ".-", label="QMM")
plt.semilogy(scipy_time, scipy_res_norm, ".-", label="Scipy 'CG'")
plt.grid("on")
plt.xlabel("Time [s]")
plt.title(f"Gradient norm")
plt.legend()
plt.tight_layout()
