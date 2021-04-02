#!/usr/bin/env python3

# pylint: disable=invalid-name

import time

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import scipy.misc  # type: ignore
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
pot = qmm.Huber(delta=10)

#%% Criterions definition
data_adeq = qmm.QuadCriterion(obs.forward, obs.adjoint, obs.fwback, data=data)
priorr_adeq = qmm.Criterion(diffr.forward, diffr.adjoint, pot, hyper=0.01)
priorc_adeq = qmm.Criterion(diffc.forward, diffc.adjoint, pot, hyper=0.01)

#%% Optimization algorithm
res = qmm.mmmg([data_adeq, priorr_adeq, priorc_adeq], init, max_iter=300, tol=5e-5)

#%% Plot
plt.figure(1)
plt.clf()
plt.subplot(2, 2, 1)
plt.imshow(imag, vmin=imag.min(), vmax=imag.max())
plt.axis([50, 250, 450, 250])
plt.axis("off")
plt.title("Original")
plt.colorbar()
plt.subplot(2, 2, 2)
plt.imshow(data, vmin=imag.min(), vmax=imag.max())
plt.axis([50, 250, 450, 250])
plt.axis("off")
plt.title("Data")
plt.colorbar()
plt.subplot(2, 2, 3)
plt.imshow(res.x, vmin=imag.min(), vmax=imag.max())
plt.axis([50, 250, 450, 250])
plt.axis("off")
plt.colorbar()
plt.title("Restored")
plt.subplot(2, 2, 4)
plt.plot(res.grad_norm, ".-")
plt.xlabel("Iteration")
plt.title(f"Gradient norm (total time {res.time[-1]:.2f} sec., {res.nit} it.)")
