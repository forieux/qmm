#!/usr/bin/env python3

# pylint: disable=invalid-name
#

import numpy as np  # type: ignore
import scipy.misc  # type: ignore
import matplotlib.pyplot as plt

from mmqmg import mmq, operators  # type: ignore

imag = scipy.misc.ascent()
shape = imag.shape

# Forward model
ir = np.ones((5, 5)) / 25
obs = operators.Conv2(ir, shape)

# Simulated data
data = obs(imag)
data += np.random.standard_normal(data.shape)
init = obs.adjoint(data)

# Regularization
diffr = operators.Diff(axis=0)
diffc = operators.Diff(axis=1)
pot = mmq.Huber(delta=10)

# Criterions definition
data_adeq = mmq.QuadCriterion(obs.forward, obs.adjoint, obs.fwback, mean=data)
priorr_adeq = mmq.Criterion(diffr.forward, diffr.adjoint, pot, pot.gradient, hyper=0.01)
priorc_adeq = mmq.Criterion(diffc.forward, diffc.adjoint, pot, pot.gradient, hyper=0.01)

# The Majorize-Minimize Quadratic Memory Gradient algorithm
res, norm_grad = mmq.mmmg([data_adeq, priorr_adeq, priorc_adeq], init, max_iter=200)

# Plot
plt.figure(1)
plt.clf()
plt.subplot(2, 2, 1)
plt.imshow(imag)
# plt.axis([300, 500, 500, 300])
plt.colorbar()
plt.subplot(2, 2, 2)
plt.imshow(data)
# plt.axis([300, 500, 500, 300])
plt.colorbar()
plt.subplot(2, 2, 3)
plt.imshow(res)
# plt.axis([300, 500, 500, 300])
plt.colorbar()
plt.subplot(2, 2, 4)
plt.plot(norm_grad)
