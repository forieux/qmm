==========
 Tutorial
==========

The toolbox is just one file, the ``qmm.py`` module. The module contains
essentially three part:

- the optimization algorithms, implemented as functions, that minimize

.. math::
   J(x) = \sum_k \mu_k \Psi_k(V_k x - \omega_k)

- the ``Criterion`` class that implements

.. math::
   \mu \Psi(V x - \omega)\quad \text{ with }\quad \Psi(u) = \sum_i \phi(u_i)

where :math:`u` is a vector and :math:`\phi` a scalar function.

- the ``Potential`` class that implement :math:`\phi`.


Operators
=========

The first thing to do is to implement the forward operator :math:`V` and adjoint
:math:`V^t`. User is obviously in charge of it. They are callable that could be
Python functions or methods of objects.

.. code-block:: python

   def forward(array):
       # ...
       # do computation
       # ...
       return out  # Must be an array or a list of array

   def adjoint(out):
       # ...
       # do computation
       # ...
       return array

The forward parameter must accept a ``numpy.ndarray`` :math:`x`, of any shape,
as unique parameter . The output of the forward operator must be a ``ndarray``
of any shape, or a list of ``ndarray`` (of any shape also). Consequently, the
adjoint operator must accept as parameter a ``ndarray`` or a list of ``ndarray``
and returns a unique ``ndarray``, of any shape, as output.

.. note::

   The list of array allows mixed operators, like combination of forward models
   of different instruments, or multiple regularization.

   Everything is internally vectorized. Therefore, the use of list of array implies
   memory copies of arrays, however.


Potentials
==========

The second step is to instantiate potential :math:`\phi`, Huber for instance

.. code-block:: python

   from qmm import Huber
   phi = Huber(delta=10)

Several potential are implemented, see TODO.

Criterions
==========

Then, a criterion :math:`\mu \Psi(Vx)` named ``prior`` can be instanced.

.. code-block:: python

   from qmm import Criterion

   prior = Criterion(forward, adjoint, phi, hyper=0.01)

If a quadratic criterion like :math:`\|y - H x\|_2^2` is needed, the specific
class ``QuadCriterion`` can be used

.. code-block:: python

   from qmm import QuadCriterion
   data_adeq = QuadCriterion(H, Ht, data=data)

.. note::

   In the example above, the hyperparameter value is set to :math:`\mu = 1` and
   the data term is different that 0.

Optimization algorithms
=======================

Then you can run the algorithm

.. code:: python

   from qmm import mmcg

   minimizer, _ = qmm.mmcg([data_adeq, prior], init, max_iter=200)

where :code:`[data_adeq, prior]` means that the two criteria are summed.

Two algorithms are proposed :

- ``mmcg`` that implements a Polak-Ribière Conjugate Gradient.
- ``mmmg`` that implements a subspace by Memory-Gradient with 2D step (that,
  therefore, include the conjugacy parameter).

Both algorithms have close form formula for the 1D or 2D step by
Majorization-Minimization Quadratic.
