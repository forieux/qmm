==========
 Tutorial
==========

.. py:currentmodule:: qmm

The toolbox is just one file, the ``qmm.py`` module. The module contains
essentially three part.

- The optimization algorithms, implemented as functions, that minimize

.. math::
   J(x) = \sum_k \mu_k \Psi_k(V_k x - \omega_k).

- The ``Objective`` class that implements

.. math::
   \mu \Psi(V x - \omega)\quad \text{ with }\quad \Psi(u) = \sum_i \varphi(u_i)

where :math:`u` is a vector and :math:`\varphi` a scalar function.

- The ``Loss`` class that implements :math:`\varphi`.


Operators
=========

The first thing to do is to implement the forward operator :math:`V` and adjoint
:math:`V^T`. User is in charge of it. They are callable that could be Python
functions or methods of objects.

.. code-block:: python

   def forward(array):
       # ...
       # do computation
       # ...
       return out  # An array or a list of array

   def adjoint(out):
       # ...
       # do computation
       # ...
       return array

The forward parameter must accept a ``numpy.ndarray`` :math:`x`, of any shape,
as unique parameter . The output of the forward operator must be

* a ``ndarray`` of any shape,
* or a list of ``ndarray`` (of any shape also).

Consequently, the adjoint operator must accept as parameter a ``ndarray`` or a
list of ``ndarray`` and returns a unique ``ndarray``, of any shape, as output.

.. note::

   The list of array allows mixed operators, like combination of forward models
   of different instruments, or multiple regularization.

   Everything is internally vectorized. Therefore, the use of list of array
   implies memory copies of arrays.


Losses
======

The second step is to instantiate loss function :math:`\varphi`, :class:`Huber`
for instance

.. code-block:: python

   from qmm.qmm import Huber, Objective, QuadObjective, mmmg
   phi = Huber(delta=10)

Several losses are implemented, see :doc:`Background <background>` and the
:doc:`qmm <qmm>` module.

Objectives
==========

Then, a :class:`Objective` :math:`\mu \Psi(Vx)` named ``prior`` can be instanced

.. code-block:: python

   prior = Objective(forward, adjoint, phi, hyper=0.01)

If a quadratic objective like :math:`\|y - H x\|_2^2` is needed, the specific
class :class:`QuadObjective` can be used

.. code-block:: python

   data_adeq = QuadObjective(H, Ht, data=data)

.. note::

   In the example above, the hyperparameter value is set to :math:`\mu = 1` and
   the data term is different that 0. For the prior term, the data is 0 by
   default and the hyperparameter is set to 0.01.

Optimization algorithms
=======================

Then you can run the algorithm, :func:`mmmg` for instance,

.. code:: python

   result = mmmg([data_adeq, prior], init, max_iter=200)

where the list :code:`[data_adeq, prior]` means that the two objective functions
are summed. The output `result` is an instance of :class:`OptimizeResult`.

.. note::

   :class:`BaseObjective` can be summed returning a :class:`MixedObjective` that
   behave like a list with additional functionalities. The above is equivalent
   to

   .. code-block:: python

      fun = data_adeq + prior
      result = mmmg(fun, init, max_iter=200)

Two algorithms are proposed :

- :func:`mmcg` that implements a Polak-Ribière Conjugate Gradient.
- :func:`mmmg` that implements a subspace by Memory-Gradient with 2D step (that,
  therefore, include the conjugacy parameter).

Both algorithms have close form formula for the 1D or 2D step by
Majorization-Minimization Quadratic.

In addition a Linear Conjugate Gradient :func:`lcg` is implemented for quadratic
objective.
