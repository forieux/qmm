================
 ``qmm`` module
================

.. py:currentmodule:: qmm

.. _label-opt-alg:

Optimization algorithms
=======================

Three algorithms are implemented.

1. :func:`mmcg` that use the Majorize-Minimize Conjugate Gradient (MM-CG) and
2. :func:`mmmg` that use the Majorize Minimize Memory Gradient (3MG).
3. :func:`lcg` that use the Linear Conjugate Gradient (CG) for quadratic
   objective only, with exact optimal step and conjugacy parameters.

The 3MG algorithm is usually faster but use more memory. The MM-CG can be faster
and use less memory.

.. autofunction:: mmcg

.. autofunction:: mmmg

.. autofunction:: lcg


Optimization results
====================

The output are instance of :class:`OptimizeResult` that behave like
`OptimizeResult
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult>`_
of `scipy <https://www.scipy.org/>`_. They behave like Python dictionary and are
implemented to avoid dependency to scipy.

.. autoclass:: OptimizeResult

Objective classes
=================

Objective functions are defined from the abstract class :class:`BaseObjective`
that have three abstract methods that must be implemented by the subclass. If
users want to implements it's own objective, he is encouraged to subclass
:class:`BaseObjective`.

Four generic concrete classes of :class:`BaseObjective` can be used. The
:class:`Objective` class is the more general and prefered way, and
:class:`QuadObjective` is a specialized subclass that allows simplification and
slightly faster computation. :class:`Vmax` and :class:`Vmin` are for bound
penalties.

.. note::

   The property ``lastgv`` is used by algorithms to compute the objective
   function value at each iteration with low overhead, if the flag ``calc_fun``
   is set to ``True`` (``False`` by default). It is not required by the
   algorithms.

.. autoclass:: BaseObjective
   :members:


Main objective class
--------------------

.. autoclass:: Objective
   :members:

.. note::

   The :class:`Objective` class implements ``__call__`` interface allowing
   objects to behave like callable (function), returning the objective value

   .. code-block:: python

      identity = lambda x: x

      objv = qmm.Objective(identity, identity, qmm.Square())
      x = np.random.standard_normal((100, ))
      objv(x) == objv.value(x)


Quadratic objective
-------------------

This class implements specific properties or methods associated to quadratic
objective function.

.. autoclass:: QuadObjective
   :members:

.. note::

   The ``operator`` argument for :class:`Objective` and :class:`QuadObjective`
   must be a callable that accept an ``array`` as input and return an array as
   output. However, the operator **can also return** a ``list`` of array (for
   data fusion for instance). In that case, all these arrays are internally
   vectorized and the data are therefore memory copied.

   If operator returns a list of array, the ``adjoint`` **must** accept a list of
   array also. Again, everything is vectorized and the `Objective` and
   `QuadObjective` rebuild the list of array internally.

   If given, the ``hessp`` callable argument for :class:`QuadObjective` must
   accept an array and returns an array.


Specific objective classes
--------------------------

.. autoclass:: Vmin
    :members:


.. autoclass:: Vmax
    :members:

Sum of objectives
-----------------

The :class:`MixedObjective` is a convenient (not required) list-like class that
represent the sum of :class:`BaseObjective`. Moreoever, :class:`BaseObjective`
and :class:`MixedObjective` support the "+" operator and returns a
:class:`MixedObjective` instance, or update the instance, respectively. Since
:class:`MixedObjective` is a list, it can be used with :ref:`optimization
algorithms<label-opt-alg>`.

.. code-block:: python

   likelihood = QuadObjective(...)
   prior1 = Objective(...)
   prior2 = Objective(...)

   # Equivalent to objective = MixedObjective([likelihood, prior1])
   objective = likelihood + prior1

   # Equivalent to objective.append(prior2)
   objective = objective + prior2

   # Equivalent to res = mmmg([likelihood, prior1, prior2], ...)
   res = mmmg(objective, ...)


.. autoclass:: MixedObjective
    :members:


Losses classes
==============

The class :class:`Loss` is an abstract base class and serve as parent class for
all losses. At that time, the provided concrete loss functions are
:class:`Square`, :class:`Huber`, :class:`Hyperbolic`, :class:`HebertLeahy`,
:class:`GemanMcClure`, and :class:`TruncSquareApprox`.

.. note::

   The :class:`Loss` class implements ``__call__`` interface allowing objects to
   behave like callable (function), returning the function value

   .. code-block:: python

      u = np.linspace(-5, 5, 1000)
      pot = qmm.Huber(1)
      plt.plot(u, pot(u))


.. autoclass:: Loss
   :members:


Square
------

.. autoclass:: Square
   :members:


Huber
-----

.. autoclass:: Huber
   :members:


Hyperbolic or Pseudo-Huber
--------------------------

.. autoclass:: Hyperbolic
   :members:


Hebert & Leahy
--------------

.. autoclass:: HebertLeahy
   :members:


Geman & Mc Clure
----------------

.. autoclass:: GemanMcClure
   :members:


Truncated Square approximation
------------------------------

.. autoclass:: TruncSquareApprox
   :members:
