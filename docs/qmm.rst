================
 ``qmm`` module
================

.. py:currentmodule:: qmm

Optimization algorithms
=======================

Two algorithms are implemented.

1. :func:`mmcg` that use the Majorize-Minimize Conjugate Gradient (MM-CG) and
2. :func:`mmmg` that use the Majorize Minimize Memory Gradient (3MG).
3. :func:`lcg` that use the Linear Conjugate Gradient (CG) for quadratic
   objective, with explicit and optimal step and conjugacy parameters.

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
   :members:
   :exclude-members: __init__


Objective classes
=================

Objective functions are defined from the abstract class :class:`BaseObjective`
that have three abstract methods that must be implemented by the subclass. If
users want to implements it's own objective, he is encouraged to subclass
:class:`BaseObjective`.

Two generic concrete classes of :class:`BaseObjective` can be used. The
:class:`Objective` class is the more general and :class:`QuadObjective` is a
specialized subclass that allows simplification and slightly faster computation.

.. autoclass:: BaseObjective
   :members:


Main objective class
--------------------

.. autoclass:: Objective
   :members:


Quadratic objective
-------------------

This class implements specific properties or methods associated to quadratic
objective function.

.. autoclass:: QuadObjective
   :members:


.. note::

   The :class:`Objective` class implements ``__call__`` interface allowing
   objects to behave like callable (function), returning the objective value

   .. code-block:: python

      identity = lambda x: x

      objv = qmm.Objective(identity, identity, qmm.Square())
      x = np.random.standard_normal((100, ))
      objv(x) == objv.value(x)


.. note::

   The ``operator`` argument for :class:`Objective` and :class:`QuadObjective`
   must be a callable that accept an ``array`` as input and return an array as
   output. However, the operator **can also return** a ``list`` of array (for
   data fusion for instance). In that case, all these arrays are internally
   vectorized and the data are therefore memory copied.

   If operator returns a list of array, the ``adjoint`` **must** accept a list of
   array also. Again, everything is vectorized and the `Objective` and
   `QuadObjective` rebuild the list of array internally.

   If given, the ``normal`` argument for :class:`QuadObjective` must accept an
   array and returns an array.


Specific objective classes
--------------------------

.. autoclass:: Vmin
    :members:


.. autoclass:: Vmax
    :members:


Losses classes
==============

The :class:`Loss` is an abstract base class that can't be instanced and
serve as parent class for all losses.

At that time, the provided concrete loss functions are :class:`Square`,
:class:`Huber`, :class:`Hyperbolic`, :class:`HerbertLeahy`,
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
