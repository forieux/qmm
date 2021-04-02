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
   criterion, with explicit and optimal step and conjugacy parameters.

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

Criterion classes
=================

Criterion are defined from the abstract class :class:`BaseCrit` that have three
abstract methods that must be implemented by the subclass. If users want to
implements it's own criterion, he is encouraged to subclass :class:`BaseCrit`.

Two generic concrete classes of :class:`BaseCrit` can be used. The
:class:`Criterion` class is the more general and :class:`QuadCriterion` is a
specialized subclass that allows simplification and slightly faster computation.

.. autoclass:: BaseCrit
   :members:

Main criterion class
--------------------

.. autoclass:: Criterion
   :members:

Quadratic criterion
-------------------

This class implements specific properties or methods associated to quadratic
criterion.

.. autoclass:: QuadCriterion
   :members:

.. note::

   The :class:`Criterion` class implements ``__call__`` interface allowing
   objects to behave like callable (function), returning the criterion value

   .. code-block:: python

      identity = lambda x: x

      crit = qmm.Criterion(identity, identity, qmm.Square())
      x = np.random.standard_normal((100, ))
      crit(x) == crit.value(x)


.. note::

   The ``operator`` argument for :class:`Criterion` and :class:`QuadCriterion`
   must be a callable that accept an ``array`` as input and return an array as
   output. However, the operator **can also return** a ``list`` of array (for
   data fusion for instance). In that case, all these arrays are internally
   vectorized and the data are therefore memory copied.

   If operator returns a list of array, the ``adjoint`` **must** accept a list of
   array also. Again, everything is vectorized and the `Criterion` and
   `QuadCriterion` rebuild the list of array internally.

   If given, the ``normal`` argument for :class:`QuadCriterion` must accept an
   array and returns an array.


Specific criterion classes
--------------------------

.. autoclass:: Vmin
    :members:

.. autoclass:: Vmax
    :members:


Potential classes
=================

The :class:`Potential` is an abstract base class that can't be instanced and
serve as parent class for all potential.

At that time, the provided concrete potential are :class:`Square`,
:class:`Huber`, :class:`Hyperbolic`, :class:`HerbertLeahy`,
:class:`GemanMcClure`, and :class:`TruncSquareApprox`.

.. note::

   The :class:`Potential` class implements ``__call__`` interface allowing
   objects to behave like callable (function), returning the function value

   .. code-block:: python

      u = np.linspace(-5, 5, 1000)
      pot = qmm.Huber(1)
      plt.plot(u, pot(u))


.. autoclass:: Potential
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

