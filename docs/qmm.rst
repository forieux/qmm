====================
 The ``qmm`` module
====================

.. py:currentmodule:: qmm

The optimization algorithms
===========================

Two algorithms are implemented.

1. :func:`mmcg` that use the Majorize-Minimize Conjugate Gradient (MM-CG) and
2. :func:`mmmg` that use the Majorize Minimize Memory Gradient (3MG).

The 3MG algorithm is usually faster but use more memory. The MM-CG can be faster
and use less memory.

.. autofunction:: mmcg

.. autofunction:: mmmg

.. note::

   The use of this list is necessary to allow efficient implementation and reuse
   of calculations. See notes section for details.

The :class:`Criterion` classes
==============================

Two class can be used. The ``Criterion`` class is the more general and
``QuadCriterion`` is a specialized subclass that allow simplification an
slightly faster computation.


.. autoclass:: Criterion
   :members:

.. autoclass:: QuadCriterion
   :members:

The ``operator`` member
=======================

.. note::
    Comments on `Criterion`

    The module provides a `Criterion` object for convenience. This object has be
    made to help the usage of optimization algorithm. However, thanks to dynamic
    nature of python, the algorithms need in practice any object with three
    specific methods.

    - `operator` : a callable with a point `x` as unique array parameter, that must return the application of `V` (that is `V·x`).
    - `gradient` : a callable with a point `x` as unique array parameter, that must return the gradient of the criterion (that is `Vᵗ·φ'(V·x - ω)`).
    - `norm_mat_major` : a callable with two parameters. The first one is the result of the operator applied on the subspace vectors. The second is the point `x`, where the normal matrix of the quadratic major function must be returned.

The :class:`Potential` classes
==============================

The :class:`Potential` is an abstract base class that can't be instancied and
serve as parent class for all potential.

.. autoclass:: Potential
   :members:

.. autoclass:: Square
   :members:

.. autoclass:: Huber
   :members:

.. autoclass:: Hyperbolic
   :members:

.. autoclass:: HerbertLeahy
   :members:

.. autoclass:: GemanMcClure
   :members:

.. autoclass:: SquareTruncApprox
   :members:

.. autoclass:: VminProj
   :members:

.. autoclass:: VmaxProj
   :members:
