================
 ``qmm`` module
================

.. py:currentmodule:: qmm

=========================
 Optimization algorithms
=========================

Two algorithms are implemented.

1. :func:`mmcg` that use the Majorize-Minimize Conjugate Gradient (MM-CG) and
2. :func:`mmmg` that use the Majorize Minimize Memory Gradient (3MG).

The 3MG algorithm is usually faster but use more memory. The MM-CG can be faster
and use less memory.

.. autofunction:: mmcg

.. autofunction:: mmmg

Criterion classes
=================

Criterion are defined from the abstract class :class:`BaseCrit` that have three
abstract methods that must be implemented by the subclass. If users want to
implements it's own criterion, he is encouraged to subclass :class:`BaseCrit`.

Two generic concrete classes of :class:`BaseCrit` can be used. The
:class:`Criterion` class is the more general and :class:`QuadCriterion` is a
specialized subclass that allows simplification an slightly faster computation.

.. autoclass:: BaseCrit
   :members:

.. autoclass:: Criterion
   :members:

.. autoclass:: QuadCriterion
   :members:

.. note::

   The ``operator`` argument for :class:`Criterion` and :class:`QuadCriterion`
   must be a callable that accept an ``array`` as input and return an array as
   output. However, the operator **can also return** a ``list`` of array (for
   data fusion for instance). In that case, all these array are internally
   vectorized and the data are therefor memory copied.

   If operator returns a list of array, the ``adjoint`` **must** accept a list of
   array also. Again, everything is vectorized and the `Criterion` and
   `QuadCriterion` rebuild the list of array internally.

   If given, the ``normal`` argument for :class:`QuadCriterion` must accept an
   array and returns an array.

..
   Specialized criterion
   ---------------------

   .. autoclass:: Vmin
       :members:

   .. autoclass:: Vmax
       :members:



:class:`Potential` classes
==========================

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

.. autoclass:: TruncSquareApprox
   :members:

