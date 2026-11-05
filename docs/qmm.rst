API references (``qmm`` module)
===============================

.. py:currentmodule:: qmm

.. _label-opt-alg:

All the functionalities are provided by the unique ``qmm`` module described
below.

Optimization algorithms
-----------------------

Four algorithms are implemented.

1. :func:`mmcg` that use the Majorize-Minimize Conjugate Gradient (MM-CG),
2. :func:`mmmg` that use the Majorize-Minimize Memory Gradient (3MG), and
3. :func:`lcg` that use the Linear Conjugate Gradient (CG) for quadratic
   objective :class:`QuadObjective` only, with exact optimal step and conjugacy
   parameters, and
4. :func:`pcg` that is the same algorithm with a different API, for a linear
   system :math:`Ax = b` given by the user.

The 3MG algorithm is usually faster but use more memory. The MM-CG can be faster
and use less memory.

Majorize-Minimize Conjugate Gradient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: mmcg

Majorize-Minimize Memory Gradient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: mmmg


Linear Conjugate Gradient
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: lcg


Preconditioned Conjugate Gradient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pcg


Real and complex variables
~~~~~~~~~~~~~~~~~~~~~~~~~~

All the algorithms and :class:`QuadObjective` accept complex variables: the
``adjoint`` must then apply the conjugate transpose :math:`V^H`, and the
objective values stay real.

:class:`Objective` requires a real residual :math:`Vx - \omega`, since losses are
not defined for complex values. :class:`Vmin` and :class:`Vmax` require real
variables.

If ``x0`` is a half-spectrum from ``numpy.fft.rfftn``, use the ``hdot_shape``
argument of :func:`lcg` and :func:`pcg` to compute the objective value correctly.
It has no effect on the solution.


Optimization results
--------------------

The output are instance of :class:`OptimizeResult` that behave like
`OptimizeResult
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult>`_
of `scipy <https://www.scipy.org/>`_. They behave like Python dictionary and are
implemented to avoid dependency to scipy.

.. autoclass:: OptimizeResult

Objective classes
-----------------

Objective functions are defined from the abstract class :class:`BaseObjective`
that have four abstract methods that must be implemented by the subclass. Users
who want to implement their own objective are encouraged to subclass
:class:`BaseObjective`, see :doc:`extend`.

Four generic concrete classes of :class:`BaseObjective` can be used. The
:class:`Objective` class is the more general and prefered way, and
:class:`QuadObjective` is a specialized class that allows simplification and
slightly faster computation. :class:`Vmax` and :class:`Vmin` are for bound
penalties.

.. note::

   The property ``lastgv`` is used by algorithms to compute the objective
   function value at each iteration with low overhead, if the flag ``calc_objv``
   is set to ``True`` (``False`` by default). It is not required by the
   algorithms.

   The property ``hyper`` is read-only: instantiate a new objective to change
   :math:`\mu`.

.. autoclass:: BaseObjective
   :members:


Main objective
~~~~~~~~~~~~~~

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

   The ``val_grad`` method returns both the value and the gradient.


Quadratic objective
~~~~~~~~~~~~~~~~~~~

This class implements specific properties or methods associated to quadratic
objective function.

.. autoclass:: QuadObjective
   :members:

.. note::

   The ``operator`` argument for :class:`Objective` and :class:`QuadObjective`
   must be a callable that accept an ``array`` as input. The operator can return
   an array as output but **can also return** a ``list`` of array (for data
   fusion for instance). However, for needs of optimization algorithm
   implementation, everything must be an array internally. In case of ``list``
   or arrays, all these arrays are handled by a `Stacked` class, internally
   vectorized and the data are therefore memory copied, at each iteration.

   If ``operator`` returns a list of array, ``data`` **must** be a list of array
   and the ``adjoint`` **must** accept a list of array also. Again, everything is
   vectorized and `Objective` rebuild the list of array internally. Without
   data, use one objective for each output. With :class:`QuadObjective`, the
   ``invcovp`` argument must also accept and return a list of array.

   .. code-block:: python

      def forward(x):
          return [H1(x), H2(x)]  # The shapes can be different

      def adjoint(out):
          return H1t(out[0]) + H2t(out[1])

      objv = QuadObjective(forward, adjoint, data=[y1, y2])

   `QuadObjective` handle this ``list`` of array more efficiently since gradients
   only use the data :math:`\omega` through :math:`\mu V^T B \omega`, that is an
   array like `x` computed once.

   If given, the ``hessp`` callable argument for :class:`QuadObjective` must
   accept an array and returns an array.


Specific objective classes
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Vmin
    :members:


.. autoclass:: Vmax
    :members:

Sum of objectives
~~~~~~~~~~~~~~~~~

The :class:`MixedObjective` is a convenient (not required) list-like class that
represent the sum of :class:`BaseObjective`. Moreover, :class:`BaseObjective`
and :class:`MixedObjective` support the "+" operator and `sum()`, that return a
new flat :class:`MixedObjective` instance without modifying the operands. The
"+=" operator updates a :class:`MixedObjective` in place. Since
:class:`MixedObjective` is a list, it can be used with :ref:`optimization
algorithms<label-opt-alg>`.

.. code-block:: python

   likelihood = QuadObjective(...)
   prior1 = Objective(...)
   prior2 = Objective(...)

   # Equivalent to objective = MixedObjective([likelihood, prior1])
   objective = likelihood + prior1

   # Equivalent to objective.append(prior2)
   objective += prior2

   # Equivalent to res = mmmg([likelihood, prior1, prior2], ...)
   res = mmmg(objective, ...)


.. autoclass:: MixedObjective
    :members:


Utilities
---------

.. autofunction:: vectorize


Losses classes
--------------

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
~~~~~~

.. autoclass:: Square
   :members:


Huber
~~~~~

.. autoclass:: Huber
   :members:


Hyperbolic or Pseudo-Huber
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Hyperbolic
   :members:


Hebert & Leahy
~~~~~~~~~~~~~~

.. autoclass:: HebertLeahy
   :members:


Geman & Mc Clure
~~~~~~~~~~~~~~~~

.. autoclass:: GemanMcClure
   :members:


Truncated Square approximation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TruncSquareApprox
   :members:
