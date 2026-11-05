How to extend Q-MM
==================

To extend Q-MM you must subclass :class:`qmm.BaseObjective` or
:class:`qmm.Loss`. Then :func:`qmm.mmmg` and :func:`qmm.mmcg` should work as
expected.

Objectives
----------

A subclass of :class:`qmm.BaseObjective` must implement ``operator``, ``value``,
``gradient`` and ``norm_mat_major``.

.. code-block:: python

   class Tikhonov(qmm.BaseObjective):
       """The objective `½ μ ||x||²`."""

       def operator(self, point):
           return point

       def value(self, point):
           return self.hyper * np.sum(point ** 2) / 2

       def gradient(self, point):
           if self.calc_objv:
               self.lastgv = self.value(point)
           return self.hyper * point

       def norm_mat_major(self, vecs, point):
           return self.hyper * vecs.T @ vecs

   res = qmm.mmmg([data_adeq, Tikhonov(hyper=0.1)], init)

- ``operator`` must be linear and return an array, not a list.
- ``gradient`` must set ``lastgv`` if ``calc_objv`` is true. If not, the objective
  values in :class:`qmm.OptimizeResult` will be wrong.

Losses
------

A subclass of :class:`qmm.Loss` must implement ``value`` and ``gradient``, and
must give :math:`\lim_{u \to 0} \dot{\varphi}(u) / u` as ``inf`` argument. The
loss must meet the conditions of :doc:`Background <background>`.

.. code-block:: python

   class Cauchy(qmm.Loss):
       """The loss `φ(u) = log(1 + u²) / 2`."""

       def __init__(self):
           super().__init__(inf=1, convex=False, coercive=True)

       def value(self, point):
           return np.log(1 + point ** 2) / 2

       def gradient(self, point):
           return point / (1 + point ** 2)

   prior = qmm.Objective(V, Vt, Cauchy(), hyper=0.01)

Contribute
----------

Feel free to `contribute <https://github.com/forieux/qmm/>`_ with issues or pull
request.
