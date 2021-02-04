MM-Q
===

MM-Q is a python implementation of Majorize-Minimize Quadratic optimization algorithms. Algorithms
provided here come from that research.

.. [1] C. Labat and J. Idier, “Convergence of Conjugate Gradient Methods with a
   Closed-Form Stepsize Formula,” J Optim Theory Appl, p. 18, 2008.

.. [2] E. Chouzenoux, J. Idier, and S. Moussaoui, “A Majorize–Minimize Strategy
   for Subspace Optimization Applied to Image Restoration,” IEEE Trans. on
   Image Process., vol. 20, no. 6, pp. 1517–1528, Jun. 2011, doi:
   10.1109/TIP.2010.2103083.

If you use this code, please cite it and the references above.

Majorize-Minimize Quadratic
---------------------------

The MM-Q optimization algorithm allow to find the minimum of criteria like

.. math::
   J(x) = \sum_k \phi_k(V_k x - \omega_k)

where ``x`` is the unkown vector of size ``N``, ``V_k`` a linear operator of size ``M × N``, ``omega_k`` a fixed vector of size ``M``, and ``phi_k`` a function that must be differentiable, even, coercive, ``phi(sqrt(·))`` concave, and ``0 < phi'(u) / u < +∞``. If all ``phi_k`` are convex, the criterion is convex and the MM-Q algorithms converge to the global and uniq minimizer. If ``phi_k``, MM-Q algorithm convege to a local minimzer.

A classical example is the resolution of an inverse problem with the minimization of

.. math::
   J(x) = ||y - H x||^2 + \mu phi(V x)

where ``H`` is the low-pass forward model, ``V`` a regularization operator that approximate gradient and ``phi`` an edge preserving function like Huber.

Features
--------

- The ``mmmg``, Majorize-Minimize Memory Gradient algorithm.
- No restriction on the number of regularizer, input shape, ...
- Base class for criterion.
- Several classical criteria like Huber, Geman & McClure, ...
- Comes with examples of linear operator.

Example
-------

The ``demo.py`` presents an example on image deconvolution. The first steps is to implements the operators ``V`` and the adjoint ``Vᵗ`` as callable (function or methods). 

After import of ``mmq``, you must instanciate ``Potential`` object that implement ``phi` and ``Criterion`` object that implements ``phi(V x - ω)``

.. code-block:: python
   import mmq
   phi = mmq.Huber(delta=10)

   data_adeq = mmq.QuadCriterion(H, Ht, HtH, mean=data)
   prior = mmq.Criterion(V, Vt, phi, phi.gradient, hyper=0.01)
   
Then you can run the algorithm

.. code-block:: python
   res, norm_grad = mmq.mmmg([data_adeq, prior], init, max_iter=200)

where :code-block:`[data_adeq, prior]` means that the criterion are summed.

Installation
------------

No installation procedure has been implemented at that time. To install, just
copy the ``mmq`` directory or the ``mmq.py`` file where your code can access it.

MMQ only depends on ``numpy``.

Documentation
-------------

Documentation is in ``./docs`` directory and is generated from the source files. You
can see the ``demo.py`` file for an example.

Contribute
----------

- Issue Tracker: github.com/forieux/mmq/issues
- Source Code: github.com/forieux/mmq

Support
-------

If you are having issues, please let us know

orieux AT l2s.centralesupelec.fr

License
-------

The project is licensed under the GPL3 license.
