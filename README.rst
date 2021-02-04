MMQ
===

MMQ provides Majorize-Minimize Quadratic optimization algorithms. Algorithms
provided here come from that research.

.. [1] C. Labat and J. Idier, “Convergence of Conjugate Gradient Methods with a
   Closed-Form Stepsize Formula,” J Optim Theory Appl, p. 18, 2008.

.. [2] E. Chouzenoux, J. Idier, and S. Moussaoui, “A Majorize–Minimize Strategy
   for Subspace Optimization Applied to Image Restoration,” IEEE Trans. on
   Image Process., vol. 20, no. 6, pp. 1517–1528, Jun. 2011, doi:
   10.1109/TIP.2010.2103083.

If you use this code, please cite it and the references above.

Features
--------

- The ``mmmg``, Majorize-Minimize Memory Gradient algorithm.
- No restriction on the number of regularizer, input shape, ...
- Base class for criterion.
- Several classical criteria like Huber, Geman & McClure, ...
- Comes with examples of linear operator.


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
