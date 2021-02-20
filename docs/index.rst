.. toctree::
   :maxdepth: 2
   :caption: Table of contents
   :hidden:

   background
   installation
   tutorial
   qmm
   extend

======================
 Q-MM's documentation
======================

Q-MM is a small Python toolbox to optimise differentiable criterion

.. math::
   \hat x = \underset{x \in \mathbb{R}}{\text{arg min}}\ J(x)

by Majorization-Minimization with quadratic surrogate function. In particular,
**no linesearch** is necessary and **close form formula for the step** are used
with guaranteed convergence. The explicit step formula allows fast convergence
of the algorithm to a minimiser with minimal tuning parameters. However, the
criterion must meet the conditions, see :doc:`Background <background>`.

Features
========

- The ``mmmg`` (or `3mg`), Majorize-Minimize Memory Gradient algorithm.
- The ``mmcg``, Majorize-Minimize Conjugate Gradient algorithm.
- *No linesearch*: the step is obtained from a close form formula.
- *No conjugacy choice*: a conjugacy strategy is not necessary thanks to the
  subspace nature of the algorithms. The ``mmcg`` algorithm use a Polak-Ribière
  formula.
- Generic and flexible: there is no restriction on the number of regularizer,
  their type, .., as well as for data adequacy.
- Provided base class for criterion allowing easier and fast implementation.
- Comes with examples of implemented linear operator.

Contribute
==========

The code is hosted on `Github <https://github.com/forieux/qmm/>`_ under MIT
Licence. Feel free to contribute or submit `issue
<https://github.com/forieux/qmm/issues>`_.

Author and support
==================

If you are having issues, please let us know

orieux AT l2s.centralesupelec.fr

More information about me `here <https://pro.orieux.fr>`_.

License
=======

The project is licensed under the MIT license. If you use the library, please
cite it

::

   @software{qmm,
      title = {Q-MM: The Python Quadratic Majorize-Minimize toolbox},
      author = {Orieux, Fran\c{c}ois},
      url = {https://github.com/forieux/qmm},
   }

and associated publications, see :doc:`Background <background>`.
