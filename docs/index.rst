======================
 Q-MM's documentation
======================

Q-MM is a small Python toolbox to optimize differentiable objective functions

.. math::

   \hat x = \underset{x \in \mathbb{R}}{\text{arg min}}\ J(x)

by Majorization-Minimization with quadratic surrogate function. In particular,
**no linesearch** is necessary and **close form formula for the step** are used
with guaranteed convergence without sub-iteration. The explicit step formula
allows fast convergence of the algorithm to a minimizer with minimal tuning
parameters. However, the objective function must meet conditions, see
:doc:`Background <background>`.

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
- Provided base class for objective allowing easier and fast implementation.
- Just one file if you like quick and dirty installation, but available with
  ``pip``.
- Comes with examples of implemented linear operator.

Contribute
==========

The code is hosted on `Github <https://github.com/forieux/qmm/>`_ under GPLv3
License. Feel free to contribute or submit `issue
<https://github.com/forieux/qmm/issues>`_.

Author
======

If you are having issues, please let us know

orieux AT l2s.centralesupelec.fr

`F. Orieux <https://pro.orieux.fr>`_ and R. Abirizk are affiliated to the Signal
and Systems Laboratory `L2S <https://l2s.centralesupelec.fr/>`_.

Acknowledgement
===============

Authors would like to thanks `Jérôme Idier
<https://pagespersowp.ls2n.fr/jeromeidier/en/jerome-idier-3/>`_, `Saïd Moussaoui
<https://scholar.google.fr/citations?user=Vkr8yxkAAAAJ&hl=fr>`_ and `Émilie
Chouzenoux <http://www-syscom.univ-mlv.fr/~chouzeno/>`_. E. Chouzenoux has also
a Matlab package that implements 3MG for image deconvolution on here `webpage
<http://www-syscom.univ-mlv.fr/~chouzeno/Logiciel.html>`_.

License
=======

The project is licensed under the GPLv3 license and has a DOI with Zenodo. If
you use the library, please cite it (you can change the version number).

::

    @software{francois_orieux_2022_6373070,
      author       = {François Orieux and Ralph Abirizk},
      title        = {Q-MM: The Quadratic Majorize-Minimize Python toolbox},
      month        = mar,
      year         = 2022,
      publisher    = {Zenodo},
      version      = {0.12.0},
      doi          = {10.5281/zenodo.6373069},
      url          = {https://doi.org/10.5281/zenodo.6373069}
    }

and associated publications, see :doc:`Background <background>`.

.. toctree::
   :maxdepth: 2
   :caption: Table of contents
   :hidden:

   background
   installation
   tutorial
   qmm
   operators
   extend
