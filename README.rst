MMQ
===

MMQ provides Majorize-Minimize Quadratic optimization algorithms.

Features
--------

- Make things faster

Installation
------------

Install MMQ by running:

    install project

MMQ only depends on numpy.

Contribute
----------

- Issue Tracker: github.com/$project/$project/issues
- Source Code: github.com/$project/$project

Support
-------

If you are having issues, please let us know.
We have a mailing list located at: project@google-groups.com

License
-------

The project is licensed under the BSD license.

- `operator` : a callable with the current point :math:`x` as unique
  parameter, that must return the application of :math:`V`.
- `gradient` : a callable with the current point :math:`x` as unique
  parameter, that must return the gradient of the criterion.
- `norm_mat_major` : a callable with two parameters, the first one is the
  operator applied on the subspace vectors, and a current point :math:`x`, and a
  the normal matrix of the quadratic major function

  :math:`x`, that must return the gradient of the criterion
