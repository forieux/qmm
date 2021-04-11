Installation
============

Q-MM only depends on ``numpy`` and requires Python 3.6. The recommended way is
to use `poetry <https://python-poetry.org/>`_

.. code-block:: sh

   poetry add qmm

but you can also use ``pip`` to install in system path

.. code-block:: sh

   pip install qmm

or user's home

.. code-block:: sh

   pip install --user qmm

Finally, since the toolbox is essentially just one file, and if ``numpy`` is
installed, you can also just copy the ``qmm`` directory from `Github realease
<https://github.com/forieux/qmm/releases>`_ (or by cloning) where your code can
find it and do

.. code-block:: python

   from qmm import qmm

or copy the ``qmm.py`` file and do

.. code-block:: python

   import qmm
