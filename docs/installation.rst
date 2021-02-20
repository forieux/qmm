==============
 Installation
==============

MM-Q only depends on ``numpy`` and requires Python 3.6. The recommended way is
to use `poetry <https://python-poetry.org/>`_

.. code-block:: sh

   poetry add mmq

but you can also use ``pip`` to install in system path

.. code-block:: sh

   pip install mmq

or user's home

.. code-block:: sh

   pip install --user mmq

Finally, since the toolbox is essentially just one file, and if ``numpy`` is
installed, you can also just copy the ``mmq`` directory from Github where your
code can find it and do

.. code-block:: python

   from mmq import mmq

or copy the ``mmq.py`` file and do

.. code-block:: python

   import mmq
