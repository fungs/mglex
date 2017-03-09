Installing MGLEX
================

Dependencies
------------

MGLEX is a Python 3 package, it **does not run with Python 2 versions**. It depends on

* NumPy (for data types and math operations)
* SciPy (for few math functions)
* docopt (for command line parsing)

Installation
------------

We show how to install MLGEX under Debian and Ubuntu, but other platforms are similar.

You can simply install the requirements as system packages.

.. code-block:: sh

   sudo apt install python3 python3-numpy python3-scipy

We recommend to create a `Python virtual installation enviroment <https://docs.python.org/3/library/venv.html>`_ for MGLEX. In order to do so, install the venv package for your Python version (e.g. the Debian package python3.4-venv), if not included (or use `virtualenv <https://pypi.python.org/pypi>`_). The following command will make use of the installed system packages.

.. code-block:: sh

   python3 -m venv --system-site-packages my-mglex-environment
   source my-mglex-environment/bin/activate

MGLEX is deposited on the `Python Package Index <https://pypi.python.org/pypi>`_ and we recommend to install it via `pip <https://docs.python.org/3/installing/>`_.

.. code-block:: sh

   python3 -m pip install mglex
