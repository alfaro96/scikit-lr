.. _install:

Installing scikit-lr
====================

scikit-lr can be installed from PyPI with ``pip``:

.. code-block:: console

   $ pip install -U scikit-lr

There are wheels for Linux, macOS and Windows, so ``pip`` only compiles the
package when there is none for the platform, from its source distribution.

Dependencies
------------

scikit-lr requires Python, NumPy, SciPy and scikit-learn, which ``pip``
installs along with it. The supported versions are listed in the metadata of
the package on `PyPI <https://pypi.org/project/scikit-lr>`_.

The extension modules of scikit-lr extend the internals of scikit-learn, which
may change between its minor releases, so each release of scikit-lr is built
against a single minor release of scikit-learn and only works with it.
Importing :mod:`sklr` with another one raises an :class:`ImportError` that tells
which one is needed.
