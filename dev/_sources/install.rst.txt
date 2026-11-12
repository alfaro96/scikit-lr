.. _install:

Installing scikit-lr
====================

scikit-lr can be installed from PyPI with ``pip``:

.. code-block:: console

   $ pip install -U scikit-lr

There are wheels for Linux, macOS and Windows, so ``pip`` only compiles the
package when there is none for the platform, from its source distribution.

.. _install_dependencies:

Dependencies
------------

scikit-lr requires Python, NumPy, SciPy and scikit-learn, which ``pip``
installs along with it. The supported versions are listed in the metadata of
the package on `PyPI <https://pypi.org/project/scikit-lr>`_.

Following `SPEC 0 <https://scientific-python.org/specs/spec-0000/>`_, each minor
release of Python is supported for three years after its first release, and each
minor release of NumPy and SciPy for two years, unless scikit-learn requires a newer
one. The supported versions only change in the minor releases of scikit-lr, which
add the new Python versions and raise the minimum versions, while its patch
releases keep those of their minor release.

The extension modules of scikit-lr extend the internals of scikit-learn, which
may change between its minor releases, so each release of scikit-lr is built
against a single minor release of scikit-learn and only works with it.
Importing :mod:`sklr` with another one raises an :class:`ImportError` that tells
which one is needed.

.. _sklearn_compatibility:

Each minor release of scikit-lr works with these versions of scikit-learn and
Python:

=========  ============  ==========
scikit-lr  scikit-learn  Python
=========  ============  ==========
0.3        1.9           3.13--3.15
=========  ============  ==========

Only the latest minor release of scikit-lr receives bug fixes, so with an older
release of scikit-learn the fixes are only available by upgrading both.
