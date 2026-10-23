.. _cython:

Cython, scikit-learn and SciPy
==============================

The extension modules of scikit-lr are written in Cython and build on the Cython code
of scikit-learn and SciPy instead of copying it: the internals of scikit-learn, and
the public Cython API of SciPy. This page explains how they are used and what it
implies, and the conventions of the Cython code. The scikit-learn guides on
:ref:`Cython <sklearn:cython>` and on :ref:`performance <sklearn:performance-howto>`
also apply.

.. _cimport_sklearn:

Using the internals of scikit-learn
-----------------------------------

The extension modules ``cimport`` the declarations of the ``.pxd`` files that
scikit-learn installs, and extend its extension types by subclassing them. The
``meson.build`` of ``sklr`` passes the directories where scikit-learn and SciPy are
installed to Cython, so that it finds their ``.pxd`` files. Only the ``.pxd`` files
included in the wheels of scikit-learn can be used, so check the installed package
rather than its repository.

Code is only copied from scikit-learn when what is needed is not declared in a
``.pxd`` file, or when it must diverge too much to be extended, and only after
discussing it in an issue. The copied file starts with a comment that tells where it
comes from, and the license notice of scikit-learn is distributed along with
scikit-lr.

The Python code follows the same idea with the private API of scikit-learn, which is
only imported through ``sklr/utils/_sklearn_compat.py`` (see :ref:`coding_guidelines`).

.. _cimport_scipy:

Using the Cython API of SciPy
-----------------------------

The extension modules also ``cimport`` the functions that SciPy exposes for Cython,
such as the root finders of ``scipy.optimize.cython_optimize``, which can be called
without the GIL. Unlike the internals of scikit-learn, they are a public API of
SciPy, documented for this use.

.. _version_coupling:

Coupling with the versions of scikit-learn and SciPy
----------------------------------------------------

The layout of the extension types and structures of scikit-learn may change between
its minor releases, without any deprecation. An extension module compiled against
one of them may then crash with another one, instead of raising an error. The
private API of scikit-learn imported by ``sklr/utils/_sklearn_compat.py`` may also
change between its minor releases without deprecation, and is adapted there when
moving to a new one. For these reasons:

* scikit-lr is built against a single minor release of scikit-learn, which
  ``pyproject.toml`` requires both to build and to run it.

* The version of scikit-learn used to build scikit-lr is recorded in the generated
  module ``sklr/_build_info.py``. Importing :mod:`sklr` with another minor release
  raises an :class:`ImportError`.

* The workflow "Upcoming scikit-learn" builds and tests scikit-lr every night
  against the next release of scikit-learn, to find the changes that affect it before
  that release is published.

* The workflow "Unit tests" also runs the tests of scikit-lr, built against the
  scikit-learn of PyPI as in the wheels, with the scikit-learn of conda-forge, which
  is built apart, possibly with another release of Cython.

Each minor release of scikit-learn thus has its own release line of scikit-lr.

SciPy keeps the ABI of its public Cython API stable across its releases, so it is
not restricted to a single minor release, and scikit-lr only sets a minimum version
of it. The workflow "Upcoming NumPy and SciPy" still builds and tests scikit-lr every
night against the nightly wheels of SciPy, to find the changes that affect it before
they are released.

.. _cython_conventions:

Writing Cython code
-------------------

The Cython code follows these conventions:

* Arrays are passed as typed memoryviews, declared ``const`` when they are only
  read.

* The inner loops run without the GIL and do not use Python objects.

* The ``boundscheck``, ``wraparound`` and ``cdivision`` directives are only disabled
  where it is safe, with a comment that says why.

Before considering a module finished, read the annotated report that Cython
generates for it, which highlights the lines that interact with the Python
interpreter:

.. code-block:: console

   $ SITE=$(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")
   $ cython -a --include-dir "$SITE" sklr/<subpackage>/<module>.pyx

The report is written next to the source, as an ``.html`` file that ``git`` ignores.
