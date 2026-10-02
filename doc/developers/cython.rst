.. _cython:

Cython and the internals of scikit-learn
========================================

The extension modules of scikit-lr are written in Cython and build on the internals
of scikit-learn instead of copying them. This page explains how they are used and
what it implies, and the conventions of the Cython code. The scikit-learn guides on
:ref:`Cython <sklearn:cython>` and on :ref:`performance <sklearn:performance-howto>`
also apply.

.. _cimport_sklearn:

Using the internals of scikit-learn
-----------------------------------

The extension modules ``cimport`` the declarations of the ``.pxd`` files that
scikit-learn installs, and extend its extension types by subclassing them. The
``meson.build`` of ``sklr`` passes the directory where scikit-learn is installed to
Cython, so that it finds them. Only the ``.pxd`` files included in the wheels of
scikit-learn can be used, so check the installed package rather than its
repository.

Code is only copied from scikit-learn when what is needed is not declared in a
``.pxd`` file, or when it must diverge too much to be extended, and only after
discussing it in an issue. The copied file starts with a comment that tells where it
comes from, and the license notice of scikit-learn is distributed along with
scikit-lr.

The Python code follows the same idea with the private API of scikit-learn, which is
only imported through ``sklr/utils/_sklearn_compat.py`` (see :ref:`coding_guidelines`).

.. _sklearn_version_coupling:

Coupling with the version of scikit-learn
-----------------------------------------

The layout of the extension types and structures of scikit-learn may change between
its minor releases, without any deprecation. An extension module compiled against
one of them may then crash with another one, instead of raising an error. For this
reason:

* scikit-lr is built against a single minor release of scikit-learn, which
  ``pyproject.toml`` requires both to build and to run it.

* The version of scikit-learn used to build scikit-lr is recorded in the generated
  module ``sklr/_build_info.py``. Importing :mod:`sklr` with another minor release
  raises an :class:`ImportError`.

* The workflow "Upcoming scikit-learn" builds and tests scikit-lr every night
  against the next release of scikit-learn, to find the changes that affect it before
  that release is published.

Each minor release of scikit-learn thus has its own release line of scikit-lr.

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
