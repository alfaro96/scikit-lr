.. _development_setup:

Setting up the development environment
======================================

scikit-lr contains Cython extensions, so it must be compiled to be used from the
source tree. The development environment is declared in ``environment.yml`` and uses
packages from `conda-forge <https://conda-forge.org>`_, compilers included, so the
same steps work on Linux, macOS and Windows. On Windows, the compilers of
conda-forge use the ones of Microsoft, so the `Build Tools for Visual Studio
<https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_ must be installed
with the "Desktop development with C++" workload.

.. _development_setup_clone:

Cloning the repository
----------------------

`Fork the repository <https://github.com/alfaro96/scikit-lr/fork>`_ on GitHub, clone
your fork and add the main repository as the ``upstream`` remote, to keep your
branches up to date with it:

.. code-block:: console

   $ git clone https://github.com/<your-username>/scikit-lr.git
   $ cd scikit-lr
   $ git remote add upstream https://github.com/alfaro96/scikit-lr.git

.. _development_setup_install:

Installing scikit-lr
--------------------

Create the environment with `micromamba <https://mamba.readthedocs.io>`_ (or
``conda``), activate it and install scikit-lr in editable mode:

.. code-block:: console

   $ micromamba create -f environment.yml
   $ micromamba activate scikit-lr
   $ pip install --no-build-isolation --no-deps -e .

``--no-build-isolation`` builds scikit-lr with the packages of the environment,
instead of with packages that ``pip`` would download, so the extensions are compiled
against the scikit-learn that runs them. ``--no-deps`` keeps ``pip`` from
installing the dependencies, which come from conda-forge.

The editable install recompiles the extensions when :mod:`sklr` is imported, so there
is no need to reinstall after changing a Python or Cython file, or a ``meson.build``
file. It is only needed after changing the environment, such as the version of
Python, scikit-learn or Cython. When ``environment.yml`` changes, update the
environment with:

.. code-block:: console

   $ micromamba env update -f environment.yml

Finally, install the ``pre-commit`` hooks, which check every commit:

.. code-block:: console

   $ pre-commit install

.. _development_setup_meson:

Building with Meson
-------------------

scikit-lr is built with `meson-python <https://mesonbuild.com/meson-python>`_. The
``meson.build`` file of each directory lists the files that are installed, so a new
Python module or Cython extension must be added to it. Otherwise it cannot be
imported, and the CI reports that it is missing from the wheels.
The scikit-learn guide explains :ref:`how Meson works <sklearn:meson_build_backend>`.

.. _development_setup_troubleshooting:

Troubleshooting
---------------

If importing :mod:`sklr` fails because a ``meson`` executable inside a
``pip-build-env-*`` directory does not exist, scikit-lr was installed without
``--no-build-isolation``. ``pip`` built it in a temporary environment that no longer
exists, so the editable install cannot recompile it. Install it again with the
command above.

If importing :mod:`sklr` raises an :class:`ImportError` because it was built against
another scikit-learn, the environment changed after installing it. Install it again,
so that it is compiled against the installed scikit-learn.
