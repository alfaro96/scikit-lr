.. _installation:

============
Installation
============

There are different ways to install scikit-lr:

    * :ref:`Install the latest official release <install_latest_release>`.
      This is the best approach for most users. It will provide a stable
      version and pre-built packages are available for most platforms.

    * :ref:`Install the development build <install_development_build>`.
      This is best for users who want the latest-and-greatest features
      and are not afraid of running brand-new code.

    * :ref:`Building the package from source <building_from_source>`.
      This is needed for users who wish to contribute to the project.

.. _install_latest_release:

Installing the latest release
=============================

Scikit-lr can be installed via pip from `PyPi`_::

    pip install -U scikit-lr

In order to check your installation, you can use::

    pip show scikit-lr
    pip freeze
    python -c "import sklr; sklr.__version__"

Note that in order to avoid potential conflicts with other packages, it is
strongly recommended to use a `virtual environment`_ or `conda environment`_.

Using an isolated environment makes possible to install a specific version of
scikit-lr and its dependencies independently of any previously installed Python
packages. In particular, under Linux is it discouraged to install pip packages
alongside the packages managed by the package manager of the distribution.

If you have not installed NumPy or SciPy yet, you can also install these.
Please, ensure that *binary wheels* are used, and NumPy and SciPy are not
recompiled from source, which can happen when using particular configurations
of operating system and hardware. You can install scikit-lr and its
dependencies with ``scikit-lr[alldeps]``.

.. warning::

    Scikit-lr requires Python 3.6 or newer, and Windows is not currently
    supported.

.. note::

    You should always remember to activate the environment of your choice
    prior to running any Python command whenever you start a new terminal
    session.

.. note::

    To make it easier to get up and running with scikit-lr, you can use the
    `official Docker image`_.

.. _install_development_build:

Installing the development build
================================

The continuous deployment servers of the scikit-lr project build, test
and upload wheel packages for the supported Python versions based on
push to the master branch.

Installing a development build is the quickest way to:

- Try a new feature that will be shipped in the next release.

- Check whether a bug you encountered has been fixed since the last release.

To install the development build from the `Anaconda repository`_::

    pip install --index https://pypi.anaconda.org/alfaro96/simple scikit-lr

.. note::

    The development build of scikit-lr is also provided via the
    `official Docker image`_.

.. _building_from_source:

Building from source
====================

Building from source is required to work on a contribution
(bug fix, new feature, code or documentation improvement):

1. Use `Git`_ to checkout the latest source from the `scikit-lr repository`_
   on GitHub::

    git clone git://github.com/alfaro96/scikit-lr.git
    cd scikit-lr

2. Optional (but recommended): Create and activate a dedicated
   `virtual environment`_  or `conda environment`_.

3. Install a C and C++ compiler for your platform, either system-wise (see
   instructions for `Linux`_ and `macOS`_) or with `conda-force`_ to get full
   isolation.

4. Install `Cython`_ and build the project with pip in editable mode::

    pip install cython
    pip install --verbose --no-build-isolation --editable .

5. Check that the installed scikit-lr has a version number ending with
   ``.dev0``::

    python -c "import sklr; sklr.__version__"

6. Run the tests on the module of your choice via `pytest`_.

.. note::

    You can use the `official Docker image`_ to setup a development
    environment.

.. note::

    If you plan on submitting a pull-request, you should clone from your fork
    instead.

.. note::

    If you want to build a stable version, you can ``git checkout <VERSION>``
    after checking out the latest source to get the code for that particular
    version.

.. note::

    You will have to run the ``pip install --no-build-isolation --editable .``
    command every time the source code of a Cython file is updated (ending in
    `.pyx` or `.pxd`). Use the ``--no-build-isolation`` flag to avoid compiling
    the whole project each time, only the files you have modified.

Dependencies
------------

Scikit-lr requires the following dependencies:

+---------------------+------------------------+---------------------+
| Runtime             | Build                  | Tests               |
+=====================+========================+=====================+
| | Python (>= 3.6)   | | Cython (>= 0.29.14)  | | Pytest (>= 4.6.4) |
| | NumPy (>= 1.17.3) | | C and C++ compiler   | |                   |
| | SciPy (>= 1.3.2)  | |                      | |                   |
+---------------------+------------------------+---------------------+

.. note::

    The runtime dependencies are automatically installed by pip
    if they were missing when building scikit-lr from source.

.. References

.. _Anaconda repository: https://anaconda.org/alfaro96/scikit-lr
.. _conda environment: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
.. _conda-force: https://anaconda.org/conda-forge/compilers
.. _Cython: https://cython.org
.. _official Docker image: https://hub.docker.com/repository/docker/alfaro96/scikit-lr
.. _Git: https://git-scm.com
.. _Linux: https://gcc.gnu.org/wiki/InstallingGCC
.. _macOS: https://clang.llvm.org/get_started.html
.. _PyPi: https://pypi.org/project/scikit-lr/
.. _pytest: https://docs.pytest.org/en/latest/
.. _scikit-lr repository: https://github.com/alfaro96/scikit-lr
.. _virtual environment: https://docs.python.org/3/tutorial/venv.html
