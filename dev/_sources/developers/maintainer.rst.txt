.. _maintainer:

Maintainer information
======================

.. _releasing:

Releasing
---------

A release is built and published by the "Wheels and source distribution"
workflow of GitHub Actions, from a tag that must be the version of the package,
``__version__`` in ``sklr/__init__.py``. Before uploading anything, the
workflow checks the tag, and that there is a wheel for every platform and
Python version.

Every release of a minor release is made from its release branch, named after it
as ``<major>.<minor>.X``, so that the development branch can go on with the next
minor release in the meantime. Only the latest minor release receives bug fixes, in
its patch releases: a fix is merged into the development branch first, and then
picked onto the release branch with ``git cherry-pick``, in a pull request against
it. The workflows run on the pull requests that target a release branch as they do
on those that target the development branch.

1. For the first release of a minor release, raise the minimum supported versions
   that the policy allows raising by the date of the release, if any (see
   :ref:`python_numpy_scipy_versions`). Then create its release branch from the
   development branch and push it:

   .. code-block:: console

      $ git switch --create <major>.<minor>.X master
      $ git push origin <major>.<minor>.X

   Then, in a pull request against the development branch, start the development of
   the next minor release: set ``__version__`` in ``sklr/__init__.py`` to its
   development version, such as ``X.Y.dev0``, add its file to ``doc/whats_new``,
   with a header like the one of the previous file, list it in
   ``doc/whats_new.rst``, and point ``filename`` in ``[tool.towncrier]`` of
   ``pyproject.toml`` to it, so that the new fragments are written there.

2. Prepare the release in a pull request against the release branch, from a branch
   created from it, whose description lists these steps as a checklist to follow
   the progress of the release. Its first commit has the release history of the
   version, generated from the :ref:`changelog fragments <changelog>`.
   ``towncrier`` writes it into the file of its minor release in ``doc/whats_new``,
   above the previous versions and dated in the current month, and removes the
   fragments with ``git rm``. Review the generated section and commit it, along
   with, for the first release of a minor release, its row in the
   :ref:`compatibility table <sklearn_compatibility>` and its version in the
   supported versions of ``SECURITY.md``:

   .. code-block:: console

      $ towncrier build --version <version>

3. Its second commit sets ``__version__`` to the version, and gives each build
   dependency in ``build-system.requires`` of ``pyproject.toml`` an upper bound
   below its next minor release, such as ``<X.Y+1.0`` when its latest release is
   ``X.Y.Z``, so that a future release of a dependency cannot break the build of
   the source distribution. scikit-learn already has one. The ``check-versions``
   hook of ``pre-commit`` reports the environment files whose build dependencies
   must follow. The checks of the pull request build and test the wheels and the
   source distribution with these bounds.

   Once they pass, merge the pull request keeping both commits, so that the commit
   of the release history can be picked onto the development branch. GitHub does
   not sign the commits that it rebases, so push the branch of the pull request to
   the release branch instead, which GitHub also marks as merged:

   .. code-block:: console

      $ git push origin <branch>:<major>.<minor>.X

   Then, in another pull request against the development branch, pick the commit of
   the release history with ``git cherry-pick``, so that the release history of the
   development branch has the version too and the released fragments are removed
   from it.

4. Reproduce the experiments of the papers from the release branch with the
   "Reproducibility" workflow, described in ``reproducibility/README.md``, which
   stores their results under the version and opens a pull request with them
   against the branch. Review the reports in
   ``reproducibility/<paper>/results/<version>`` and merge it, so that the tag and
   the documentation of the version include the results. A version that does not
   change the estimators may skip this step: the documentation then shows the
   results of the latest version that has them.

   .. code-block:: console

      $ gh workflow run reproducibility.yml --ref <major>.<minor>.X

5. Create the tag on the last commit of the release branch and push it:

   .. code-block:: console

      $ git tag -s <version> -m "scikit-lr <version>"
      $ git push origin <version>

6. Run the workflow by hand from the tag, with ``testpypi`` as the repository
   to upload the distributions to. It builds and tests the wheels and the
   source distribution, uploads them to `TestPyPI <https://test.pypi.org>`_,
   then installs the release from there and runs the tests again. This checks
   what testing the files alone cannot: that the index accepts them, that
   ``pip`` selects the right wheel, and that the dependencies are resolved
   from PyPI.

   .. code-block:: console

      $ gh workflow run wheels.yml --ref <version> -f publish=testpypi

7. If it passes, publish a GitHub release from the same tag. The workflow
   builds and tests the distributions again, and then waits for a maintainer
   to approve the deployment to the ``pypi`` environment before uploading them
   to PyPI. A version cannot be uploaded again to PyPI, even after deleting
   it, so this is the last chance to stop the release.

The project uploads to PyPI and TestPyPI with trusted publishing, so no token
is stored in the repository: each index only accepts the uploads of this
workflow from the environment of the same name.

Publishing the GitHub release also makes the "Documentation" workflow build the
documentation of the tag and publish it on GitHub Pages, in the directory of its
minor release. A final release, not a pre-release, also becomes the stable
version, where the root of the site redirects to, unless a newer minor release
already is. The pushes to the development branch publish the development
version.

.. _supported_versions:

Supported versions
------------------

The supported versions of Python, NumPy and SciPy follow the policy described in
:ref:`install_dependencies`, while scikit-lr supports a single minor release of
scikit-learn, for the reasons given in :ref:`version_coupling`. The supported
versions thus depend on the releases of these projects, which the tests do not
notice while they pass, so the "Supported versions" workflow checks those releases
every week and keeps an issue, labeled "dependencies", with the changes that are
due: a new minor release of scikit-learn, a new Python version that NumPy, SciPy
and the supported scikit-learn have wheels for, and the minimum versions that the
policy allows raising. The workflow comments on the issue when that list differs
from the previous one, so that its subscribers are notified, and closes it once
there is nothing left. The same report is available locally, also for the planned
date of a release:

.. code-block:: console

   $ python build_tools/github/check_supported_versions.py --date <YYYY-MM-DD>

.. _python_numpy_scipy_versions:

Python, NumPy and SciPy
~~~~~~~~~~~~~~~~~~~~~~~

The supported versions only change in the development branch, in a pull request,
and thus with the minor releases: the release branches only receive bug fixes, and
their patch releases keep the versions of their minor release. A new Python
version is added as soon as the issue reports it, and is thus supported from the
next minor release. The minimum versions are raised before the first release of a
minor release, to those that the policy allows by its date. In both cases, change
``requires-python``, the classifiers or ``project.dependencies`` in
``pyproject.toml``: the ``check-versions`` hook of ``pre-commit`` then reports the
other files that declare them, which must follow.

.. _new_sklearn_release:

Moving to a new minor release of scikit-learn
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each new minor release of scikit-learn opens a new release line of scikit-lr:

1. Check the latest run of the "Upcoming scikit-learn" workflow, which builds and
   tests scikit-lr against the latest release of scikit-learn on PyPI, and fix its
   failures first.

2. Review what scikit-lr takes from scikit-learn outside its public API, the
   cimported ``.pxd`` files and the private API imported by
   ``sklr/utils/_sklearn_compat.py``, for the changes between both minor releases,
   even if the tests pass: a change in the layout of an extension type may only
   show up as a crash or a wrong result.

3. Set the range of scikit-learn to the new minor release, such as ``~=X.Y.0``, in
   both the runtime and the build dependencies of ``pyproject.toml``. The
   ``check-versions`` hook of ``pre-commit`` then reports the other files that
   declare it, which must follow.

4. Update the development environment, build scikit-lr again as described in
   :ref:`development_setup_install`, and run the tests.

5. Start a new minor release of scikit-lr, as described in :ref:`releasing`, whose
   row in the :ref:`compatibility table <sklearn_compatibility>` has the new minor
   release of scikit-learn.

Bundled libraries
-----------------

The tools that repair the wheels bundle with them the OpenMP runtime of their
platform: ``libgomp`` on Linux, ``libomp`` on macOS and ``vcomp140.dll`` on Windows.
Its license is appended to the license of scikit-lr in the wheels, from the files in
``build_tools/wheels``, and the workflow checks that each wheel has the runtime of
its platform and its license. On macOS, the runtime of LLVM is installed from
conda-forge before building, in its oldest version for each architecture, so that it
supports the oldest macOS version of the wheels. If the wheels bundle another
library, add its license to those files.
