.. _maintainer:

Maintainer information
======================

Releasing
---------

A release is built and published by the "Wheels and source distribution"
workflow of GitHub Actions, from a tag that must be the version of the package,
``__version__`` in ``sklr/__init__.py``. Before uploading anything, the
workflow checks the tag, and that there is a wheel for every platform and
Python version.

1. Generate the release history of the version from the :ref:`changelog fragments
   <changelog>`. ``towncrier`` writes it into the file of its minor release in
   ``doc/whats_new``, above the previous versions and dated in the current month, and
   removes the fragments with ``git rm``. Review the generated section and commit it,
   along with ``__version__`` set to the version and, for the first release of a
   minor release, its row in the :ref:`compatibility table <sklearn_compatibility>`:

   .. code-block:: console

      $ towncrier build --version <version>

2. Push that commit to its branch and reproduce the experiments of the papers
   from it with the "Reproducibility" workflow, described in
   ``reproducibility/README.md``, which stores their results under the version
   and opens a pull request with them against the branch. Review the reports in
   ``reproducibility/<paper>/results/<version>`` and merge it, so that the tag
   and the documentation of the version include the results. A version that does
   not change the estimators may skip this step: the documentation then shows the
   results of the latest version that has them.

   .. code-block:: console

      $ gh workflow run reproducibility.yml --ref <branch>

3. Create the tag on the commit to release and push it:

   .. code-block:: console

      $ git tag -s <version> -m "scikit-lr <version>"
      $ git push origin <version>

4. Run the workflow by hand from the tag, with ``testpypi`` as the repository
   to upload the distributions to. It builds and tests the wheels and the
   source distribution, uploads them to `TestPyPI <https://test.pypi.org>`_,
   then installs the release from there and runs the tests again. This checks
   what testing the files alone cannot: that the index accepts them, that
   ``pip`` selects the right wheel, and that the dependencies are resolved
   from PyPI.

   .. code-block:: console

      $ gh workflow run wheels.yml --ref <version> -f publish=testpypi

5. If it passes, publish a GitHub release from the same tag. The workflow
   builds and tests the distributions again, and then waits for a maintainer
   to approve the deployment to the ``pypi`` environment before uploading them
   to PyPI. A version cannot be uploaded again to PyPI, even after deleting
   it, so this is the last chance to stop the release.

When the development of a new minor release starts, add its file to
``doc/whats_new``, with a header like the one of the previous file, list it in
``doc/whats_new.rst``, and point ``filename`` in ``[tool.towncrier]`` of
``pyproject.toml`` to it, so that the fragments are written there.

Only the latest minor release receives bug fixes, in patch releases made from its
release branch, named after it as ``<major>.<minor>.X`` and created from the
commit of its first release. A fix is merged into the development branch first,
and then picked onto the release branch with ``git cherry-pick``. The workflows
run on the pull requests that target a release branch as they do on those that
target the development branch.

The project uploads to PyPI and TestPyPI with trusted publishing, so no token
is stored in the repository: each index only accepts the uploads of this
workflow from the environment of the same name.

Publishing the GitHub release also makes the "Documentation" workflow build the
documentation of the tag and publish it on GitHub Pages, in the directory of its
minor release. A final release, not a pre-release, also becomes the stable
version, where the root of the site redirects to, unless a newer minor release
already is. The pushes to the development branch publish the development
version.

.. _new_sklearn_release:

Moving to a new minor release of scikit-learn
---------------------------------------------

scikit-lr is built against a single minor release of scikit-learn, for the reasons
given in :ref:`version_coupling`, so each new minor release of scikit-learn opens a
new release line of scikit-lr:

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

5. Start a new minor release of scikit-lr, as described above, whose row in the
   :ref:`compatibility table <sklearn_compatibility>` has the new minor release of
   scikit-learn.

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
