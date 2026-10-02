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
   along with ``__version__`` set to the version:

   .. code-block:: console

      $ towncrier build --version <version>

2. Create the tag on the commit to release and push it:

   .. code-block:: console

      $ git tag -s <version> -m "scikit-lr <version>"
      $ git push origin <version>

3. Run the workflow by hand from the tag, with ``testpypi`` as the repository
   to upload the distributions to. It builds and tests the wheels and the
   source distribution, uploads them to `TestPyPI <https://test.pypi.org>`_,
   then installs the release from there and runs the tests again. This checks
   what testing the files alone cannot: that the index accepts them, that
   ``pip`` selects the right wheel, and that the dependencies are resolved
   from PyPI.

   .. code-block:: console

      $ gh workflow run wheels.yml --ref <version> -f publish=testpypi

4. If it passes, publish a GitHub release from the same tag. The workflow
   builds and tests the distributions again, and then waits for a maintainer
   to approve the deployment to the ``pypi`` environment before uploading them
   to PyPI. A version cannot be uploaded again to PyPI, even after deleting
   it, so this is the last chance to stop the release.

When the development of a new minor release starts, add its file to
``doc/whats_new``, with a header like the one of the previous file, list it in
``doc/whats_new.rst``, and point ``filename`` in ``[tool.towncrier]`` of
``pyproject.toml`` to it, so that the fragments are written there.

The project uploads to PyPI and TestPyPI with trusted publishing, so no token
is stored in the repository: each index only accepts the uploads of this
workflow from the environment of the same name.

Publishing the GitHub release also makes the "Documentation" workflow build the
documentation of the tag and publish it on GitHub Pages, in the directory of its
minor release. A final release, not a pre-release, also becomes the stable
version, where the root of the site redirects to, unless a newer minor release
already is. The pushes to the development branch publish the development
version.
