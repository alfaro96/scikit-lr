"""Check that ``sklr`` runs with the ``scikit-learn`` it was built against."""

import sklearn

from sklr._build_info import SKLEARN_BUILD_VERSION


def check_sklearn_version(build_version, runtime_version):
    """Raise an :class:`ImportError` if the versions differ in major or minor."""
    build_minor = ".".join(build_version.split(".")[:2])
    if build_minor != ".".join(runtime_version.split(".")[:2]):
        raise ImportError(
            f"scikit-lr was built against scikit-learn {build_version}, but "
            f"scikit-learn {runtime_version} is installed. Its extension "
            "modules depend on the internals of scikit-learn, which may change "
            "between minor releases, so both versions must match up to the "
            f"minor release. Install scikit-learn {build_minor}.x, or a "
            f"scikit-lr release built against scikit-learn {runtime_version}."
        )


check_sklearn_version(SKLEARN_BUILD_VERSION, sklearn.__version__)
