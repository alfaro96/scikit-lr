"""Give a helpful message when the extension modules of ``sklr`` are not built."""

import os


def raise_build_error(error):
    """Raise an :class:`ImportError` explaining that ``sklr`` is not built correctly.

    Parameters
    ----------
    error : ImportError
        Error raised when importing an extension module.

    Raises
    ------
    ImportError
        Always, with `error` as its cause and the contents of this directory,
        to help debugging.
    """
    local_dir = os.path.dirname(__file__)
    dir_content = "\n".join(sorted(os.listdir(local_dir)))
    raise ImportError(
        f"{error}\n"
        f"Contents of {local_dir}:\n"
        f"{dir_content}\n"
        "It seems that scikit-lr has not been built correctly. If you have "
        "installed it from source, build it before using it. If you have used "
        "an installer, check that it is suited for your Python version, your "
        "operating system and your platform."
    ) from error


try:
    from sklr.__check_build._check_build import check_build  # noqa: F401
except ImportError as error:
    raise_build_error(error)
