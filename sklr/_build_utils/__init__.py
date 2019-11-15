"""Utilities useful during the build."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from pkg_resources import parse_version
from os.path import splitext, exists, join
from traceback import print_exc


# =============================================================================
# Constants
# =============================================================================

# Main module of the scikit-lr package
DEFAULT_ROOT = "sklr"

# Minimum required version of Cython
# to cythonize the extension modules
CYTHON_MIN_VERSION = "0.29"


# =============================================================================
# Methods
# =============================================================================

def build_from_c_and_cpp_files(extensions):
    """Modify the extensions to build from the .c and .cpp files.

    This is useful for releases. This way, Cython is not required to
    run python setup.py install.
    """
    # Obtain the source files when they
    # correspond to a Cython extension module
    for extension in extensions:
        # Initialize the source files from
        # the current extension module
        sources = []
        # Check whether the source files
        # correspond to a Cython module
        for source_file in extension.sources:
            # Obtain the path to the source
            # file and its extension
            (path, ext) = splitext(source_file)
            # If there exist a .pyx file or .py,
            # obtain the .c or .cpp extension
            if ext in {".pyx", ".py"}:
                if extension.language == "c++":
                    ext = ".cpp"
                else:
                    ext = ".c"
                # Replace the current source
                # file by the .c or .cpp one
                source_file = path + ext
            # Append the source file to
            # the list of source files
            sources.append(source_file)
        # Add them to the extension module
        extension.sources = sources


def get_cython_version():
    """Return a string containing the Cython version
    string (empty string if not installed)."""
    # Try to import Cython and, if it is not
    # found, show the error message to the user
    try:
        # Import
        import Cython
        # Get the current version
        cython_version = Cython.__version__
    except ImportError:
        # Show the error message
        print_exc()
        # Set the version as an empty string
        cython_version = ""

    # Return the installed version
    return cython_version


def maybe_cythonize_extensions(top_path, config):
    """Tweaks for building extensions between release and development mode."""
    # Check whether the current package is a release or development
    # looking for the existance of the "PKG-INFO" file
    # (only available in release mode)
    is_release = exists(join(top_path, "PKG-INFO"))

    # If it is a release, obtain the extensions modules
    # from the corresponding .c and .cpp files
    if is_release:
        build_from_c_and_cpp_files(config.ext_modules)
    # Otherwise, "cythonize" the .pyx files
    else:
        # Get the Cython version needed to
        # "cythonize" the extension modules
        cython_version = get_cython_version()
        # If it is not installed, show the
        # corresponding message to the user
        if not cython_version:
            raise ImportError("Cython is not installed. "
                              "At least version {} is required."
                              .format(CYTHON_MIN_VERSION))
        # Otherwise, if the version is not the required one, show the
        # got and the required version with the corresponding message
        elif parse_version(cython_version) < parse_version(CYTHON_MIN_VERSION):
            raise ImportError("Your installation of Cython is not "
                              "the required. Got {} but requires >={}."
                              .format(cython_version, CYTHON_MIN_VERSION))

        # Import the required "cythonize" method
        from Cython.Build import cythonize

        # "Cythonize" the extension modules
        config.ext_modules = cythonize(config.ext_modules)
