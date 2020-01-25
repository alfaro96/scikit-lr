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
CYTHON_MIN_VERSION = "0.29.14"


# =============================================================================
# Methods
# =============================================================================

def build_from_c_and_cpp_files(extensions):
    """Modify the extensions to build
    from the .c and .cpp files."""
    # Modify the extensions to build from .c
    # and .cpp files instead of .pyx or .py
    for extension in extensions:
        # Initialize the list of modified
        # source files for this extension
        sources = []
        # Modify the source files of this extension
        # (according to the previous criteria)
        for source in extension.sources:
            # Obtain the path to this source
            # file and the file extension
            (path, ext) = splitext(source)
            # Extract the .c or .cpp extension
            # of this .pyx or .py source file
            if ext in {".pyx", ".py"}:
                if extension.language == "c++":
                    ext = ".cpp"
                else:
                    ext = ".c"
                # Replace the extension
                # of this source file
                source = path + ext
            # Append the modified source
            # file to the list of modified
            # source files of this extension
            sources.append(source)
        # Add the modified source
        # files to this extension
        extension.sources = sources


def get_cython_version():
    """Return a string containing the Cython
    version (empty string if not installed)."""
    # Try to import Cython and, if it is not
    # found, show an error message to the user
    try:
        # Import
        import Cython
        # Get the installed version
        cython_version = Cython.__version__
    except ImportError:
        # Show the error message
        print_exc()
        # Set the version to an empty string
        cython_version = ""

    # Return installed
    # version of Cython
    return cython_version


def maybe_cythonize_extensions(top_path, config):
    """Tweaks for building extensions between release and development mode."""
    # Check whether the package is in release or development
    # looking to the existance of the "PKG-INFO" file
    # (which is only available in release mode)
    is_release = exists(join(top_path, "PKG-INFO"))

    # If it is a release, directly build the
    # extensions from the .c and .cpp source files
    if is_release:
        build_from_c_and_cpp_files(config.ext_modules)
    # Otherwise, "cythonize" the .pyx or .py source files
    else:
        # Get the installed version of Cython
        # (needed to "cythonize" the extensions)
        cython_version = get_cython_version()
        # If it is not installed, show the
        # proper error message to the user
        if not cython_version:
            raise ImportError("Cython is not installed. "
                              "At least version {} is required."
                              .format(CYTHON_MIN_VERSION))
        # Otherwise, if the installed version is not the minimum required one,
        # inform to the user about the got and the minimum required version
        elif parse_version(cython_version) < parse_version(CYTHON_MIN_VERSION):
            raise ImportError("Your installation of Cython is not "
                              "the required. Got {} but requires >={}."
                              .format(cython_version, CYTHON_MIN_VERSION))

        # Import the "cythonize" method
        from Cython.Build import cythonize

        # "Cythonize" the extension modules from
        # the corresponding .pyx and .py source files
        config.ext_modules = cythonize(config.ext_modules)
