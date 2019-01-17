"""
    Utilities useful during the build.
"""

# =============================================================================
# Imports
# =============================================================================

# Operating system
from   pkg_resources import parse_version
import os
import traceback

# =============================================================================
# Public objects
# =============================================================================

# Metadata
CYTHON_MIN_VERSION = "0.29"

# =============================================================================
# Methods
# =============================================================================

def build_from_c_and_cpp_files(extensions):
    """
        Modify the extensions to build from the ".c" and ".cpp" files,
        instead of cythonizing the ".pyx" files.
    """
    # Iterate all the extensions to obtain the source files
    # when they corresponding to a cython module
    for extension in extensions:
        # Initialize the source files from the current extension module
        sources = []
        # Iterate all the source files to check whether they
        # corresponding to a cython module
        for source_file in extension.sources:
            # Obtain the path and the extension
            (path, ext) = os.path.splitext(source_file)
            # If there exist a ".pyx" file or ".py",
            # obtain the ".c" or ".cpp" extension
            if ext in [".pyx", ".py"]:
                if extension.language == "c++":
                    ext = ".cpp"
                else:
                    ext = ".c"
                # Replace the current source file by the
                # ".c" or ".cpp" one
                source_file = path + ext
            # Append to the list of source files
            sources.append(source_file)
        # Add them to the extension module
        extension.sources = sources

def get_cython_version():
    """
        Get the version of Cython.
    """
    # Try to import Cython and if it is not found,
    # show the error message to the user
    try:
        # Import
        import Cython
        # Get the version
        cython_version = Cython.__version__
    except:
        # Show the error message
        traceback.print_exc()
        # Set the version as an empty string
        cython_version = ""

    # Return the obtained version 
    return cython_version   

def maybe_cythonize_extensions(top_path, config):
    """
        Tweaks for building extensions between release and development mode.
    """
    # Obtain whether the current package is a release or development
    is_release = os.path.exists(os.path.join(top_path, 'PKG-INFO'))

    # If it is a release, obtain the extensions modules from
    # the corresponding ".c" and ".cpp" files
    if is_release:
        build_from_c_and_cpp_files(config.ext_modules)
    # Otherwise, cythonize the ".pyx" files
    else:
        # Get the version
        cython_version = get_cython_version()
        # If Cython is not installed, show the corresponding message to the user
        if not cython_version:
            raise ImportError("Cython is not installed. At least version '{}' is required.".format(CYTHON_MIN_VERSION))
        # Otherwise, if the version is not the one required,
        # show the got and required version with the corresponding message
        elif parse_version(cython_version) < parse_version(CYTHON_MIN_VERSION):
            raise ImportError("Your installation of Cython is not the required, got '{}' but requires '>={}'".format(cython_version, CYTHON_MIN_VERSION))

        # Import the required "cythonize" method from Cython
        from Cython.Build import cythonize

        # Cythonize the extension modules
        config.ext_modules = cythonize(config.ext_modules)
