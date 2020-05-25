"""Utilities useful during the build."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from distutils.core import Extension
import glob
import os
import sys

# Third party
from Cython.Build import cythonize
import numpy as np


# =============================================================================
# Constants
# =============================================================================

# The extension modules that need to compile against NumPy should locate
# the corresponding include directory (source directory or core headers)
if np.show_config is None:
    d = os.path.join(os.path.dirname(np.__file__), "core", "include")
else:
    d = os.path.join(os.path.dirname(np.core.__file__), "include")

NUMPY_HEADERS_PATH = d


# =============================================================================
# Methods
# =============================================================================

def create_extension(file_path):
    """Create and return an extension module."""
    # Replace the slash by a point in the
    # file path to get the extension name
    (extension_name, _) = os.path.splitext(file_path)
    extension_name = extension_name.replace("/", ".")

    file_path = [file_path]
    include_dirs = [NUMPY_HEADERS_PATH]

    # Compile against C++11 and produce code
    # with the highest level of optimization
    extra_link_args = ["-std=c++11"]
    extra_compile_args = ["-O3", "-std=c++11"]

    return Extension(extension_name,
                     file_path,
                     language="c++",
                     include_dirs=include_dirs,
                     extra_link_args=extra_link_args,
                     extra_compile_args=extra_compile_args)


def cythonize_extensions(module_name):
    """Find and cythonize the extension modules."""
    # Skip cythonization in the release tarballs since
    # the generated C++ source files are not necessary
    if "sdist" not in sys.argv:
        pattern = os.path.join(module_name, "**/*.pyx")
        file_paths = glob.glob(pattern, recursive=True)

        for (extension, file_path) in enumerate(file_paths):
            file_paths[extension] = create_extension(file_path)

        extensions = cythonize(file_paths)

        return extensions
