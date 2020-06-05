"""Utilities useful during the build."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from distutils.core import Extension
from glob import glob
from os.path import dirname, join, split, splitext
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
    NUMPY_HEADERS_PATH = join(dirname(np.__file__), "core", "include")
else:
    NUMPY_HEADERS_PATH = join(dirname(np.core.__file__), "include")


# =============================================================================
# Functions
# =============================================================================

def create_extension(sources):
    """Create and return an extension module."""
    (name, _) = splitext(sources[0].replace("/", "."))

    include_dirs = [NUMPY_HEADERS_PATH]

    extra_link_args = ["-std=c++11"]
    extra_compile_args = ["-O3", "-std=c++11"]

    return Extension(name,
                     sources,
                     language="c++",
                     include_dirs=include_dirs,
                     extra_link_args=extra_link_args,
                     extra_compile_args=extra_compile_args)


def cythonize_extensions(module_name):
    """Locate and cythonize the extension modules."""
    # Skip cythonization in source distributions
    # (release tarballs) for later compatibility
    if "sdist" not in sys.argv:
        pyx_pattern = join(module_name, "**/*.pyx")
        pyx_files = glob(pyx_pattern, recursive=True)

        for (pyx_index, pyx_file) in enumerate(pyx_files):
            (pyx_path, _) = split(pyx_file)
            cpp_pattern = join(pyx_path, "src/**/*.cpp")
            cpp_files = glob(cpp_pattern, recursive=True)

            sources = [pyx_file] + cpp_files
            pyx_files[pyx_index] = create_extension(sources)

        extensions = cythonize(pyx_files)

        return extensions
