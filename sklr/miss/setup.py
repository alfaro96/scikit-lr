# =============================================================================
# Imports
# =============================================================================

# Standard
from os import name

# Third party
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration


# =============================================================================
# Methods
# =============================================================================

def configuration(parent_package="", top_path=None):
    """Configure the sklr.miss module."""
    # Create the configuration file of the sklr.miss module,
    # where transformers to miss classes from rankings will hold
    config = Configuration("miss", parent_package, top_path)

    # Fix POSIX Operating Systems including
    # the macros of the modules in the library
    libraries = ["m" if name == "posix" else ""]

    # Define the extra flags passed to the compiler
    # and linker of the extension modules with
    # underlying C++ programming language
    extra_cpp_compile_args = ["-O3", "-std=c++11"]
    extra_cpp_link_args = ["-std=c++11"]

    # Add the extension module with the base and
    # fast methods to miss classes from rankings
    config.add_extension(
        "_base_fast",
        language="c++",
        sources=["_base_fast.pyx"],
        libraries=libraries,
        extra_link_args=extra_cpp_link_args,
        extra_compile_args=extra_cpp_compile_args)

    # Add the data files containing the headers for
    # the declarations of the extension modules
    config.add_data_files("_base_fast.pxd")

    # Add the submodule with tests
    # for the sklr.miss module
    config.add_subpackage("tests")

    # Return the configuration file
    # of the sklr.miss module
    return config


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    setup(**configuration().todict())
