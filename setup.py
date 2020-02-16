# =============================================================================
# Imports
# =============================================================================

# Standard
from distutils.command.clean import clean as Clean
from distutils.version import LooseVersion
from importlib import import_module
from platform import python_version
from os import unlink, walk
from os.path import abspath, dirname, exists, join, splitext
import sys
import shutil

# Local application
import sklr


# =============================================================================
# Constants
# =============================================================================

# The following constants are the metadata of the package

# The name of the package, the name of the main
# module, the current version of the package and
# the distributing license of the project
DISTNAME = "scikit-lr"
MODNAME = "sklr"
VERSION = sklr.__version__
LICENSE = "MIT"

# The short and the long description of the package (the latest loaded
# from the README.md file for maintenance) and the content type of the
# long description
DESCRIPTION = "A set of Python modules for Label Ranking problems."
with open("README.md", mode="r", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"

# The name and the email of the package maintainer
MAINTAINER = "Juan Carlos Alfaro Jiménez"
MAINTAINER_EMAIL = "JuanCarlos.Alfaro@uclm.es"

# The different URLs of the project
URL = "https://github.com/alfaro96/scikit-lr"
BUG_TRACKER_URL = "https://github.com/alfaro96/scikit-lr/issues"
DOWNLOAD_URL = "https://pypi.org/project/scikit-lr/#files"
SOURCE_CODE_URL = "https://github.com/alfaro96/scikit-lr"
PROJECT_URLS = {"Bug Tracker": BUG_TRACKER_URL, "Source Code": SOURCE_CODE_URL}

# Trove classifiers to categorize each release
CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Unix",
    "Programming Language :: C",
    "Programming Language :: C++",
    "Programming Language :: Cython",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: Implementation :: CPython",
    "Topic :: Scientific/Engineering :: Artificial Intelligence"
]

# The following places must be sync with regard
# to the Python, Numpy and SciPy versions:
#   - .github/workflows/integration.yml
#   - docker/requirements/build.txt
#   - docker/requirements/test.txt
#   - setup.py
PYTHON_MIN_VERSION = (3, 6)
NUMPY_MIN_VERSION = "1.17.3"
SCIPY_MIN_VERSION = "1.3.2"

# The following constants are required by the package manager
# to know the minimum required version of the packages
PYTHON_REQUIRES = ">=" + ".".join(map(str, PYTHON_MIN_VERSION))
NUMPY_REQUIRES = "numpy>=" + NUMPY_MIN_VERSION
SCIPY_REQUIRES = "scipy>=" + SCIPY_MIN_VERSION
INSTALL_REQUIRES = [NUMPY_REQUIRES, SCIPY_REQUIRES]


# =============================================================================
# Others
# =============================================================================

# Import setuptools early if their features are
# wanted, as it monkey-patches the "setup" function
SETUPTOOLS_COMMANDS = {
    "develop", "release", "bdist_egg", "bdist_rpm",
    "bdist_wininst", "install_egg_info", "build_sphinx",
    "egg_info", "easy_install", "upload", "bdist_wheel",
    "--single-version-externally-managed",
}

if SETUPTOOLS_COMMANDS.intersection(sys.argv):
    import setuptools

    extra_setuptools_args = dict(zip_safe=False,
                                 include_package_data=True,
                                 extras_require={"alldeps": INSTALL_REQUIRES})
else:
    extra_setuptools_args = dict()


# =============================================================================
# Classes
# =============================================================================

class CleanCommand(Clean):
    """Custom clean command to remove build artifacts."""

    def run(self):
        """Execute the custom clean command."""
        # Call to the method of the parent to
        # clean common files and directories
        Clean.run(self)

        # Remove C and C++ files if the current working directory
        # is not a source distribution, since the source files
        # are needed by the package in release mode
        cwd = abspath(dirname(__file__))
        remove_c_files = not exists(join(cwd, "PKG-INFO"))

        if exists("build"):
            shutil.rmtree("build")

        for (dirpath, dirnames, filenames) in walk(MODNAME):
            for filename in filenames:
                extension = splitext(filename)[1]
                if filename.endswith((".so", ".pyd", ".dll", ".pyc")):
                    unlink(join(dirpath, filename))
                elif remove_c_files and extension in {".c", ".cpp"}:
                    pyx_file = str.replace(filename, extension, ".pyx")
                    # Remove the C and C++ files only when they are
                    # generated from a Cython extension, because in
                    # any other case, they really correspond to the
                    # source code
                    if exists(join(dirpath, pyx_file)):
                        unlink(join(dirpath, filename))
            for ddirname in dirnames:
                if ddirname in {"__pycache__"}:
                    shutil.rmtree(join(dirpath, ddirname))


CMDCLASS = {"clean": CleanCommand}


# =============================================================================
# Methods
# =============================================================================

def _check_python_version(min_version):
    """Check that the Python version installed in the system
    is greater than or equal the minimum required version."""
    if sys.version_info < min_version:
        raise RuntimeError("Scikit-lr requires Python {0} or later. "
                           "The current Python version is {1} installed "
                           "in {2}.".format(python_version(), min_version,
                                            sys.executable))


def _check_package_version(package, min_version):
    """Check that the version of the package installed in the
    system is greater than or equal the minimum required version."""
    # Re-raise with a more informative message when the package is not
    # installed
    try:
        module = import_module(package)
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Please install {0} with a version >= "
                                  "{1} in order to install scikit-lr."
                                  .format(package, min_version))

    if LooseVersion(module.__version__) < min_version:
        raise ValueError("The current version of {0} is {1} installed in {2}."
                         .format(package, module.__version__, module.__path__))


def configuration(parent_package="", top_path=None):
    """Configure the package."""
    # Remove the manifest before building the extensions.
    # Otherwise it may not be properly updated when the
    # contents of directories change
    if exists("MANIFEST"):
        unlink("MANIFEST")

    # Locally import the distribution utils of NumPy
    # to avoid that an import error is shown before
    # checking whether it is installed in the system
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)

    # Avoid useless messages
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage(MODNAME)

    return config


def setup_package():
    """Setup the package."""
    # Need to create a dictionary with the metadata of the package
    # to ensure that it is properly incorporated in the index
    metadata = dict(name=DISTNAME,
                    version=VERSION,
                    license=LICENSE,
                    maintainer=MAINTAINER,
                    maintainer_email=MAINTAINER_EMAIL,
                    description=DESCRIPTION,
                    long_description=LONG_DESCRIPTION,
                    url=URL,
                    download_url=DOWNLOAD_URL,
                    project_urls=PROJECT_URLS,
                    classifiers=CLASSIFIERS,
                    cmdclass=CMDCLASS,
                    python_requires=PYTHON_REQUIRES,
                    install_requires=INSTALL_REQUIRES,
                    **extra_setuptools_args)

    # For these actions, NumPy is not required. They are required to succeed,
    # for example, when pip manager is used to install the package when NumPy
    # is not yet present in the system
    if (len(sys.argv) == 1 or
            len(sys.argv) >= 2 and ("--help" in sys.argv[1:] or
                                    sys.argv[1] in {"--help-commands",
                                                    "egg_info", "dist_info",
                                                    "--version", "clean"})):
        # Use setuptools because these commands do
        # not work well or at all with distutils
        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup
        metadata["version"] = VERSION
    else:
        _check_python_version(PYTHON_MIN_VERSION)
        _check_package_version("numpy", NUMPY_MIN_VERSION)
        _check_package_version("scipy", SCIPY_MIN_VERSION)

        from numpy.distutils.core import setup

        metadata["configuration"] = configuration

    setup(**metadata)


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    setup_package()
