"""Sphinx extension to populate the links in the API reference."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from functools import partial
from importlib import import_module
from inspect import getsourcefile, getsourcelines, unwrap
from packaging.version import parse
from operator import attrgetter
from os.path import dirname, relpath


# =============================================================================
# Methods
# =============================================================================

def linkcode_resolve(domain, info, author, package):
    """Determine the URL corresponding to a Python object."""
    try:
        module = import_module(info["module"])
    except ModuleNotFoundError:
        # No URL is returned when the object is not in
        # any module, or the module cannot be imported
        return None

    # Unwrap the object to get the correct source
    # file in case that is wrapped by a decorator
    obj = unwrap(attrgetter(info["fullname"])(module))

    try:
        filename = getsourcefile(obj)
    except TypeError:
        # No URL is returned when the object is
        # a built-in module, class, or function
        return None

    # Get the relative path to the filename starting
    # from the top-level module to determine the URL
    module = import_module(module.__package__.split(".")[0])
    filename = relpath(filename, start=dirname(module.__file__))

    version = module.__version__
    module = module.__name__

    if parse(version).is_devrelease:
        # Ensure that the master branch is
        # used for the development version
        version = "master"

    (_, line_number) = getsourcelines(obj)

    return f"https://github.com/{author}/{package}/blob/" \
           f"{version}/{module}/{filename}#L{line_number}"


def make_linkcode_resolve(author, package):
    """Returns the function to populate the links in the API reference."""
    return partial(linkcode_resolve, author=author, package=package)
