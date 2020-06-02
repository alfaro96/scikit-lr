"""Configuration file for the Sphinx documentation builder."""


# =============================================================================
# Imports
# =============================================================================

# Standard
import os
import sys

# Local application
import sklr

sys.path.insert(0, os.path.abspath("../sphinxext"))

from github_link import make_linkcode_resolve  # noqa


# =============================================================================
# Project information
# =============================================================================

# The project name, the author name and the copyright statement
project = "scikit-lr"
author = "Juan Carlos Alfaro Jiménez"
copyright = "2019-2020, Juan Carlos Alfaro Jiménez (MIT License)"

# The major project version and the full project version
version = sklr.__version__
release = version


# =============================================================================
# General configuration
# =============================================================================

# The extensions module names
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon"
]

# The extra templates
templates_path = ["../templates"]

# The style for highlighting of source code
pygments_style = "sphinx"

# Remove parentheses from functions and methods role text
add_function_parentheses = False


# =============================================================================
# Options for HTML output
# =============================================================================

# The output theme
html_theme = "sphinx_rtd_theme"


# =============================================================================
# Extensions options
# =============================================================================

# Map the directives to the default values
autodoc_default_options = {
    "members": None,
    "inherited-members": None
}

# Generate a stub page for all found documents
autosummary_generate = True

# Map the name to the location of other projects
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "scikit-learn": ("https://scikit-learn.org/stable", None)
}

# Link the source code of an object to the repository
linkcode_resolve = make_linkcode_resolve(author="alfaro96",
                                         package="scikit-lr")

# Show the class variables (attributes)
napoleon_use_ivar = True
