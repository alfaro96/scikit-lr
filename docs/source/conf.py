"""Configuration file for the Sphinx documentation builder."""


# =============================================================================
# Imports
# =============================================================================

# Standard
import os
import sys

# Local application
import sklr

# If the extensions (or modules to document with autodoc) are in another
# directory, add these directories here. If the directory is relative to
# the documentation root, use os.path.abspath to make this path absolute
sys.path.insert(0, os.path.abspath("../sphinxext"))

from github_link import make_linkcode_resolve  # noqa


# =============================================================================
# Project information
# =============================================================================

# The documented project's name, the author name(s) of the document and
# the copyright statement, in concordance with the main "setup.py" file
project = "scikit-lr"
author = "Juan Carlos Alfaro Jiménez"
copyright = "2019-2020, Juan Carlos Alfaro Jiménez (MIT License)"

# The major project version and the full project version, used
# as the replacement for |version| and |release|, respectively
version = sklr.__version__
release = version


# =============================================================================
# General configuration
# =============================================================================

# A list of strings that are module names of extensions,
# coming with Sphinx (named sphinx.ext.*) or custom ones
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon"
]

# The document name of the "master" document
master_doc = "index"

# A list of paths that contain extra templates
templates_path = ["../templates"]

# The style name to use for Pygments highlighting of source code
pygments_style = "sphinx"

# Whether parentheses are appended to function and
# method text to signify that the name is callable
add_function_parentheses = False

# Whether doctest flags at the ends of lines are detached
# for all code blocks showing interactive Python sessions
trim_doctest_flags = True


# =============================================================================
# Options for autodoc extension
# =============================================================================

# Map the option names for autodoc directives to the default values
autodoc_default_options = {
    "members": None,
    "inherited-members": None
}


# =============================================================================
# Options for autosummary extension
# =============================================================================

# Whether to scan all found documents for autosummary
# directives, and to generate stub pages for each one
autosummary_generate = True


# =============================================================================
# Options for intersphinx extension
# =============================================================================

# Map the name to the location of other projects
# that should be linked to in this documentation
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None)
}


# =============================================================================
# Options for linkcode extension
# =============================================================================

# Link the source code of an object to the GitHub repository
linkcode_resolve = make_linkcode_resolve(author="alfaro96",
                                         package="scikit-lr")


# =============================================================================
# Options for napoleon extension
# =============================================================================

# This is required to format the attributes of a class.
# See: https://github.com/sphinx-doc/sphinx/issues/7582
napoleon_use_ivar = True


# =============================================================================
# Options for HTML output
# =============================================================================

# The "theme" that the HTML output should use
html_theme = "sphinx_rtd_theme"

# Whether links to the reST sources will be added to the sidebar
html_show_sourcelink = False

# Whether "Created using Sphinx" is shown in the HTML footer
html_show_sphinx = False
