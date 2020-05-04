"""Configuration file for the Sphinx documentation builder."""


# =============================================================================
# Imports
# =============================================================================

# Local application
import sklr


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

# A list of strings that are module names of extensions. Note
# that extensions coming with Sphinx are named "sphinx.ext.*"
extensions = [
    "sphinx.ext.intersphinx"
]

# Map the name to the location of other projects
# that should be linked to in this documentation
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None)
}

# The document name of the "master" document
master_doc = "index"

# Whether parentheses are appended to function and method role text
add_function_parentheses = False


# =============================================================================
# Options for HTML output
# =============================================================================

# The "theme" that the HTML output should use
html_theme = "sphinx_rtd_theme"

# Whether links to the reST sources will be added to the sidebar
html_show_sourcelink = False

# Whether "Created using Sphinx" is shown in the HTML footer
html_show_sphinx = False
