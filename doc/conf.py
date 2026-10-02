"""Sphinx configuration of the scikit-lr documentation."""

from datetime import UTC, datetime

import sklearn

import sklr

project = "scikit-lr"
copyright = f"2019-{datetime.now(tz=UTC).year}, Juan Carlos Alfaro Jiménez"
# The short version, without the release level, and the full one
version = ".".join(sklr.__version__.split(".")[:2])
release = sklr.__version__

extensions = [
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_gallery.gen_gallery",
]

root_doc = "index"
templates_path = ["templates"]
exclude_patterns = ["_build", "templates", "sg_execution_times.rst"]

# Document the methods of a class on its own page, as the autosummary template
# does, instead of in a table of numpydoc with a separate page for each one
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

# scikit-lr is built against a single minor release of scikit-learn, so the
# links point to the documentation of that release
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": (
        "https://scikit-learn.org/{}.{}".format(*sklearn.__version__.split(".")),
        None,
    ),
}

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    # Link the names of scikit-lr in the code of the examples to the API
    # reference, and list the examples that use each one on its page
    "doc_module": ("sklr",),
    "reference_url": {"sklr": None},
    "backreferences_dir": "modules/generated",
}

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/alfaro96/scikit-lr",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/scikit-lr",
            "icon": "fa-brands fa-python",
        },
    ],
}
