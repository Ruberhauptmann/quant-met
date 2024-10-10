# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import importlib.metadata

project = "quant-met"
copyright = "2024, Tjark Sievers"
author = "Tjark Sievers"
version = importlib.metadata.version("quant-met")
language = "en"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx.ext.autosummary",
    "myst_parser",
    "sphinx.ext.intersphinx",
    "pydata_sphinx_theme",
    "nbsphinx",
    "sphinx_gallery.load_style",
    "sphinx_design",
]

intersphinx_mapping = {
    "h5py": ("https://docs.h5py.org/en/latest", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

autodoc_typehints = "none"

templates_path = ["_templates"]

html_sidebars = {
    "index": ["search-button-field"],
    "**": ["search-button-field", "sidebar-nav-bs"],
}

html_theme_options = {
    "github_url": "https://github.com/Ruberhauptmann/quant-met",
    "logo": {
        "text": "Quant-Met",
    },
    "navbar_end": ["theme-switcher", "version-switcher", "navbar-icon-links"],
    "navbar_persistent": [],
    "check_switcher": False,
    "switcher": {
        "version_match": version,
        "json_url": "https://quant-met.tjarksievers.de/en/docs-visuals/versions.json",
    },
}

html_show_sourcelink = False

# add_module_names = False
napoleon_numpy_docstring = True

add_function_parentheses = False
# modindex_common_prefix = ["quant-met."]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_extra_path = ["extra"]
