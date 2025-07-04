# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(".."))  # So autodoc can find your modules

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DMD_toolbox'
copyright = '2025, Gabriel Seigneur'
author = 'Gabriel Seigneur'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',       # For Google/Numpy docstring styles
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',  # Optional, nicer type hints
    'myst_nb',                   # Parse notebooks and markdown
]
templates_path = ['_templates']
autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "icon_links": [],
    "use_edit_page_button": False,
}
# MyST-NB options (for notebooks/Markdown)
nb_execution_mode = "off"  # Optional: don’t re-run notebooks on build

html_static_path = ['_static']

# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

