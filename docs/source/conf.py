# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../yambopy/'))


project = 'yambopy'
copyright = '2024, Henrique Miranda, Alejandro Molina Sánchez, Fulvio Paleari, Riccardo Reho'
author = 'Henrique Miranda, Alejandro Molina Sánchez, Fulvio Paleari, Riccardo Reho'
release = '3.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',  # Add napoleon to support Google style docstrings
]

templates_path = ['_templates']
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']

pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_sidebars = { '**': ['globaltoc.html','relations.html', 'sourcelink.html', 'searchbox.html'] }
#html_static_path = ['_static']

# -- Autodoc options ---------------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',  # Order by source to improve visualization
    'undoc-members': True,
    'show-inheritance': True,
}

autosummary_generate = True
# -- Napolean options --------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
