import os
import sys

# Add project root to path for autodoc
sys.path.insert(0, os.path.abspath('../..'))

# Project information
project = 'OCI Management App'
copyright = '2025, Your Name'
author = 'Your Name'
release = '0.1.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # For Google-style docstrings
]

templates_path = ['_templates']
exclude_patterns = []

# HTML output options
html_theme = 'alabaster'
html_static_path = ['_static']

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True