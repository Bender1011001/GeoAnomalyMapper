# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'GeoAnomalyMapper'
copyright = '2025, GAM Contributors'
author = 'GAM Contributors'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',  # For Google/NumPy docstrings
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'nbsphinx',  # For Jupyter notebooks
    'sphinx_copybutton',  # Copy code blocks
    'sphinxcontrib.mermaid',  # Mermaid diagrams
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints*']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'  # Modern theme
html_static_path = ['_static']

# Theme options
html_theme_options = {
    'sidebar_hide_name': False,
}

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
}
autodoc_mock_imports = ['simpeg', 'obspy', 'pygimli', 'pygmt', 'pyvista']  # Mock heavy deps for build

# -- Options for autosummary -------------------------------------------------
autosummary_generate = True

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'xarray': ('https://docs.xarray.dev/en/stable/', None),
    'dask': ('https://docs.dask.org/en/stable/', None),
}

# -- Napoleon (docstrings) ---------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# -- nbsphinx (notebooks) ----------------------------------------------------
nbsphinx_execute = 'never'  # Don't execute notebooks during build
nbsphinx_allow_errors = True

# -- Path setup --------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('../gam'))  # Add gam package to path for autodoc

# -- Build API docs ----------------------------------------------------------
# Run `sphinx-apidoc -o api/ ../gam/` manually to generate API RST files