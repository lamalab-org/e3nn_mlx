"""Sphinx configuration for the e3nn-mlx documentation."""

from __future__ import annotations

from importlib.metadata import version


project = "e3nn-mlx"
author = "e3nn-mlx contributors"
copyright = "2026, lamalab-org"
release = version("e3nn-mlx")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autosectionlabel_prefix_document = True
myst_enable_extensions = ["colon_fence", "dollarmath", "fieldlist"]
myst_heading_anchors = 3

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]
templates_path = ["_templates"]
html_static_path = ["_static"]
html_theme = "furo"
html_title = f"e3nn-mlx {release}"
html_theme_options = {
    "source_repository": "https://github.com/lamalab-org/e3nn_mlx/",
    "source_branch": "main",
    "source_directory": "docs/",
}

nitpicky = False
suppress_warnings = ["myst.header"]
