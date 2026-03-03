project = "isotomics"
author = "Timothy Csernica, Sarah S. Zeichner"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
]

autosummary_generate = True
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = ["_static"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
