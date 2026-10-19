project = "isotomics"
author = "Timothy Csernica, Sarah S. Zeichner"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
]

autosummary_generate = True
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = ["_static"]

# Tutorials are notebooks rendered with their stored outputs (run them locally to update)
nb_execution_mode = "off"
nb_output_stderr = "remove"

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "deflist",
]
