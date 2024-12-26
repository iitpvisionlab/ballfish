# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
from pathlib import Path

root = Path(__file__).parents[1]


project = "Ballfish"
copyright = "2024, iitpvisionlab"
author = "iitpvisionlab"
release = "unknown"
with (root / "pyproject.toml").open() as f:
    for line in f:
        if line.startswith("version = "):
            release = line.strip()[11:-1]
print(f'release: "{release}"')

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

sys.path.insert(0, str(root))

from doc.gen_images import generate_images

print("generating images")
generate_images()
print("done")
