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
copyright = "2024, Institute for Information Transmission Problems of the Russian Academy of Sciences"
author = "Arseniy Terekhin"

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

from ballfish import __version__ as ballfish_version

# https://www.sphinx-doc.org/en/master/usage/configuration.html
# If your project does not draw a meaningful distinction between between a
# ‘full’ and ‘major’ version, set both version and release to the same value.
version = release = ballfish_version

from doc.gen_images import generate_images

print("generating images")
generate_images()
print("done")
