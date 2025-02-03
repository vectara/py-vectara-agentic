"""
vectara_agentic package.
"""

from .agent import Agent
from .tools import VectaraToolFactory, VectaraTool

# Define the __all__ variable for wildcard imports
__all__ = ['Agent', 'VectaraToolFactory', 'VectaraTool']

# Ensure package version is available
try:
    import importlib.metadata
    __version__ = importlib.metadata.version("vectara_agentic")
except Exception:
    __version__ = "0.0.0"  # fallback if not installed
