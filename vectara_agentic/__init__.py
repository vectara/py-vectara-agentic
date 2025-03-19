"""
vectara_agentic package.
"""

from .agent import Agent
from .tools import VectaraToolFactory, VectaraTool, ToolsFactory
from .tools_catalog import ToolsCatalog
from .agent_config import AgentConfig
from .agent_endpoint import create_app, start_app
from .types import (
    AgentType, ObserverType, ModelProvider, AgentStatusType, LLMRole, ToolType
)

# Define the __all__ variable for wildcard imports
__all__ = [
    'Agent', 'VectaraToolFactory', 'VectaraTool', 'ToolsFactory', 'AgentConfig',
    'create_app', 'start_app', 'ToolsCatalog',
    'AgentType', 'ObserverType', 'ModelProvider', 'AgentStatusType', 'LLMRole', 'ToolType'
]

# Ensure package version is available
try:
    import importlib.metadata
    __version__ = importlib.metadata.version("vectara_agentic")
except Exception:
    __version__ = "0.0.0"  # fallback if not installed
