"""
This module contains the types used in the Vectara Agentic.
"""
from enum import Enum


class AgentType(Enum):
    """Enumeration for different types of agents."""

    REACT = "REACT"
    OPENAI = "OPENAI"
    LLMCOMPILER = "LLMCOMPILER"
    LATS = "LATS"

class ObserverType(Enum):
    """Enumeration for different types of observability integrations."""

    NO_OBSERVER = "NO_OBSERVER"
    ARIZE_PHOENIX = "ARIZE_PHOENIX"


class ModelProvider(Enum):
    """Enumeration for different types of model providers."""

    OPENAI = "OPENAI"
    ANTHROPIC = "ANTHROPIC"
    TOGETHER = "TOGETHER"
    GROQ = "GROQ"
    FIREWORKS = "FIREWORKS"
    COHERE = "COHERE"
    GEMINI = "GEMINI"


class AgentStatusType(Enum):
    """Enumeration for different types of agent statuses."""

    AGENT_UPDATE = "agent_update"
    TOOL_CALL = "tool_call"
    TOOL_OUTPUT = "tool_output"
    AGENT_STEP = "agent_step"


class LLMRole(Enum):
    """Enumeration for different types of LLM roles."""

    MAIN: str = "MAIN"
    TOOL: str = "TOOL"


class ToolType(Enum):
    """Enumeration for different types of tools."""
    QUERY = "query"
    ACTION = "action"
