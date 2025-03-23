"""
This module contains the types used in the Vectara Agentic.
"""
from enum import Enum

from llama_index.core.tools.types import ToolOutput as LI_ToolOutput
from llama_index.core.chat_engine.types import AgentChatResponse as LI_AgentChatResponse
from llama_index.core.chat_engine.types import StreamingAgentChatResponse as LI_StreamingAgentChatResponse

class AgentType(Enum):
    """Enumeration for different types of agents."""

    REACT = "REACT"
    OPENAI = "OPENAI"
    FUNCTION_CALLING = "FUNCTION_CALLING"
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
    BEDROCK = "BEDROCK"
    PRIVATE = "PRIVATE"


class AgentStatusType(Enum):
    """Enumeration for different types of agent statuses."""

    AGENT_UPDATE = "agent_update"
    TOOL_CALL = "tool_call"
    TOOL_OUTPUT = "tool_output"
    AGENT_STEP = "agent_step"


class LLMRole(Enum):
    """Enumeration for different types of LLM roles."""

    MAIN = "MAIN"
    TOOL = "TOOL"


class ToolType(Enum):
    """Enumeration for different types of tools."""
    QUERY = "query"
    ACTION = "action"

class AgentConfigType(Enum):
    """Enumeration for different types of agent configurations."""
    DEFAULT = "default"
    FALLBACK = "fallback"


# classes for Agent responses
ToolOutput = LI_ToolOutput
AgentResponse = LI_AgentChatResponse
AgentStreamingResponse = LI_StreamingAgentChatResponse
