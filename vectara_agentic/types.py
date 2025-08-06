"""
This module contains the types used in the Vectara Agentic.
"""

from enum import Enum
from typing import Any, Dict, Callable, AsyncIterator, Protocol, cast
from dataclasses import dataclass, field

from llama_index.core.schema import Document as LI_Document
from llama_index.core.tools.types import ToolOutput as LI_ToolOutput
from llama_index.core.chat_engine.types import (
    AgentChatResponse as LI_AgentChatResponse,
)


class AgentType(Enum):
    """Enumeration for different types of agents."""

    REACT = "REACT"
    FUNCTION_CALLING = "FUNCTION_CALLING"


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


class HumanReadableOutput(Protocol):
    """Protocol for tool outputs that can provide human-readable representations."""

    def to_human_readable(self) -> str:
        """Convert the output to a human-readable format."""

    def get_raw_output(self) -> Any:
        """Get the raw output data."""


class _StreamProto(Protocol):
    """What we actually use from the LI streaming response."""

    response: str | None
    response_id: str | None

    def get_response(self) -> LI_AgentChatResponse:
        """Get the agent chat response."""

    def to_response(self) -> LI_AgentChatResponse:
        """Convert to agent chat response."""

    def get_final_response(self) -> LI_AgentChatResponse:
        """Get the final agent chat response."""

    def __iter__(self):
        """Return an iterator over the stream."""

    async def async_response_gen(self) -> AsyncIterator[str]:
        """Async generator that yields response strings."""


# classes for Agent responses
ToolOutput = LI_ToolOutput
AgentResponse = LI_AgentChatResponse
Document = LI_Document


@dataclass
class AgentStreamingResponse:
    """Our stream wrapper with writable metadata and a typed get_response()."""

    base: _StreamProto
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str):
        return getattr(self.base, name)

    def get_response(self) -> AgentResponse:
        """Get the response from the base stream, merging metadata."""
        # choose whichever method the base actually has
        if hasattr(self.base, "get_response"):
            resp = cast(AgentResponse, self.base.get_response())
        elif hasattr(self.base, "to_response"):
            resp = cast(AgentResponse, self.base.to_response())
        else:
            resp = cast(AgentResponse, self.base.get_final_response())

        resp.metadata = (resp.metadata or {}) | self.metadata
        return resp

    async def aget_response(self) -> AgentResponse:
        """Get the response from the base stream, merging metadata (async version)."""
        # prefer async version if available
        if hasattr(self.base, "aget_response"):
            resp = cast(AgentResponse, await self.base.aget_response())
        elif hasattr(self.base, "get_response"):
            resp = cast(AgentResponse, self.base.get_response())
        elif hasattr(self.base, "to_response"):
            resp = cast(AgentResponse, self.base.to_response())
        elif hasattr(self.base, "get_final_response"):
            resp = cast(AgentResponse, self.base.get_final_response())
        else:
            # Fallback for StreamingAgentChatResponse objects that don't have standard methods
            # Try to get the response directly from the object's response attribute
            if hasattr(self.base, "response"):
                response_text = self.base.response if isinstance(self.base.response, str) else str(self.base.response)
                resp = AgentResponse(response=response_text, metadata=getattr(self.base, "metadata", {}))
            else:
                resp = AgentResponse(response="", metadata={})

        resp.metadata = (resp.metadata or {}) | self.metadata
        return resp

    @property
    def async_response_gen(self) -> Callable[[], AsyncIterator[str]]:
        """Get the async response generator from the base stream."""
        return self.base.async_response_gen

    @async_response_gen.setter
    def async_response_gen(self, fn: Callable[[], AsyncIterator[str]]):
        """Set the async response generator for the base stream."""
        self.base.async_response_gen = fn

    @classmethod
    def from_error(cls, msg: str) -> "AgentStreamingResponse":
        """Create an AgentStreamingResponse from an error message."""

        async def _empty_gen():
            if False:  # pylint: disable=using-constant-test
                yield ""

        class _ErrStream:
            def __init__(self, msg: str):
                self.response = msg
                self.response_id = None

            async def async_response_gen(self):
                """Async generator that yields an error message."""
                if False:  # pylint: disable=using-constant-test
                    yield ""

            def get_response(self) -> AgentResponse:
                """Return an AgentResponse with the error message."""
                return AgentResponse(response=self.response, metadata={})

        return cls(base=_ErrStream(msg), metadata={})
