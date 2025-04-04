"""
Define the AgentConfig dataclass for the Vectara Agentic utilities.
"""

import os
from dataclasses import dataclass, field
from .types import ModelProvider, AgentType, ObserverType

@dataclass(eq=True, frozen=True)
class AgentConfig:
    """
    Centralized configuration for the Vectara Agentic utilities.

    Each field can default to either a hard-coded value or an environment
    variable. For example, if you have environment variables you want to
    fall back on, you can default to them here.
    """

    # Agent type
    agent_type: AgentType = field(
        default_factory=lambda: AgentType(
            os.getenv("VECTARA_AGENTIC_AGENT_TYPE", AgentType.OPENAI.value)
        )
    )

    # Main LLM provider & model name
    main_llm_provider: ModelProvider = field(
        default_factory=lambda: ModelProvider(
            os.getenv("VECTARA_AGENTIC_MAIN_LLM_PROVIDER", ModelProvider.OPENAI.value)
        )
    )

    main_llm_model_name: str = field(
        default_factory=lambda: os.getenv("VECTARA_AGENTIC_MAIN_MODEL_NAME", "")
    )

    # Tool LLM provider & model name
    tool_llm_provider: ModelProvider = field(
        default_factory=lambda: ModelProvider(
            os.getenv("VECTARA_AGENTIC_TOOL_LLM_PROVIDER", ModelProvider.OPENAI.value)
        )
    )
    tool_llm_model_name: str = field(
        default_factory=lambda: os.getenv("VECTARA_AGENTIC_TOOL_MODEL_NAME", "")
    )

    # Params for Private LLM endpoint if used
    private_llm_api_base: str = field(
        default_factory=lambda: os.getenv("VECTARA_AGENTIC_PRIVATE_LLM_API_BASE",
                                          "http://private-endpoint.company.com:5000/v1")
    )
    private_llm_api_key: str = field(
        default_factory=lambda: os.getenv("VECTARA_AGENTIC_PRIVATE_LLM_API_KEY", "<private-api-key>")
    )

    # Observer
    observer: ObserverType = field(
        default_factory=lambda: ObserverType(
            os.getenv("VECTARA_AGENTIC_OBSERVER_TYPE", "NO_OBSERVER")
        )
    )

    # Endpoint API key
    endpoint_api_key: str = field(
        default_factory=lambda: os.getenv("VECTARA_AGENTIC_API_KEY", "dev-api-key")
    )

    # max reasoning steps
    # used for both OpenAI and React Agent types
    max_reasoning_steps: int = field(
        default_factory=lambda: int(os.getenv("VECTARA_AGENTIC_MAX_REASONING_STEPS", "50"))
    )

    def __post_init__(self):
        # Use object.__setattr__ since the dataclass is frozen
        if isinstance(self.agent_type, str):
            object.__setattr__(self, "agent_type", AgentType(self.agent_type))
        if isinstance(self.main_llm_provider, str):
            object.__setattr__(self, "main_llm_provider", ModelProvider(self.main_llm_provider))
        if isinstance(self.tool_llm_provider, str):
            object.__setattr__(self, "tool_llm_provider", ModelProvider(self.tool_llm_provider))
        if isinstance(self.observer, str):
            object.__setattr__(self, "observer", ObserverType(self.observer))

    def to_dict(self) -> dict:
        """
        Convert the AgentConfig to a dictionary.
        """
        return {
            "agent_type": self.agent_type.value,
            "main_llm_provider": self.main_llm_provider.value,
            "main_llm_model_name": self.main_llm_model_name,
            "tool_llm_provider": self.tool_llm_provider.value,
            "tool_llm_model_name": self.tool_llm_model_name,
            "observer": self.observer.value,
            "endpoint_api_key": self.endpoint_api_key,
            "max_reasoning_steps": self.max_reasoning_steps
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "AgentConfig":
        """
        Create an AgentConfig from a dictionary.
        """
        return cls(
            agent_type=AgentType(config_dict["agent_type"]),
            main_llm_provider=ModelProvider(config_dict["main_llm_provider"]),
            main_llm_model_name=config_dict["main_llm_model_name"],
            tool_llm_provider=ModelProvider(config_dict["tool_llm_provider"]),
            tool_llm_model_name=config_dict["tool_llm_model_name"],
            observer=ObserverType(config_dict["observer"]),
            endpoint_api_key=config_dict["endpoint_api_key"],
            max_reasoning_steps=config_dict["max_reasoning_steps"]
        )
