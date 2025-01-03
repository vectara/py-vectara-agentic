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
        default_factory=lambda: os.getenv("VECTARA_AGENTIC_MAIN_LLM_MODEL_NAME", "")
    )

    # Tool LLM provider & model name
    tool_llm_provider: ModelProvider = field(
        default_factory=lambda: ModelProvider(
            os.getenv("VECTARA_AGENTIC_TOOL_LLM_PROVIDER", ModelProvider.OPENAI.value)
        )
    )
    tool_llm_model_name: str = field(
        default_factory=lambda: os.getenv("VECTARA_AGENTIC_TOOL_LLM_MODEL_NAME", "")
    )

    # Example Vectara-related configs
    vectara_customer_id: str = field(
        default_factory=lambda: os.getenv("VECTARA_CUSTOMER_ID", "")
    )
    vectara_corpus_id: str = field(
        default_factory=lambda: os.getenv("VECTARA_CORPUS_ID", "")
    )
    vectara_api_key: str = field(
        default_factory=lambda: os.getenv("VECTARA_API_KEY", "")
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
