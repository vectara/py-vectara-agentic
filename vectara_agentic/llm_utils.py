"""
Utilities for the Vectara agentic.
"""

from typing import Tuple, Optional
import os
from functools import lru_cache
import hashlib

from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic

# LLM provider imports are now lazy-loaded in get_llm() function

from .types import LLMRole, ModelProvider
from .agent_config import AgentConfig

provider_to_default_model_name = {
    ModelProvider.OPENAI: "gpt-4.1-mini",
    ModelProvider.ANTHROPIC: "claude-sonnet-4-20250514",
    ModelProvider.TOGETHER: "deepseek-ai/DeepSeek-V3",
    ModelProvider.GROQ: "openai/gpt-oss-20b",
    ModelProvider.BEDROCK: "us.anthropic.claude-sonnet-4-20250514-v1:0",
    ModelProvider.COHERE: "command-a-03-2025",
    ModelProvider.GEMINI: "models/gemini-2.5-flash-lite",
}

DEFAULT_MODEL_PROVIDER = ModelProvider.OPENAI

# Manual cache for LLM instances to handle mutable AgentConfig objects
_llm_cache = {}


def _create_llm_cache_key(role: LLMRole, config: Optional[AgentConfig] = None) -> str:
    """Create a hash-based cache key for LLM instances."""
    if config is None:
        config = AgentConfig()

    # Extract only the relevant config parameters for the cache key
    cache_data = {
        "role": role.value,
        "main_llm_provider": config.main_llm_provider.value,
        "main_llm_model_name": config.main_llm_model_name,
        "tool_llm_provider": config.tool_llm_provider.value,
        "tool_llm_model_name": config.tool_llm_model_name,
        "private_llm_api_base": config.private_llm_api_base,
        "private_llm_api_key": config.private_llm_api_key,
    }

    # Create a stable hash from the cache data
    cache_str = str(sorted(cache_data.items()))
    return hashlib.md5(cache_str.encode()).hexdigest()


@lru_cache(maxsize=None)
def _get_llm_params_for_role(
    role: LLMRole, config: Optional[AgentConfig] = None
) -> Tuple[ModelProvider, str]:
    """
    Get the model provider and model name for the specified role.

    If config is None, a new AgentConfig() is instantiated using environment defaults.
    """
    config = config or AgentConfig()  # fallback to default config

    if role == LLMRole.TOOL:
        model_provider = ModelProvider(config.tool_llm_provider)
        # If the user hasn’t explicitly set a tool_llm_model_name,
        # fallback to provider default from provider_to_default_model_name
        model_name = config.tool_llm_model_name or provider_to_default_model_name.get(
            model_provider
        )
    else:
        model_provider = ModelProvider(config.main_llm_provider)
        model_name = config.main_llm_model_name or provider_to_default_model_name.get(
            model_provider
        )

    return model_provider, model_name


def get_llm(role: LLMRole, config: Optional[AgentConfig] = None) -> LLM:
    """
    Get the LLM for the specified role, using the provided config
    or a default if none is provided.

    Uses a cache based on configuration parameters to avoid repeated LLM instantiation.
    """
    if config is None:
        config = AgentConfig()
    # Check cache first
    cache_key = _create_llm_cache_key(role, config)
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]
    model_provider, model_name = _get_llm_params_for_role(role, config)
    max_tokens = (
        16384
        if model_provider
        in [
            ModelProvider.GEMINI,
            ModelProvider.TOGETHER,
            ModelProvider.OPENAI,
            ModelProvider.ANTHROPIC,
        ]
        else 8192
    )
    if model_provider == ModelProvider.OPENAI:
        additional_kwargs = {"reasoning_effort": "minimal"} if model_name.startswith("gpt-5") else {}
        llm = OpenAI(
            model=model_name,
            temperature=0,
            is_function_calling_model=True,
            strict=False,
            max_tokens=max_tokens,
            pydantic_program_mode="openai",
            additional_kwargs=additional_kwargs,
        )
    elif model_provider == ModelProvider.ANTHROPIC:
        llm = Anthropic(
            model=model_name,
            temperature=0,
            max_tokens=max_tokens,
        )
    elif model_provider == ModelProvider.GEMINI:
        try:
            from llama_index.llms.google_genai import GoogleGenAI
        except ImportError as e:
            raise ImportError(
                "google_genai not available. Install with: pip install llama-index-llms-google-genai"
            ) from e
        llm = GoogleGenAI(
            model=model_name,
            temperature=0,
            is_function_calling_model=True,
            max_tokens=max_tokens,
        )
    elif model_provider == ModelProvider.TOGETHER:
        try:
            from llama_index.llms.together import TogetherLLM
        except ImportError as e:
            raise ImportError(
                "together not available. Install with: pip install llama-index-llms-together"
            ) from e
        additional_kwargs = {"reasoning_effort": "low"} if model_name.startswith("gpt-oss") else {}
        llm = TogetherLLM(
            model=model_name,
            temperature=0,
            is_function_calling_model=True,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs,
        )
    elif model_provider == ModelProvider.GROQ:
        try:
            from llama_index.llms.groq import Groq
        except ImportError as e:
            raise ImportError(
                "groq not available. Install with: pip install llama-index-llms-groq"
            ) from e
        llm = Groq(
            model=model_name,
            temperature=0,
            is_function_calling_model=True,
            max_tokens=max_tokens,
        )
    elif model_provider == ModelProvider.BEDROCK:
        try:
            from llama_index.llms.bedrock_converse import BedrockConverse
        except ImportError as e:
            raise ImportError(
                "bedrock_converse not available. Install with: pip install llama-index-llms-bedrock"
            ) from e
        aws_profile_name = os.getenv("AWS_PROFILE", None)
        aws_region = os.getenv("AWS_REGION", "us-east-2")

        llm = BedrockConverse(
            model=model_name,
            temperature=0,
            max_tokens=max_tokens,
            profile_name=aws_profile_name,
            region_name=aws_region,
        )
    elif model_provider == ModelProvider.COHERE:
        try:
            from llama_index.llms.cohere import Cohere
        except ImportError as e:
            raise ImportError(
                "cohere not available. Install with: pip install llama-index-llms-cohere"
            ) from e
        llm = Cohere(model=model_name, temperature=0, max_tokens=max_tokens)
    elif model_provider == ModelProvider.PRIVATE:
        try:
            from llama_index.llms.openai_like import OpenAILike
        except ImportError as e:
            raise ImportError(
                "openai_like not available. Install with: pip install llama-index-llms-openai-like"
            ) from e
        if not config or not config.private_llm_api_base or not config.private_llm_api_key:
            raise ValueError(
                "Private LLM requires both private_llm_api_base and private_llm_api_key to be set in AgentConfig."
            )
        llm = OpenAILike(
            model=model_name,
            temperature=0,
            is_function_calling_model=True,
            is_chat_model=True,
            api_base=config.private_llm_api_base,
            api_key=config.private_llm_api_key,
            max_tokens=max_tokens,
        )

    else:
        raise ValueError(f"Unknown LLM provider: {model_provider}")

    # Cache the created LLM instance
    _llm_cache[cache_key] = llm
    return llm
