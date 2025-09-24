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
    ModelProvider.ANTHROPIC: "claude-sonnet-4-0",
    ModelProvider.TOGETHER: "deepseek-ai/DeepSeek-V3",
    ModelProvider.GROQ: "openai/gpt-oss-20b",
    ModelProvider.BEDROCK: "us.anthropic.claude-sonnet-4-20250514-v1:0",
    ModelProvider.COHERE: "command-a-03-2025",
    ModelProvider.GEMINI: "models/gemini-2.5-flash",
}

models_to_max_tokens = {
    "gpt-5": 128000,
    "gpt-5-mini": 128000,
    "gpt-4.1": 32768,
    "gpt-4o": 16384,
    "gpt-4.1-mini": 32768,
    "claude-sonnet-4-20250514": 64000,
    "claude-sonnet-4-0": 64000,
    "deepseek-ai/deepseek-v3": 8192,
    "models/gemini-2.5-flash": 65536,
    "models/gemini-2.5-flash-lite": 65536,
    "models/gemini-2.5-pro": 65536,
    "openai/gpt-oss-20b": 65536,
    "openai/gpt-oss-120b": 65536,
    "us.anthropic.claude-sonnet-4-20250514-v1:0": 64000,
    "command-a-03-2025": 8192,
}


def get_max_tokens(model_name: str, model_provider: str) -> int:
    """Get the maximum token limit for a given model name and provider."""
    if model_provider in [
        ModelProvider.GEMINI,
        ModelProvider.TOGETHER,
        ModelProvider.OPENAI,
        ModelProvider.ANTHROPIC,
        ModelProvider.GROQ,
        ModelProvider.BEDROCK,
        ModelProvider.COHERE,
    ]:
        # Try exact match first (case-insensitive)
        max_tokens = models_to_max_tokens.get(model_name.lower(), 16384)
    else:
        max_tokens = 8192
    return max_tokens


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
    max_tokens = get_max_tokens(model_name, model_provider)
    if model_provider == ModelProvider.OPENAI:
        additional_kwargs = (
            {"reasoning_effort": "minimal"} if model_name.startswith("gpt-5") else {}
        )
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
        import google.genai.types as google_types
        generation_config = google_types.GenerateContentConfig(
            temperature=0.0,
            seed=123,
            max_output_tokens=max_tokens,
            thinking_config=google_types.ThinkingConfig(thinking_budget=0, include_thoughts=False),
        )
        llm = GoogleGenAI(
            model=model_name,
            temperature=0,
            is_function_calling_model=True,
            max_tokens=max_tokens,
            generation_config=generation_config,
            context_window=1_000_000,
        )
    elif model_provider == ModelProvider.TOGETHER:
        try:
            from llama_index.llms.together import TogetherLLM
        except ImportError as e:
            raise ImportError(
                "together not available. Install with: pip install llama-index-llms-together"
            ) from e
        additional_kwargs = {"seed": 42}
        if model_name in [
            "deepseek-ai/DeepSeek-V3.1",
            "deepseek-ai/DeepSeek-R1", "Qwen/Qwen3-235B-A22B-Thinking-2507"
            "openai/gpt-oss-120b", "openai/gpt-oss-20b",
        ]:
            additional_kwargs['reasoning_effort'] = "low"
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
        if (
            not config
            or not config.private_llm_api_base
            or not config.private_llm_api_key
        ):
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
