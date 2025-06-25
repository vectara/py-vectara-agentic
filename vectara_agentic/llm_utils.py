"""
Utilities for the Vectara agentic.
"""

from typing import Tuple, Callable, Optional
import os
from functools import lru_cache
import tiktoken

from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic

# Optional provider imports with graceful fallback
try:
    from llama_index.llms.google_genai import GoogleGenAI
except ImportError:
    GoogleGenAI = None

try:
    from llama_index.llms.together import TogetherLLM
except ImportError:
    TogetherLLM = None

try:
    from llama_index.llms.groq import Groq
except ImportError:
    Groq = None

try:
    from llama_index.llms.fireworks import Fireworks
except ImportError:
    Fireworks = None

try:
    from llama_index.llms.bedrock_converse import BedrockConverse
except ImportError:
    BedrockConverse = None

try:
    from llama_index.llms.cohere import Cohere
except ImportError:
    Cohere = None

try:
    from llama_index.llms.openai_like import OpenAILike
except ImportError:
    OpenAILike = None

from .types import LLMRole, AgentType, ModelProvider
from .agent_config import AgentConfig

provider_to_default_model_name = {
    ModelProvider.OPENAI: "gpt-4.1",
    ModelProvider.ANTHROPIC: "claude-sonnet-4-20250514",
    ModelProvider.TOGETHER: "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    ModelProvider.GROQ: "deepseek-r1-distill-llama-70b",
    ModelProvider.FIREWORKS: "accounts/fireworks/models/firefunction-v2",
    ModelProvider.BEDROCK: "us.anthropic.claude-sonnet-4-20250514-v1:0",
    ModelProvider.COHERE: "command-a-03-2025",
    ModelProvider.GEMINI: "models/gemini-2.5-flash",
}

DEFAULT_MODEL_PROVIDER = ModelProvider.OPENAI


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

    # If the agent type is OpenAI, check that the main LLM provider is also OpenAI.
    if role == LLMRole.MAIN and config.agent_type == AgentType.OPENAI:
        if model_provider != ModelProvider.OPENAI:
            raise ValueError(
                "OpenAI agent requested but main model provider is not OpenAI."
            )

    return model_provider, model_name


@lru_cache(maxsize=None)
def get_tokenizer_for_model(
    role: LLMRole, config: Optional[AgentConfig] = None
) -> Optional[Callable]:
    """
    Get the tokenizer for the specified model, as determined by the role & config.
    """
    model_name = "Unknown model"
    try:
        model_provider, model_name = _get_llm_params_for_role(role, config)
        if model_provider == ModelProvider.OPENAI:
            return tiktoken.encoding_for_model('gpt-4o').encode
        if model_provider == ModelProvider.ANTHROPIC:
            return Anthropic().tokenizer
    except Exception:
        print(f"Error getting tokenizer for model {model_name}, ignoring")
        return None
    return None


@lru_cache(maxsize=None)
def get_llm(role: LLMRole, config: Optional[AgentConfig] = None) -> LLM:
    """
    Get the LLM for the specified role, using the provided config
    or a default if none is provided.
    """
    max_tokens = 8192
    model_provider, model_name = _get_llm_params_for_role(role, config)
    if model_provider == ModelProvider.OPENAI:
        llm = OpenAI(
            model=model_name,
            temperature=0,
            is_function_calling_model=True,
            strict=True,
            max_tokens=max_tokens,
            pydantic_program_mode="openai",
        )
    elif model_provider == ModelProvider.ANTHROPIC:
        llm = Anthropic(
            model=model_name,
            temperature=0,
            max_tokens=max_tokens,
        )
    elif model_provider == ModelProvider.GEMINI:
        if GoogleGenAI is None:
            raise ImportError("google_genai not available. Install with: pip install llama-index-llms-google-genai")
        llm = GoogleGenAI(
            model=model_name,
            temperature=0,
            is_function_calling_model=True,
            allow_parallel_tool_calls=True,
            max_tokens=max_tokens,
        )
    elif model_provider == ModelProvider.TOGETHER:
        if TogetherLLM is None:
            raise ImportError("together not available. Install with: pip install llama-index-llms-together")
        llm = TogetherLLM(
            model=model_name,
            temperature=0,
            is_function_calling_model=True,
            max_tokens=max_tokens,
        )
    elif model_provider == ModelProvider.GROQ:
        if Groq is None:
            raise ImportError("groq not available. Install with: pip install llama-index-llms-groq")
        llm = Groq(
            model=model_name,
            temperature=0,
            is_function_calling_model=True,
            max_tokens=max_tokens,
        )
    elif model_provider == ModelProvider.FIREWORKS:
        if Fireworks is None:
            raise ImportError("fireworks not available. Install with: pip install llama-index-llms-fireworks")
        llm = Fireworks(model=model_name, temperature=0, max_tokens=max_tokens)
    elif model_provider == ModelProvider.BEDROCK:
        if BedrockConverse is None:
            raise ImportError("bedrock_converse not available. Install with: pip install llama-index-llms-bedrock")
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
        if Cohere is None:
            raise ImportError("cohere not available. Install with: pip install llama-index-llms-cohere")
        llm = Cohere(model=model_name, temperature=0, max_tokens=max_tokens)
    elif model_provider == ModelProvider.PRIVATE:
        if OpenAILike is None:
            raise ImportError("openai_like not available. Install with: pip install llama-index-llms-openai-like")
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
    return llm
