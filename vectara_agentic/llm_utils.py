"""
Utilities for the Vectara agentic.
"""
from types import MethodType
from typing import Tuple, Callable, Optional
from functools import lru_cache
import tiktoken

from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic

from .types import LLMRole, AgentType, ModelProvider
from .agent_config import AgentConfig
from .tool_utils import _updated_openai_prepare_chat_with_tools

provider_to_default_model_name = {
    ModelProvider.OPENAI: "gpt-4o",
    ModelProvider.ANTHROPIC: "claude-3-7-sonnet-latest",
    ModelProvider.TOGETHER: "Qwen/Qwen2.5-72B-Instruct-Turbo",
    ModelProvider.GROQ: "meta-llama/llama-4-scout-17b-16e-instruct",
    ModelProvider.FIREWORKS: "accounts/fireworks/models/firefunction-v2",
    ModelProvider.BEDROCK: "anthropic.claude-3-7-sonnet-20250219-v1:0",
    ModelProvider.COHERE: "command-a-03-2025",
    ModelProvider.GEMINI: "models/gemini-2.0-flash",
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
    try:
        model_provider, model_name = _get_llm_params_for_role(role, config)
        if model_provider == ModelProvider.OPENAI:
            # This might raise an exception if the model_name is unknown to tiktoken
            return tiktoken.encoding_for_model(model_name).encode
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
        from llama_index.llms.google_genai import GoogleGenAI

        llm = GoogleGenAI(
            model=model_name,
            temperature=0,
            is_function_calling_model=True,
            allow_parallel_tool_calls=True,
            max_tokens=max_tokens,
        )
    elif model_provider == ModelProvider.TOGETHER:
        from llama_index.llms.together import TogetherLLM

        llm = TogetherLLM(
            model=model_name,
            temperature=0,
            is_function_calling_model=True,
            max_tokens=max_tokens,
        )
        # pylint: disable=protected-access
        llm._prepare_chat_with_tools = MethodType(
            _updated_openai_prepare_chat_with_tools,
            llm,
        )
    elif model_provider == ModelProvider.GROQ:
        from llama_index.llms.groq import Groq

        llm = Groq(
            model=model_name,
            temperature=0,
            is_function_calling_model=True,
            max_tokens=max_tokens,
        )
        # pylint: disable=protected-access
        llm._prepare_chat_with_tools = MethodType(
            _updated_openai_prepare_chat_with_tools,
            llm,
        )
    elif model_provider == ModelProvider.FIREWORKS:
        from llama_index.llms.fireworks import Fireworks

        llm = Fireworks(model=model_name, temperature=0, max_tokens=max_tokens)
    elif model_provider == ModelProvider.BEDROCK:
        from llama_index.llms.bedrock import Bedrock

        llm = Bedrock(model=model_name, temperature=0, max_tokens=max_tokens)
    elif model_provider == ModelProvider.COHERE:
        from llama_index.llms.cohere import Cohere

        llm = Cohere(model=model_name, temperature=0, max_tokens=max_tokens)
    elif model_provider == ModelProvider.PRIVATE:
        from llama_index.llms.openai_like import OpenAILike

        llm = OpenAILike(
            model=model_name,
            temperature=0,
            is_function_calling_model=True,
            is_chat_model=True,
            api_base=config.private_llm_api_base,
            api_key=config.private_llm_api_key,
            max_tokens=max_tokens,
        )
        # pylint: disable=protected-access
        llm._prepare_chat_with_tools = MethodType(
            _updated_openai_prepare_chat_with_tools,
            llm,
        )

    else:
        raise ValueError(f"Unknown LLM provider: {model_provider}")
    return llm
