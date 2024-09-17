"""
Utilities for the Vectara agentic.
"""

import os

from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.together import TogetherLLM
from llama_index.llms.groq import Groq
from llama_index.llms.fireworks import Fireworks
from llama_index.llms.cohere import Cohere

import tiktoken
from typing import Tuple, Callable, Optional

from .types import LLMRole, AgentType, ModelProvider

provider_to_default_model_name = {
    ModelProvider.OPENAI: "gpt-4o-2024-08-06",
    ModelProvider.ANTHROPIC: "claude-3-5-sonnet-20240620",
    ModelProvider.TOGETHER: "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    ModelProvider.GROQ: "llama-3.1-70b-versatile",
    ModelProvider.FIREWORKS: "accounts/fireworks/models/firefunction-v2",
    ModelProvider.COHERE: "command-r-plus",
}

DEFAULT_MODEL_PROVIDER = ModelProvider.OPENAI


def _get_llm_params_for_role(role: LLMRole) -> Tuple[ModelProvider, str]:
    """Get the model provider and model name for the specified role."""
    if role == LLMRole.TOOL:
        model_provider = ModelProvider(
            os.getenv("VECTARA_AGENTIC_TOOL_LLM_PROVIDER", DEFAULT_MODEL_PROVIDER)
        )
        model_name = os.getenv(
            "VECTARA_AGENTIC_TOOL_MODEL_NAME",
            provider_to_default_model_name.get(model_provider),
        )
    else:
        model_provider = ModelProvider(
            os.getenv("VECTARA_AGENTIC_MAIN_LLM_PROVIDER", DEFAULT_MODEL_PROVIDER)
        )
        model_name = os.getenv(
            "VECTARA_AGENTIC_MAIN_MODEL_NAME",
            provider_to_default_model_name.get(model_provider),
        )

    agent_type = AgentType(os.getenv("VECTARA_AGENTIC_AGENT_TYPE", AgentType.OPENAI))
    if (
        role == LLMRole.MAIN
        and agent_type == AgentType.OPENAI
        and model_provider != ModelProvider.OPENAI
    ):
        raise ValueError(
            "OpenAI agent requested but main model provider is not OpenAI."
        )

    return model_provider, model_name


def get_tokenizer_for_model(role: LLMRole) -> Optional[Callable]:
    """Get the tokenizer for the specified model."""
    model_provider, model_name = _get_llm_params_for_role(role)
    if model_provider == ModelProvider.OPENAI:
        return tiktoken.encoding_for_model(model_name).encode
    elif model_provider == ModelProvider.ANTHROPIC:
        return Anthropic().tokenizer
    else:
        return None


def get_llm(role: LLMRole) -> LLM:
    """Get the LLM for the specified role."""
    model_provider, model_name = _get_llm_params_for_role(role)

    if model_provider == ModelProvider.OPENAI:
        llm = OpenAI(model=model_name, temperature=0)
    elif model_provider == ModelProvider.ANTHROPIC:
        llm = Anthropic(model=model_name, temperature=0)
    elif model_provider == ModelProvider.TOGETHER:
        llm = TogetherLLM(model=model_name, temperature=0)
    elif model_provider == ModelProvider.GROQ:
        llm = Groq(model=model_name, temperature=0)
    elif model_provider == ModelProvider.FIREWORKS:
        llm = Fireworks(model=model_name, temperature=0)
    elif model_provider == ModelProvider.COHERE:
        llm = Cohere(model=model_name, temperature=0)
    else:
        raise ValueError(f"Unknown LLM provider: {model_provider}")

    return llm
