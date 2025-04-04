"""
Utilities for the Vectara agentic.
"""

from typing import Tuple, Callable, Optional
from functools import lru_cache
from inspect import signature
import json
import asyncio
import tiktoken
import aiohttp

from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic

from .types import LLMRole, AgentType, ModelProvider
from .agent_config import AgentConfig

provider_to_default_model_name = {
    ModelProvider.OPENAI: "gpt-4o",
    ModelProvider.ANTHROPIC: "claude-3-7-sonnet-20250219",
    ModelProvider.TOGETHER: "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    ModelProvider.GROQ: "llama-3.3-70b-versatile",
    ModelProvider.FIREWORKS: "accounts/fireworks/models/firefunction-v2",
    ModelProvider.BEDROCK: "anthropic.claude-3-5-sonnet-20241022-v2:0",
    ModelProvider.COHERE: "command-r-plus",
    ModelProvider.GEMINI: "models/gemini-2.0-flash",
}

DEFAULT_MODEL_PROVIDER = ModelProvider.OPENAI

@lru_cache(maxsize=None)
def _get_llm_params_for_role(
    role: LLMRole,
    config: Optional[AgentConfig] = None
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
        model_name = (
            config.tool_llm_model_name
            or provider_to_default_model_name.get(model_provider)
        )
    else:
        model_provider = ModelProvider(config.main_llm_provider)
        model_name = (
            config.main_llm_model_name
            or provider_to_default_model_name.get(model_provider)
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
    role: LLMRole,
    config: Optional[AgentConfig] = None
) -> Optional[Callable]:
    """
    Get the tokenizer for the specified model, as determined by the role & config.
    """
    model_provider, model_name = _get_llm_params_for_role(role, config)
    if model_provider == ModelProvider.OPENAI:
        # This might raise an exception if the model_name is unknown to tiktoken
        return tiktoken.encoding_for_model(model_name).encode
    if model_provider == ModelProvider.ANTHROPIC:
        return Anthropic().tokenizer
    return None


@lru_cache(maxsize=None)
def get_llm(
    role: LLMRole,
    config: Optional[AgentConfig] = None
) -> LLM:
    """
    Get the LLM for the specified role, using the provided config
    or a default if none is provided.
    """
    max_tokens = 8192
    model_provider, model_name = _get_llm_params_for_role(role, config)
    if model_provider == ModelProvider.OPENAI:
        llm = OpenAI(model=model_name, temperature=0,
                     is_function_calling_model=True,
                     strict=True,
                     max_tokens=max_tokens
            )
    elif model_provider == ModelProvider.ANTHROPIC:
        llm = Anthropic(
            model=model_name, temperature=0,
            max_tokens=max_tokens, cache_idx=2,
        )
    elif model_provider == ModelProvider.GEMINI:
        from llama_index.llms.gemini import Gemini
        llm = Gemini(
            model=model_name, temperature=0,
            is_function_calling_model=True,
            allow_parallel_tool_calls=True,
            max_tokens=max_tokens,
        )
    elif model_provider == ModelProvider.TOGETHER:
        from llama_index.llms.together import TogetherLLM
        llm = TogetherLLM(
            model=model_name, temperature=0,
            is_function_calling_model=True,
            max_tokens=max_tokens
        )
    elif model_provider == ModelProvider.GROQ:
        from llama_index.llms.groq import Groq
        llm = Groq(
            model=model_name, temperature=0,
            is_function_calling_model=True, max_tokens=max_tokens
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
        llm = OpenAILike(model=model_name, temperature=0, is_function_calling_model=True,is_chat_model=True,
                         api_base=config.private_llm_api_base, api_key=config.private_llm_api_key,
                         max_tokens=max_tokens)
    else:
        raise ValueError(f"Unknown LLM provider: {model_provider}")
    return llm

def is_float(value: str) -> bool:
    """Check if a string can be converted to a float."""
    try:
        float(value)
        return True
    except ValueError:
        return False

def remove_self_from_signature(func):
    """Decorator to remove 'self' from a method's signature for introspection."""
    sig = signature(func)
    params = list(sig.parameters.values())
    # Remove the first parameter if it is named 'self'
    if params and params[0].name == "self":
        params = params[1:]
    new_sig = sig.replace(parameters=params)
    func.__signature__ = new_sig
    return func

async def summarize_vectara_document(corpus_key: str, vectara_api_key: str, doc_id: str) -> str:
    """
    Summarize a document in a Vectara corpus using the Vectara API.
    """
    url = f"https://api.vectara.io/v2/corpora/{corpus_key}/documents/{doc_id}/summarize"

    payload = json.dumps({
        "llm_name": "gpt-4o",
        "model_parameters": {},
        "stream_response": False
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'x-api-key': vectara_api_key
    }
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, headers=headers, data=payload) as response:
            if response.status != 200:
                error_json = await response.json()
                return (
                    f"Vectara Summarization failed with error code {response.status}, "
                    f"error={error_json['messages'][0]}"
                )
            data = await response.json()
            return data["summary"]
    return json.loads(response.text)["summary"]

async def summarize_documents(
    vectara_corpus_key: str,
    vectara_api_key: str,
    doc_ids: list[str]
) -> dict[str, str]:
    """
    Summarize multiple documents in a Vectara corpus using the Vectara API.
    """
    if not doc_ids:
        return {}
    tasks = [
        summarize_vectara_document(vectara_corpus_key, vectara_api_key, doc_id)
        for doc_id in doc_ids
    ]
    summaries = await asyncio.gather(*tasks, return_exceptions=True)
    return dict(zip(doc_ids, summaries))
