"""
Agent factory functions for creating different types of agents.

This module provides specialized functions for creating various agent types
with proper configuration, prompt formatting, and structured planning setup.
"""

import os
import re
from datetime import date
from typing import List, Optional, Dict, Any

from llama_index.core.tools import FunctionTool
from llama_index.core.memory import Memory
from llama_index.core.callbacks import CallbackManager
from llama_index.core.agent.workflow import FunctionAgent, ReActAgent
from llama_index.core.agent import BaseWorkflowAgent

from pydantic import Field, create_model

from ..agent_config import AgentConfig
from ..types import AgentType
from .prompts import (
    REACT_PROMPT_TEMPLATE,
    GENERAL_PROMPT_TEMPLATE,
    get_general_instructions,
)
from ..tools import VectaraToolFactory
from .utils.schemas import PY_TYPES


def format_prompt(
    prompt_template: str,
    general_instructions: str,
    topic: str,
    custom_instructions: str,
) -> str:
    """
    Generate a prompt by replacing placeholders with topic and date.

    Args:
        prompt_template: The template for the prompt
        general_instructions: General instructions to be included in the prompt
        topic: The topic to be included in the prompt
        custom_instructions: The custom instructions to be included in the prompt

    Returns:
        str: The formatted prompt
    """
    return (
        prompt_template.replace("{chat_topic}", topic)
        .replace("{today}", date.today().strftime("%A, %B %d, %Y"))
        .replace("{custom_instructions}", custom_instructions)
        .replace("{INSTRUCTIONS}", general_instructions)
    )


def create_react_agent(
    tools: List[FunctionTool],
    llm,
    memory: Memory,
    config: AgentConfig,
    callback_manager: CallbackManager,
    general_instructions: str,
    topic: str,
    custom_instructions: str,
    verbose: bool = True,
) -> ReActAgent:
    """
    Create a ReAct (Reasoning and Acting) agent.

    Args:
        tools: List of tools available to the agent
        llm: Language model instance
        memory: Agent memory
        config: Agent configuration
        callback_manager: Callback manager for events
        general_instructions: General instructions for the agent
        topic: Topic expertise area
        custom_instructions: Custom user instructions
        verbose: Whether to enable verbose output

    Returns:
        ReActAgent: Configured ReAct agent
    """
    prompt = format_prompt(
        REACT_PROMPT_TEMPLATE,
        general_instructions,
        topic,
        custom_instructions,
    )

    # Create ReActAgent with correct parameters based on current LlamaIndex API
    # Note: ReActAgent is a workflow-based agent and doesn't have from_tools method
    return ReActAgent(
        tools=tools,
        llm=llm,
        system_prompt=prompt,
        verbose=verbose,
    )


def create_function_agent(
    tools: List[FunctionTool],
    llm,
    memory: Memory,
    config: AgentConfig,
    callback_manager: CallbackManager,
    general_instructions: str,
    topic: str,
    custom_instructions: str,
    verbose: bool = True,
    enable_parallel_tool_calls: bool = False,
) -> FunctionAgent:
    """
    Create a unified Function Calling agent.

    Modern workflow-based function calling agent implementation using LlamaIndex 0.13.0+ architecture.

    Args:
        tools: List of tools available to the agent
        llm: Language model instance
        memory: Agent memory (maintained via Context during agent execution)
        config: Agent configuration
        callback_manager: Callback manager for events (not directly supported by FunctionAgent)
        general_instructions: General instructions for the agent
        topic: Topic expertise area
        custom_instructions: Custom user instructions
        verbose: Whether to enable verbose output
        enable_parallel_tool_calls: Whether to enable parallel tool execution

    Returns:
        FunctionAgent: Configured Function Calling agent

    Notes:
        - Works with any LLM provider (OpenAI, Anthropic, Together, etc.)
        - Memory/state is managed via Context object during workflow execution
        - Parallel tool calls depend on LLM provider support
        - Modern workflow-based agent implementation using LlamaIndex 0.13.0+ architecture
    """
    prompt = format_prompt(
        GENERAL_PROMPT_TEMPLATE,
        general_instructions,
        topic,
        custom_instructions,
    )

    # Create FunctionAgent with correct parameters based on current LlamaIndex API
    # Note: FunctionAgent is a workflow-based agent and doesn't have from_tools method
    return FunctionAgent(
        tools=tools,
        llm=llm,
        system_prompt=prompt,
        verbose=verbose,
    )

def create_agent_from_config(
    tools: List[FunctionTool],
    llm,
    memory: Memory,
    config: AgentConfig,
    callback_manager: CallbackManager,
    general_instructions: str,
    topic: str,
    custom_instructions: str,
    verbose: bool = True,
    agent_type: Optional[AgentType] = None,  # For compatibility with existing interface
) -> BaseWorkflowAgent:
    """
    Create an agent based on configuration.

    This is the main factory function that delegates to specific agent creators
    based on the agent type in the configuration.

    Args:
        tools: List of tools available to the agent
        llm: Language model instance
        memory: Agent memory
        config: Agent configuration
        callback_manager: Callback manager for events
        general_instructions: General instructions for the agent
        topic: Topic expertise area
        custom_instructions: Custom user instructions
        verbose: Whether to enable verbose output
        agent_type: Override agent type (for backward compatibility)

    Returns:
        BaseWorkflowAgent: Configured agent

    Raises:
        ValueError: If unknown agent type is specified
    """
    # Use override agent type if provided, otherwise use config
    effective_agent_type = agent_type or config.agent_type

    # Create base agent based on type
    if effective_agent_type == AgentType.FUNCTION_CALLING:
        agent = create_function_agent(
            tools,
            llm,
            memory,
            config,
            callback_manager,
            general_instructions,
            topic,
            custom_instructions,
            verbose,
            enable_parallel_tool_calls=True,  # Enable parallel calls for FUNCTION_CALLING type
        )
    elif effective_agent_type == AgentType.REACT:
        agent = create_react_agent(
            tools,
            llm,
            memory,
            config,
            callback_manager,
            general_instructions,
            topic,
            custom_instructions,
            verbose,
        )
    else:
        raise ValueError(f"Unknown agent type: {effective_agent_type}")

    return agent


def create_agent_from_corpus(
    tool_name: str,
    data_description: str,
    assistant_specialty: str,
    general_instructions: Optional[str] = None,
    vectara_corpus_key: str = str(os.environ.get("VECTARA_CORPUS_KEY", "")),
    vectara_api_key: str = str(os.environ.get("VECTARA_API_KEY", "")),
    agent_config: AgentConfig = AgentConfig(),
    fallback_agent_config: Optional[AgentConfig] = None,
    verbose: bool = False,
    vectara_filter_fields: List[dict] = [],
    vectara_offset: int = 0,
    vectara_lambda_val: float = 0.005,
    vectara_semantics: str = "default",
    vectara_custom_dimensions: Dict = {},
    vectara_reranker: str = "slingshot",
    vectara_rerank_k: int = 50,
    vectara_rerank_limit: Optional[int] = None,
    vectara_rerank_cutoff: Optional[float] = None,
    vectara_diversity_bias: float = 0.2,
    vectara_udf_expression: Optional[str] = None,
    vectara_rerank_chain: Optional[List[Dict]] = None,
    vectara_n_sentences_before: int = 2,
    vectara_n_sentences_after: int = 2,
    vectara_summary_num_results: int = 10,
    vectara_summarizer: str = "vectara-summary-ext-24-05-med-omni",
    vectara_summary_response_language: str = "eng",
    vectara_summary_prompt_text: Optional[str] = None,
    vectara_max_response_chars: Optional[int] = None,
    vectara_max_tokens: Optional[int] = None,
    vectara_temperature: Optional[float] = None,
    vectara_frequency_penalty: Optional[float] = None,
    vectara_presence_penalty: Optional[float] = None,
    vectara_save_history: bool = True,
    return_direct: bool = False,
) -> Dict[str, Any]:
    """
    Create agent configuration from a single Vectara corpus.

    This function creates the necessary tools and configuration for an agent
    that can interact with a Vectara corpus for RAG operations.

    Args:
        tool_name: The name of Vectara tool used by the agent
        data_description: The description of the data
        assistant_specialty: The specialty of the assistant
        general_instructions: General instructions for the agent
        vectara_corpus_key: The Vectara corpus key (or comma separated list of keys)
        vectara_api_key: The Vectara API key
        agent_config: The configuration of the agent
        fallback_agent_config: The fallback configuration of the agent
        verbose: Whether to print verbose output
        vectara_filter_fields: The filterable attributes
        vectara_offset: Number of results to skip
        vectara_lambda_val: Lambda value for Vectara hybrid search
        vectara_semantics: Indicates whether the query is intended as a query or response
        vectara_custom_dimensions: Custom dimensions for the query
        vectara_reranker: The Vectara reranker name
        vectara_rerank_k: The number of results to use with reranking
        vectara_rerank_limit: The maximum number of results to return after reranking
        vectara_rerank_cutoff: The minimum score threshold for results to include
        vectara_diversity_bias: The MMR diversity bias
        vectara_udf_expression: The user defined expression for reranking results
        vectara_rerank_chain: A list of Vectara rerankers to be applied sequentially
        vectara_n_sentences_before: The number of sentences before the matching text
        vectara_n_sentences_after: The number of sentences after the matching text
        vectara_summary_num_results: The number of results to use in summarization
        vectara_summarizer: The Vectara summarizer name
        vectara_summary_response_language: The response language for the Vectara summary
        vectara_summary_prompt_text: The custom prompt, using appropriate prompt variables
        vectara_max_response_chars: The desired maximum number of characters
        vectara_max_tokens: The maximum number of tokens to be returned by the LLM
        vectara_temperature: The sampling temperature
        vectara_frequency_penalty: How much to penalize repeating tokens
        vectara_presence_penalty: How much to penalize repeating tokens for diversity
        vectara_save_history: Whether to save the query in history
        return_direct: Whether the agent should return the tool's response directly

    Returns:
        Dict[str, Any]: Agent creation parameters including tools and instructions
    """
    # Create Vectara tool factory
    vec_factory = VectaraToolFactory(
        vectara_api_key=vectara_api_key,
        vectara_corpus_key=vectara_corpus_key,
    )

    # Build field definitions for the tool schema
    field_definitions = {}
    field_definitions["query"] = (str, Field(description="The user query"))

    for field in vectara_filter_fields:
        field_definitions[field["name"]] = (
            PY_TYPES.get(field["type"], Any),
            Field(description=field["description"]),
        )

    # Sanitize tool name
    if tool_name:
        tool_name = re.sub(r"[^A-Za-z0-9_]", "_", tool_name)
    query_args = create_model(f"{tool_name}_QueryArgs", **field_definitions)

    # Create the Vectara RAG tool
    vectara_tool = vec_factory.create_rag_tool(
        tool_name=tool_name or f"vectara_{vectara_corpus_key}",
        tool_description=f"""
        Given a user query,
        returns a response (str) to a user question about {data_description}.
        """,
        tool_args_schema=query_args,
        reranker=vectara_reranker,
        rerank_k=vectara_rerank_k,
        rerank_limit=vectara_rerank_limit,
        rerank_cutoff=vectara_rerank_cutoff,
        mmr_diversity_bias=vectara_diversity_bias,
        udf_expression=vectara_udf_expression,
        rerank_chain=vectara_rerank_chain,
        n_sentences_before=vectara_n_sentences_before,
        n_sentences_after=vectara_n_sentences_after,
        offset=vectara_offset,
        lambda_val=vectara_lambda_val,
        semantics=vectara_semantics,
        custom_dimensions=vectara_custom_dimensions,
        summary_num_results=vectara_summary_num_results,
        vectara_summarizer=vectara_summarizer,
        summary_response_lang=vectara_summary_response_language,
        vectara_prompt_text=vectara_summary_prompt_text,
        max_response_chars=vectara_max_response_chars,
        max_tokens=vectara_max_tokens,
        temperature=vectara_temperature,
        frequency_penalty=vectara_frequency_penalty,
        presence_penalty=vectara_presence_penalty,
        save_history=vectara_save_history,
        include_citations=True,
        verbose=verbose,
        return_direct=return_direct,
    )

    # Create assistant instructions
    assistant_instructions = f"""
    - You are a helpful {assistant_specialty} assistant.
    - You can answer questions about {data_description}.
    - Never discuss politics, and always respond politely.
    """

    # Determine general instructions based on available tools
    tools = [vectara_tool]
    effective_general_instructions = (
        general_instructions if general_instructions is not None
        else get_general_instructions(tools)
    )

    return {
        "tools": tools,
        "agent_config": agent_config,
        "topic": assistant_specialty,
        "custom_instructions": assistant_instructions,
        "general_instructions": effective_general_instructions,
        "verbose": verbose,
        "fallback_agent_config": fallback_agent_config,
        "vectara_api_key": vectara_api_key,
    }
