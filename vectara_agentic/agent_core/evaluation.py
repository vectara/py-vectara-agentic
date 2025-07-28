"""
Agent evaluation and response post-processing utilities.

This module provides functions for evaluating agent responses,
calculating factual consistency scores, and formatting responses
for specific agent types like LATS.
"""

import logging
from typing import Optional

from llama_index.core.memory import Memory
from llama_index.core.llms import MessageRole

from ..types import AgentType, AgentResponse
from .utils.hallucination import Hallucination
from ..tool_utils import _is_human_readable_output


async def format_response_for_lats(
    prompt: str,
    agent_response: AgentResponse,
    agent_instance,
    current_agent_type: AgentType,
) -> None:
    """
    Format agent response for LATS (Language Agent Tree Search) agents.

    LATS agents require special post-processing to format their responses
    in a more readable and coherent manner.

    Args:
        prompt: Original user prompt
        agent_response: Agent response to format
        agent_instance: Agent instance to get current agent from
        current_agent_type: Current agent type to check if LATS formatting is needed
    """
    if current_agent_type != AgentType.LATS:
        return

    llm_prompt = f"""
    Given the question '{prompt}', and agent response '{agent_response.response}',
    Please provide a well formatted final response to the query.
    final response:
    """

    # pylint: disable=protected-access
    current_agent = agent_instance._get_current_agent()
    agent_response.response = (await current_agent.llm.acomplete(llm_prompt)).text


def calculate_factual_consistency_score(
    memory: Memory, agent_response: str, vectara_api_key: str
) -> Optional[float]:
    """
    Calculate the Factual Consistency Score (FCS) for the agent response.

    FCS measures how well the agent's response is supported by the context
    retrieved from tools and previous conversation history.

    Args:
        memory: Agent memory containing chat history
        agent_response: The agent's response text to evaluate
        vectara_api_key: Vectara API key for FCS calculation

    Returns:
        Optional[float]: FCS score between 0 and 1, or None if calculation fails
    """
    if not vectara_api_key:
        logging.debug("FCS calculation skipped: 'vectara_api_key' is missing.")
        return None  # can't calculate FCS without Vectara API key

    context, num_tool_calls = extract_chat_context_for_evaluation(memory)

    # Skip FCS calculation if no context or no tool calls
    if not context or num_tool_calls == 0:
        return None

    # Calculate FCS using Hallucination Detection
    context_str = "\n".join(context)
    try:
        score = Hallucination(vectara_api_key).compute(context_str, agent_response)
        return score
    except Exception as e:
        logging.error(f"Failed to calculate FCS: {e}")
        return None


def extract_chat_context_for_evaluation(memory: Memory) -> tuple[list[str], int]:
    """
    Extract context and tool call count from chat history for evaluation.

    Args:
        memory: Agent memory containing chat history

    Returns:
        tuple[list[str], int]: List of context strings and number of tool calls
    """
    chat_history = memory.get()
    context = []
    num_tool_calls = 0

    for msg in chat_history:
        if msg.role == MessageRole.TOOL:
            num_tool_calls += 1
            content = msg.content

            # Handle special human-readable output formats
            if _is_human_readable_output(content):
                try:
                    content = content.to_human_readable()
                except Exception as e:
                    logging.debug(
                        f"Failed to get human-readable format for context extraction: {e}"
                    )
                    content = str(content)

            context.append(content)
        elif msg.role in [MessageRole.USER, MessageRole.ASSISTANT] and msg.content:
            context.append(msg.content)

    return context, num_tool_calls
