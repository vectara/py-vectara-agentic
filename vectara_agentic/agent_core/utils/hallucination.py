"""Vectara Hallucination Detection and Correction client."""

import logging
import re
from typing import List, Optional, Tuple
import requests

from llama_index.core.llms import MessageRole


# Compiled regex patterns for better performance
_MARKDOWN_LINK_PATTERN = re.compile(r'\[([^\]]*)\]\([^)]*\)')
_WHITESPACE_CLEANUP_PATTERN = re.compile(r'\s+')


def clean_urls_from_text(text: str) -> str:
    """
    Remove markdown URLs [text](URL) from text, preserving the link text.
    This prevents interference with hallucination detection while keeping useful text content.

    Args:
        text (str): The input text potentially containing markdown URLs

    Returns:
        str: Text with markdown URLs replaced by their text content
    """
    if not text:
        return text

    # Replace markdown links [text](url) with just the text part
    cleaned_text = _MARKDOWN_LINK_PATTERN.sub(r'\1', text)

    # Clean up any extra whitespace that might result from the replacement
    cleaned_text = _WHITESPACE_CLEANUP_PATTERN.sub(' ', cleaned_text).strip()

    return cleaned_text


class Hallucination:
    """Vectara Hallucination Correction."""

    def __init__(self, vectara_api_key: str):
        self._vectara_api_key = vectara_api_key

    def compute(
        self, query: str, context: list[str], hypothesis: str
    ) -> Tuple[str, list[str]]:
        """
        Calls the Vectara VHC (Vectara Hallucination Correction)

        Returns:
            str: The corrected hypothesis text.
            list[str]: the list of corrections from VHC
        """

        payload = {
            "generated_text": hypothesis,
            "query": query,
            "documents": [{"text": c} for c in context],
            "model_name": "vhc-large-1.0",
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-api-key": self._vectara_api_key,
        }

        response = requests.post(
            "https://api.vectara.io/v2/hallucination_correctors/correct_hallucinations",
            json=payload,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        corrected_text = data.get("corrected_text", "")
        corrections = data.get("corrections", [])

        logging.info(f"VHC: query={query}\n")
        logging.info(f"VHC: response={hypothesis}\n")
        logging.info("VHC: Context:")
        for i, ctx in enumerate(context):
            logging.info(f"VHC: context {i}: {ctx[:200]}\n\n")

        logging.info(f"VHC: outputs: {len(corrections)} corrections")
        logging.info(f"VHC: corrected_text: {corrected_text}\n")
        for correction in corrections:
            logging.info(f"VHC: correction: {correction}\n")

        return corrected_text, corrections


def check_tool_eligibility(tool_name: Optional[str], tools: List) -> bool:
    """Check if a tool output is eligible to be included in VHC, by looking up in tools list."""
    if not tool_name or not tools:
        return False

    # Try to find the tool and check its VHC eligibility
    for tool in tools:
        if (
            hasattr(tool, "metadata")
            and hasattr(tool.metadata, "name")
            and tool.metadata.name == tool_name
        ):
            if hasattr(tool.metadata, "vhc_eligible"):
                is_vhc_eligible = tool.metadata.vhc_eligible
                return is_vhc_eligible
            break

    return True


def analyze_hallucinations(
    query: str,
    chat_history: List,
    agent_response: str,
    tools: List,
    vectara_api_key: str,
    tool_outputs: Optional[List[dict]] = None,
) -> Tuple[Optional[str], List[str]]:
    """Use VHC to compute corrected_text and corrections using provided tool data."""

    if not vectara_api_key:
        logging.warning("VHC: No Vectara API key - returning None")
        return None, []

    context = []

    # Process tool outputs if provided
    if tool_outputs:
        tool_output_count = 0
        for tool_output in tool_outputs:
            if tool_output.get("status_type") == "TOOL_OUTPUT" and tool_output.get(
                "content"
            ):
                tool_output_count += 1
                tool_name = tool_output.get("tool_name")
                is_vhc_eligible = check_tool_eligibility(tool_name, tools)

                if is_vhc_eligible:
                    content = str(tool_output["content"])
                    if content and content.strip():
                        context.append(content)

        logging.info(
            f"VHC: Processed {tool_output_count} tool outputs, added {len(context)} to context so far"
        )
    else:
        logging.info("VHC: No tool outputs provided")

    # Add user messages and previous assistant messages from chat_history for context
    last_assistant_index = -1
    for i, msg in enumerate(chat_history):
        if msg.role == MessageRole.ASSISTANT and msg.content:
            last_assistant_index = i

    for i, msg in enumerate(chat_history):
        if msg.role == MessageRole.USER and msg.content:
            # Don't include the current query in context since it's passed separately as query parameter
            if msg.content != query:
                context.append(msg.content)

        elif msg.role == MessageRole.ASSISTANT and msg.content:
            if i != last_assistant_index:  # do not include the last assistant message
                context.append(msg.content)

    logging.info(f"VHC: Final VHC context has {len(context)} items")

    # If no context, we cannot compute VHC
    if len(context) == 0:
        logging.info("VHC: No context available for VHC - returning None")
        return None, []

    try:
        # Clean URLs from agent response to prevent interference with hallucination detection
        cleaned_agent_response = clean_urls_from_text(agent_response)

        h = Hallucination(vectara_api_key)
        corrected_text, corrections = h.compute(
            query=query, context=context, hypothesis=cleaned_agent_response
        )
        return corrected_text, corrections

    except Exception as e:
        logging.warning(
            f"VHC call failed: {e}. "
            "Ensure you have a valid Vectara API key and the Hallucination Correction service is available."
        )
        return None, []
