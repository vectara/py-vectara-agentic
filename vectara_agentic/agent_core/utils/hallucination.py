"""Vectara Hallucination Detection and Correction client."""

import logging
from typing import List, Dict, Optional, Tuple
import requests

from llama_index.core.llms import MessageRole

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

        logging.debug(
            f"VHC: query={query}\n"
        )
        logging.debug(
            f"VHC: response={hypothesis}\n"
        )
        logging.debug("VHC: Context:")
        for i, ctx in enumerate(context):
            logging.info(f"VHC: context {i}: {ctx}\n\n")

        logging.debug(
            f"VHC: outputs: {len(corrections)} corrections"
        )
        logging.debug(
            f"VHC: corrected_text: {corrected_text}\n"
        )
        for correction in corrections:
            logging.debug(f"VHC: correction: {correction}\n")

        return corrected_text, corrections

def extract_tool_call_mapping(chat_history) -> Dict[str, str]:
    """Extract tool_call_id to tool_name mapping from chat history."""
    tool_call_id_to_name = {}
    for msg in chat_history:
        if (
            msg.role == MessageRole.ASSISTANT
            and hasattr(msg, "additional_kwargs")
            and msg.additional_kwargs
        ):
            tool_calls = msg.additional_kwargs.get("tool_calls", [])
            for tool_call in tool_calls:
                if (
                    isinstance(tool_call, dict)
                    and "id" in tool_call
                    and "function" in tool_call
                ):
                    tool_call_id = tool_call["id"]
                    tool_name = tool_call["function"].get("name")
                    if tool_call_id and tool_name:
                        tool_call_id_to_name[tool_call_id] = tool_name

    return tool_call_id_to_name


def identify_tool_name(msg, tool_call_id_to_name: Dict[str, str]) -> Optional[str]:
    """Identify tool name from message using multiple strategies."""
    tool_name = None

    # First try: standard tool_name attribute (for backwards compatibility)
    tool_name = getattr(msg, "tool_name", None)

    # Second try: additional_kwargs (LlamaIndex standard location)
    if (
        tool_name is None
        and hasattr(msg, "additional_kwargs")
        and msg.additional_kwargs
    ):
        tool_name = msg.additional_kwargs.get("name") or msg.additional_kwargs.get(
            "tool_name"
        )

        # If no direct tool name, try to map from tool_call_id
        if tool_name is None:
            tool_call_id = msg.additional_kwargs.get("tool_call_id")
            if tool_call_id and tool_call_id in tool_call_id_to_name:
                tool_name = tool_call_id_to_name[tool_call_id]

    # Third try: extract from content if it's a ToolOutput object
    if tool_name is None and hasattr(msg.content, "tool_name"):
        tool_name = msg.content.tool_name

    return tool_name


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
    query: str, chat_history: List,
    agent_response: str, tools: List, vectara_api_key: str
) -> Tuple[Optional[str], List[str]]:
    """Use VHC to compute corrected_text and corrections."""
    if not vectara_api_key:
        logging.debug("No Vectara API key - returning None")
        return None, []

    # Build a mapping from tool_call_id to tool_name for better tool identification
    tool_call_id_to_name = extract_tool_call_mapping(chat_history)

    context = []
    last_assistant_index = -1
    for i, msg in enumerate(chat_history):
        if msg.role == MessageRole.ASSISTANT and msg.content:
            last_assistant_index = i

    for i, msg in enumerate(chat_history):
        if msg.role == MessageRole.TOOL:
            tool_name = identify_tool_name(msg, tool_call_id_to_name)
            is_vhc_eligible = check_tool_eligibility(tool_name, tools)

            # Only count tool calls from VHC-eligible tools
            if is_vhc_eligible:
                content = msg.content

                # Since tools with human-readable output now convert to formatted strings immediately
                # in VectaraTool._format_tool_output(), we just use the content directly
                content = str(content) if content is not None else ""

                # Only add non-empty content to context
                if content and content.strip():
                    context.append(content)

        elif msg.role == MessageRole.USER and msg.content:
            context.append(msg.content)

        elif msg.role == MessageRole.ASSISTANT and msg.content:
            if i == last_assistant_index:  # do not include the last assistant message
                continue
            context.append(msg.content)

    # If no context or no tool calls, we cannot compute VHC
    if len(context) == 0:
        return None, []

    try:
        h = Hallucination(vectara_api_key)
        corrected_text, corrections = h.compute(
            query=query, context=context, hypothesis=agent_response
        )
        return corrected_text, corrections

    except Exception as e:
        logging.error(
            f"VHC call failed: {e}. "
            "Ensure you have a valid Vectara API key and the Hallucination Correction service is available."
        )
        return None, []
