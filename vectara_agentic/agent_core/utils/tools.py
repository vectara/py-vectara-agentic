"""
Tool processing and validation utilities for agent functionality.

This module provides utilities for tool validation, processing, and
compatibility adjustments for different LLM providers.
"""

import inspect
from typing import Any, List
from inspect import Signature, Parameter, ismethod
from collections import Counter
import logging

from pydantic import Field, create_model
from llama_index.core.tools import FunctionTool

from ...llm_utils import get_llm
from ...types import LLMRole


def sanitize_tools_for_gemini(tools: List[FunctionTool]) -> List[FunctionTool]:
    """
    Strip all default values from tools for Gemini LLM compatibility.

    Gemini requires that tools only show required parameters without defaults.
    This function modifies:
    - tool.fn signature
    - tool.async_fn signature
    - tool.metadata.fn_schema

    Args:
        tools: List of FunctionTool objects to sanitize

    Returns:
        List[FunctionTool]: Sanitized tools with no default values
    """
    for tool in tools:
        # 1) Strip defaults off the actual callables
        for func in (tool.fn, tool.async_fn):
            if not func:
                continue
            try:
                orig_sig = inspect.signature(func)
                new_params = [
                    p.replace(default=Parameter.empty) for p in orig_sig.parameters.values()
                ]
                new_sig = Signature(
                    new_params, return_annotation=orig_sig.return_annotation
                )
                if ismethod(func):
                    func.__func__.__signature__ = new_sig
                else:
                    func.__signature__ = new_sig
            except Exception as e:
                logging.warning(
                    f"Could not modify signature for tool '{tool.metadata.name}': {e}. "
                    "Proceeding with Pydantic schema modification."
                )

        # 2) Rebuild the Pydantic schema so that *every* field is required
        schema_cls = getattr(tool.metadata, "fn_schema", None)
        if schema_cls and hasattr(schema_cls, "model_fields"):
            # Collect (name → (type, Field(...))) for all fields
            new_fields: dict[str, tuple[type, Any]] = {}
            for name, mf in schema_cls.model_fields.items():
                typ = mf.annotation
                desc = getattr(mf, "description", "")
                # Force required (no default) with Field(...)
                new_fields[name] = (typ, Field(..., description=desc))

            # Make a brand-new schema class where every field is required
            no_default_schema = create_model(
                f"{schema_cls.__name__}",  # new class name
                **new_fields,  # type: ignore
            )

            # Give it a clean __signature__ so inspect.signature sees no defaults
            params = [
                Parameter(n, Parameter.POSITIONAL_OR_KEYWORD, annotation=typ)
                for n, (typ, _) in new_fields.items()
            ]
            no_default_schema.__signature__ = Signature(params)

            # Swap it back onto the tool
            tool.metadata.fn_schema = no_default_schema

    return tools


def validate_tool_consistency(
    tools: List[FunctionTool], custom_instructions: str, agent_config
) -> None:
    """
    Validate that tools mentioned in instructions actually exist.

    Args:
        tools: List of available tools
        custom_instructions: Custom instructions that may reference tools
        agent_config: Agent configuration for LLM access

    Raises:
        ValueError: If invalid tools are referenced in instructions
    """
    tool_names = [tool.metadata.name for tool in tools]

    # Check for duplicate tools
    duplicates = [tool for tool, count in Counter(tool_names).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate tools detected: {', '.join(duplicates)}")

    # Validate tools mentioned in instructions exist
    if custom_instructions:
        prompt = f"""
        You are provided these tools:
        <tools>{','.join(tool_names)}</tools>
        And these instructions:
        <instructions>
        {custom_instructions}
        </instructions>
        Your task is to identify invalid tools.
        A tool is invalid if it is mentioned in the instructions but not in the tools list.
        A tool's name must have at least two characters.
        Your response should be a comma-separated list of the invalid tools.
        If no invalid tools exist, respond with "<OKAY>" (and nothing else).
        """
        llm = get_llm(LLMRole.MAIN, config=agent_config)
        bad_tools_str = llm.complete(prompt).text.strip("\n")
        if bad_tools_str and bad_tools_str != "<OKAY>":
            bad_tools = [tool.strip() for tool in bad_tools_str.split(",")]
            numbered = ", ".join(f"({i}) {tool}" for i, tool in enumerate(bad_tools, 1))
            raise ValueError(
                f"The Agent custom instructions mention these invalid tools: {numbered}"
            )
