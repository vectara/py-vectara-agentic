"""
Agent serialization and deserialization utilities.

This module handles saving and restoring agent state, including tools,
memory, configuration, and all associated metadata. It provides both
modern JSON-based serialization and legacy pickle fallbacks.
"""

import logging
import importlib
import inspect
from typing import Dict, Any, List, Optional, Callable

import cloudpickle as pickle
from pydantic import Field, create_model, BaseModel
from llama_index.core.memory import Memory
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool

from ..agent_config import AgentConfig
from ..tools import VectaraTool
from ..types import ToolType
from .utils.schemas import get_field_type

def restore_memory_from_dict(data: Dict[str, Any], session_id: str, token_limit: int = 65536) -> Memory:
    """
    Restore agent memory from serialized dictionary data.

    Supports both modern JSON format and legacy pickle format for backward compatibility.

    Args:
        data: Serialized agent data dictionary
        session_id: Session ID to use for the memory
        token_limit: Token limit for the memory instance

    Returns:
        Memory: Restored memory instance
    """
    mem = Memory.from_defaults(
        session_id=session_id,
        token_limit=token_limit
    )

    # New JSON dump format
    dump = data.get("memory_dump", [])
    if dump:
        mem.put_messages([ChatMessage(**m) for m in dump])

    # Legacy pickle fallback
    legacy_blob = data.get("memory")
    if legacy_blob and not dump:
        try:
            legacy_mem = pickle.loads(legacy_blob.encode("latin-1"))
            mem.put_messages(legacy_mem.get())
        except Exception:
            logging.debug("Legacy memory pickle could not be loaded; ignoring.")

    return mem


def serialize_tools(tools: List[FunctionTool]) -> List[Dict[str, Any]]:
    """
    Serialize a list of tools to dictionary format.

    Args:
        tools: List of FunctionTool objects to serialize

    Returns:
        List[Dict[str, Any]]: Serialized tool data
    """
    tool_info = []
    for tool in tools:
        # Serialize function schema if available
        if hasattr(tool.metadata, "fn_schema"):
            fn_schema_cls = tool.metadata.fn_schema
            fn_schema_serialized = {
                "schema": (
                    fn_schema_cls.model_json_schema()
                    if fn_schema_cls and hasattr(fn_schema_cls, "model_json_schema")
                    else None
                ),
                "metadata": {
                    "module": fn_schema_cls.__module__ if fn_schema_cls else None,
                    "class": fn_schema_cls.__name__ if fn_schema_cls else None,
                },
            }
        else:
            fn_schema_serialized = None

        # Serialize tool data
        tool_dict = {
            "tool_type": tool.metadata.tool_type.value,
            "name": tool.metadata.name,
            "description": tool.metadata.description,
            "fn": (
                pickle.dumps(getattr(tool, "fn", None)).decode("latin-1")
                if getattr(tool, "fn", None)
                else None
            ),
            "async_fn": (
                pickle.dumps(getattr(tool, "async_fn", None)).decode("latin-1")
                if getattr(tool, "async_fn", None)
                else None
            ),
            "fn_schema": fn_schema_serialized,
        }
        tool_info.append(tool_dict)

    return tool_info


def deserialize_tools(tool_data_list: List[Dict[str, Any]]) -> List[FunctionTool]:
    """
    Deserialize tools from dictionary format.

    Args:
        tool_data_list: List of serialized tool dictionaries

    Returns:
        List[FunctionTool]: Deserialized tools
    """
    tools: List[FunctionTool] = []

    for tool_data in tool_data_list:
        query_args_model = None

        # Reconstruct function schema if available
        if tool_data.get("fn_schema"):
            query_args_model = _reconstruct_tool_schema(tool_data)

        # If fn_schema was not in tool_data or reconstruction failed, default to empty pydantic model
        if query_args_model is None:
            query_args_model = create_model(f"{tool_data['name']}_QueryArgs")

        # Deserialize function objects with error handling
        fn = None
        async_fn = None

        try:
            if tool_data["fn"]:
                fn = pickle.loads(tool_data["fn"].encode("latin-1"))
        except Exception as e:
            logging.warning(
                f"[TOOL_DESERIALIZE] Failed to deserialize fn for tool '{tool_data['name']}': {e}"
            )

        try:
            if tool_data["async_fn"]:
                async_fn = pickle.loads(tool_data["async_fn"].encode("latin-1"))
        except Exception as e:
            logging.warning(
                f"[TOOL_DESERIALIZE] Failed to deserialize async_fn for tool '{tool_data['name']}': {e}"
            )

        # Create tool instance with enhanced error handling
        try:
            tool = VectaraTool.from_defaults(
                name=tool_data["name"],
                description=tool_data["description"],
                fn=fn,
                async_fn=async_fn,
                fn_schema=query_args_model,
                tool_type=ToolType(tool_data["tool_type"]),
            )
        except ValueError as e:
            if "invalid method signature" in str(e):
                logging.warning(
                    f"Skipping tool '{tool_data['name']}' due to invalid method signature"
                )
                continue  # Skip this tool and continue with others
        tools.append(tool)

    return tools


def _reconstruct_tool_schema(tool_data: Dict[str, Any]):
    """
    Reconstruct Pydantic schema for a tool from serialized data.

    First attempts to import the original class, falls back to JSON schema reconstruction.

    Args:
        tool_data: Serialized tool data containing schema information

    Returns:
        Pydantic model class or None if reconstruction fails
    """
    schema_info = tool_data["fn_schema"]

    try:
        # Try to import original class
        module_name = schema_info["metadata"]["module"]
        class_name = schema_info["metadata"]["class"]
        mod = importlib.import_module(module_name)
        candidate_cls = getattr(mod, class_name)

        if inspect.isclass(candidate_cls) and issubclass(candidate_cls, BaseModel):
            return candidate_cls
        else:
            # Force fallback to JSON schema reconstruction
            raise ImportError(
                f"Retrieved '{class_name}' from '{module_name}' is not a Pydantic BaseModel class. "
                "Falling back to JSON schema reconstruction."
            )
    except Exception:
        # Fallback: rebuild using the JSON schema
        return _rebuild_schema_from_json(schema_info, tool_data["name"])


def _rebuild_schema_from_json(schema_info: Dict[str, Any], tool_name: str):
    """
    Rebuild Pydantic schema from JSON schema information.

    Args:
        schema_info: Schema information dictionary
        tool_name: Name of the tool for fallback naming

    Returns:
        Pydantic model class
    """
    field_definitions = {}
    json_schema_to_rebuild = schema_info.get("schema")

    if json_schema_to_rebuild and isinstance(json_schema_to_rebuild, dict):
        for field, values in json_schema_to_rebuild.get("properties", {}).items():
            field_type = get_field_type(values)
            field_description = values.get("description")  # Defaults to None

            if "default" in values:
                field_definitions[field] = (
                    field_type,
                    Field(
                        description=field_description,
                        default=values["default"],
                    ),
                )
            else:
                field_definitions[field] = (
                    field_type,
                    Field(description=field_description),
                )

        return create_model(
            json_schema_to_rebuild.get("title", f"{tool_name}_QueryArgs"),
            **field_definitions,
        )
    else:
        # If schema part is missing or not a dict, create a default empty model
        return create_model(f"{tool_name}_QueryArgs")


def serialize_agent_to_dict(agent) -> Dict[str, Any]:
    """
    Serialize an Agent instance to a dictionary.

    Args:
        agent: Agent instance to serialize

    Returns:
        Dict[str, Any]: Serialized agent data
    """
    return {
        "agent_type": agent.agent_config.agent_type.value,
        "memory_dump": [m.model_dump() for m in agent.memory.get()],
        "session_id": agent.session_id,
        "tools": serialize_tools(agent.tools),
        # pylint: disable=protected-access
        "topic": agent._topic,
        # pylint: disable=protected-access
        "custom_instructions": agent._custom_instructions,
        "verbose": agent.verbose,
        "agent_config": agent.agent_config.to_dict(),
        "fallback_agent_config": (
            agent.fallback_agent_config.to_dict()
            if agent.fallback_agent_config
            else None
        ),
        "workflow_cls": agent.workflow_cls if agent.workflow_cls else None,
        "vectara_api_key": agent.vectara_api_key,
        # Custom metadata for agent-specific settings (e.g., use_waii for EV agent)
        "custom_metadata": getattr(agent, "_custom_metadata", {}),
    }


def deserialize_agent_from_dict(
    agent_cls,
    data: Dict[str, Any],
    agent_progress_callback: Optional[Callable] = None,
    query_logging_callback: Optional[Callable] = None,
):
    """
    Create an Agent instance from a dictionary.

    Args:
        agent_cls: Agent class to instantiate
        data: Serialized agent data
        agent_progress_callback: Optional progress callback
        query_logging_callback: Optional query logging callback

    Returns:
        Agent instance restored from data
    """
    # Restore configurations
    agent_config = AgentConfig.from_dict(data["agent_config"])
    fallback_agent_config = (
        AgentConfig.from_dict(data["fallback_agent_config"])
        if data.get("fallback_agent_config")
        else None
    )

    # Restore tools with error handling and fallback
    try:
        tools = deserialize_tools(data["tools"])
    except Exception as e:
        raise ValueError(f"[AGENT_DESERIALIZE] Tool deserialization failed: {e}") from e

    # Create agent instance
    agent = agent_cls(
        tools=tools,
        agent_config=agent_config,
        topic=data["topic"],
        custom_instructions=data["custom_instructions"],
        verbose=data["verbose"],
        fallback_agent_config=fallback_agent_config,
        workflow_cls=data["workflow_cls"],
        agent_progress_callback=agent_progress_callback,
        query_logging_callback=query_logging_callback,
        vectara_api_key=data.get("vectara_api_key"),
        session_id=data.get("session_id"),
    )

    # Restore custom metadata (backward compatible)
    # pylint: disable=protected-access
    agent._custom_metadata = data.get("custom_metadata", {})

    # Restore memory with the agent's session_id
    # Support both new and legacy serialization formats
    session_id_from_data = data.get("session_id") or data.get("memory_session_id", "default")
    mem = restore_memory_from_dict(data, session_id_from_data, token_limit=65536)
    agent.memory = mem

    # Keep inner agent (if already built) in sync
    # pylint: disable=protected-access
    if getattr(agent, "_agent", None) is not None:
        agent._agent.memory = mem
    # pylint: disable=protected-access
    if getattr(agent, "_fallback_agent", None):
        agent._fallback_agent.memory = mem

    return agent
