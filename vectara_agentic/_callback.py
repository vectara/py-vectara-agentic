"""
Module to handle agent callbacks
"""

import inspect
import logging
from typing import Any, Dict, Optional, List, Callable
from functools import wraps
import traceback

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload

from .types import AgentStatusType


def wrap_callback_fn(callback):
    """
    Wrap a callback function to ensure it only receives the parameters it can accept.
    This is useful for ensuring that the callback function does not receive unexpected
    parameters, especially when the callback is called from different contexts.
    """
    if callback is None:
        return None
    try:
        sig = inspect.signature(callback)
        allowed_params = set(sig.parameters.keys())
    except Exception:
        # If we cannot determine the signature, return the callback as is.
        return callback

    @wraps(callback)
    def new_callback(*args, **kwargs):
        # Filter kwargs to only those that the original callback accepts.
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_params}
        return callback(*args, **filtered_kwargs)

    return new_callback


def _extract_content_from_response(response) -> str:
    """
    Extract text content from various LLM response formats.

    Handles different provider response objects and extracts the text content consistently.

    Args:
        response: Response object from LLM provider

    Returns:
        str: Extracted text content
    """
    # Handle case where response is a string
    if isinstance(response, str):
        return response

    # Handle ChatMessage objects with blocks (Anthropic, etc.)
    if hasattr(response, "blocks") and response.blocks:
        text_parts = []
        for block in response.blocks:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        return "".join(text_parts)

    # Handle responses with content attribute
    if hasattr(response, "content"):
        return str(response.content)

    # Handle responses with message attribute that has content
    if hasattr(response, "message") and hasattr(response.message, "content"):
        return str(response.message.content)

    # Handle delta attribute for streaming responses
    if hasattr(response, "delta"):
        return str(response.delta)

    # Fallback to string conversion
    return str(response)


class AgentCallbackHandler(BaseCallbackHandler):
    """
    Callback handler to track agent status

    This handler simply keeps track of event starts/ends, separated by event types.
    You can use this callback handler to keep track of agent progress.

    Args:
        fn: callable function agent will call back to report on agent progress
    """

    def __init__(self, fn: Optional[Callable] = None) -> None:
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.fn = wrap_callback_fn(fn)

    # Existing synchronous methods
    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        if self.fn is not None and payload is not None:
            if inspect.iscoroutinefunction(self.fn):
                raise ValueError(
                    "Synchronous callback handler cannot use async callback function"
                )
            # Handle events as before
            self._handle_event(event_type, payload, event_id)
        return event_id

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        pass

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Handle the end of an event

        Args:
            event_type: the type of event
            payload: the event payload
            event_id: the event ID
            kwargs: additional keyword arguments

        Returns:
            None
        """
        if self.fn is not None and payload is not None:
            if inspect.iscoroutinefunction(self.fn):
                raise ValueError(
                    "Synchronous callback handler cannot use async callback function"
                )
            # Handle events as before
            self._handle_event(event_type, payload, event_id)

    # New asynchronous methods
    async def aon_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """
        Handle the start of an event

        Args:
            event_type: the type of event
            payload: the event payload
            event_id: the event ID
            parent_id: the parent event ID
            kwargs: additional keyword arguments

        Returns:
            event_id: the event ID
        """
        if self.fn is not None and payload is not None:
            await self._ahandle_event(event_type, payload, event_id)
        return event_id

    async def aon_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Handle the end of an event (async)
        """
        if self.fn is not None and payload is not None:
            await self._ahandle_event(event_type, payload, event_id)

    # Helper methods for handling events
    def _handle_event(
        self, event_type: CBEventType, payload: Dict[str, Any], event_id: str
    ) -> None:
        try:
            if event_type == CBEventType.LLM:
                self._handle_llm(payload, event_id)
            elif event_type == CBEventType.FUNCTION_CALL:
                self._handle_function_call(payload, event_id)
            elif event_type == CBEventType.AGENT_STEP:
                self._handle_agent_step(payload, event_id)
            else:
                pass
        except Exception as e:
            logging.error(f"Exception in callback handler: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            # Continue execution to prevent callback failures from breaking the agent

    async def _ahandle_event(
        self, event_type: CBEventType, payload: Dict[str, Any], event_id: str
    ) -> None:
        try:
            if event_type == CBEventType.LLM:
                await self._ahandle_llm(payload, event_id)
            elif event_type == CBEventType.FUNCTION_CALL:
                await self._ahandle_function_call(payload, event_id)
            elif event_type == CBEventType.AGENT_STEP:
                await self._ahandle_agent_step(payload, event_id)
            else:
                pass
        except Exception as e:
            logging.error(f"Exception in async callback handler: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            # Continue execution to prevent callback failures from breaking the agent

    # Synchronous handlers
    def _handle_llm(
        self,
        payload: dict,
        event_id: str,
    ) -> None:
        if EventPayload.MESSAGES in payload:
            response = payload.get(EventPayload.RESPONSE)
            if response and str(response) not in ["None", "assistant: None"]:
                if self.fn:
                    # Convert response to consistent dict format
                    content = _extract_content_from_response(response)
                    self.fn(
                        status_type=AgentStatusType.AGENT_UPDATE,
                        msg={"content": content},
                        event_id=event_id,
                    )
        elif EventPayload.PROMPT in payload:
            prompt = payload.get(EventPayload.PROMPT)
            if self.fn:
                # Convert prompt to consistent dict format
                content = str(prompt) if prompt else ""
                self.fn(
                    status_type=AgentStatusType.AGENT_UPDATE,
                    msg={"content": content},
                    event_id=event_id,
                )
        else:
            pass

    def _handle_function_call(self, payload: dict, event_id: str) -> None:
        try:
            if EventPayload.FUNCTION_CALL in payload:
                fcall = payload.get(EventPayload.FUNCTION_CALL)
                tool = payload.get(EventPayload.TOOL)

                if tool:
                    tool_name = tool.name
                    if self.fn:
                        self.fn(
                            status_type=AgentStatusType.TOOL_CALL,
                            msg={"tool_name": tool_name, "arguments": fcall},
                            event_id=event_id,
                        )

            elif EventPayload.FUNCTION_OUTPUT in payload:
                response = payload.get(EventPayload.FUNCTION_OUTPUT)
                tool = payload.get(EventPayload.TOOL)

                if tool and self.fn:
                    self.fn(
                        status_type=AgentStatusType.TOOL_OUTPUT,
                        msg={"tool_name": tool.name, "content": response},
                        event_id=event_id,
                    )

        except Exception as e:
            logging.error(f"Exception in _handle_function_call: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            # Continue execution to prevent callback failures from breaking the agent

    def _handle_agent_step(self, payload: dict, event_id: str) -> None:
        if EventPayload.MESSAGES in payload:
            msg = payload.get(EventPayload.MESSAGES)
            if self.fn:
                self.fn(
                    status_type=AgentStatusType.AGENT_STEP,
                    msg=msg,
                    event_id=event_id,
                )
        elif EventPayload.RESPONSE in payload:
            response = payload.get(EventPayload.RESPONSE)
            if self.fn:
                self.fn(
                    status_type=AgentStatusType.AGENT_STEP,
                    msg=response,
                    event_id=event_id,
                )

    # Asynchronous handlers
    async def _ahandle_llm(self, payload: dict, event_id: str) -> None:
        if EventPayload.MESSAGES in payload:
            response = payload.get(EventPayload.RESPONSE)
            if response and str(response) not in ["None", "assistant: None"]:
                if self.fn:
                    # Convert response to consistent dict format
                    content = _extract_content_from_response(response)
                    if inspect.iscoroutinefunction(self.fn):
                        await self.fn(
                            status_type=AgentStatusType.AGENT_UPDATE,
                            msg={"content": content},
                            event_id=event_id,
                        )
                    else:
                        self.fn(
                            status_type=AgentStatusType.AGENT_UPDATE,
                            msg={"content": content},
                            event_id=event_id,
                        )
        elif EventPayload.PROMPT in payload:
            prompt = payload.get(EventPayload.PROMPT)
            if self.fn:
                # Convert prompt to consistent dict format
                content = str(prompt) if prompt else ""
                self.fn(
                    status_type=AgentStatusType.AGENT_UPDATE,
                    msg={"content": content},
                    event_id=event_id,
                )

    async def _ahandle_function_call(self, payload: dict, event_id: str) -> None:
        try:
            if EventPayload.FUNCTION_CALL in payload:
                fcall = payload.get(EventPayload.FUNCTION_CALL)
                tool = payload.get(EventPayload.TOOL)

                if tool and self.fn:
                    if inspect.iscoroutinefunction(self.fn):
                        await self.fn(
                            status_type=AgentStatusType.TOOL_CALL,
                            msg={"tool_name": tool.name, "arguments": fcall},
                            event_id=event_id,
                        )
                    else:
                        self.fn(
                            status_type=AgentStatusType.TOOL_CALL,
                            msg={"tool_name": tool.name, "arguments": fcall},
                            event_id=event_id,
                        )

            elif EventPayload.FUNCTION_OUTPUT in payload:
                response = payload.get(EventPayload.FUNCTION_OUTPUT)
                tool = payload.get(EventPayload.TOOL)

                if tool and self.fn:
                    if inspect.iscoroutinefunction(self.fn):
                        await self.fn(
                            status_type=AgentStatusType.TOOL_OUTPUT,
                            msg={
                                "tool_name": tool.name,
                                "content": response,
                            },
                            event_id=event_id,
                        )
                    else:
                        self.fn(
                            status_type=AgentStatusType.TOOL_OUTPUT,
                            msg={
                                "tool_name": tool.name,
                                "content": response,
                            },
                            event_id=event_id,
                        )

        except Exception as e:
            logging.error(f"Exception in _ahandle_function_call: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            # Continue execution to prevent callback failures from breaking the agent

    async def _ahandle_agent_step(self, payload: dict, event_id: str) -> None:
        if EventPayload.MESSAGES in payload:
            if self.fn:
                msg = payload.get(EventPayload.MESSAGES)
                if inspect.iscoroutinefunction(self.fn):
                    await self.fn(
                        status_type=AgentStatusType.AGENT_STEP,
                        msg=msg,
                        event_id=event_id,
                    )
                else:
                    self.fn(
                        status_type=AgentStatusType.AGENT_STEP,
                        msg=msg,
                        event_id=event_id,
                    )
        elif EventPayload.RESPONSE in payload:
            if self.fn:
                response = payload.get(EventPayload.RESPONSE)
                if inspect.iscoroutinefunction(self.fn):
                    await self.fn(
                        status_type=AgentStatusType.AGENT_STEP,
                        msg=response,
                        event_id=event_id,
                    )
                else:
                    self.fn(
                        status_type=AgentStatusType.AGENT_STEP,
                        msg=response,
                        event_id=event_id,
                    )
