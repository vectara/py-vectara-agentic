"""
Module to handle agent callbacks
"""

import inspect
from typing import Any, Dict, Optional, List, Callable
from functools import wraps

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
        if event_type == CBEventType.LLM:
            self._handle_llm(payload, event_id)
        elif event_type == CBEventType.FUNCTION_CALL:
            self._handle_function_call(payload, event_id)
        elif event_type == CBEventType.AGENT_STEP:
            self._handle_agent_step(payload, event_id)
        elif event_type == CBEventType.EXCEPTION:
            print(f"Exception in handle_event: {payload.get(EventPayload.EXCEPTION)}")
        else:
            print(f"Unknown event type: {event_type}, payload={payload}")

    async def _ahandle_event(
        self, event_type: CBEventType, payload: Dict[str, Any], event_id: str
    ) -> None:
        if event_type == CBEventType.LLM:
            await self._ahandle_llm(payload, event_id)
        elif event_type == CBEventType.FUNCTION_CALL:
            await self._ahandle_function_call(payload, event_id)
        elif event_type == CBEventType.AGENT_STEP:
            await self._ahandle_agent_step(payload, event_id)
        elif event_type == CBEventType.EXCEPTION:
            print(f"Exception in ahandle_event: {payload.get(EventPayload.EXCEPTION)}")
        else:
            print(f"Unknown event type: {event_type}, payload={payload}")

    # Synchronous handlers
    def _handle_llm(
        self,
        payload: dict,
        event_id: str,
    ) -> None:
        if EventPayload.MESSAGES in payload:
            response = str(payload.get(EventPayload.RESPONSE))
            if response and response not in ["None", "assistant: None"]:
                if self.fn:
                    self.fn(
                        status_type=AgentStatusType.AGENT_UPDATE,
                        msg=response,
                        event_id=event_id,
                    )
        elif EventPayload.PROMPT in payload:
            prompt = str(payload.get(EventPayload.PROMPT))
            if self.fn:
                self.fn(
                    status_type=AgentStatusType.AGENT_UPDATE,
                    msg=prompt,
                    event_id=event_id,
                )
        else:
            print(
                f"vectara-agentic llm callback: no messages or prompt found in payload {payload}"
            )

    def _handle_function_call(self, payload: dict, event_id: str) -> None:
        if EventPayload.FUNCTION_CALL in payload:
            fcall = str(payload.get(EventPayload.FUNCTION_CALL))
            tool = payload.get(EventPayload.TOOL)
            if tool:
                tool_name = tool.name
                if self.fn:
                    self.fn(
                        status_type=AgentStatusType.TOOL_CALL,
                        msg=f"Executing '{tool_name}' with arguments: {fcall}",
                        event_id=event_id,
                    )
        elif EventPayload.FUNCTION_OUTPUT in payload:
            response = str(payload.get(EventPayload.FUNCTION_OUTPUT))
            if self.fn:
                self.fn(
                    status_type=AgentStatusType.TOOL_OUTPUT,
                    msg=response,
                    event_id=event_id,
                )
        else:
            print(
                f"Vectara-agentic callback handler: no function call or output found in payload {payload}"
            )

    def _handle_agent_step(self, payload: dict, event_id: str) -> None:
        if EventPayload.MESSAGES in payload:
            msg = str(payload.get(EventPayload.MESSAGES))
            if self.fn:
                self.fn(
                    status_type=AgentStatusType.AGENT_STEP,
                    msg=msg,
                    event_id=event_id,
                )
        elif EventPayload.RESPONSE in payload:
            response = str(payload.get(EventPayload.RESPONSE))
            if self.fn:
                self.fn(
                    status_type=AgentStatusType.AGENT_STEP,
                    msg=response,
                    event_id=event_id,
                )
        else:
            print(
                f"Vectara-agentic agent_step: no messages or prompt found in payload {payload}"
            )

    # Asynchronous handlers
    async def _ahandle_llm(self, payload: dict, event_id: str) -> None:
        if EventPayload.MESSAGES in payload:
            response = str(payload.get(EventPayload.RESPONSE))
            if response and response not in ["None", "assistant: None"]:
                if self.fn:
                    if inspect.iscoroutinefunction(self.fn):
                        await self.fn(
                            status_type=AgentStatusType.AGENT_UPDATE,
                            msg=response,
                            event_id=event_id,
                        )
                    else:
                        self.fn(
                            status_type=AgentStatusType.AGENT_UPDATE,
                            msg=response,
                            event_id=event_id,
                        )
        elif EventPayload.PROMPT in payload:
            prompt = str(payload.get(EventPayload.PROMPT))
            if self.fn:
                self.fn(
                    status_type=AgentStatusType.AGENT_UPDATE,
                    msg=prompt,
                    event_id=event_id,
                )
        else:
            print(
                f"vectara-agentic llm callback: no messages or prompt found in payload {payload}"
            )

    async def _ahandle_function_call(self, payload: dict, event_id: str) -> None:
        if EventPayload.FUNCTION_CALL in payload:
            fcall = str(payload.get(EventPayload.FUNCTION_CALL))
            tool = payload.get(EventPayload.TOOL)
            if tool:
                tool_name = tool.name
                if self.fn:
                    if inspect.iscoroutinefunction(self.fn):
                        await self.fn(
                            status_type=AgentStatusType.TOOL_CALL,
                            msg=f"Executing '{tool_name}' with arguments: {fcall}",
                            event_id=event_id,
                        )
                    else:
                        self.fn(
                            status_type=AgentStatusType.TOOL_CALL,
                            msg=f"Executing '{tool_name}' with arguments: {fcall}",
                            event_id=event_id,
                        )
        elif EventPayload.FUNCTION_OUTPUT in payload:
            if self.fn:
                response = str(payload.get(EventPayload.FUNCTION_OUTPUT))
                if inspect.iscoroutinefunction(self.fn):
                    await self.fn(
                        status_type=AgentStatusType.TOOL_OUTPUT,
                        msg=response,
                        event_id=event_id,
                    )
                else:
                    self.fn(
                        status_type=AgentStatusType.TOOL_OUTPUT,
                        msg=response,
                        event_id=event_id,
                    )
        else:
            print(f"No function call or output found in payload {payload}")

    async def _ahandle_agent_step(self, payload: dict, event_id: str) -> None:
        if EventPayload.MESSAGES in payload:
            if self.fn:
                msg = str(payload.get(EventPayload.MESSAGES))
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
                response = str(payload.get(EventPayload.RESPONSE))
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
        else:
            print(f"No messages or prompt found in payload {payload}")
