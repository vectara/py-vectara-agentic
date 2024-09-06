"""
Callback handler to track agent status
"""

from typing import Any, Dict, Callable, Optional, List

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload

from .types import AgentStatusType


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
        self.fn = fn

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        pass

    def _handle_llm(self, payload: dict) -> None:
        """Calls self.fn() with the message from the LLM."""
        if EventPayload.MESSAGES in payload:
            response = str(payload.get(EventPayload.RESPONSE))
            if response and response != "None" and response != "assistant: None":
                if self.fn:
                    self.fn(AgentStatusType.AGENT_UPDATE, response)
        else:
            print("No messages or prompt found in payload")

    def _handle_function_call(self, payload: dict) -> None:
        """Calls self.fn() with the information about tool calls."""
        if EventPayload.FUNCTION_CALL in payload:
            fcall = str(payload.get(EventPayload.FUNCTION_CALL))
            tool = payload.get(EventPayload.TOOL)
            if tool:
                tool_name = tool.name
                if self.fn:
                    self.fn(
                        AgentStatusType.TOOL_CALL,
                        f"Executing '{tool_name}' with arguments: {fcall}",
                    )
        elif EventPayload.FUNCTION_OUTPUT in payload:
            response = str(payload.get(EventPayload.FUNCTION_OUTPUT))
            if self.fn:
                self.fn(AgentStatusType.TOOL_OUTPUT, response)
        else:
            print("No function call or output found in payload")

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        if self.fn is not None and payload is not None:
            if event_type == CBEventType.LLM:
                self._handle_llm(payload)
            elif event_type == CBEventType.FUNCTION_CALL:
                self._handle_function_call(payload)
            elif event_type == CBEventType.AGENT_STEP:
                pass  # Do nothing
            elif event_type == CBEventType.EXCEPTION:
                print(f"Exception: {payload.get(EventPayload.EXCEPTION)}")
            else:
                print(f"Unknown event type: {event_type}, payload={payload}")
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Count the LLM or Embedding tokens as needed."""
        if self.fn is not None and payload is not None:
            if event_type == CBEventType.LLM:
                self._handle_llm(payload)
            elif event_type == CBEventType.FUNCTION_CALL:
                self._handle_function_call(payload)
