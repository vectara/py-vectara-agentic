"""
Streaming utilities for agent responses.

This module provides streaming response handling, adapters, and utilities
for managing asynchronous agent interactions with proper synchronization.
"""

import asyncio
import logging
import uuid
import json
import traceback

from typing import Callable, Any, Dict, AsyncIterator
from collections import OrderedDict

from llama_index.core.agent.workflow import (
    ToolCall,
    ToolCallResult,
    AgentInput,
    AgentOutput,
)
from ..types import AgentResponse

class ToolEventTracker:
    """
    Tracks event IDs for tool calls to ensure consistent pairing of tool calls and outputs.

    This class maintains a mapping between tool identifiers and event IDs to ensure
    that related tool call and tool output events share the same event_id for proper
    frontend grouping.
    """

    def __init__(self):
        self.event_ids = OrderedDict()  # tool_call_id -> event_id mapping
        self.fallback_counter = 0  # For events without identifiable tool_ids

    def get_event_id(self, event) -> str:
        """
        Get a consistent event ID for a tool event.

        Args:
            event: The tool event object

        Returns:
            str: Consistent event ID for this tool execution
        """
        # Try to get tool_id from the event first
        tool_id = getattr(event, "tool_id", None)

        # If we have a tool_id, use it directly (any format from any LLM provider)
        if tool_id:
            pass  # We already have tool_id, just use it
        # If no tool_id, try to derive one from tool_name (for LlamaIndex events)
        elif hasattr(event, "tool_name") and event.tool_name:
            tool_id = f"{event.tool_name}_{self.fallback_counter}"
            self.fallback_counter += 1
        # If still no tool_id, create a generic one based on event type
        else:
            event_type = type(event).__name__
            tool_id = f"{event_type.lower()}_{self.fallback_counter}"
            self.fallback_counter += 1

        # Get or create event_id for this tool_id
        if tool_id not in self.event_ids:
            self.event_ids[tool_id] = str(uuid.uuid4())

        return self.event_ids[tool_id]

    def clear_old_entries(self, max_entries: int = 100):
        """Clear old entries to prevent unbounded memory growth."""
        while len(self.event_ids) > max_entries // 2:
            self.event_ids.popitem(last=False)  # Remove oldest entry


class StreamingResponseAdapter:
    """
    Adapter class that provides a LlamaIndex-compatible streaming response interface.

    This class bridges custom streaming logic with AgentStreamingResponse expectations
    by implementing the required protocol methods and properties.
    """

    def __init__(
        self,
        async_response_gen: Callable[[], Any] | None = None,
        response: str = "",
        metadata: Dict[str, Any] | None = None,
        post_process_task: Any = None,
    ) -> None:
        """
        Initialize the streaming response adapter.

        Args:
            async_response_gen: Async generator function for streaming tokens
            response: Final response text (filled after streaming completes)
            metadata: Response metadata dictionary
            post_process_task: Async task that will populate response/metadata
        """
        self.async_response_gen = async_response_gen
        self.response = response
        self.metadata = metadata or {}
        self.post_process_task = post_process_task

    async def aget_response(self) -> AgentResponse:
        """
        Async version that waits for post-processing to complete.
        """
        if self.post_process_task:
            final_response = await self.post_process_task
            # Update our state with the final response
            self.response = final_response.response
            self.metadata = final_response.metadata or {}
        return AgentResponse(response=self.response, metadata=self.metadata)

    def get_response(self) -> AgentResponse:
        """
        Return an AgentResponse using the current state.

        Required by the _StreamProto protocol for AgentStreamingResponse compatibility.
        """
        return AgentResponse(response=self.response, metadata=self.metadata)

    def wait_for_completion(self) -> None:
        """
        Wait for post-processing to complete and update metadata.
        This should be called after streaming finishes but before accessing metadata.
        """
        if self.post_process_task and not self.post_process_task.done():
            return
        if self.post_process_task and self.post_process_task.done():
            try:
                final_response = self.post_process_task.result()
                if hasattr(final_response, "metadata") and final_response.metadata:
                    # Update our metadata from the completed task
                    self.metadata.update(final_response.metadata)
            except Exception as e:
                logging.error(
                    f"Error during post-processing: {e}. "
                    "Ensure the post-processing task is correctly implemented."
                )


def extract_response_text_from_chat_message(response_text: Any) -> str:
    """
    Extract text content from various response formats.

    Handles ChatMessage objects with blocks, content attributes, or plain strings.

    Args:
        response_text: Response object that may be ChatMessage, string, or other format

    Returns:
        str: Extracted text content
    """
    # Handle case where response is a ChatMessage object
    if hasattr(response_text, "content"):
        return response_text.content
    elif hasattr(response_text, "blocks"):
        # Extract text from ChatMessage blocks
        text_parts = []
        for block in response_text.blocks:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        return "".join(text_parts)
    elif not isinstance(response_text, str):
        return str(response_text)

    return response_text


async def execute_post_stream_processing(
    result: Any,
    prompt: str,
    agent_instance,
    user_metadata: Dict[str, Any],
) -> AgentResponse:
    """
    Execute post-stream processing on a completed result.

    This function consolidates the common post-processing steps that happen
    after streaming completes, including response extraction, formatting,
    callbacks, and FCS calculation.

    Args:
        result: The completed result object from streaming
        prompt: Original user prompt
        agent_instance: Agent instance for callbacks and processing
        user_metadata: User metadata to update with FCS scores

    Returns:
        AgentResponse: Processed final response
    """
    if result is None:
        logging.warning(
            "Received None result from streaming, returning empty response."
        )
        return AgentResponse(
            response="No response generated",
            metadata=getattr(result, "metadata", {}),
        )

    # Ensure we have an AgentResponse object with a string response
    if hasattr(result, "response"):
        response_text = result.response
    else:
        response_text = str(result)

    # Extract text from various response formats
    response_text = extract_response_text_from_chat_message(response_text)

    final = AgentResponse(
        response=response_text,
        metadata=getattr(result, "metadata", {}),
    )

    # Post-processing steps

    if agent_instance.query_logging_callback:
        agent_instance.query_logging_callback(prompt, final.response)

    # Let LlamaIndex handle agent memory naturally - no custom capture needed

    if not final.metadata:
        final.metadata = {}
    final.metadata.update(user_metadata)

    if agent_instance.observability_enabled:
        from .._observability import eval_fcs

        eval_fcs()

    return final


def create_stream_post_processing_task(
    stream_complete_event: asyncio.Event,
    final_response_container: Dict[str, Any],
    prompt: str,
    agent_instance,
    user_metadata: Dict[str, Any],
) -> asyncio.Task:
    """
    Create an async task for post-stream processing.

    Args:
        stream_complete_event: Event to wait for stream completion
        final_response_container: Container with final response data
        prompt: Original user prompt
        agent_instance: Agent instance for callbacks and processing
        user_metadata: User metadata to update with FCS scores

    Returns:
        asyncio.Task: Task that will process the final response
    """

    async def _post_process():
        # Wait until the generator has finished and final response is populated
        await stream_complete_event.wait()
        result = final_response_container.get("resp")
        return await execute_post_stream_processing(
            result, prompt, agent_instance, user_metadata
        )

    async def _safe_post_process():
        try:
            return await _post_process()
        except Exception:
            traceback.print_exc()
            # Return empty response on error
            return AgentResponse(response="", metadata={})

    return asyncio.create_task(_safe_post_process())


class FunctionCallingStreamHandler:
    """
    Handles streaming for function calling agents with proper event processing.
    """

    def __init__(self, agent_instance, handler, prompt: str):
        self.agent_instance = agent_instance
        self.handler = handler
        self.prompt = prompt
        self.final_response_container = {"resp": None}
        self.stream_complete_event = asyncio.Event()
        self.event_tracker = ToolEventTracker()

    async def process_stream_events(self) -> AsyncIterator[str]:
        """
        Process streaming events and yield text tokens.

        Yields:
            str: Text tokens from the streaming response
        """
        had_tool_calls = False
        transitioned_to_prose = False

        async for ev in self.handler.stream_events():
            # Store tool outputs for VHC regardless of progress callback
            if isinstance(ev, ToolCallResult):
                if hasattr(self.agent_instance, '_add_tool_output'):
                    # pylint: disable=W0212
                    self.agent_instance._add_tool_output(ev.tool_name, str(ev.tool_output))

            # Handle progress callbacks if available
            if self.agent_instance.agent_progress_callback:
                # Only track events that are actual tool-related events
                if self._is_tool_related_event(ev):
                    event_id = self.event_tracker.get_event_id(ev)
                    await self._handle_progress_callback(ev, event_id)

            # Process streaming text events
            if hasattr(ev, "__class__") and "AgentStream" in str(ev.__class__):
                if hasattr(ev, "tool_calls") and ev.tool_calls:
                    had_tool_calls = True
                elif (
                    hasattr(ev, "tool_calls")
                    and not ev.tool_calls
                    and had_tool_calls
                    and not transitioned_to_prose
                ):
                    yield "\n\n"
                    transitioned_to_prose = True
                    if hasattr(ev, "delta"):
                        yield ev.delta
                elif (
                    hasattr(ev, "tool_calls")
                    and not ev.tool_calls
                    and hasattr(ev, "delta")
                ):
                    yield ev.delta

        # When stream is done, await the handler to get the final response
        try:
            self.final_response_container["resp"] = await self.handler
        except Exception as e:
            logging.error(f"🔍 [STREAM_ERROR] Error processing stream events: {e}")
            logging.error(f"🔍 [STREAM_ERROR] Full traceback: {traceback.format_exc()}")
            self.final_response_container["resp"] = type(
                "AgentResponse",
                (),
                {
                    "response": "Response completion Error",
                    "source_nodes": [],
                    "metadata": None,
                },
            )()
        finally:
            # Clean up event tracker to prevent memory leaks
            self.event_tracker.clear_old_entries()
            # Signal that stream processing is complete
            self.stream_complete_event.set()

    def _is_tool_related_event(self, event) -> bool:
        """
        Determine if an event is actually tool-related and should be tracked.

        This should only return True for events that represent actual tool calls or tool outputs,
        not for streaming text deltas or other LLM response events.

        Args:
            event: The stream event to check

        Returns:
            bool: True if this event should be tracked for tool purposes
        """
        # Track explicit tool events from LlamaIndex workflow
        if isinstance(event, (ToolCall, ToolCallResult)):
            return True

        has_tool_id = hasattr(event, "tool_id") and event.tool_id
        has_delta = hasattr(event, "delta") and event.delta
        has_tool_name = hasattr(event, "tool_name") and event.tool_name

        # We're not seeing ToolCall/ToolCallResult events in the stream, so let's be more liberal
        # but still avoid streaming deltas
        if (has_tool_id or has_tool_name) and not has_delta:
            return True

        # Everything else (streaming deltas, agent outputs, workflow events, etc.)
        # should NOT be tracked as tool events
        return False

    async def _handle_progress_callback(self, event, event_id: str):
        """Handle progress callback events for different event types with proper context propagation."""
        # Import here to avoid circular imports
        from ..types import AgentStatusType

        try:
            if isinstance(event, ToolCall):
                # Check if callback is async or sync
                if asyncio.iscoroutinefunction(
                    self.agent_instance.agent_progress_callback
                ):
                    await self.agent_instance.agent_progress_callback(
                        status_type=AgentStatusType.TOOL_CALL,
                        msg={
                            "tool_name": event.tool_name,
                            "arguments": json.dumps(event.tool_kwargs),
                        },
                        event_id=event_id,
                    )
                else:
                    # For sync callbacks, ensure we call them properly
                    self.agent_instance.agent_progress_callback(
                        status_type=AgentStatusType.TOOL_CALL,
                        msg={
                            "tool_name": event.tool_name,
                            "arguments": json.dumps(event.tool_kwargs),
                        },
                        event_id=event_id,
                    )

            elif isinstance(event, ToolCallResult):
                # Check if callback is async or sync
                if asyncio.iscoroutinefunction(
                    self.agent_instance.agent_progress_callback
                ):
                    await self.agent_instance.agent_progress_callback(
                        status_type=AgentStatusType.TOOL_OUTPUT,
                        msg={
                            "tool_name": event.tool_name,
                            "content": str(event.tool_output),
                        },
                        event_id=event_id,
                    )
                else:
                    self.agent_instance.agent_progress_callback(
                        status_type=AgentStatusType.TOOL_OUTPUT,
                        msg={
                            "tool_name": event.tool_name,
                            "content": str(event.tool_output),
                        },
                        event_id=event_id,
                    )

            elif isinstance(event, AgentInput):
                self.agent_instance.agent_progress_callback(
                    status_type=AgentStatusType.AGENT_UPDATE,
                    msg={"content": f"Agent input: {event.input}"},
                    event_id=event_id,
                )

            elif isinstance(event, AgentOutput):
                self.agent_instance.agent_progress_callback(
                    status_type=AgentStatusType.AGENT_UPDATE,
                    msg={"content": f"Agent output: {event.response}"},
                    event_id=event_id,
                )

        except Exception as e:

            logging.error(f"Exception in progress callback: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            # Continue execution despite callback errors

    def create_streaming_response(
        self, user_metadata: Dict[str, Any]
    ) -> "StreamingResponseAdapter":
        """
        Create a StreamingResponseAdapter with proper post-processing.

        Args:
            user_metadata: User metadata dictionary to update

        Returns:
            StreamingResponseAdapter: Configured streaming adapter
        """
        post_process_task = create_stream_post_processing_task(
            self.stream_complete_event,
            self.final_response_container,
            self.prompt,
            self.agent_instance,
            user_metadata,
        )

        return StreamingResponseAdapter(
            async_response_gen=self.process_stream_events,
            response="",  # will be filled post-stream
            metadata={},
            post_process_task=post_process_task,
        )
