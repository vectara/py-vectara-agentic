"""
Streaming utilities for agent responses.

This module provides streaming response handling, adapters, and utilities
for managing asynchronous agent interactions with proper synchronization.
"""

import asyncio
import uuid
import json
from typing import Callable, Any, Dict, AsyncIterator

from ..types import AgentResponse


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

        if result is None:
            # Fallback response if something went wrong
            result = type(
                "AgentResponse",
                (),
                {
                    "response": "Response completed",
                    "source_nodes": [],
                    "metadata": None,
                },
            )()

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
        # pylint: disable=protected-access
        await agent_instance._aformat_for_lats(prompt, final)
        if agent_instance.query_logging_callback:
            agent_instance.query_logging_callback(prompt, final.response)

        # Calculate factual consistency score
        # pylint: disable=protected-access
        fcs = agent_instance._calc_fcs(final.response)
        if fcs is not None:
            user_metadata["fcs"] = fcs
        if agent_instance.observability_enabled:
            from .._observability import eval_fcs

            eval_fcs()

        return final

    async def _safe_post_process():
        try:
            return await _post_process()
        except Exception:
            import traceback

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

    async def process_stream_events(self) -> AsyncIterator[str]:
        """
        Process streaming events and yield text tokens.

        Yields:
            str: Text tokens from the streaming response
        """
        had_tool_calls = False
        transitioned_to_prose = False

        async for ev in self.handler.stream_events():
            # Handle progress callbacks if available
            if self.agent_instance.agent_progress_callback:
                event_id = str(uuid.uuid4())
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
        except Exception:
            self.final_response_container["resp"] = type(
                "AgentResponse",
                (),
                {
                    "response": "Response completed",
                    "source_nodes": [],
                    "metadata": None,
                },
            )()
        finally:
            # Signal that stream processing is complete
            self.stream_complete_event.set()

    async def _handle_progress_callback(self, event, event_id: str):
        """Handle progress callback events for different event types."""
        # Import here to avoid circular imports
        from ..types import AgentStatusType
        from llama_index.core.agent.workflow import (
            ToolCall,
            ToolCallResult,
            AgentInput,
            AgentOutput,
        )

        if isinstance(event, ToolCall):
            self.agent_instance.agent_progress_callback(
                AgentStatusType.TOOL_CALL,
                {
                    "tool_name": event.tool_name,
                    "arguments": json.dumps(event.tool_kwargs),
                },
                event_id,
            )
        elif isinstance(event, ToolCallResult):
            self.agent_instance.agent_progress_callback(
                AgentStatusType.TOOL_OUTPUT,
                {
                    "content": str(event.tool_output),
                },
                event_id,
            )
        elif isinstance(event, AgentInput):
            self.agent_instance.agent_progress_callback(
                AgentStatusType.AGENT_UPDATE,
                f"Agent input: {event.input}",
                event_id,
            )
        elif isinstance(event, AgentOutput):
            self.agent_instance.agent_progress_callback(
                AgentStatusType.AGENT_UPDATE,
                f"Agent output: {event.response}",
                event_id,
            )

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


class StandardStreamHandler:
    """
    Handles streaming for standard LlamaIndex agents.
    """

    def __init__(self, agent_instance, li_stream, prompt: str):
        self.agent_instance = agent_instance
        self.li_stream = li_stream
        self.prompt = prompt

    async def create_wrapped_stream_generator(
        self, user_metadata: Dict[str, Any]
    ) -> AsyncIterator[str]:
        """
        Create wrapped stream generator with post-processing.

        Args:
            user_metadata: User metadata dictionary to update

        Yields:
            str: Text tokens from the original stream
        """
        orig_async = self.li_stream.async_response_gen

        async for tok in orig_async():
            yield tok

        # Post-stream hooks
        # pylint: disable=protected-access
        await self.agent_instance._aformat_for_lats(self.prompt, self.li_stream)
        if self.agent_instance.query_logging_callback:
            self.agent_instance.query_logging_callback(
                self.prompt, self.li_stream.response
            )

        # pylint: disable=protected-access
        fcs = self.agent_instance._calc_fcs(self.li_stream.response)
        if fcs is not None:
            user_metadata["fcs"] = fcs
        if self.agent_instance.observability_enabled:
            from .._observability import eval_fcs

            eval_fcs()

    def wrap_stream_response(self, user_metadata: Dict[str, Any]):
        """
        Wrap the LlamaIndex stream response with post-processing.

        Args:
            user_metadata: User metadata dictionary to update
        """
        self.li_stream.async_response_gen = (
            lambda: self.create_wrapped_stream_generator(user_metadata)
        )
        return self.li_stream
