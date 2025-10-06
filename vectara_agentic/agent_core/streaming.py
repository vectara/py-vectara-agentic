"""
Streaming utilities for agent responses.

This module provides streaming response handling, adapters, and utilities
for managing asynchronous agent interactions with proper synchronization.
"""

import asyncio
import logging
import json
import traceback
import uuid

from typing import Callable, Any, Dict, AsyncIterator

from llama_index.core.agent.workflow import (
    ToolCall,
    ToolCallResult,
    AgentInput,
    AgentOutput,
)
from ..types import AgentResponse, AgentStatusType


def get_event_id(event) -> str:
    """
    Get event ID from LlamaIndex event.

    Args:
        event: The event object from LlamaIndex

    Returns:
        str: Event ID from the event, or creates a new one if it does not exist
    """
    # Check for direct event_id first
    if hasattr(event, "event_id") and event.event_id:
        return event.event_id

    # Check for tool_id for tool-related events
    if hasattr(event, "tool_id") and event.tool_id:
        return event.tool_id

    return str(uuid.uuid4())


def is_tool_related_event(event) -> bool:
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

    has_tool_id = getattr(event, "tool_id", None)
    has_tool_name = getattr(event, "tool_name", None)
    has_delta = getattr(event, "delta", None)

    # Some providers don't emit ToolCall/ToolCallResult; avoid treating deltas as tool events
    if (has_tool_id or has_tool_name) and not has_delta:
        return True

    return False


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
    if hasattr(response_text, "blocks"):
        # Extract text from ChatMessage blocks
        text_parts = []
        for block in response_text.blocks:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        return "".join(text_parts)
    elif hasattr(response_text, "content"):
        return response_text.content
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
        except Exception as e:
            logging.error(f"Error {e} occurred during post-processing: {traceback.format_exc()}")
            return AgentResponse(response="", metadata={})

    return asyncio.create_task(_safe_post_process())


class FunctionCallingStreamHandler:
    """
    Streaming handler for function-calling agents with strict "no leaks" gating.

    Core ideas:
    - Buffer tokens PER LLM STEP.
    - Commit the buffer ONLY if that step ends with AgentOutput.tool_calls == [].
    - Drop the buffer if the step triggers tool calls (planning/tool-selection).
    - Track pending tool results; handle multi-round (tool -> read -> tool -> ...) loops.
    - Support return_direct tools (tool output is the final answer, no synthesis step).
    - Two streaming modes:
      - final_only: Buffer all tokens and commit only after step completes with no tool calls
      - optimistic_live: Stream tokens live after all tool calls are complete
    """

    def __init__(
        self,
        agent_instance,
        handler,
        prompt: str,
        *,
        stream_policy: str = "optimistic_live",            # "final_only" | "optimistic_live"
    ):
        self.agent_instance = agent_instance
        self.handler = handler  # awaitable; also has .stream_events()
        self.prompt = prompt

        self.stream_policy = stream_policy

        # Plumbing for your existing adapter/post-processing
        self.final_response_container = {"resp": None}
        self.stream_complete_event = asyncio.Event()

    async def process_stream_events(self) -> AsyncIterator[str]:
        """
        Process streaming events and yield only valid, final tokens.

        Contract:
        - Never surface "planning" tokens (tool arguments, scratchpads, etc).
        - Only surface tokens produced in the last, post-tool LLM step,
          or a return_direct tool's output.
        """
        # Step-scoped state
        step_buffer: list[str] = []
        step_has_tool_calls = False

        # Run-scoped state
        pending_tools = 0
        committed_any_text = False

        def _reset_step():
            nonlocal step_has_tool_calls
            step_buffer.clear()
            step_has_tool_calls = False

        async for ev in self.handler.stream_events():
            # ---- 1) Capture tool outputs for downstream logging/telemetry ----
            if isinstance(ev, ToolCallResult):
                if hasattr(self.agent_instance, "_add_tool_output"):
                    # pylint: disable=W0212
                    self.agent_instance._add_tool_output(ev.tool_name, str(ev.tool_output))

                pending_tools = max(0, pending_tools - 1)

                # Return-direct short-circuit: surface tool output as the final answer
                if getattr(ev, "return_direct", False):
                    yield str(ev.tool_output)
                    committed_any_text = True
                    # Do not early-break; keep draining events safely.

            # ---- 2) Progress callback plumbing (safe and optional) ----
            if self.agent_instance.agent_progress_callback:
                if is_tool_related_event(ev):
                    try:
                        event_id = get_event_id(ev)
                        await self._handle_progress_callback(ev, event_id)
                    except Exception as e:
                        logging.warning(f"[progress-callback] skipping event: {e}")

            # ---- 3) Step boundaries & gating logic ----
            # New step starts: clear per-step state
            if isinstance(ev, AgentInput):
                _reset_step()
                continue

            # Streaming deltas (provisional)
            if hasattr(ev, "__class__") and "AgentStream" in str(ev.__class__):
                # If the model is constructing a function call, LlamaIndex will attach tool_calls here
                if getattr(ev, "tool_calls", None):
                    step_has_tool_calls = True

                delta = getattr(ev, "delta", None)
                if not delta:
                    continue

                # Always buffer first
                step_buffer.append(delta)

                # Stream live only after all tools are complete
                if self.stream_policy == "optimistic_live" and pending_tools == 0:
                    yield delta

                continue

            # Step end: decide to commit or drop
            if isinstance(ev, AgentOutput):
                n_calls = len(getattr(ev, "tool_calls", []) or [])

                if n_calls == 0:
                    # Final text step -> commit
                    if self.stream_policy == "final_only":
                        # We held everything; now stream it out in order.
                        for chunk in step_buffer:
                            yield chunk
                    # In optimistic mode, tokens were streamed live after tools completed.

                    committed_any_text = committed_any_text or bool(step_buffer)
                    _reset_step()

                else:
                    # Planning/tool step -> drop buffer
                    _reset_step()
                    pending_tools += n_calls

                continue

        # ---- 4) Finish: await the underlying handler for the final result ----
        try:
            self.final_response_container["resp"] = await self.handler
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                logging.error(f"[RATE_LIMIT_ERROR] {e}")
                self.final_response_container["resp"] = AgentResponse(
                    response="Rate limit exceeded. Please try again later.",
                    source_nodes=[],
                    metadata={"error_type": "rate_limit", "original_error": str(e)},
                )
            else:
                logging.error(f"[STREAM_ERROR] {e}")
                self.final_response_container["resp"] = AgentResponse(
                    response="Response completion Error",
                    source_nodes=[],
                    metadata={"error_type": "general", "original_error": str(e)},
                )
        finally:
            # If nothing was ever committed and we ended right after a tool,
            # assume that tool's output is the "final answer" (common with return_direct).
            self.stream_complete_event.set()

    async def _handle_progress_callback(self, event, event_id: str):
        """
        Fan out progress events to the user's callback (sync or async). Mirrors your existing logic.
        """
        cb = self.agent_instance.agent_progress_callback
        is_async = asyncio.iscoroutinefunction(cb)

        try:
            if isinstance(event, ToolCall):
                payload = {
                    "tool_name": event.tool_name,
                    "arguments": json.dumps(getattr(event, "tool_kwargs", {})),
                }
                if is_async:
                    await cb(status_type=AgentStatusType.TOOL_CALL, msg=payload, event_id=event_id)
                else:
                    cb(status_type=AgentStatusType.TOOL_CALL, msg=payload, event_id=event_id)

            elif isinstance(event, ToolCallResult):
                payload = {
                    "tool_name": event.tool_name,
                    "content": str(event.tool_output),
                }
                if is_async:
                    await cb(status_type=AgentStatusType.TOOL_OUTPUT, msg=payload, event_id=event_id)
                else:
                    cb(status_type=AgentStatusType.TOOL_OUTPUT, msg=payload, event_id=event_id)

            elif isinstance(event, AgentInput):
                payload = {"content": f"Agent input: {getattr(event, 'input', '')}"}
                if is_async:
                    await cb(status_type=AgentStatusType.AGENT_UPDATE, msg=payload, event_id=event_id)
                else:
                    cb(status_type=AgentStatusType.AGENT_UPDATE, msg=payload, event_id=event_id)

            elif isinstance(event, AgentOutput):
                payload = {"content": f"Agent output: {getattr(event, 'response', '')}"}
                if is_async:
                    await cb(status_type=AgentStatusType.AGENT_UPDATE, msg=payload, event_id=event_id)
                else:
                    cb(status_type=AgentStatusType.AGENT_UPDATE, msg=payload, event_id=event_id)

        except Exception as e:
            logging.error(f"[progress-callback] Exception: {e}")
            logging.error(traceback.format_exc())

    def create_streaming_response(self, user_metadata: Dict[str, Any]) -> "StreamingResponseAdapter":
        """
        Build the adapter with post-processing wired in.
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
            response="",      # will be set by post-processing
            metadata={},      # will be set by post-processing
            post_process_task=post_process_task,
        )


class ReActStreamHandler:
    """
    Handles streaming for ReAct agents with proper event processing.

    ReAct agents use a workflow-based approach and emit ToolCall/ToolCallResult events
    that need to be captured and converted to progress callbacks.
    """

    def __init__(self, agent_instance, handler, prompt: str):
        self.agent_instance = agent_instance
        self.handler = handler
        self.prompt = prompt
        self.final_response_container = {"resp": None}
        self.stream_complete_event = asyncio.Event()

    async def process_stream_events(self) -> AsyncIterator[str]:
        """
        Process streaming events from ReAct workflow and yield text tokens.

        Yields:
            str: Text tokens from the streaming response
        """
        async for event in self.handler.stream_events():
            # Store tool outputs for VHC regardless of progress callback
            if isinstance(event, ToolCallResult):
                if hasattr(self.agent_instance, "_add_tool_output"):
                    # pylint: disable=W0212
                    self.agent_instance._add_tool_output(
                        event.tool_name, str(event.tool_output)
                    )
            # Handle progress callbacks if available - this is the key missing piece!
            if self.agent_instance.agent_progress_callback:
                # Only track events that are actual tool-related events
                if is_tool_related_event(event):
                    try:
                        # Get event ID from LlamaIndex event
                        event_id = get_event_id(event)

                        # Handle different types of workflow events using same logic as achat method
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
                            if asyncio.iscoroutinefunction(
                                self.agent_instance.agent_progress_callback
                            ):
                                await self.agent_instance.agent_progress_callback(
                                    status_type=AgentStatusType.AGENT_UPDATE,
                                    msg={"content": f"Agent input: {event.input}"},
                                    event_id=event_id,
                                )
                            else:
                                self.agent_instance.agent_progress_callback(
                                    status_type=AgentStatusType.AGENT_UPDATE,
                                    msg={"content": f"Agent input: {event.input}"},
                                    event_id=event_id,
                                )
                        elif isinstance(event, AgentOutput):
                            if asyncio.iscoroutinefunction(
                                self.agent_instance.agent_progress_callback
                            ):
                                await self.agent_instance.agent_progress_callback(
                                    status_type=AgentStatusType.AGENT_UPDATE,
                                    msg={"content": f"Agent output: {event.response}"},
                                    event_id=event_id,
                                )
                            else:
                                self.agent_instance.agent_progress_callback(
                                    status_type=AgentStatusType.AGENT_UPDATE,
                                    msg={"content": f"Agent output: {event.response}"},
                                    event_id=event_id,
                                )
                    except ValueError as e:
                        logging.warning(f"Skipping event due to missing ID: {e}")
                        continue
                    except Exception as e:
                        logging.error(f"Exception in ReAct progress callback: {e}")
                        logging.error(f"Traceback: {traceback.format_exc()}")
                        # Continue execution despite callback errors

            # For ReAct agents, we typically don't have streaming text like function calling
            # ReAct usually processes in steps and then provides complete responses
            # So we just yield empty strings to maintain streaming interface
            yield ""

        # When stream is done, await the handler to get the final response
        try:
            self.final_response_container["resp"] = await self.handler
        except Exception as e:
            logging.error(
                f"[REACT_STREAM_ERROR] Error processing ReAct stream events: {e}"
            )
            logging.error(
                f"[REACT_STREAM_ERROR] Full traceback: {traceback.format_exc()}"
            )
            self.final_response_container["resp"] = AgentResponse(
                response="ReAct Response completion Error", source_nodes=[], metadata={}
            )
        finally:
            # Signal that stream processing is complete
            self.stream_complete_event.set()

    def create_streaming_response(
        self, user_metadata: Dict[str, Any]
    ) -> "StreamingResponseAdapter":
        """
        Create a StreamingResponseAdapter for ReAct agents with proper post-processing.

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
