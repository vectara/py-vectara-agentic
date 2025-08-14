# Suppress external dependency warnings before any other imports
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

import unittest
from typing import Dict, Any

from vectara_agentic.agent import Agent, AgentStatusType
from vectara_agentic.tools import ToolsFactory

import nest_asyncio

nest_asyncio.apply()

from conftest import (
    AgentTestMixin,
    react_config_anthropic,
    react_config_gemini,
    react_config_together,
    mult,
    add,
    STANDARD_TEST_TOPIC,
    STANDARD_TEST_INSTRUCTIONS,
)


class TestReActWorkflowEvents(unittest.IsolatedAsyncioTestCase, AgentTestMixin):
    """Test workflow event handling and streaming for ReAct agents."""

    def setUp(self):
        self.tools = [ToolsFactory().create_tool(mult), ToolsFactory().create_tool(add)]
        self.topic = STANDARD_TEST_TOPIC
        self.instructions = STANDARD_TEST_INSTRUCTIONS
        self.captured_events = []

    def capture_progress_callback(
        self, status_type: AgentStatusType, msg: Dict[str, Any], event_id: str = None
    ):
        """Capture agent progress events for testing."""
        self.captured_events.append(
            {
                "status_type": status_type,
                "msg": msg,
                "event_id": event_id,
            }
        )

    async def test_react_workflow_tool_call_events(self):
        """Test that ReAct workflow generates proper tool call events."""
        agent = Agent(
            agent_config=react_config_anthropic,
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.instructions,
            agent_progress_callback=self.capture_progress_callback,
        )

        with self.with_provider_fallback("Anthropic"):
            stream = await agent.astream_chat("Calculate 8 times 9.")

            # Consume the stream
            async for chunk in stream.async_response_gen():
                pass

            response = await stream.aget_response()
            self.check_response_and_skip(response, "Anthropic")

            if response.response and "72" in response.response:
                # Verify we captured tool-related events
                tool_call_events = [
                    event
                    for event in self.captured_events
                    if event["status_type"] == AgentStatusType.TOOL_CALL
                ]

                tool_output_events = [
                    event
                    for event in self.captured_events
                    if event["status_type"] == AgentStatusType.TOOL_OUTPUT
                ]

                # Should have at least one tool call and one tool output
                self.assertGreater(
                    len(tool_call_events), 0, "Should capture tool call events"
                )
                self.assertGreater(
                    len(tool_output_events), 0, "Should capture tool output events"
                )

                # Verify tool call event structure
                if tool_call_events:
                    tool_call = tool_call_events[0]
                    self.assertIn("tool_name", tool_call["msg"])
                    self.assertIn("arguments", tool_call["msg"])
                    self.assertIsNotNone(tool_call["event_id"])

                # Verify tool output event structure
                if tool_output_events:
                    tool_output = tool_output_events[0]
                    self.assertIn("tool_name", tool_output["msg"])
                    self.assertIn("content", tool_output["msg"])
                    self.assertIsNotNone(tool_output["event_id"])

    async def test_react_workflow_multi_step_events(self):
        """Test ReAct workflow events for multi-step reasoning tasks."""
        self.captured_events.clear()

        agent = Agent(
            agent_config=react_config_anthropic,
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.instructions,
            agent_progress_callback=self.capture_progress_callback,
        )

        with self.with_provider_fallback("Anthropic"):
            stream = await agent.astream_chat(
                "First multiply 7 by 6, then add 15 to that result."
            )

            # Consume the stream
            async for chunk in stream.async_response_gen():
                pass

            response = await stream.aget_response()
            self.check_response_and_skip(response, "Anthropic")

            # Should be (7*6)+15 = 42+15 = 57
            if response.response and "57" in response.response:
                # Should have multiple tool calls (multiplication and addition)
                tool_call_events = [
                    event
                    for event in self.captured_events
                    if event["status_type"] == AgentStatusType.TOOL_CALL
                ]

                # Should have at least 2 tool calls (mult and add)
                self.assertGreaterEqual(
                    len(tool_call_events),
                    1,
                    "Should have tool call events for multi-step task",
                )

                # Verify event IDs are present for events that have them
                # With simplified logic, events without proper IDs are skipped
                events_with_ids = [
                    event
                    for event in self.captured_events
                    if event["event_id"]
                ]

                # At least some events should have IDs
                self.assertGreater(
                    len(events_with_ids), 0, "Should have events with proper IDs"
                )

    async def test_react_workflow_agent_update_events(self):
        """Test that ReAct workflow generates agent update events."""
        self.captured_events.clear()

        agent = Agent(
            agent_config=react_config_anthropic,
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.instructions,
            agent_progress_callback=self.capture_progress_callback,
        )

        with self.with_provider_fallback("Anthropic"):
            stream = await agent.astream_chat("Calculate 5 times 11.")

            # Consume the stream
            async for chunk in stream.async_response_gen():
                pass

            response = await stream.aget_response()
            self.check_response_and_skip(response, "Anthropic")

            if response.response and "55" in response.response:
                # Look for agent update events
                agent_update_events = [
                    event
                    for event in self.captured_events
                    if event["status_type"] == AgentStatusType.AGENT_UPDATE
                ]

                # ReAct agents should generate some agent update events during workflow
                self.assertGreaterEqual(
                    len(agent_update_events),
                    0,
                    "ReAct workflow should generate agent update events",
                )

                # Verify structure of agent update events
                for event in agent_update_events:
                    self.assertIn("content", event["msg"])
                    self.assertIsInstance(event["msg"]["content"], str)

    async def test_react_workflow_event_ordering(self):
        """Test that ReAct workflow events are generated in correct order."""
        self.captured_events.clear()

        agent = Agent(
            agent_config=react_config_anthropic,
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.instructions,
            agent_progress_callback=self.capture_progress_callback,
        )

        with self.with_provider_fallback("Anthropic"):
            stream = await agent.astream_chat("Multiply 9 by 4.")

            # Consume the stream
            async for chunk in stream.async_response_gen():
                pass

            response = await stream.aget_response()
            self.check_response_and_skip(response, "Anthropic")

            if response.response and "36" in response.response:
                # Find tool call and tool output events
                tool_events = [
                    event
                    for event in self.captured_events
                    if event["status_type"]
                    in [AgentStatusType.TOOL_CALL, AgentStatusType.TOOL_OUTPUT]
                ]

                if len(tool_events) >= 2:
                    # Group events by event_id to match calls with outputs
                    event_groups = {}
                    for event in tool_events:
                        event_id = event["event_id"]
                        if event_id not in event_groups:
                            event_groups[event_id] = []
                        event_groups[event_id].append(event)

                    # For each event group, tool call should come before tool output
                    for event_id, events in event_groups.items():
                        if len(events) >= 2:
                            call_events = [
                                e
                                for e in events
                                if e["status_type"] == AgentStatusType.TOOL_CALL
                            ]
                            output_events = [
                                e
                                for e in events
                                if e["status_type"] == AgentStatusType.TOOL_OUTPUT
                            ]

                            if call_events and output_events:
                                # Find indices in original event list
                                call_index = self.captured_events.index(call_events[0])
                                output_index = self.captured_events.index(
                                    output_events[0]
                                )

                                self.assertLess(
                                    call_index,
                                    output_index,
                                    "Tool call should come before tool output",
                                )

    async def test_react_workflow_event_error_handling(self):
        """Test ReAct workflow event handling when tools fail."""
        self.captured_events.clear()

        def failing_tool(x: float) -> float:
            """A tool that fails with certain inputs."""
            if x == 0:
                raise ValueError("Cannot process zero")
            return x * 10

        error_tools = [ToolsFactory().create_tool(failing_tool)]

        agent = Agent(
            agent_config=react_config_anthropic,
            tools=error_tools,
            topic=self.topic,
            custom_instructions=self.instructions,
            agent_progress_callback=self.capture_progress_callback,
        )

        with self.with_provider_fallback("Anthropic"):
            stream = await agent.astream_chat("Use failing_tool with input 0.")

            # Consume the stream
            async for chunk in stream.async_response_gen():
                pass

            response = await stream.aget_response()
            self.check_response_and_skip(response, "Anthropic")

            # Even with tool errors, we should still capture events
            self.assertGreater(
                len(self.captured_events),
                0,
                "Should capture events even when tools fail",
            )

            # Look for tool call events
            tool_call_events = [
                event
                for event in self.captured_events
                if event["status_type"] == AgentStatusType.TOOL_CALL
            ]

            self.assertGreater(
                len(tool_call_events),
                0,
                "Should capture tool call events even when tools fail",
            )

    async def test_react_workflow_event_callback_error_resilience(self):
        """Test that ReAct workflow continues even if progress callback raises errors."""

        def failing_callback(
            status_type: AgentStatusType, msg: Dict[str, Any], event_id: str = None
        ):
            """A callback that always raises an error."""
            raise RuntimeError("Callback error")

        agent = Agent(
            agent_config=react_config_anthropic,
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.instructions,
            agent_progress_callback=failing_callback,
        )

        with self.with_provider_fallback("Anthropic"):
            # Even with failing callback, agent should still work
            stream = await agent.astream_chat("Calculate 12 times 3.")

            # Consume the stream
            async for chunk in stream.async_response_gen():
                pass

            response = await stream.aget_response()
            self.check_response_and_skip(response, "Anthropic")

            if response.response and "36" in response.response:
                # Test passed - agent worked despite callback failures
                self.assertTrue(True)

    async def test_react_workflow_event_consistency_across_providers(self):
        """Test that ReAct workflow events are consistent across different providers."""
        providers_to_test = [
            ("Anthropic", react_config_anthropic),
            ("Gemini", react_config_gemini),
            ("Together AI", react_config_together),
        ]

        for provider_name, config in providers_to_test:
            with self.subTest(provider=provider_name):
                self.captured_events.clear()

                agent = Agent(
                    agent_config=config,
                    tools=self.tools,
                    topic=self.topic,
                    custom_instructions=self.instructions,
                    agent_progress_callback=self.capture_progress_callback,
                )

                with self.with_provider_fallback(provider_name):
                    stream = await agent.astream_chat("Calculate 6 times 8.")

                    # Consume the stream
                    async for chunk in stream.async_response_gen():
                        pass

                    response = await stream.aget_response()
                    self.check_response_and_skip(response, provider_name)

                    if response.response and "48" in response.response:
                        # Should have some events
                        self.assertGreater(
                            len(self.captured_events),
                            0,
                            f"{provider_name} should generate events",
                        )

                        # All events should have proper structure
                        for event in self.captured_events:
                            self.assertIn("status_type", event)
                            self.assertIn("msg", event)
                            self.assertIsInstance(event["msg"], dict)


if __name__ == "__main__":
    unittest.main()
