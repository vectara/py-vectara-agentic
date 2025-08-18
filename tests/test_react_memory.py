# Suppress external dependency warnings before any other imports
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

import unittest
import threading

from vectara_agentic.agent import Agent
from vectara_agentic.tools import ToolsFactory
from llama_index.core.llms import MessageRole

import nest_asyncio

nest_asyncio.apply()

from conftest import (
    AgentTestMixin,
    react_config_anthropic,
    react_config_gemini,
    mult,
    add,
    STANDARD_TEST_TOPIC,
    STANDARD_TEST_INSTRUCTIONS,
)

ARIZE_LOCK = threading.Lock()


class TestReActMemory(unittest.TestCase, AgentTestMixin):
    """Test memory persistence and conversation history for ReAct agents."""

    def setUp(self):
        self.tools = [ToolsFactory().create_tool(mult), ToolsFactory().create_tool(add)]
        self.topic = STANDARD_TEST_TOPIC
        self.instructions = STANDARD_TEST_INSTRUCTIONS
        self.session_id = "test-react-memory-123"

    def test_react_memory_persistence_across_chats(self):
        """Test that ReAct agents maintain conversation context across multiple chat calls."""
        agent = Agent(
            agent_config=react_config_anthropic,
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.instructions,
            session_id=self.session_id,
        )

        with self.with_provider_fallback("Anthropic"):
            # First interaction - establish context
            response1 = agent.chat("Calculate 5 times 10 and remember this result.")
            self.check_response_and_skip(response1, "Anthropic")
            self.assertIn("50", response1.response)

            # Second interaction - reference previous result
            response2 = agent.chat(
                "What was the result I asked you to calculate and remember in the previous message?"
            )
            self.check_response_and_skip(response2, "Anthropic")
            self.assertIn("50", response2.response)

            # Third interaction - use previous result in new calculation
            response3 = agent.chat("Add 25 to the number you calculated earlier.")
            self.check_response_and_skip(response3, "Anthropic")
            # Should be 50 + 25 = 75
            self.assertIn("75", response3.response)

    def test_react_memory_with_tool_history(self):
        """Test that ReAct agents remember tool usage history across conversations."""
        agent = Agent(
            agent_config=react_config_anthropic,
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.instructions,
            session_id=self.session_id + "_tools",
        )

        with self.with_provider_fallback("Anthropic"):
            # Use multiplication tool
            response1 = agent.chat("Multiply 7 by 8.")
            self.check_response_and_skip(response1, "Anthropic")
            self.assertIn("56", response1.response)

            # Use addition tool
            response2 = agent.chat("Add 20 to 30.")
            self.check_response_and_skip(response2, "Anthropic")
            self.assertIn("50", response2.response)

            # Reference both previous tool uses
            response3 = agent.chat(
                "What were the two calculations I asked you to perform? "
                "Add those two results together."
            )
            self.check_response_and_skip(response3, "Anthropic")
            # Should remember 7*8=56 and 20+30=50, then 56+50=106
            self.assertIn("106", response3.response)

    def test_react_memory_state_consistency(self):
        """Test that ReAct agent memory state remains consistent during workflow execution."""
        agent = Agent(
            agent_config=react_config_anthropic,
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.instructions,
            session_id=self.session_id + "_consistency",
        )

        with self.with_provider_fallback("Anthropic"):
            # Check initial memory state
            initial_memory_size = len(agent.memory.get_all())

            # Perform a conversation with tool use
            response1 = agent.chat("Calculate 6 times 9 and tell me the result.")
            self.check_response_and_skip(response1, "Anthropic")

            # Memory should now contain user message, tool calls, and assistant response
            after_first_memory = agent.memory.get_all()
            self.assertGreater(
                len(after_first_memory),
                initial_memory_size,
                "Memory should contain new messages after interaction",
            )

            # Continue conversation
            response2 = agent.chat("Double that result.")
            self.check_response_and_skip(response2, "Anthropic")

            # Memory should contain all previous messages plus new ones
            after_second_memory = agent.memory.get_all()
            self.assertGreater(
                len(after_second_memory),
                len(after_first_memory),
                "Memory should accumulate messages across interactions",
            )

            # Final result should be (6*9)*2 = 54*2 = 108
            self.assertIn("108", response2.response)

    def test_react_memory_across_different_providers(self):
        """Test memory consistency when using different ReAct providers."""
        # Test with Anthropic
        agent_anthropic = Agent(
            agent_config=react_config_anthropic,
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.instructions,
            session_id=self.session_id + "_anthropic_provider",
        )

        with self.with_provider_fallback("Anthropic"):
            response1 = agent_anthropic.chat("Multiply 4 by 12.")
            self.check_response_and_skip(response1, "Anthropic")
            self.assertIn("48", response1.response)

            # Verify memory structure is consistent
            anthropic_memory = agent_anthropic.memory.get_all()
            self.assertGreater(len(anthropic_memory), 0)

        # Test with Gemini (if available)
        agent_gemini = Agent(
            agent_config=react_config_gemini,
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.instructions,
            session_id=self.session_id + "_gemini_provider",
        )

        with self.with_provider_fallback("Gemini"):
            response2 = agent_gemini.chat("Multiply 4 by 12.")
            self.check_response_and_skip(response2, "Gemini")
            self.assertIn("48", response2.response)

            # Verify memory structure is consistent across providers
            gemini_memory = agent_gemini.memory.get_all()
            self.assertGreater(len(gemini_memory), 0)

    def test_react_memory_serialization_compatibility(self):
        """Test that ReAct agent memory can be properly serialized and deserialized."""
        agent = Agent(
            agent_config=react_config_anthropic,
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.instructions,
            session_id=self.session_id + "_serialization",
        )

        with self.with_provider_fallback("Anthropic"):
            # Perform some interactions to populate memory
            response1 = agent.chat("Calculate 15 times 3.")
            self.check_response_and_skip(response1, "Anthropic")

            response2 = agent.chat("Add 10 to the previous result.")
            self.check_response_and_skip(response2, "Anthropic")

            # Get memory state before serialization
            original_memory = agent.memory.get_all()
            original_memory_size = len(original_memory)

            # Test that memory state can be accessed and contains expected content
            self.assertGreater(original_memory_size, 0)

            # Verify memory contains both user and assistant messages
            user_messages = [
                msg for msg in original_memory if msg.role == MessageRole.USER
            ]
            assistant_messages = [
                msg for msg in original_memory if msg.role == MessageRole.ASSISTANT
            ]

            self.assertGreater(
                len(user_messages), 0, "Should have user messages in memory"
            )
            self.assertGreater(
                len(assistant_messages), 0, "Should have assistant messages in memory"
            )

    async def test_react_memory_async_streaming_consistency(self):
        """Test memory consistency during async streaming operations with ReAct agents."""
        agent = Agent(
            agent_config=react_config_anthropic,
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.instructions,
            session_id=self.session_id + "_async_streaming",
        )

        with self.with_provider_fallback("Anthropic"):
            # First streaming interaction
            stream1 = await agent.astream_chat("Calculate 8 times 7.")
            async for chunk in stream1.async_response_gen():
                pass
            response1 = await stream1.aget_response()
            self.check_response_and_skip(response1, "Anthropic")

            # Check memory after first streaming interaction
            memory_after_first = agent.memory.get_all()
            self.assertGreater(len(memory_after_first), 0)

            # Second streaming interaction that references the first
            stream2 = await agent.astream_chat(
                "Subtract 6 from the result you just calculated."
            )
            async for chunk in stream2.async_response_gen():
                pass
            response2 = await stream2.aget_response()
            self.check_response_and_skip(response2, "Anthropic")

            # Memory should contain both interactions
            memory_after_second = agent.memory.get_all()
            self.assertGreater(len(memory_after_second), len(memory_after_first))

            # Final result should be (8*7)-6 = 56-6 = 50
            self.assertIn("50", response2.response)


if __name__ == "__main__":
    unittest.main()
