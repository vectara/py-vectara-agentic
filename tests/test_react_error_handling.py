# Suppress external dependency warnings before any other imports
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

import unittest
import asyncio

from vectara_agentic.agent import Agent
from vectara_agentic.tools import ToolsFactory
from vectara_agentic.agent_config import AgentConfig
from vectara_agentic.types import AgentType, ModelProvider

import nest_asyncio

nest_asyncio.apply()

from conftest import (
    AgentTestMixin,
    react_config_anthropic,
    react_config_gemini,
    react_config_together,
    mult,
    STANDARD_TEST_TOPIC,
    STANDARD_TEST_INSTRUCTIONS,
    is_rate_limited,
    is_api_key_error,
)


class TestReActErrorHandling(unittest.TestCase, AgentTestMixin):
    """Test error handling and recovery for ReAct agents."""

    def setUp(self):
        self.tools = [ToolsFactory().create_tool(mult)]
        self.topic = STANDARD_TEST_TOPIC
        self.instructions = STANDARD_TEST_INSTRUCTIONS

    def test_react_anthropic_rate_limit_handling(self):
        """Test ReAct agent handling of Anthropic rate limits."""
        agent = Agent(
            agent_config=react_config_anthropic,
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.instructions,
        )

        with self.with_provider_fallback("Anthropic"):
            response = agent.chat("What is 5 times 10?")
            self.check_response_and_skip(response, "Anthropic")

            # If we get a response, check it's valid
            if response.response and not is_rate_limited(response.response):
                self.assertIn("50", response.response)

    def test_react_openai_error_handling(self):
        """Test ReAct agent handling of OpenAI-specific errors."""
        openai_react_config = AgentConfig(
            agent_type=AgentType.REACT,
            main_llm_provider=ModelProvider.OPENAI,
            tool_llm_provider=ModelProvider.OPENAI,
        )

        agent = Agent(
            agent_config=openai_react_config,
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.instructions,
        )

        with self.with_provider_fallback("OpenAI"):
            response = agent.chat("Calculate 7 times 8.")
            self.check_response_and_skip(response, "OpenAI")

            # If we get a response, check it's valid
            if response.response and not is_rate_limited(response.response):
                self.assertIn("56", response.response)

    def test_react_together_error_handling(self):
        """Test ReAct agent handling of Together.AI errors."""
        agent = Agent(
            agent_config=react_config_together,
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.instructions,
        )

        with self.with_provider_fallback("Together AI"):
            response = agent.chat("Calculate 4 times 15.")
            self.check_response_and_skip(response, "Together AI")

            # If we get a response, check it's valid
            if response.response and not is_rate_limited(response.response):
                self.assertIn("60", response.response)

    def test_react_gemini_error_handling(self):
        """Test ReAct agent handling of Gemini-specific errors."""
        agent = Agent(
            agent_config=react_config_gemini,
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.instructions,
        )

        with self.with_provider_fallback("Gemini"):
            response = agent.chat("What is 12 times 4?")
            self.check_response_and_skip(response, "Gemini")

            # If we get a response, check it's valid
            if response.response and not is_rate_limited(response.response):
                self.assertIn("48", response.response)

    def test_react_async_error_handling(self):
        """Test ReAct agent error handling during async operations."""
        async def _async_test():
            agent = Agent(
                agent_config=react_config_anthropic,
                tools=self.tools,
                topic=self.topic,
                custom_instructions=self.instructions,
            )

            with self.with_provider_fallback("Anthropic"):
                # Test async chat error handling
                response = await agent.achat("Calculate 11 times 3.")
                self.check_response_and_skip(response, "Anthropic")

                if response.response and not is_rate_limited(response.response):
                    self.assertIn("33", response.response)

        asyncio.run(_async_test())

    def test_react_streaming_error_handling(self):
        """Test ReAct agent error handling during streaming operations."""
        async def _async_test():
            agent = Agent(
                agent_config=react_config_anthropic,
                tools=self.tools,
                topic=self.topic,
                custom_instructions=self.instructions,
            )

            with self.with_provider_fallback("Anthropic"):
                try:
                    stream = await agent.astream_chat("Calculate 13 times 2.")

                    # Consume the stream
                    chunks = []
                    async for chunk in stream.async_response_gen():
                        chunks.append(str(chunk))

                    response = await stream.aget_response()
                    self.check_response_and_skip(response, "Anthropic")

                    if response.response and not is_rate_limited(response.response):
                        self.assertIn("26", response.response)

                except Exception as e:
                    error_msg = str(e)
                    if is_rate_limited(error_msg) or is_api_key_error(error_msg):
                        self.skipTest(f"Anthropic streaming error: {error_msg}")
                    raise

        asyncio.run(_async_test())

    def test_react_tool_execution_error_handling(self):
        """Test ReAct agent handling of tool execution errors."""

        def error_tool(x: float) -> float:
            """A tool that intentionally raises an error."""
            if x < 0:
                raise ValueError("Cannot process negative numbers")
            return x * 2

        error_tools = [ToolsFactory().create_tool(error_tool)]

        agent = Agent(
            agent_config=react_config_anthropic,
            tools=error_tools,
            topic=self.topic,
            custom_instructions=self.instructions,
        )

        with self.with_provider_fallback("Anthropic"):
            # Test with valid input first
            response1 = agent.chat("Use the error_tool with input 5.")
            self.check_response_and_skip(response1, "Anthropic")

            if not is_rate_limited(response1.response):
                self.assertIn("10", response1.response)  # 5 * 2 = 10

            # Test with invalid input that should cause tool error
            response2 = agent.chat("Use the error_tool with input -1.")
            self.check_response_and_skip(response2, "Anthropic")

            if not is_rate_limited(response2.response):
                # ReAct agent should handle the tool error gracefully
                # and provide some kind of error message or explanation
                self.assertTrue(len(response2.response) > 0)
                # Should mention error or inability to process
                error_indicators = ["error", "cannot", "unable", "negative", "problem"]
                has_error_indication = any(
                    indicator in response2.response.lower()
                    for indicator in error_indicators
                )
                self.assertTrue(
                    has_error_indication,
                    f"Response should indicate error handling: {response2.response}",
                )

    def test_react_workflow_interruption_handling(self):
        """Test ReAct agent handling of workflow interruptions."""
        agent = Agent(
            agent_config=react_config_anthropic,
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.instructions,
        )

        with self.with_provider_fallback("Anthropic"):
            # Test a complex multi-step task that might be interrupted
            response = agent.chat(
                "Calculate 3 times 7, then multiply that by 4, then add 10 to the result."
            )
            self.check_response_and_skip(response, "Anthropic")

            if response.response and not is_rate_limited(response.response):
                # Final result should be: (3*7)*4+10 = 21*4+10 = 84+10 = 94
                self.assertIn("94", response.response)

    def test_react_memory_corruption_recovery(self):
        """Test ReAct agent recovery from memory-related errors."""
        agent = Agent(
            agent_config=react_config_anthropic,
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.instructions,
        )

        with self.with_provider_fallback("Anthropic"):
            # Normal operation
            response1 = agent.chat("Calculate 6 times 7.")
            self.check_response_and_skip(response1, "Anthropic")

            if not is_rate_limited(response1.response):
                self.assertIn("42", response1.response)

                # Continue conversation to test memory consistency
                response2 = agent.chat(
                    "What was the result of the previous calculation?"
                )
                self.check_response_and_skip(response2, "Anthropic")

                if not is_rate_limited(response2.response):
                    # Should remember the previous result
                    self.assertIn("42", response2.response)

    def test_react_fallback_behavior_on_provider_failure(self):
        """Test ReAct agent behavior when provider fails completely."""

        # Create a config with an invalid API configuration to simulate failure
        invalid_config = AgentConfig(
            agent_type=AgentType.REACT,
            main_llm_provider=ModelProvider.ANTHROPIC,
            tool_llm_provider=ModelProvider.ANTHROPIC,
        )

        agent = Agent(
            agent_config=invalid_config,
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.instructions,
        )

        with self.with_provider_fallback("Anthropic"):
            try:
                response = agent.chat("What is 2 times 3?")
                self.check_response_and_skip(response, "Anthropic")

                # If we get here without error, the test passed
                if response.response and not is_rate_limited(response.response):
                    self.assertIn("6", response.response)

            except Exception as e:
                error_msg = str(e)
                if is_rate_limited(error_msg) or is_api_key_error(error_msg):
                    self.skipTest(f"Anthropic provider failure: {error_msg}")
                # For other exceptions, let them bubble up as actual test failures
                raise


if __name__ == "__main__":
    unittest.main()
