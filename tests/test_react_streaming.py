# Suppress external dependency warnings before any other imports
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

import unittest
import asyncio
import gc

from vectara_agentic.agent import Agent
from vectara_agentic.tools import ToolsFactory
from vectara_agentic.llm_utils import clear_llm_cache

import nest_asyncio

nest_asyncio.apply()

from tests.conftest import (
    AgentTestMixin,
    react_config_openai,
    react_config_anthropic,
    react_config_gemini,
    react_config_together,
    mult,
    STANDARD_TEST_TOPIC,
    STANDARD_TEST_INSTRUCTIONS,
)


class TestReActStreaming(unittest.IsolatedAsyncioTestCase, AgentTestMixin):
    """Test streaming functionality for ReAct agents across all providers."""

    def setUp(self):
        super().setUp()
        self.tools = [ToolsFactory().create_tool(mult)]
        self.topic = STANDARD_TEST_TOPIC
        self.instructions = STANDARD_TEST_INSTRUCTIONS
        # Clear any cached LLM instances before each test
        clear_llm_cache()
        gc.collect()

    def tearDown(self):
        """Clean up after each test."""
        super().tearDown()
        # Clear cached LLM instances after each test
        clear_llm_cache()
        gc.collect()

    async def _test_react_streaming_workflow(self, config, provider_name):
        """Common workflow for testing ReAct streaming with any provider."""
        agent = Agent(
            agent_config=config,
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.instructions,
        )

        with self.with_provider_fallback(provider_name):
            # First calculation: 5 * 10 = 50
            stream1 = await agent.astream_chat(
                "What is 5 times 10. Only give the answer, nothing else"
            )
            # Consume the stream
            async for chunk in stream1.async_response_gen():
                pass
            response1 = await stream1.aget_response()
            self.check_response_and_skip(response1, provider_name)

            # Second calculation: 3 * 7 = 21
            stream2 = await agent.astream_chat(
                "what is 3 times 7. Only give the answer, nothing else"
            )
            # Consume the stream
            async for chunk in stream2.async_response_gen():
                pass
            response2 = await stream2.aget_response()
            self.check_response_and_skip(response2, provider_name)

            # Final calculation: 50 * 21 = 1050
            stream3 = await agent.astream_chat(
                "multiply the results of the last two multiplications. Only give the answer, nothing else."
            )
            # Consume the stream and collect chunks for verification
            chunks = []
            async for chunk in stream3.async_response_gen():
                chunks.append(chunk)

            response3 = await stream3.aget_response()
            self.check_response_and_skip(response3, provider_name)

            # Verify the final result
            self.assertIn("1050", response3.response)

            # Verify we actually got streaming chunks
            self.assertGreater(
                len(chunks), 0, f"{provider_name} should produce streaming chunks"
            )

    async def test_anthropic_react_streaming(self):
        """Test ReAct agent streaming with Anthropic."""
        await self._test_react_streaming_workflow(react_config_anthropic, "Anthropic")

    async def test_openai_react_streaming(self):
        """Test ReAct agent streaming with OpenAI."""
        await self._test_react_streaming_workflow(react_config_openai, "OpenAI")

    async def test_gemini_react_streaming(self):
        """Test ReAct agent streaming with Gemini."""
        # Extra cleanup for Gemini before starting
        clear_llm_cache()
        gc.collect()
        await asyncio.sleep(0.1)  # Give a moment for cleanup

        try:
            await self._test_react_streaming_workflow(react_config_gemini, "Gemini")
        finally:
            # Extra cleanup for Gemini after test
            clear_llm_cache()
            gc.collect()

    async def test_together_react_streaming(self):
        """Test ReAct agent streaming with Together.AI."""
        await self._test_react_streaming_workflow(react_config_together, "Together AI")

    async def test_react_streaming_reasoning_pattern(self):
        """Test that ReAct agents demonstrate reasoning patterns in streaming responses."""
        agent = Agent(
            agent_config=react_config_anthropic,
            tools=self.tools,
            topic=self.topic,
            custom_instructions="Think step by step and show your reasoning before using tools.",
        )

        with self.with_provider_fallback("Anthropic"):
            # Ask a question that requires multi-step reasoning
            stream = await agent.astream_chat(
                "I need to calculate 7 times 8, then add 12 to that result, then multiply by 2. "
                "Show me your reasoning process."
            )

            chunks = []
            async for chunk in stream.async_response_gen():
                chunks.append(str(chunk))

            response = await stream.aget_response()
            self.check_response_and_skip(response, "Anthropic")

            # Verify we got streaming content
            self.assertGreater(len(chunks), 0)

            # For ReAct agents, we should see reasoning patterns in the response
            full_content = "".join(chunks).lower()

            # The final answer should be correct: (7*8 + 12) * 2 = (56 + 12) * 2 = 68 * 2 = 136
            self.assertTrue("136" in response.response or "136" in full_content)


if __name__ == "__main__":
    unittest.main()
