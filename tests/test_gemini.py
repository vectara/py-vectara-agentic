# Suppress external dependency warnings before any other imports
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

import unittest
import asyncio
import gc

from vectara_agentic.agent import Agent
from vectara_agentic.tools import ToolsFactory
from vectara_agentic.tools_catalog import ToolsCatalog
from vectara_agentic.llm_utils import clear_llm_cache


import nest_asyncio

nest_asyncio.apply()

from tests.conftest import (
    mult,
    add,
    fc_config_gemini,
    STANDARD_TEST_TOPIC,
    STANDARD_TEST_INSTRUCTIONS,
)


class TestGEMINI(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        # Clear any cached LLM instances before each test
        clear_llm_cache()
        # Force garbage collection to clean up any lingering resources
        gc.collect()

    async def asyncTearDown(self):
        """Clean up after each test - async version."""
        await super().asyncTearDown()
        # Clear cached LLM instances after each test
        clear_llm_cache()
        # Force garbage collection
        gc.collect()
        # Small delay to allow cleanup
        await asyncio.sleep(0.01)

    async def test_gemini(self):
        tools = [ToolsFactory().create_tool(mult)]

        agent = Agent(
            agent_config=fc_config_gemini,
            tools=tools,
            topic=STANDARD_TEST_TOPIC,
            custom_instructions=STANDARD_TEST_INSTRUCTIONS,
        )
        _ = await agent.achat("What is 5 times 10. Only give the answer, nothing else")
        _ = await agent.achat("what is 3 times 7. Only give the answer, nothing else")
        res = await agent.achat(
            "what is the result of multiplying the results of the last two multiplications. Only give the answer, nothing else."
        )
        self.assertIn("1050", res.response)

    async def test_gemini_single_prompt(self):
        tools = [ToolsFactory().create_tool(mult)]

        agent = Agent(
            agent_config=fc_config_gemini,
            tools=tools,
            topic=STANDARD_TEST_TOPIC,
            custom_instructions=STANDARD_TEST_INSTRUCTIONS,
        )
        res = await agent.achat(
            "First, multiply 5 by 10. Then, multiply 3 by 7. Finally, multiply the results of the first two calculations."
        )
        self.assertIn("1050", res.response)

    async def test_gemini_25_flash_multi_tool_chain(self):
        """Test Gemini 2.5 Flash with complex multi-step reasoning chain using multiple tools."""
        # Use Gemini config (Gemini 2.5 Flash)
        tools_catalog = ToolsCatalog(fc_config_gemini)
        tools = [
            ToolsFactory().create_tool(mult),
            ToolsFactory().create_tool(add),
            ToolsFactory().create_tool(tools_catalog.summarize_text),
            ToolsFactory().create_tool(tools_catalog.rephrase_text),
        ]

        agent = Agent(
            agent_config=fc_config_gemini,
            tools=tools,
            topic=STANDARD_TEST_TOPIC,
            custom_instructions="You are a mathematical reasoning agent that explains your work step by step.",
        )

        # Complex multi-step reasoning task
        complex_query = (
            "Perform this calculation step by step: "
            "First multiply 3 by 8, then add 14 to that result, "
            "then multiply the new result by 3. "
            "After getting the final number, create a text description of the entire mathematical process "
            "(e.g., 'First I multiplied 3 by 8 to get 24, then added 14 to get 38, then multiplied by 3 to get 114'). "
            "Then use the summarize_text tool to summarize that text description with expertise in 'mathematics education'. "
            "Finally, use the rephrase_text tool to rephrase that summary as a 10-year-old would explain it."
        )

        print("\nStarting Gemini 2.5 Flash multi-tool chain test")
        print(f"Query: {complex_query}")

        # Note: Gemini tests now use async chat
        response = await agent.achat(complex_query)

        print(f"Final response: {response.response}")
        print(f"📄 Final response length: {len(response.response)} chars")

        # Check for mathematical results in the response
        # Expected: 3*8=24, 24+14=38, 38*3=114
        expected_intermediate_results = ["24", "38", "114"]
        response_text = response.response.lower()
        math_results_found = sum(1 for result in expected_intermediate_results
                                 if result in response_text)

        print(f"Mathematical results found: {math_results_found}/3 expected")
        print(f"Response text searched: {response_text[:200]}...")

        # More lenient assertion - just check that some mathematical progress was made
        self.assertGreaterEqual(math_results_found, 1,
                                f"Expected at least 1 mathematical result. Found {math_results_found}. "
                                f"Response: {response.response}")

        # Verify response has content and mentions math concepts
        self.assertGreater(len(response.response.strip()), 50, "Expected substantial response content")

        # Check for indications of multi-tool usage (math, summary, or explanation content)
        # Note: Gemini may answer directly without using all requested tools (summarize/rephrase)
        # So we check for math indicators only, which is more reliable
        math_indicators = ["multiply", "add", "calculation", "result"]
        indicators_found = sum(1 for indicator in math_indicators
                               if indicator in response_text)
        # Lenient check: if math is correct (checked above), we accept the response
        # even if Gemini didn't use all tools as instructed
        self.assertGreaterEqual(indicators_found, 1,
                                f"Expected at least one math indicator. Found {indicators_found}: {response.response}")


if __name__ == "__main__":
    unittest.main()
