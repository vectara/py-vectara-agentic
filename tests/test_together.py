# Suppress external dependency warnings before any other imports
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

import unittest
import threading

from vectara_agentic.agent import Agent
from vectara_agentic.tools import ToolsFactory

import nest_asyncio

nest_asyncio.apply()

from conftest import (
    fc_config_together,
    mult,
    STANDARD_TEST_TOPIC,
    STANDARD_TEST_INSTRUCTIONS,
)
from vectara_agentic.agent_config import AgentConfig
from vectara_agentic.types import AgentType, ModelProvider


ARIZE_LOCK = threading.Lock()


class TestTogether(unittest.IsolatedAsyncioTestCase):

    async def test_multiturn(self):
        with ARIZE_LOCK:
            tools = [ToolsFactory().create_tool(mult)]
            agent = Agent(
                agent_config=fc_config_together,
                tools=tools,
                topic=STANDARD_TEST_TOPIC,
                custom_instructions=STANDARD_TEST_INSTRUCTIONS,
            )

            # First calculation: 5 * 10 = 50
            stream1 = await agent.astream_chat(
                "What is 5 times 10. Only give the answer, nothing else"
            )
            # Consume the stream
            async for chunk in stream1.async_response_gen():
                pass
            _ = await stream1.aget_response()

            # Second calculation: 3 * 7 = 21
            stream2 = await agent.astream_chat(
                "what is 3 times 7. Only give the answer, nothing else"
            )
            # Consume the stream
            async for chunk in stream2.async_response_gen():
                pass
            _ = await stream2.aget_response()

            # Final calculation: 50 * 21 = 1050
            stream3 = await agent.astream_chat(
                "multiply the results of the last two questions. Output only the answer."
            )
            # Consume the stream
            async for chunk in stream3.async_response_gen():
                pass
            response3 = await stream3.aget_response()

            self.assertEqual(response3.response, "1050")

    async def test_qwen3_coder(self):
        """Test Qwen3-Coder-480B-A35B-Instruct-FP8 model with Together AI provider."""
        with ARIZE_LOCK:
            # Create config specifically for Qwen3-Coder
            qwen_config = AgentConfig(
                agent_type=AgentType.FUNCTION_CALLING,
                main_llm_provider=ModelProvider.TOGETHER,
                main_llm_model_name="Qwen/Qwen3-235B-A22B-fp8-tput",
                tool_llm_provider=ModelProvider.TOGETHER,
                tool_llm_model_name="Qwen/Qwen3-235B-A22B-fp8-tput",
            )

            tools = [ToolsFactory().create_tool(mult)]
            agent = Agent(
                agent_config=qwen_config,
                tools=tools,
                topic=STANDARD_TEST_TOPIC,
                custom_instructions=STANDARD_TEST_INSTRUCTIONS,
            )

            # Test simple multiplication: 7 * 9 = 63
            stream = await agent.astream_chat(
                "What is 7 times 9? Only give the answer, nothing else"
            )
            # Consume the stream
            async for chunk in stream.async_response_gen():
                pass
            response = await stream.aget_response()

            # Verify the response contains the correct answer
            self.assertIn("63", response.response)

    async def test_llama4_scout(self):
        """Test Llama-4-Scout-17B-16E-Instruct model with Together AI provider."""
        with ARIZE_LOCK:
            # Create config specifically for Llama 4 Scout
            llama4_config = AgentConfig(
                agent_type=AgentType.FUNCTION_CALLING,
                main_llm_provider=ModelProvider.TOGETHER,
                main_llm_model_name="meta-llama/Llama-4-Scout-17B-16E-Instruct",
                tool_llm_provider=ModelProvider.TOGETHER,
                tool_llm_model_name="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            )

            tools = [ToolsFactory().create_tool(mult)]
            agent = Agent(
                agent_config=llama4_config,
                tools=tools,
                topic=STANDARD_TEST_TOPIC,
                custom_instructions=STANDARD_TEST_INSTRUCTIONS,
            )

            # Test simple multiplication: 8 * 6 = 48
            stream = await agent.astream_chat(
                "What is 8 times 6? Only give the answer, nothing else"
            )
            # Consume the stream
            async for chunk in stream.async_response_gen():
                pass
            response = await stream.aget_response()

            # Verify the response contains the correct answer
            self.assertIn("48", response.response)

    async def test_gpt_oss_120b(self):
        """Test OpenAI GPT-OSS-120B model with Together AI provider."""
        with ARIZE_LOCK:
            # Create config specifically for GPT-OSS-120B
            gpt_oss_120b_config = AgentConfig(
                agent_type=AgentType.FUNCTION_CALLING,
                main_llm_provider=ModelProvider.TOGETHER,
                main_llm_model_name="openai/gpt-oss-120b",
                tool_llm_provider=ModelProvider.TOGETHER,
                tool_llm_model_name="openai/gpt-oss-120b",
            )

            tools = [ToolsFactory().create_tool(mult)]
            agent = Agent(
                agent_config=gpt_oss_120b_config,
                tools=tools,
                topic=STANDARD_TEST_TOPIC,
                custom_instructions=STANDARD_TEST_INSTRUCTIONS,
            )

            # Test simple multiplication: 9 * 11 = 99
            stream = await agent.astream_chat(
                "What is 9 times 11? Only give the answer, nothing else"
            )
            # Consume the stream
            async for chunk in stream.async_response_gen():
                pass
            response = await stream.aget_response()

            # Verify the response contains the correct answer
            self.assertIn("99", response.response)


if __name__ == "__main__":
    unittest.main()
