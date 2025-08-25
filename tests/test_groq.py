# Suppress external dependency warnings before any other imports
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

import unittest
import threading

from vectara_agentic.agent import Agent
from vectara_agentic.tools import ToolsFactory
from vectara_agentic.agent_config import AgentConfig
from vectara_agentic.types import AgentType, ModelProvider

import nest_asyncio

nest_asyncio.apply()

from conftest import (
    mult,
    fc_config_groq,
    STANDARD_TEST_TOPIC,
    STANDARD_TEST_INSTRUCTIONS,
)

ARIZE_LOCK = threading.Lock()


class TestGROQ(unittest.IsolatedAsyncioTestCase):

    async def test_multiturn(self):
        with ARIZE_LOCK:
            tools = [ToolsFactory().create_tool(mult)]
            agent = Agent(
                tools=tools,
                topic=STANDARD_TEST_TOPIC,
                custom_instructions=STANDARD_TEST_INSTRUCTIONS,
                agent_config=fc_config_groq,
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

    async def test_gpt_oss_120b(self):
        """Test GPT-OSS-120B model with GROQ provider."""
        with ARIZE_LOCK:
            # Create config specifically for GPT-OSS-120B via GROQ
            gpt_oss_config = AgentConfig(
                agent_type=AgentType.FUNCTION_CALLING,
                main_llm_provider=ModelProvider.GROQ,
                main_llm_model_name="openai/gpt-oss-120b",
                tool_llm_provider=ModelProvider.GROQ,
                tool_llm_model_name="openai/gpt-oss-120b",
            )

            tools = [ToolsFactory().create_tool(mult)]
            agent = Agent(
                agent_config=gpt_oss_config,
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


if __name__ == "__main__":
    unittest.main()
