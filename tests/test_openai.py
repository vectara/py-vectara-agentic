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
    fc_config_openai,
    mult,
    STANDARD_TEST_TOPIC,
    STANDARD_TEST_INSTRUCTIONS,
)


ARIZE_LOCK = threading.Lock()


class TestOpenAI(unittest.IsolatedAsyncioTestCase):

    async def test_multiturn(self):
        """Test multi-turn conversation with default OpenAI model."""
        with ARIZE_LOCK:
            tools = [ToolsFactory().create_tool(mult)]
            agent = Agent(
                agent_config=fc_config_openai,
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

    async def test_gpt_4o(self):
        """Test GPT-4o model with OpenAI provider."""
        with ARIZE_LOCK:
            config = AgentConfig(
                agent_type=AgentType.FUNCTION_CALLING,
                main_llm_provider=ModelProvider.OPENAI,
                main_llm_model_name="gpt-4o",
                tool_llm_provider=ModelProvider.OPENAI,
                tool_llm_model_name="gpt-4o",
            )

            tools = [ToolsFactory().create_tool(mult)]
            agent = Agent(
                agent_config=config,
                tools=tools,
                topic=STANDARD_TEST_TOPIC,
                custom_instructions=STANDARD_TEST_INSTRUCTIONS,
            )

            # Test simple multiplication: 4 * 3 = 12
            stream = await agent.astream_chat(
                "What is 4 times 3? Only give the answer, nothing else"
            )
            async for chunk in stream.async_response_gen():
                pass
            response = await stream.aget_response()

            self.assertIn("12", response.response)

    async def test_gpt_4_1(self):
        """Test GPT-4.1 model with OpenAI provider."""
        with ARIZE_LOCK:
            config = AgentConfig(
                agent_type=AgentType.FUNCTION_CALLING,
                main_llm_provider=ModelProvider.OPENAI,
                main_llm_model_name="gpt-4.1",
                tool_llm_provider=ModelProvider.OPENAI,
                tool_llm_model_name="gpt-4.1",
            )

            tools = [ToolsFactory().create_tool(mult)]
            agent = Agent(
                agent_config=config,
                tools=tools,
                topic=STANDARD_TEST_TOPIC,
                custom_instructions=STANDARD_TEST_INSTRUCTIONS,
            )

            # Test simple multiplication: 6 * 2 = 12
            stream = await agent.astream_chat(
                "What is 6 times 2? Only give the answer, nothing else"
            )
            async for chunk in stream.async_response_gen():
                pass
            response = await stream.aget_response()

            self.assertIn("12", response.response)

    async def test_gpt_5_minimal_reasoning(self):
        """Test GPT-5 model with minimal reasoning effort."""
        with ARIZE_LOCK:
            config = AgentConfig(
                agent_type=AgentType.FUNCTION_CALLING,
                main_llm_provider=ModelProvider.OPENAI,
                main_llm_model_name="gpt-5",
                tool_llm_provider=ModelProvider.OPENAI,
                tool_llm_model_name="gpt-5",
            )

            tools = [ToolsFactory().create_tool(mult)]
            agent = Agent(
                agent_config=config,
                tools=tools,
                topic=STANDARD_TEST_TOPIC,
                custom_instructions=STANDARD_TEST_INSTRUCTIONS,
            )

            # Test simple multiplication: 5 * 5 = 25
            stream = await agent.astream_chat(
                "What is 5 times 5? Only give the answer, nothing else"
            )
            async for chunk in stream.async_response_gen():
                pass
            response = await stream.aget_response()

            self.assertIn("25", response.response)


if __name__ == "__main__":
    unittest.main()
