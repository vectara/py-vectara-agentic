# Suppress external dependency warnings before any other imports
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import unittest
import asyncio

from vectara_agentic.agent import Agent, AgentType
from vectara_agentic.agent_config import AgentConfig
from vectara_agentic.tools import ToolsFactory
from vectara_agentic.types import ModelProvider

import nest_asyncio
nest_asyncio.apply()

from conftest import (
    fcs_config_openai,
    fc_config_anthropic,
    fc_config_gemini,
    fc_config_together,
)


def mult(x: float, y: float) -> float:
    "Multiply two numbers"
    return x * y

class TestAgentStreaming(unittest.TestCase):

    async def test_anthropic(self):
        tools = [ToolsFactory().create_tool(mult)]
        topic = "AI topic"
        instructions = "Always do as your father tells you, if your mother agrees!"
        agent = Agent(
            agent_config=fc_config_anthropic,
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
        )

        # First calculation: 5 * 10 = 50
        stream1 = await agent.astream_chat("What is 5 times 10. Only give the answer, nothing else")
        # Consume the stream
        async for chunk in stream1.async_response_gen():
            pass
        _ = await stream1.aget_response()

        # Second calculation: 3 * 7 = 21
        stream2 = await agent.astream_chat("what is 3 times 7. Only give the answer, nothing else")
        # Consume the stream
        async for chunk in stream2.async_response_gen():
            pass
        _ = await stream2.aget_response()

        # Final calculation: 50 * 21 = 1050
        stream3 = await agent.astream_chat("multiply the results of the last two multiplications. Only give the answer, nothing else.")
        # Consume the stream
        async for chunk in stream3.async_response_gen():
            pass
        response3 = await stream3.aget_response()

        self.assertIn("1050", response3.response)

    async def test_openai(self):
        tools = [ToolsFactory().create_tool(mult)]
        topic = "AI topic"
        instructions = "Always do as your father tells you, if your mother agrees!"
        agent = Agent(
            agent_config=fcs_config_openai,
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
        )

        # First calculation: 5 * 10 = 50
        stream1 = await agent.astream_chat("What is 5 times 10. Only give the answer, nothing else")
        # Consume the stream
        async for chunk in stream1.async_response_gen():
            pass
        _ = await stream1.aget_response()

        # Second calculation: 3 * 7 = 21
        stream2 = await agent.astream_chat("what is 3 times 7. Only give the answer, nothing else")
        # Consume the stream
        async for chunk in stream2.async_response_gen():
            pass
        _ = await stream2.aget_response()

        # Final calculation: 50 * 21 = 1050
        stream3 = await agent.astream_chat("multiply the results of the last two multiplications. Only give the answer, nothing else.")
        # Consume the stream
        async for chunk in stream3.async_response_gen():
            pass
        response3 = await stream3.aget_response()

        self.assertIn("1050", response3.response)

    def test_openai_sync(self):
        """Synchronous wrapper for the async test"""
        asyncio.run(self.test_openai())


if __name__ == "__main__":
    unittest.main()
