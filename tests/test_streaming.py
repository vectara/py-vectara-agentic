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

def mult(x: float, y: float) -> float:
    "Multiply two numbers"
    return x * y


config_function_calling_openai = AgentConfig(
    agent_type=AgentType.FUNCTION_CALLING,
    main_llm_provider=ModelProvider.OPENAI,
    tool_llm_provider=ModelProvider.OPENAI,
)

fc_config_anthropic = AgentConfig(
    agent_type=AgentType.FUNCTION_CALLING,
    main_llm_provider=ModelProvider.ANTHROPIC,
    tool_llm_provider=ModelProvider.ANTHROPIC,
)

fc_config_gemini = AgentConfig(
    agent_type=AgentType.FUNCTION_CALLING,
    main_llm_provider=ModelProvider.GEMINI,
    tool_llm_provider=ModelProvider.GEMINI,
)

fc_config_together = AgentConfig(
    agent_type=AgentType.FUNCTION_CALLING,
    main_llm_provider=ModelProvider.TOGETHER,
    tool_llm_provider=ModelProvider.TOGETHER,
)


class TestAgentStreaming(unittest.TestCase):

    async def test_anthropic(self):
        tools = [ToolsFactory().create_tool(mult)]
        topic = "AI topic"
        instructions = "Always do as your father tells you, if your mother agrees!"
        agent = Agent(
            agent_config=fc_config_anthropic,  # Use function calling which has better streaming
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
        asyncio.run(self.test_anthropic())


if __name__ == "__main__":
    unittest.main()
