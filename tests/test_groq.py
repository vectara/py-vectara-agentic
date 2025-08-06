# Suppress external dependency warnings before any other imports
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import unittest
import threading

from vectara_agentic.agent import Agent
from vectara_agentic.tools import ToolsFactory

import nest_asyncio
nest_asyncio.apply()

from conftest import mult, fc_config_groq, STANDARD_TEST_TOPIC, STANDARD_TEST_INSTRUCTIONS

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


if __name__ == "__main__":
    unittest.main()
