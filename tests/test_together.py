# Suppress external dependency warnings before any other imports
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import unittest
import threading

from vectara_agentic.agent import Agent
from vectara_agentic.tools import ToolsFactory

import nest_asyncio
nest_asyncio.apply()

from conftest import fc_config_together, mult, STANDARD_TEST_TOPIC, STANDARD_TEST_INSTRUCTIONS


ARIZE_LOCK = threading.Lock()

class TestTogether(unittest.TestCase):

    def test_multiturn(self):
        with ARIZE_LOCK:
            tools = [ToolsFactory().create_tool(mult)]
            agent = Agent(
                agent_config=fc_config_together,
                tools=tools,
                topic=STANDARD_TEST_TOPIC,
                custom_instructions=STANDARD_TEST_INSTRUCTIONS,
            )

            agent.chat("What is 5 times 10. Only give the answer, nothing else")
            agent.chat("what is 3 times 7. Only give the answer, nothing else")
            res = agent.chat("multiply the results of the last two questions. Output only the answer.")
            self.assertEqual(res.response, "1050")


if __name__ == "__main__":
    unittest.main()
