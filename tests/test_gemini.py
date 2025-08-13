# Suppress external dependency warnings before any other imports
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import unittest

from vectara_agentic.agent import Agent
from vectara_agentic.tools import ToolsFactory


import nest_asyncio
nest_asyncio.apply()

from conftest import mult, fc_config_gemini, STANDARD_TEST_TOPIC, STANDARD_TEST_INSTRUCTIONS

class TestGEMINI(unittest.TestCase):
    def test_gemini(self):
        tools = [ToolsFactory().create_tool(mult)]

        agent = Agent(
            agent_config=fc_config_gemini,
            tools=tools,
            topic=STANDARD_TEST_TOPIC,
            custom_instructions=STANDARD_TEST_INSTRUCTIONS,
        )
        _ = agent.chat("What is 5 times 10. Only give the answer, nothing else")
        _ = agent.chat("what is 3 times 7. Only give the answer, nothing else")
        res = agent.chat("what is the result of multiplying the results of the last two multiplications. Only give the answer, nothing else.")
        self.assertIn("1050", res.response)

    def test_gemini_single_prompt(self):
        tools = [ToolsFactory().create_tool(mult)]

        agent = Agent(
            agent_config=fc_config_gemini,
            tools=tools,
            topic=STANDARD_TEST_TOPIC,
            custom_instructions=STANDARD_TEST_INSTRUCTIONS,
        )
        res = agent.chat("First, multiply 5 by 10. Then, multiply 3 by 7. Finally, multiply the results of the first two calculations.")
        self.assertIn("1050", res.response)


if __name__ == "__main__":
    unittest.main()
