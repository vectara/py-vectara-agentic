# Suppress external dependency warnings before any other imports
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

import unittest

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from conftest import (
    AgentTestMixin,
    mult,
    STANDARD_TEST_TOPIC,
    STANDARD_TEST_INSTRUCTIONS,
    default_config,
    react_config_anthropic,
    react_config_gemini,
    react_config_together,
    fc_config_anthropic,
    fc_config_gemini,
    fc_config_together,
)
from vectara_agentic.agent import Agent
from vectara_agentic.tools import ToolsFactory


class TestAgentType(unittest.TestCase, AgentTestMixin):

    def setUp(self):
        self.tools = [ToolsFactory().create_tool(mult)]
        self.topic = STANDARD_TEST_TOPIC
        self.instructions = STANDARD_TEST_INSTRUCTIONS

    def test_default_function_calling(self):
        agent = Agent(
            agent_config=default_config,
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.instructions,
        )

        agent.chat("What is 5 times 10. Only give the answer, nothing else")
        agent.chat("what is 3 times 7. Only give the answer, nothing else")
        res = agent.chat(
            "multiply the results of the last two multiplications. Only give the answer, nothing else."
        )
        self.assertIn("1050", res.response)

    def test_gemini(self):
        agent = Agent(
            agent_config=react_config_gemini,
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.instructions,
        )
        agent.chat("What is 5 times 10. Only give the answer, nothing else")
        agent.chat("what is 3 times 7. Only give the answer, nothing else")
        res = agent.chat(
            "what is the result of multiplying the results of the last two "
            "multiplications. Only give the answer, nothing else."
        )
        self.assertIn("1050", res.response)

        agent = Agent(
            agent_config=fc_config_gemini,
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.instructions,
        )
        agent.chat("What is 5 times 10. Only give the answer, nothing else")
        agent.chat("what is 3 times 7. Only give the answer, nothing else")
        res = agent.chat(
            "what is the result of multiplying the results of the last two "
            "multiplications. Only give the answer, nothing else."
        )
        self.assertIn("1050", res.response)

    def test_together(self):
        # Test ReAct agent with Together
        agent = Agent(
            agent_config=react_config_together,
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.instructions,
        )

        with self.with_provider_fallback("Together AI"):
            agent.chat("What is 5 times 10. Only give the answer, nothing else")
            agent.chat("what is 3 times 7. Only give the answer, nothing else")
            res = agent.chat(
                "multiply the results of the last two multiplications. Only give the answer, nothing else."
            )
            self.check_response_and_skip(res, "Together AI")
            self.assertIn("1050", res.response)

        # Test Function Calling agent with Together
        agent = Agent(
            agent_config=fc_config_together,
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.instructions,
        )

        with self.with_provider_fallback("Together AI"):
            agent.chat("What is 5 times 10. Only give the answer, nothing else")
            agent.chat("what is 3 times 7. Only give the answer, nothing else")
            res = agent.chat(
                "multiply the results of the last two multiplications. Only give the answer, nothing else."
            )
            self.check_response_and_skip(res, "Together AI")
            self.assertIn("1050", res.response)

    def test_anthropic(self):
        agent = Agent(
            agent_config=react_config_anthropic,
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.instructions,
        )
        agent.chat("What is 5 times 10. Only give the answer, nothing else")
        agent.chat("what is 3 times 7. Only give the answer, nothing else")
        res = agent.chat(
            "multiply the results of the last two multiplications. Only give the answer, nothing else."
        )
        self.assertIn("1050", res.response)

        agent = Agent(
            agent_config=fc_config_anthropic,
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.instructions,
        )
        agent.chat("What is 5 times 10. Only give the answer, nothing else")
        agent.chat("what is 3 times 7. Only give the answer, nothing else")
        res = agent.chat(
            "multiply the results of the last two multiplications. Only give the answer, nothing else."
        )
        self.assertIn("1050", res.response)


if __name__ == "__main__":
    unittest.main()
