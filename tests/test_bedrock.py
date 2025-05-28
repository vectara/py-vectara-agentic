import unittest

from vectara_agentic.agent import Agent, AgentType
from vectara_agentic.agent_config import AgentConfig
from vectara_agentic.tools import ToolsFactory
from vectara_agentic.types import ModelProvider

import nest_asyncio
nest_asyncio.apply()


def mult(x: float, y: float) -> float:
    "Multiply two numbers"
    return x * y


fc_config_bedrock = AgentConfig(
    agent_type=AgentType.FUNCTION_CALLING,
    main_llm_provider=ModelProvider.BEDROCK,
    tool_llm_provider=ModelProvider.BEDROCK,
)

class TestBedrock(unittest.TestCase):

    def test_multiturn(self):
        tools = [ToolsFactory().create_tool(mult)]
        topic = "AI topic"
        instructions = "Always do as your father tells you, if your mother agrees!"
        agent = Agent(
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
        )

        agent.chat("What is 5 times 10. Only give the answer, nothing else")
        agent.chat("what is 3 times 7. Only give the answer, nothing else")
        res = agent.chat("multiply the results of the last two questions. Output only the answer.")
        self.assertEqual(res.response, "1050")


if __name__ == "__main__":
    unittest.main()
