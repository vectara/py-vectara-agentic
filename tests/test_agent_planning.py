import unittest

from vectara_agentic.agent import Agent
from vectara_agentic.agent_config import AgentConfig
from vectara_agentic.tools import ToolsFactory

def mult(x, y):
    return x * y

def addition(x, y):
    return x + y

class TestAgentPlanningPackage(unittest.TestCase):

    def test_no_planning(self):
        tools = [ToolsFactory().create_tool(mult)]
        topic = "AI topic"
        instructions = "Always do as your father tells you, if your mother agrees!"
        agent = Agent(
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
            agent_config = AgentConfig()
        )

        res = agent.chat("If you multiply 5 times 7, then 3 times 2, and add the results - what do you get?")
        self.assertIn("41", res.response)

    def test_structured_planning(self):
        tools = [ToolsFactory().create_tool(mult), ToolsFactory().create_tool(addition)]
        topic = "AI topic"
        instructions = "Always do as your father tells you, if your mother agrees!"
        agent = Agent(
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
            agent_config = AgentConfig(),
            use_structured_planning = True,
        )

        res = agent.chat("If you multiply 5 times 7, then 3 times 2, and add the results - what do you get?")
        self.assertIn("41", res.response)


if __name__ == "__main__":
    unittest.main()
