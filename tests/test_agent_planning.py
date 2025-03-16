import unittest

from vectara_agentic.agent import Agent
from vectara_agentic.agent_config import AgentConfig
from vectara_agentic.tools import ToolsFactory

def mult(x, y):
    return x * y

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

        agent.chat("What is 5 times 10. Only give the answer, nothing else")
        agent.chat("what is 3 times 7. Only give the answer, nothing else")
        res = agent.chat("multiply the results of the last two multiplications. Only give the answer, nothing else.")
        self.assertIn("1050", res.response)

    def test_structured_planning(self):
        tools = [ToolsFactory().create_tool(mult)]
        topic = "AI topic"
        instructions = "Always do as your father tells you, if your mother agrees!"
        agent = Agent(
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
            agent_config = AgentConfig(),
            use_structured_planning = True
        )

        agent.chat("What is 5 times 10. Only give the answer, nothing else")
        agent.chat("what is 3 times 7. Only give the answer, nothing else")
        res = agent.chat("multiply the results of the last two multiplications. Only give the answer, nothing else.")
        self.assertIn("1050", res.response)


if __name__ == "__main__":
    unittest.main()
