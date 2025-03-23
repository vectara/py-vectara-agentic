import unittest

from vectara_agentic.agent_config import AgentConfig
from vectara_agentic.tools import ToolsFactory
from vectara_agentic.agent import Agent, AgentType
from vectara_agentic.types import ModelProvider

def mult(x: float, y: float) -> float:
    return x * y

def addition(x: float, y: float) -> float:
    return x + y

react_config_together = AgentConfig(
    agent_type=AgentType.REACT,
    main_llm_provider=ModelProvider.TOGETHER,
    tool_llm_provider=ModelProvider.TOGETHER,
)

fc_config_anthropic = AgentConfig(
    agent_type=AgentType.FUNCTION_CALLING,
    main_llm_provider=ModelProvider.ANTHROPIC,
    tool_llm_provider=ModelProvider.ANTHROPIC,
)

class TestAgentPlanningPackage(unittest.TestCase):

    def test_no_planning(self):
        tools = [ToolsFactory().create_tool(mult)]
        topic = "math"
        instructions = "Answer the user's math questions."
        agent_config = AgentConfig()

        agent = Agent(
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
            agent_config=agent_config,
        )
        res = agent.chat("Calculate the product of 5 and 7, then calculate the product of 3 and 2, and finally add the two products together.")
        self.assertIn("41", res.response)

        agent = Agent(
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
            agent_config=react_config_together,
        )
        res = agent.chat("Calculate the product of 5 and 7, then calculate the product of 3 and 2, and finally add the two products together.")
        self.assertIn("41", res.response)

        agent = Agent(
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
            agent_config=fc_config_anthropic,
        )
        res = agent.chat("Calculate the product of 5 and 7, then calculate the product of 3 and 2, and finally add the two products together.")
        self.assertIn("41", res.response)

    def test_structured_planning(self):
        tools = [ToolsFactory().create_tool(mult), ToolsFactory().create_tool(addition)]
        topic = "math"
        instructions = "Answer the user's math questions."
        agent_config = AgentConfig()

        agent = Agent(
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
            agent_config=agent_config,
            use_structured_planning = True,
        )
        res = agent.chat("Calculate the square of every number from 1 to 5, then add the results of all squares. What is the final sum?")
        self.assertIn("55", res.response)


if __name__ == "__main__":
    unittest.main()
