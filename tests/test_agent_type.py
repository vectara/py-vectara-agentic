import unittest

from vectara_agentic.agent import Agent, AgentType
from vectara_agentic.agent_config import AgentConfig
from vectara_agentic.tools import ToolsFactory
from vectara_agentic.types import ModelProvider, ObserverType

def mult(x, y):
    return x * y


react_config = AgentConfig(
    agent_type=AgentType.REACT,
    main_llm_provider=ModelProvider.ANTHROPIC,
    main_llm_model_name="claude-3-5-sonnet-20241022",
    tool_llm_provider=ModelProvider.TOGETHER,
    tool_llm_model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    observer=ObserverType.ARIZE_PHOENIX
)

openai_config = AgentConfig(
    agent_type=AgentType.OPENAI,
    observer=ObserverType.ARIZE_PHOENIX
)


class TestAgentType(unittest.TestCase):

    def test_openai(self):
        tools = [ToolsFactory().create_tool(mult)]
        topic = "AI topic"
        instructions = "Always do as your father tells you, if your mother agrees!"
        agent = Agent(
            agent_config=openai_config,
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
        )

        agent.chat("What is 5 times 10. Only give the answer, nothing else")
        agent.chat("what is 3 times 7. Only give the answer, nothing else")
        res = agent.chat("multiply the results of the last two multiplications. Only give the answer, nothing else.")
        self.assertIn("1050", res.response)

    def test_react(self):
        tools = [ToolsFactory().create_tool(mult)]
        topic = "AI topic"
        instructions = "Always do as your father tells you, if your mother agrees!"
        agent = Agent(
            agent_config=react_config,
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
        )

        agent.chat("What is 5 times 10. Only give the answer, nothing else")
        agent.chat("what is 3 times 7. Only give the answer, nothing else")
        res = agent.chat("multiply the results of the last two multiplications. Only give the answer, nothing else.")
        self.assertIn("1050", res.response)


if __name__ == "__main__":
    unittest.main()
