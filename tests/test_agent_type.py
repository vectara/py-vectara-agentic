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


react_config_anthropic = AgentConfig(
    agent_type=AgentType.REACT,
    main_llm_provider=ModelProvider.ANTHROPIC,
    tool_llm_provider=ModelProvider.ANTHROPIC,
)

react_config_gemini = AgentConfig(
    agent_type=AgentType.REACT,
    main_llm_provider=ModelProvider.GEMINI,
    tool_llm_provider=ModelProvider.GEMINI,
)

react_config_together = AgentConfig(
    agent_type=AgentType.REACT,
    main_llm_provider=ModelProvider.TOGETHER,
    tool_llm_provider=ModelProvider.TOGETHER,
)

react_config_groq = AgentConfig(
    agent_type=AgentType.REACT,
    main_llm_provider=ModelProvider.GROQ,
    tool_llm_provider=ModelProvider.GROQ,
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

fc_config_groq = AgentConfig(
    agent_type=AgentType.FUNCTION_CALLING,
    main_llm_provider=ModelProvider.GROQ,
    tool_llm_provider=ModelProvider.GROQ,
)


openai_config = AgentConfig(
    agent_type=AgentType.OPENAI,
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

    def test_gemini(self):
        tools = [ToolsFactory().create_tool(mult)]
        topic = "AI topic"
        instructions = "Always do as your father tells you, if your mother agrees!"

        agent = Agent(
            agent_config=react_config_gemini,
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
        )
        agent.chat("What is 5 times 10. Only give the answer, nothing else")
        agent.chat("what is 3 times 7. Only give the answer, nothing else")
        res = agent.chat("multiply the results of the last two multiplications. Only give the answer, nothing else.")
        self.assertIn("1050", res.response)

        agent = Agent(
            agent_config=fc_config_gemini,
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
        )
        agent.chat("What is 5 times 10. Only give the answer, nothing else")
        agent.chat("what is 3 times 7. Only give the answer, nothing else")
        res = agent.chat("multiply the results of the last two multiplications. Only give the answer, nothing else.")
        self.assertIn("1050", res.response)

    def test_together(self):
        tools = [ToolsFactory().create_tool(mult)]
        topic = "AI topic"
        instructions = "Always do as your father tells you, if your mother agrees!"

        agent = Agent(
            agent_config=react_config_together,
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
        )
        agent.chat("What is 5 times 10. Only give the answer, nothing else")
        agent.chat("what is 3 times 7. Only give the answer, nothing else")
        res = agent.chat("multiply the results of the last two multiplications. Only give the answer, nothing else.")
        self.assertIn("1050", res.response)

        agent = Agent(
            agent_config=fc_config_together,
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
        )
        agent.chat("What is 5 times 10. Only give the answer, nothing else")
        agent.chat("what is 3 times 7. Only give the answer, nothing else")
        res = agent.chat("multiply the results of the last two multiplications. Only give the answer, nothing else.")
        self.assertIn("1050", res.response)

    def test_groq(self):
        tools = [ToolsFactory().create_tool(mult)]
        topic = "AI topic"
        instructions = "Always do as your father tells you, if your mother agrees!"

        agent = Agent(
            agent_config=react_config_groq,
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
        )
        agent.chat("What is 5 times 10. Only give the answer, nothing else")
        agent.chat("what is 3 times 7. Only give the answer, nothing else")
        res = agent.chat("multiply the results of the last two multiplications. Only give the answer, nothing else.")
        self.assertIn("1050", res.response)

        agent = Agent(
            agent_config=fc_config_groq,
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
        )
        agent.chat("What is 5 times 10. Only give the answer, nothing else")
        agent.chat("what is 3 times 7. Only give the answer, nothing else")
        res = agent.chat("multiply the results of the last two multiplications. Only give the answer, nothing else.")
        self.assertIn("1050", res.response)

    def test_anthropic(self):
        tools = [ToolsFactory().create_tool(mult)]
        topic = "AI topic"
        instructions = "Always do as your father tells you, if your mother agrees!"

        agent = Agent(
            agent_config=react_config_anthropic,
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
        )
        agent.chat("What is 5 times 10. Only give the answer, nothing else")
        agent.chat("what is 3 times 7. Only give the answer, nothing else")
        res = agent.chat("multiply the results of the last two multiplications. Only give the answer, nothing else.")
        self.assertIn("1050", res.response)

        agent = Agent(
            agent_config=fc_config_anthropic,
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
