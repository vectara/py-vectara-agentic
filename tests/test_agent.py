# Suppress external dependency warnings before any other imports
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import unittest
import threading
from datetime import date

from vectara_agentic.agent import Agent, AgentType
from vectara_agentic.agent_core.factory import format_prompt
from vectara_agentic.agent_config import AgentConfig
from vectara_agentic.types import ModelProvider, ObserverType
from vectara_agentic.tools import ToolsFactory

from vectara_agentic.agent_core.prompts import GENERAL_INSTRUCTIONS
from conftest import mult, STANDARD_TEST_TOPIC, STANDARD_TEST_INSTRUCTIONS


ARIZE_LOCK = threading.Lock()

class TestAgentPackage(unittest.TestCase):
    def test_get_prompt(self):
        prompt_template = "{chat_topic} on {today} with {custom_instructions}"
        topic = "Programming"
        custom_instructions = "Always do as your mother tells you!"
        expected_output = (
            "Programming on "
            + date.today().strftime("%A, %B %d, %Y")
            + " with Always do as your mother tells you!"
        )
        self.assertEqual(
            format_prompt(prompt_template, GENERAL_INSTRUCTIONS, topic, custom_instructions), expected_output
        )

    def test_agent_init(self):
        tools = [ToolsFactory().create_tool(mult)]
        agent = Agent(tools, STANDARD_TEST_TOPIC, STANDARD_TEST_INSTRUCTIONS)
        self.assertEqual(agent.agent_type, AgentType.FUNCTION_CALLING)
        self.assertEqual(agent._topic, STANDARD_TEST_TOPIC)
        self.assertEqual(agent._custom_instructions, STANDARD_TEST_INSTRUCTIONS)

        # To run this test, you must have appropriate API key in your environment
        self.assertEqual(
            agent.chat(
                "What is 5 times 10. Only give the answer, nothing else"
            ).response.replace("$", "\\$"),
            "50",
        )

    def test_agent_config(self):
        with ARIZE_LOCK:
            tools = [ToolsFactory().create_tool(mult)]
            config = AgentConfig(
                agent_type=AgentType.REACT,
                main_llm_provider=ModelProvider.ANTHROPIC,
                main_llm_model_name="claude-sonnet-4-20250514",
                tool_llm_provider=ModelProvider.TOGETHER,
                tool_llm_model_name="moonshotai/Kimi-K2-Instruct",
                observer=ObserverType.ARIZE_PHOENIX
            )

            agent = Agent(
                tools=tools,
                topic=STANDARD_TEST_TOPIC,
                custom_instructions=STANDARD_TEST_INSTRUCTIONS,
                agent_config=config
            )
            self.assertEqual(agent._topic, STANDARD_TEST_TOPIC)
            self.assertEqual(agent._custom_instructions, STANDARD_TEST_INSTRUCTIONS)
            self.assertEqual(agent.agent_type, AgentType.REACT)
            self.assertEqual(agent.agent_config.observer, ObserverType.ARIZE_PHOENIX)
            self.assertEqual(agent.agent_config.main_llm_provider, ModelProvider.ANTHROPIC)
            self.assertEqual(agent.agent_config.tool_llm_provider, ModelProvider.TOGETHER)

            # To run this test, you must have ANTHROPIC_API_KEY and TOGETHER_API_KEY in your environment
            self.assertEqual(
                agent.chat(
                    "What is 5 times 10. Only give the answer, nothing else"
                ).response.replace("$", "\\$"),
                "50",
            )

    def test_multiturn(self):
        with ARIZE_LOCK:
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

    def test_from_corpus(self):
        agent = Agent.from_corpus(
            tool_name="RAG Tool",
            vectara_corpus_key="corpus_key",
            vectara_api_key="api_key",
            data_description="information",
            assistant_specialty="question answering",
        )

        self.assertIsInstance(agent, Agent)
        self.assertEqual(agent._topic, "question answering")

    def test_chat_history(self):
        tools = [ToolsFactory().create_tool(mult)]
        topic = "AI topic"
        instructions = "Always do as your father tells you, if your mother agrees!"
        agent = Agent(
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
            chat_history=[("What is 5 times 10", "50"), ("What is 3 times 7", "21")]
        )

        data = agent.dumps()
        clone = Agent.loads(data)
        assert clone.memory.get() == agent.memory.get()

        res = agent.chat("multiply the results of the last two questions. Output only the answer.")
        self.assertEqual(res.response, "1050")

    def test_custom_general_instruction(self):
        general_instructions = "Always respond with: I DIDN'T DO IT"
        agent = Agent.from_corpus(
            tool_name="RAG Tool",
            vectara_corpus_key="corpus_key",
            vectara_api_key="api_key",
            data_description="information",
            assistant_specialty="question answering",
            general_instructions=general_instructions,
        )

        res = agent.chat("What is the meaning of the universe?")
        self.assertEqual(res.response, "I DIDN'T DO IT")


if __name__ == "__main__":
    unittest.main()
