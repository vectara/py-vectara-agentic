import unittest
import threading
import os

from vectara_agentic.agent import Agent, AgentType
from vectara_agentic.agent_config import AgentConfig
from vectara_agentic.types import ModelProvider, ObserverType
from vectara_agentic.tools import ToolsFactory

from llama_index.core.utilities.sql_wrapper import SQLDatabase
from sqlalchemy import create_engine

def mult(x: float, y: float) -> float:
    return x * y


ARIZE_LOCK = threading.Lock()

class TestAgentSerialization(unittest.TestCase):

    @classmethod
    def tearDown(cls):
        try:
            os.remove('ev_database.db')
        except FileNotFoundError:
            pass

    def test_serialization(self):
        with ARIZE_LOCK:
            config = AgentConfig(
                agent_type=AgentType.REACT,
                main_llm_provider=ModelProvider.ANTHROPIC,
                tool_llm_provider=ModelProvider.TOGETHER,
                observer=ObserverType.ARIZE_PHOENIX
            )
            db_tools = ToolsFactory().database_tools(
                tool_name_prefix = "ev",
                content_description = 'Electric Vehicles in the state of Washington and other population information',
                sql_database = SQLDatabase(create_engine('sqlite:///ev_database.db')),
            )

            tools = [ToolsFactory().create_tool(mult)] + ToolsFactory().standard_tools() + db_tools
            topic = "AI topic"
            instructions = "Always do as your father tells you, if your mother agrees!"
            agent = Agent(
                tools=tools,
                topic=topic,
                custom_instructions=instructions,
                agent_config=config
            )

            agent_reloaded = agent.loads(agent.dumps())
            agent_reloaded_again = agent_reloaded.loads(agent_reloaded.dumps())

            self.assertIsInstance(agent_reloaded, Agent)
            self.assertEqual(agent, agent_reloaded)
            self.assertEqual(agent.agent_type, agent_reloaded.agent_type)

            self.assertEqual(agent.agent_config.observer, agent_reloaded.agent_config.observer)
            self.assertEqual(agent.agent_config.main_llm_provider, agent_reloaded.agent_config.main_llm_provider)
            self.assertEqual(agent.agent_config.tool_llm_provider, agent_reloaded.agent_config.tool_llm_provider)

            self.assertIsInstance(agent_reloaded, Agent)
            self.assertEqual(agent, agent_reloaded_again)
            self.assertEqual(agent.agent_type, agent_reloaded_again.agent_type)

            self.assertEqual(agent.agent_config.observer, agent_reloaded_again.agent_config.observer)
            self.assertEqual(agent.agent_config.main_llm_provider, agent_reloaded_again.agent_config.main_llm_provider)
            self.assertEqual(agent.agent_config.tool_llm_provider, agent_reloaded_again.agent_config.tool_llm_provider)

    def test_serialization_from_corpus(self):
        with ARIZE_LOCK:
            config = AgentConfig(
                agent_type=AgentType.REACT,
                main_llm_provider=ModelProvider.ANTHROPIC,
                tool_llm_provider=ModelProvider.TOGETHER,
                observer=ObserverType.ARIZE_PHOENIX
            )

            agent = Agent.from_corpus(
                tool_name="RAG Tool",
                agent_config=config,
                vectara_corpus_key="corpus_key",
                vectara_api_key="api_key",
                data_description="information",
                assistant_specialty="question answering",
            )

            agent_reloaded = agent.loads(agent.dumps())
            agent_reloaded_again = agent_reloaded.loads(agent_reloaded.dumps())

            self.assertIsInstance(agent_reloaded, Agent)
            self.assertEqual(agent, agent_reloaded)
            self.assertEqual(agent.agent_type, agent_reloaded.agent_type)

            self.assertEqual(agent.agent_config.observer, agent_reloaded.agent_config.observer)
            self.assertEqual(agent.agent_config.main_llm_provider, agent_reloaded.agent_config.main_llm_provider)
            self.assertEqual(agent.agent_config.tool_llm_provider, agent_reloaded.agent_config.tool_llm_provider)

            self.assertIsInstance(agent_reloaded, Agent)
            self.assertEqual(agent, agent_reloaded_again)
            self.assertEqual(agent.agent_type, agent_reloaded_again.agent_type)

            self.assertEqual(agent.agent_config.observer, agent_reloaded_again.agent_config.observer)
            self.assertEqual(agent.agent_config.main_llm_provider, agent_reloaded_again.agent_config.main_llm_provider)
            self.assertEqual(agent.agent_config.tool_llm_provider, agent_reloaded_again.agent_config.tool_llm_provider)


if __name__ == "__main__":
    unittest.main()
