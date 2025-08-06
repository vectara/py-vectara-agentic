# Suppress external dependency warnings before any other imports
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

import unittest
import threading

from vectara_agentic.agent import Agent, AgentType
from vectara_agentic.agent_config import AgentConfig
from vectara_agentic.types import ModelProvider, AgentConfigType
from vectara_agentic.tools import ToolsFactory
from llama_index.core.llms import ChatMessage, MessageRole
from conftest import mult, STANDARD_TEST_TOPIC, STANDARD_TEST_INSTRUCTIONS


ARIZE_LOCK = threading.Lock()


class TestAgentMemoryConsistency(unittest.TestCase):
    """Test memory consistency behavior for main/fallback agent switching"""

    def setUp(self):
        """Set up test fixtures"""
        self.tools = [ToolsFactory().create_tool(mult)]
        self.topic = STANDARD_TEST_TOPIC
        self.custom_instructions = STANDARD_TEST_INSTRUCTIONS

        # Main agent config
        self.main_config = AgentConfig(
            agent_type=AgentType.FUNCTION_CALLING,
            main_llm_provider=ModelProvider.ANTHROPIC,
        )

        # Fallback agent config
        self.fallback_config = AgentConfig(
            agent_type=AgentType.REACT, main_llm_provider=ModelProvider.ANTHROPIC
        )

        self.session_id = "test-memory-consistency-123"

    def test_agent_recreation_on_config_switch(self):
        """Test that agent instances are properly recreated when switching configurations"""
        agent = Agent(
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.custom_instructions,
            agent_config=self.main_config,
            fallback_agent_config=self.fallback_config,
            session_id=self.session_id,
        )

        # Load main agent first
        original_main_agent = agent.agent
        self.assertIsNotNone(original_main_agent)
        self.assertEqual(agent.agent_config_type, AgentConfigType.DEFAULT)

        # Switch to fallback - should clear fallback agent instance for recreation
        agent._switch_agent_config()
        self.assertEqual(agent.agent_config_type, AgentConfigType.FALLBACK)
        self.assertIsNone(agent._fallback_agent)  # Should be cleared for recreation

        # Load fallback agent - should be new instance
        new_fallback_agent = agent.fallback_agent
        self.assertIsNotNone(new_fallback_agent)

        # Switch back to main - should clear main agent instance for recreation
        agent._switch_agent_config()
        self.assertEqual(agent.agent_config_type, AgentConfigType.DEFAULT)
        self.assertIsNone(agent._agent)  # Should be cleared for recreation

        # Load main agent again - should be new instance
        recreated_main_agent = agent.agent
        self.assertIsNotNone(recreated_main_agent)
        self.assertIsNot(recreated_main_agent, original_main_agent)

    def test_memory_persistence_across_config_switches(self):
        """Test that Agent memory persists correctly when switching configurations"""
        agent = Agent(
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.custom_instructions,
            agent_config=self.main_config,
            fallback_agent_config=self.fallback_config,
            session_id=self.session_id,
        )

        # Add initial memory
        initial_messages = [
            ChatMessage(role=MessageRole.USER, content="Initial question"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Initial response"),
        ]
        agent.memory.put_messages(initial_messages)

        # Verify initial memory
        self.assertEqual(len(agent.memory.get()), 2)
        self.assertEqual(agent.memory.get()[0].content, "Initial question")

        # Switch to fallback configuration
        agent._switch_agent_config()
        self.assertEqual(agent.agent_config_type, AgentConfigType.FALLBACK)

        # Memory should persist at the Agent level
        self.assertEqual(len(agent.memory.get()), 2)
        self.assertEqual(agent.memory.get()[0].content, "Initial question")

        # Add more memory while in fallback mode
        fallback_messages = [
            ChatMessage(role=MessageRole.USER, content="Fallback question"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Fallback response"),
        ]
        agent.memory.put_messages(fallback_messages)

        # Verify combined memory
        self.assertEqual(len(agent.memory.get()), 4)
        self.assertEqual(agent.memory.get()[2].content, "Fallback question")

        # Switch back to main configuration
        agent._switch_agent_config()
        self.assertEqual(agent.agent_config_type, AgentConfigType.DEFAULT)

        # All memory should still be present
        self.assertEqual(len(agent.memory.get()), 4)
        self.assertEqual(agent.memory.get()[0].content, "Initial question")
        self.assertEqual(agent.memory.get()[2].content, "Fallback question")

    def test_clear_memory_resets_agent_instances(self):
        """Test that clearing memory properly resets agent instances"""
        agent = Agent(
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.custom_instructions,
            agent_config=self.main_config,
            fallback_agent_config=self.fallback_config,
            session_id=self.session_id,
        )

        # Add memory and load both agents
        test_messages = [
            ChatMessage(role=MessageRole.USER, content="Test question"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Test response"),
        ]
        agent.memory.put_messages(test_messages)

        # Load both agents
        _ = agent.agent
        _ = agent.fallback_agent

        # Verify memory exists
        self.assertEqual(len(agent.memory.get()), 2)

        # Clear memory
        agent.clear_memory()

        # Verify memory is cleared
        self.assertEqual(len(agent.memory.get()), 0)

        # Verify agent instances were reset
        self.assertIsNone(agent._agent)
        self.assertIsNone(agent._fallback_agent)

    def test_session_id_consistency(self):
        """Test that session_id remains consistent throughout agent lifecycle"""
        agent = Agent(
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.custom_instructions,
            agent_config=self.main_config,
            fallback_agent_config=self.fallback_config,
            session_id=self.session_id,
        )

        # Verify initial session_id
        self.assertEqual(agent.session_id, self.session_id)
        self.assertEqual(agent.memory.session_id, self.session_id)

        # Switch configurations multiple times
        agent._switch_agent_config()
        self.assertEqual(agent.session_id, self.session_id)
        self.assertEqual(agent.memory.session_id, self.session_id)

        agent._switch_agent_config()
        self.assertEqual(agent.session_id, self.session_id)
        self.assertEqual(agent.memory.session_id, self.session_id)

        # Clear memory
        agent.clear_memory()
        self.assertEqual(agent.session_id, self.session_id)
        self.assertEqual(agent.memory.session_id, self.session_id)

    def test_serialization_preserves_consistency(self):
        """Test that serialization/deserialization preserves memory consistency behavior"""
        agent = Agent(
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.custom_instructions,
            agent_config=self.main_config,
            fallback_agent_config=self.fallback_config,
            session_id=self.session_id,
        )

        # Add memory and switch configurations
        test_messages = [
            ChatMessage(role=MessageRole.USER, content="Serialization test"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Serialization response"),
        ]
        agent.memory.put_messages(test_messages)
        agent._switch_agent_config()  # Switch to fallback

        # Serialize and deserialize
        serialized_data = agent.dumps()
        restored_agent = Agent.loads(serialized_data)

        # Verify restored agent has same memory (config type resets to DEFAULT on deserialization)
        self.assertEqual(restored_agent.session_id, self.session_id)
        self.assertEqual(len(restored_agent.memory.get()), 2)
        self.assertEqual(restored_agent.memory.get()[0].content, "Serialization test")
        self.assertEqual(
            restored_agent.agent_config_type, AgentConfigType.DEFAULT
        )  # Resets to default

        # Verify memory consistency behavior is preserved
        restored_agent._switch_agent_config()  # Switch to fallback
        self.assertEqual(restored_agent.agent_config_type, AgentConfigType.FALLBACK)
        self.assertEqual(len(restored_agent.memory.get()), 2)  # Memory should persist


if __name__ == "__main__":
    unittest.main()
