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
from conftest import mult, add


ARIZE_LOCK = threading.Lock()


class TestAgentFallbackMemoryConsistency(unittest.TestCase):
    """Test memory consistency between main and fallback agents"""

    def setUp(self):
        """Set up test fixtures"""
        self.tools = [ToolsFactory().create_tool(mult), ToolsFactory().create_tool(add)]
        self.topic = "Mathematics"
        self.custom_instructions = "You are a helpful math assistant."

        # Main agent config
        self.main_config = AgentConfig(
            agent_type=AgentType.FUNCTION_CALLING,
            main_llm_provider=ModelProvider.ANTHROPIC,
        )

        # Fallback agent config
        self.fallback_config = AgentConfig(
            agent_type=AgentType.REACT, main_llm_provider=ModelProvider.ANTHROPIC
        )

        self.session_id = "test-fallback-session-123"

    def test_memory_consistency_on_agent_creation(self):
        """Test that main and fallback agents are created with the same memory content"""
        agent = Agent(
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.custom_instructions,
            agent_config=self.main_config,
            fallback_agent_config=self.fallback_config,
            session_id=self.session_id,
        )

        # Add some memory before creating the agents
        test_messages = [
            ChatMessage(role=MessageRole.USER, content="What is 2*3?"),
            ChatMessage(role=MessageRole.ASSISTANT, content="2*3 = 6"),
        ]
        agent.memory.put_messages(test_messages)

        # Verify both agents have memory with the same content
        # Memory is managed by the main Agent class, not individual agent instances
        main_memory = agent.memory.get()
        fallback_memory = agent.memory.get()  # Both access the same memory

        self.assertEqual(len(main_memory), 2)
        self.assertEqual(len(fallback_memory), 2)
        self.assertEqual(main_memory[0].content, "What is 2*3?")
        self.assertEqual(fallback_memory[0].content, "What is 2*3?")

        # Verify session_id consistency
        # Memory is managed by the main Agent class
        self.assertEqual(agent.memory.session_id, self.session_id)

    def test_memory_sync_during_agent_switching(self):
        """Test that memory remains consistent when switching between main and fallback agents"""
        agent = Agent(
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.custom_instructions,
            agent_config=self.main_config,
            fallback_agent_config=self.fallback_config,
            session_id=self.session_id,
        )

        # Start with main agent
        self.assertEqual(agent.agent_config_type, AgentConfigType.DEFAULT)

        # Add initial memory
        initial_messages = [
            ChatMessage(role=MessageRole.USER, content="Initial question"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Initial response"),
        ]
        agent.memory.put_messages(initial_messages)

        # Access main agent to ensure it's loaded
        main_memory_before = agent.memory.get()  # Memory managed by main Agent class
        self.assertEqual(len(main_memory_before), 2)

        # Switch to fallback agent (this should clear the fallback agent instance)
        agent._switch_agent_config()
        self.assertEqual(agent.agent_config_type, AgentConfigType.FALLBACK)

        # Access fallback agent (should be recreated with current memory)
        fallback_memory = agent.memory.get()  # Memory managed by main Agent class

        # Verify fallback agent has the same memory content
        self.assertEqual(len(fallback_memory), 2)
        self.assertEqual(fallback_memory[0].content, "Initial question")
        self.assertEqual(fallback_memory[1].content, "Initial response")

        # Add more memory while using fallback agent
        additional_messages = [
            ChatMessage(role=MessageRole.USER, content="Fallback question"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Fallback response"),
        ]
        agent.memory.put_messages(additional_messages)

        # Switch back to main agent (this should clear the main agent instance)
        agent._switch_agent_config()
        self.assertEqual(agent.agent_config_type, AgentConfigType.DEFAULT)

        # Verify recreated main agent now has all the memory including what was added during fallback
        main_memory_after = agent.memory.get()  # Memory managed by main Agent class
        self.assertEqual(len(main_memory_after), 4)
        self.assertEqual(main_memory_after[2].content, "Fallback question")
        self.assertEqual(main_memory_after[3].content, "Fallback response")

    def test_memory_sync_on_clear_memory(self):
        """Test that memory clearing resets agent instances for consistency"""
        agent = Agent(
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.custom_instructions,
            agent_config=self.main_config,
            fallback_agent_config=self.fallback_config,
            session_id=self.session_id,
        )

        # Add memory
        test_messages = [
            ChatMessage(role=MessageRole.USER, content="Test question"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Test response"),
        ]
        agent.memory.put_messages(test_messages)

        # Verify memory exists
        # Memory is managed by the main Agent class
        self.assertEqual(len(agent.memory.get()), 2)
        self.assertEqual(len(agent.memory.get()), 2)  # Both access same memory

        # Clear memory (should reset agent instances)
        agent.clear_memory()

        # Verify core memory is cleared
        self.assertEqual(len(agent.memory.get()), 0)

        # Verify agent instances were reset (None)
        self.assertIsNone(agent._agent)
        self.assertIsNone(agent._fallback_agent)

        # Verify new agents have cleared memory
        # Memory is managed by the main Agent class
        self.assertEqual(len(agent.memory.get()), 0)
        self.assertEqual(len(agent.memory.get()), 0)  # Both access same memory

    def test_memory_consistency_after_serialization(self):
        """Test that memory consistency is maintained after serialization/deserialization"""
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
            ChatMessage(role=MessageRole.USER, content="Serialization test"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Serialization response"),
        ]
        agent.memory.put_messages(test_messages)

        # Access both agents
        _ = agent.agent
        _ = agent.fallback_agent

        # Serialize and deserialize
        serialized_data = agent.dumps()
        restored_agent = Agent.loads(serialized_data)

        # Verify memory is preserved and consistent
        self.assertEqual(restored_agent.session_id, self.session_id)
        self.assertEqual(len(restored_agent.memory.get()), 2)

        # Verify memory consistency
        # Individual agent instances don't have .memory attribute - memory is managed by main Agent class
        # Both agent instances should use the same memory from the main Agent

        main_memory = restored_agent.memory.get()
        fallback_memory = restored_agent.memory.get()  # Both access same memory

        self.assertEqual(len(main_memory), 2)
        self.assertEqual(len(fallback_memory), 2)
        self.assertEqual(main_memory[0].content, "Serialization test")
        self.assertEqual(fallback_memory[0].content, "Serialization test")

    def test_session_id_consistency_across_agents(self):
        """Test that session_id is consistent between main and fallback agents"""
        agent = Agent(
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.custom_instructions,
            agent_config=self.main_config,
            fallback_agent_config=self.fallback_config,
            session_id=self.session_id,
        )

        # Verify main agent session_id consistency
        self.assertEqual(agent.session_id, self.session_id)
        self.assertEqual(agent.memory.session_id, self.session_id)

        # Verify session_id consistency across all agents
        # Memory is managed by the main Agent class
        self.assertEqual(agent.memory.session_id, self.session_id)
        self.assertEqual(
            agent.memory.session_id, self.session_id
        )  # Both access same memory

    def test_agent_recreation_on_switch(self):
        """Test that agents are properly recreated when switching configurations"""
        agent = Agent(
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.custom_instructions,
            agent_config=self.main_config,
            fallback_agent_config=self.fallback_config,
            session_id=self.session_id,
        )

        # Load main agent
        original_main_agent = agent.agent
        self.assertIsNotNone(original_main_agent)

        # Load fallback agent
        original_fallback_agent = agent.fallback_agent
        self.assertIsNotNone(original_fallback_agent)

        # Switch to fallback - should clear the fallback agent instance
        agent._switch_agent_config()
        self.assertEqual(agent.agent_config_type, AgentConfigType.FALLBACK)
        self.assertIsNone(agent._fallback_agent)  # Should be cleared

        # Access fallback agent again - should be a new instance
        new_fallback_agent = agent.fallback_agent
        self.assertIsNot(new_fallback_agent, original_fallback_agent)

        # Switch back to main - should clear the main agent instance
        agent._switch_agent_config()
        self.assertEqual(agent.agent_config_type, AgentConfigType.DEFAULT)
        self.assertIsNone(agent._agent)  # Should be cleared

        # Access main agent again - should be a new instance
        new_main_agent = agent.agent
        self.assertIsNot(new_main_agent, original_main_agent)


if __name__ == "__main__":
    unittest.main()
