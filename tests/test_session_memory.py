# Suppress external dependency warnings before any other imports
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

import unittest
import threading
from datetime import date

from vectara_agentic.agent import Agent
from vectara_agentic.agent_config import AgentConfig
from vectara_agentic.types import ModelProvider
from vectara_agentic.tools import ToolsFactory
from llama_index.core.llms import ChatMessage, MessageRole
from conftest import mult, add


ARIZE_LOCK = threading.Lock()


class TestSessionMemoryManagement(unittest.TestCase):
    """Test session_id parameter and memory management functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.tools = [ToolsFactory().create_tool(mult), ToolsFactory().create_tool(add)]
        self.topic = "Mathematics"
        self.custom_instructions = "You are a helpful math assistant."
        self.config = AgentConfig(main_llm_provider=ModelProvider.ANTHROPIC)

    def test_agent_init_with_session_id(self):
        """Test Agent initialization with custom session_id"""
        custom_session_id = "test-session-123"

        agent = Agent(
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.custom_instructions,
            agent_config=self.config,
            session_id=custom_session_id,
        )

        # Verify the agent uses the provided session_id
        self.assertEqual(agent.session_id, custom_session_id)

        # Verify memory uses the same session_id
        self.assertEqual(agent.memory.session_id, custom_session_id)

    def test_agent_init_without_session_id(self):
        """Test Agent initialization without session_id (auto-generation)"""
        agent = Agent(
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.custom_instructions,
            agent_config=self.config,
        )

        # Verify auto-generated session_id follows expected pattern
        expected_pattern = f"{self.topic}:{date.today().isoformat()}"
        self.assertEqual(agent.session_id, expected_pattern)

        # Verify memory uses the same session_id
        self.assertEqual(agent.memory.session_id, expected_pattern)

    def test_from_tools_with_session_id(self):
        """Test Agent.from_tools() with custom session_id"""
        custom_session_id = "from-tools-session-456"

        agent = Agent.from_tools(
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.custom_instructions,
            agent_config=self.config,
            session_id=custom_session_id,
        )

        # Verify the agent uses the provided session_id
        self.assertEqual(agent.session_id, custom_session_id)
        self.assertEqual(agent.memory.session_id, custom_session_id)

    def test_from_tools_without_session_id(self):
        """Test Agent.from_tools() without session_id (auto-generation)"""
        agent = Agent.from_tools(
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.custom_instructions,
            agent_config=self.config,
        )

        # Verify auto-generated session_id
        expected_pattern = f"{self.topic}:{date.today().isoformat()}"
        self.assertEqual(agent.session_id, expected_pattern)
        self.assertEqual(agent.memory.session_id, expected_pattern)

    def test_session_id_consistency_across_agents(self):
        """Test that agents with same session_id have consistent session_id attributes"""
        shared_session_id = "shared-session-id-test"

        # Create two agents with the same session_id
        agent1 = Agent(
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.custom_instructions,
            agent_config=self.config,
            session_id=shared_session_id,
        )

        agent2 = Agent(
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.custom_instructions,
            agent_config=self.config,
            session_id=shared_session_id,
        )

        # Verify both agents have the same session_id
        self.assertEqual(agent1.session_id, shared_session_id)
        self.assertEqual(agent2.session_id, shared_session_id)
        self.assertEqual(agent1.session_id, agent2.session_id)

        # Verify their memory instances also have the correct session_id
        self.assertEqual(agent1.memory.session_id, shared_session_id)
        self.assertEqual(agent2.memory.session_id, shared_session_id)

        # Note: Each agent gets its own Memory instance (this is expected behavior)
        # In production, memory persistence happens through serialization/deserialization

    def test_memory_isolation_different_sessions(self):
        """Test that agents with different session_ids have isolated memory"""
        session_id_1 = "isolated-session-1"
        session_id_2 = "isolated-session-2"

        # Create two agents with different session_ids
        agent1 = Agent(
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.custom_instructions,
            agent_config=self.config,
            session_id=session_id_1,
        )

        agent2 = Agent(
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.custom_instructions,
            agent_config=self.config,
            session_id=session_id_2,
        )

        # Add messages to agent1's memory
        agent1_messages = [
            ChatMessage(role=MessageRole.USER, content="Agent 1 question"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Agent 1 response"),
        ]
        agent1.memory.put_messages(agent1_messages)

        # Add different messages to agent2's memory
        agent2_messages = [
            ChatMessage(role=MessageRole.USER, content="Agent 2 question"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Agent 2 response"),
        ]
        agent2.memory.put_messages(agent2_messages)

        # Verify memory isolation
        retrieved_agent1_messages = agent1.memory.get()
        retrieved_agent2_messages = agent2.memory.get()

        self.assertEqual(len(retrieved_agent1_messages), 2)
        self.assertEqual(len(retrieved_agent2_messages), 2)

        # Verify agent1 only has its own messages
        self.assertEqual(retrieved_agent1_messages[0].content, "Agent 1 question")
        self.assertEqual(retrieved_agent1_messages[1].content, "Agent 1 response")

        # Verify agent2 only has its own messages
        self.assertEqual(retrieved_agent2_messages[0].content, "Agent 2 question")
        self.assertEqual(retrieved_agent2_messages[1].content, "Agent 2 response")

    def test_serialization_preserves_session_id(self):
        """Test that agent serialization preserves custom session_id"""
        custom_session_id = "serialization-test-session"

        # Create agent with custom session_id
        original_agent = Agent(
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.custom_instructions,
            agent_config=self.config,
            session_id=custom_session_id,
        )

        # Add some memory
        test_messages = [
            ChatMessage(role=MessageRole.USER, content="Test question"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Test answer"),
        ]
        original_agent.memory.put_messages(test_messages)

        # Serialize the agent
        serialized_data = original_agent.dumps()

        # Deserialize the agent
        restored_agent = Agent.loads(serialized_data)

        # Verify session_id is preserved
        self.assertEqual(restored_agent.session_id, custom_session_id)
        self.assertEqual(restored_agent.memory.session_id, custom_session_id)

        # Verify memory is preserved
        restored_messages = restored_agent.memory.get()
        self.assertEqual(len(restored_messages), 2)
        self.assertEqual(restored_messages[0].content, "Test question")
        self.assertEqual(restored_messages[1].content, "Test answer")

    def test_chat_history_initialization_with_session_id(self):
        """Test Agent initialization with chat_history and custom session_id"""
        custom_session_id = "chat-history-session"
        chat_history = [
            ("Hello", "Hi there!"),
            ("How are you?", "I'm doing well, thank you!"),
        ]

        agent = Agent(
            tools=self.tools,
            topic=self.topic,
            custom_instructions=self.custom_instructions,
            agent_config=self.config,
            session_id=custom_session_id,
            chat_history=chat_history,
        )

        # Verify session_id is correct
        self.assertEqual(agent.session_id, custom_session_id)
        self.assertEqual(agent.memory.session_id, custom_session_id)

        # Verify chat history was loaded into memory
        messages = agent.memory.get()
        self.assertEqual(len(messages), 4)  # 2 user + 2 assistant messages

        # Verify message content and roles
        self.assertEqual(messages[0].role, MessageRole.USER)
        self.assertEqual(messages[0].content, "Hello")
        self.assertEqual(messages[1].role, MessageRole.ASSISTANT)
        self.assertEqual(messages[1].content, "Hi there!")
        self.assertEqual(messages[2].role, MessageRole.USER)
        self.assertEqual(messages[2].content, "How are you?")
        self.assertEqual(messages[3].role, MessageRole.ASSISTANT)
        self.assertEqual(messages[3].content, "I'm doing well, thank you!")


if __name__ == "__main__":
    unittest.main()
