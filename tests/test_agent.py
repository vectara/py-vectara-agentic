import unittest
from datetime import date

from vectara_agentic.agent import _get_prompt, Agent, AgentType, FunctionTool


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
            _get_prompt(prompt_template, topic, custom_instructions), expected_output
        )

    def test_agent_init(self):
        def mult(x, y):
            return x * y

        tools = [
            FunctionTool.from_defaults(
                fn=mult, name="mult", description="Multiplication functions"
            )
        ]
        topic = "AI"
        custom_instructions = "Always do as your mother tells you!"
        agent = Agent(tools, topic, custom_instructions)
        self.assertEqual(agent.agent_type, AgentType.OPENAI)
        self.assertEqual(agent.tools, tools)
        self.assertEqual(agent._topic, topic)
        self.assertEqual(agent._custom_instructions, custom_instructions)

        # To run this test, you must have OPENAI_API_KEY in your environment
        self.assertEqual(
            agent.chat(
                "What is 5 times 10. Only give the answer, nothing else"
            ).replace("$", "\\$"),
            "50",
        )

    def test_from_corpus(self):
        agent = Agent.from_corpus(
            tool_name="RAG Tool",
            vectara_customer_id="4584783",
            vectara_corpus_id="4",
            vectara_api_key="api_key",
            data_description="information",
            assistant_specialty="question answering",
        )

        self.assertIsInstance(agent, Agent)
        self.assertEqual(agent._topic, "question answering")

    def test_serialization(self):
        agent = Agent.from_corpus(
            tool_name="RAG Tool",
            vectara_customer_id="4584783",
            vectara_corpus_id="4",
            vectara_api_key="api_key",
            data_description="information",
            assistant_specialty="question answering",
        )

        agent_reloaded = agent.loads(agent.dumps())
        self.assertIsInstance(agent_reloaded, Agent)
        self.assertEqual(agent, agent_reloaded)


if __name__ == "__main__":
    unittest.main()
