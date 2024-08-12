import unittest
from datetime import date

from vectara_agentic.agent import get_prompt, Agent, AgentType, FunctionTool


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
            get_prompt(prompt_template, topic, custom_instructions), expected_output
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

        # Only run this assert statement if you have an OPENAI_API_KEY in your environment
        self.assertEqual(
            agent.chat(
                "What is 5 times 10. Only give the answer, nothing else"
            ).replace("$", "\\$"),
            "50",
        )


if __name__ == "__main__":
    unittest.main()
