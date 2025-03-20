import os
import unittest
import subprocess
import time
import requests
import signal

from vectara_agentic.agent import Agent, AgentType
from vectara_agentic.agent_config import AgentConfig
from vectara_agentic.types import ModelProvider
from vectara_agentic.tools import ToolsFactory

class TestPrivateLLM(unittest.TestCase):

    @classmethod
    def setUp(cls):
        # Start the Flask server as a subprocess
        cls.flask_process = subprocess.Popen(
            ['flask', 'run', '--port=5000'],
            env={**os.environ, 'FLASK_APP': 'tests.endpoint:app', 'FLASK_ENV': 'development'},
            stdout=None, stderr=None,
        )
        # Wait for the server to start
        timeout = 10
        url = 'http://127.0.0.1:5000/'
        for _ in range(timeout):
            try:
                requests.get(url)
                return
            except requests.ConnectionError:
                time.sleep(1)
        raise RuntimeError(f"Failed to start Flask server at {url}")

    @classmethod
    def tearDown(cls):
        # Terminate the Flask server
        cls.flask_process.send_signal(signal.SIGINT)
        cls.flask_process.wait()

    def test_endpoint(self):
        def mult(x, y):
            return x * y

        tools = [ToolsFactory().create_tool(mult)]
        topic = "calculator"
        custom_instructions = "you are an agent specializing in math, assisting a user."
        config = AgentConfig(
            agent_type=AgentType.REACT,
            main_llm_provider=ModelProvider.PRIVATE,
            main_llm_model_name="gpt-4o",
            private_llm_api_base="http://127.0.0.1:5000/v1",
            private_llm_api_key="TEST_API_KEY",
        )
        agent = Agent(agent_config=config, tools=tools, topic=topic,
                      custom_instructions=custom_instructions)

        # To run this test, you must have OPENAI_API_KEY in your environment
        self.assertEqual(
            agent.chat(
                "What is 5 times 10. Only give the answer, nothing else."
            ).response.replace("$", "\\$"),
            "50",
        )


if __name__ == "__main__":
    unittest.main()
