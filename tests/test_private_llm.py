# Suppress external dependency warnings before any other imports
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

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


FLASK_PORT = 5001


class TestPrivateLLM(unittest.TestCase):

    @classmethod
    def setUp(cls):
        # Start the Flask server as a subprocess
        cls.flask_process = subprocess.Popen(
            ["flask", "run", f"--port={FLASK_PORT}"],
            env={
                **os.environ,
                "FLASK_APP": "tests.endpoint:app",
                "FLASK_ENV": "development",
            },
            stdout=None,
            stderr=None,
        )
        # Wait for the server to start
        timeout = 10
        url = f"http://127.0.0.1:{FLASK_PORT}/"
        for _ in range(timeout):
            try:
                requests.get(url)
                print("Flask server started for private LLM unit test")
                return
            except requests.ConnectionError:
                time.sleep(1)
        raise RuntimeError(f"Failed to start Flask server at {url}")

    @classmethod
    def tearDown(cls):
        # Terminate the Flask server
        cls.flask_process.send_signal(signal.SIGINT)
        cls.flask_process.wait()

    def test_endpoint_openai(self):
        """Test private LLM endpoint with OpenAI model."""
        def mult(x: float, y: float) -> float:
            return x * y

        tools = [ToolsFactory().create_tool(mult)]
        topic = "calculator"
        custom_instructions = "you are an agent specializing in math, assisting a user."
        config = AgentConfig(
            agent_type=AgentType.FUNCTION_CALLING,
            main_llm_provider=ModelProvider.PRIVATE,
            main_llm_model_name="gpt-4.1-mini",
            private_llm_api_base=f"http://127.0.0.1:{FLASK_PORT}/v1",
            private_llm_api_key="TEST_API_KEY",
        )
        agent = Agent(
            agent_config=config,
            tools=tools,
            topic=topic,
            custom_instructions=custom_instructions,
            verbose=False,
        )

        # To run this test, you must have OPENAI_API_KEY in your environment
        res = agent.chat(
            "What is 5 times 10. Only give the answer, nothing else."
        ).response
        if res is None:
            self.fail("Agent returned None response")
        # Convert to string for comparison if it's a number
        if isinstance(res, (int, float)):
            res = str(int(res))
        self.assertEqual(res, "50")

    def test_endpoint_gpt_oss_120b(self):
        """Test private LLM endpoint with GPT-OSS-120B via Together.AI."""
        def mult(x: float, y: float) -> float:
            return x * y

        tools = [ToolsFactory().create_tool(mult)]
        topic = "calculator"
        custom_instructions = "you are an agent specializing in math, assisting a user."
        config = AgentConfig(
            agent_type=AgentType.FUNCTION_CALLING,
            main_llm_provider=ModelProvider.PRIVATE,
            main_llm_model_name="openai/gpt-oss-120b",
            private_llm_api_base=f"http://127.0.0.1:{FLASK_PORT}/v1",
            private_llm_api_key="TEST_API_KEY",
        )
        agent = Agent(
            agent_config=config,
            tools=tools,
            topic=topic,
            custom_instructions=custom_instructions,
            verbose=False,
        )

        # To run this test, you must have TOGETHER_API_KEY in your environment
        res = agent.chat(
            "What is 7 times 8. Only give the answer, nothing else."
        ).response
        if res is None:
            self.fail("Agent returned None response")
        # Convert to string for comparison if it's a number
        if isinstance(res, (int, float)):
            res = str(int(res))
        # Check if "56" is contained in the response (model may add prefixes like "final")
        self.assertIn("56", res)

    def test_endpoint_deepseek_v3(self):
        """Test private LLM endpoint with DeepSeek-V3 via Together.AI."""
        def mult(x: float, y: float) -> float:
            return x * y

        tools = [ToolsFactory().create_tool(mult)]
        topic = "calculator"
        custom_instructions = "you are an agent specializing in math, assisting a user."
        config = AgentConfig(
            agent_type=AgentType.FUNCTION_CALLING,
            main_llm_provider=ModelProvider.PRIVATE,
            main_llm_model_name="deepseek-ai/DeepSeek-V3",
            private_llm_api_base=f"http://127.0.0.1:{FLASK_PORT}/v1",
            private_llm_api_key="TEST_API_KEY",
        )
        agent = Agent(
            agent_config=config,
            tools=tools,
            topic=topic,
            custom_instructions=custom_instructions,
            verbose=False,
        )

        # To run this test, you must have TOGETHER_API_KEY in your environment
        res = agent.chat(
            "What is 9 times 6. Only give the answer, nothing else."
        ).response
        if res is None:
            self.fail("Agent returned None response")
        # Convert to string for comparison if it's a number
        if isinstance(res, (int, float)):
            res = str(int(res))
        # Check if "54" is contained in the response (model may add prefixes)
        self.assertIn("54", res)


if __name__ == "__main__":
    unittest.main()
