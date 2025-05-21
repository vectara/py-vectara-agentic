import unittest
from uuid import UUID

from fastapi.testclient import TestClient

# Adjust this import to point at the file where you put create_app
from vectara_agentic import create_app
from vectara_agentic.agent import Agent
from vectara_agentic.agent_config import AgentConfig


class DummyAgent(Agent):
    def __init__(self):
        # satisfy Agent.__init__(tools: ...)
        super().__init__(tools=[])

    def chat(self, message: str) -> str:
        return f"Echo: {message}"

class APITestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = DummyAgent()
        # Override only the endpoint_api_key, leave everything else as default
        cls.config = AgentConfig(endpoint_api_key="testkey")
        app = create_app(cls.agent, cls.config)
        cls.client = TestClient(app)
        cls.headers = {"X-API-Key": cls.config.endpoint_api_key}

    def test_chat_success(self):
        r = self.client.get("/chat", params={"message": "hello"}, headers=self.headers)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"response": "Echo: hello"})

    def test_chat_empty_message(self):
        r = self.client.get("/chat", params={"message": ""}, headers=self.headers)
        self.assertEqual(r.status_code, 400)
        self.assertIn("No message provided", r.json()["detail"])

    def test_chat_unauthorized(self):
        r = self.client.get("/chat", params={"message": "hello"}, headers={"X-API-Key": "bad"})
        self.assertEqual(r.status_code, 403)

    def test_completions_success(self):
        payload = {"model": "m1", "prompt": "test"}
        r = self.client.post("/v1/completions", json=payload, headers=self.headers)
        self.assertEqual(r.status_code, 200)
        data = r.json()

        # ID prefix + valid UUID check
        self.assertTrue(data["id"].startswith("cmpl-"))
        UUID(data["id"].split("-", 1)[1])

        self.assertEqual(data["model"], "m1")
        self.assertEqual(data["choices"][0]["text"], "Echo: test")
        # prompt_tokens=1, completion_tokens=2 ("Echo:", "test")
        self.assertEqual(data["usage"]["prompt_tokens"], 1)
        self.assertEqual(data["usage"]["completion_tokens"], 2)

    def test_completions_no_prompt(self):
        payload = {"model": "m1"}  # missing prompt
        r = self.client.post("/v1/completions", json=payload, headers=self.headers)
        self.assertEqual(r.status_code, 400)
        self.assertIn("`prompt` is required", r.json()["detail"])

    def test_completions_unauthorized(self):
        payload = {"model": "m1", "prompt": "hi"}
        r = self.client.post("/v1/completions", json=payload, headers={"X-API-Key": "bad"})
        self.assertEqual(r.status_code, 403)


if __name__ == "__main__":
    unittest.main()
