import unittest

from vectara_agentic.agent import Agent
from vectara_agentic.tools import VectaraToolFactory

vectara_corpus_key = "vectara-docs_1"
vectara_api_key = "zqt_UXrBcnI2UXINZkrv4g1tQPhzj02vfdtqYJIDiA"


class TestAgentPackage(unittest.TestCase):

    def test_return_direct1(self):
        vec_factory = VectaraToolFactory(vectara_corpus_key, vectara_api_key)

        query_tool = vec_factory.create_rag_tool(
            tool_name="rag_tool",
            tool_description="""
            A dummy tool for testing return_direct.
            """,
            return_direct=True,
        )

        agent = Agent(
            tools=[query_tool],
            topic="Sample topic",
            custom_instructions="You are a helpful assistant.",
        )
        res = agent.chat("What is Vectara?")
        self.assertIn("Response:", str(res))
        self.assertIn("fcs_score", str(res))
        self.assertIn("References:", str(res))

    def test_from_corpus(self):
        agent = Agent.from_corpus(
            tool_name="rag_tool",
            vectara_corpus_key=vectara_corpus_key,
            vectara_api_key=vectara_api_key,
            data_description="stuff about Vectara",
            assistant_specialty="question answering",
            return_direct=True,
        )
        res = agent.chat("What is Vectara?")
        self.assertIn("Response:", str(res))
        self.assertIn("fcs_score", str(res))
        self.assertIn("References:", str(res))


if __name__ == "__main__":
    unittest.main()
