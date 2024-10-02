import unittest

from vectara_agentic.tools import VectaraTool, VectaraToolFactory, ToolsFactory, ToolType
from vectara_agentic.agent import Agent
from pydantic import Field, BaseModel
from llama_index.core.tools import FunctionTool


class TestToolsPackage(unittest.TestCase):
    def test_vectara_tool_factory(self):
        vectara_customer_id = "4584783"
        vectara_corpus_id = "4"
        vectara_api_key = "api_key"
        vec_factory = VectaraToolFactory(
            vectara_customer_id, vectara_corpus_id, vectara_api_key
        )

        self.assertEqual(vectara_customer_id, vec_factory.vectara_customer_id)
        self.assertEqual(vectara_corpus_id, vec_factory.vectara_corpus_id)
        self.assertEqual(vectara_api_key, vec_factory.vectara_api_key)

        class QueryToolArgs(BaseModel):
            query: str = Field(description="The user query")

        query_tool = vec_factory.create_rag_tool(
            tool_name="rag_tool",
            tool_description="""
            Returns a response (str) to the user query based on the data in this corpus.
            """,
            tool_args_schema=QueryToolArgs,
        )

        self.assertIsInstance(query_tool, VectaraTool)
        self.assertIsInstance(query_tool, FunctionTool)
        self.assertEqual(query_tool.tool_type, ToolType.QUERY)

    def test_tool_factory(self):
        def mult(x, y):
            return x * y

        tools_factory = ToolsFactory()
        other_tool = tools_factory.create_tool(mult)
        self.assertIsInstance(other_tool, VectaraTool)
        self.assertIsInstance(other_tool, FunctionTool)
        self.assertEqual(other_tool.tool_type, ToolType.QUERY)

    def test_llama_index_tools(self):
        tools_factory = ToolsFactory()

        llama_tools = tools_factory.get_llama_index_tools(
            tool_package_name="arxiv",
            tool_spec_name="ArxivToolSpec"
        )

        arxiv_tool = llama_tools[0]

        self.assertIsInstance(arxiv_tool, VectaraTool)
        self.assertIsInstance(arxiv_tool, FunctionTool)
        self.assertEqual(arxiv_tool.tool_type, ToolType.QUERY)

    def test_public_repo(self):
        vectara_customer_id = "1366999410"
        vectara_corpus_id = "1"
        vectara_api_key = "zqt_UXrBcnI2UXINZkrv4g1tQPhzj02vfdtqYJIDiA"

        class QueryToolArgs(BaseModel):
            query: str = Field(description="The user query")

        agent = Agent.from_corpus(
            vectara_customer_id=vectara_customer_id,
            vectara_corpus_id=vectara_corpus_id,
            vectara_api_key=vectara_api_key,
            tool_name="ask_vectara",
            data_description="data from Vectara website",
            assistant_specialty="RAG as a service",
            vectara_summarizer="mockingbird-1.0-2024-07-16"
        )

        self.assertIn("Vectara is an end-to-end platform", agent.chat("What is Vectara?"))


if __name__ == "__main__":
    unittest.main()
