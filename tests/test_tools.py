import unittest

from pydantic import Field, BaseModel

from vectara_agentic.tools import VectaraTool, VectaraToolFactory, ToolsFactory, ToolType
from vectara_agentic.agent import Agent
from vectara_agentic.agent_config import AgentConfig

from llama_index.core.tools import FunctionTool


class TestToolsPackage(unittest.TestCase):
    def test_vectara_tool_factory(self):
        vectara_corpus_key = "corpus_key"
        vectara_api_key = "api_key"
        vec_factory = VectaraToolFactory(
            vectara_corpus_key, vectara_api_key
        )

        self.assertEqual(vectara_corpus_key, vec_factory.vectara_corpus_key)
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
        self.assertEqual(query_tool.metadata.tool_type, ToolType.QUERY)

        search_tool = vec_factory.create_search_tool(
            tool_name="search_tool",
            tool_description="""
            Returns a list of documents (str) that match the user query.
            """,
            tool_args_schema=QueryToolArgs,
        )
        self.assertIsInstance(search_tool, VectaraTool)
        self.assertIsInstance(search_tool, FunctionTool)
        self.assertEqual(search_tool.metadata.tool_type, ToolType.QUERY)

    def test_tool_factory(self):
        def mult(x: float, y: float) -> float:
            return x * y

        tools_factory = ToolsFactory()
        other_tool = tools_factory.create_tool(mult)
        self.assertIsInstance(other_tool, VectaraTool)
        self.assertIsInstance(other_tool, FunctionTool)
        self.assertEqual(other_tool.metadata.tool_type, ToolType.QUERY)

    def test_llama_index_tools(self):
        tools_factory = ToolsFactory()

        llama_tools = tools_factory.get_llama_index_tools(
            tool_package_name="arxiv",
            tool_spec_name="ArxivToolSpec"
        )

        arxiv_tool = llama_tools[0]

        self.assertIsInstance(arxiv_tool, VectaraTool)
        self.assertIsInstance(arxiv_tool, FunctionTool)
        self.assertEqual(arxiv_tool.metadata.tool_type, ToolType.QUERY)

    def test_public_repo(self):
        vectara_corpus_key = "vectara-docs_1"
        vectara_api_key = "zqt_UXrBcnI2UXINZkrv4g1tQPhzj02vfdtqYJIDiA"

        agent = Agent.from_corpus(
            vectara_corpus_key=vectara_corpus_key,
            vectara_api_key=vectara_api_key,
            tool_name="ask_vectara",
            data_description="data from Vectara website",
            assistant_specialty="RAG as a service",
            vectara_summarizer="mockingbird-1.0-2024-07-16"
        )

        self.assertIn("Vectara is an end-to-end platform", str(agent.chat("What is Vectara?")))

    def test_class_method_as_tool(self):
        class TestClass:
            def __init__(self):
                pass

            def mult(self, x, y):
                return x * y

        test_class = TestClass()
        tools = [ToolsFactory().create_tool(test_class.mult)]
        topic = "AI topic"
        instructions = "Always do as your father tells you, if your mother agrees!"
        config = AgentConfig()
        agent = Agent(
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
            agent_config=config
        )

        self.assertEqual(
            agent.chat(
                "What is 5 times 10. Only give the answer, nothing else"
            ).response.replace("$", "\\$"),
            "50",
        )


if __name__ == "__main__":
    unittest.main()
