import unittest

from vectara_agentic.tools import (
    VectaraTool,
    VectaraToolFactory,
    ToolType,
)
from llama_index.core.tools import FunctionTool

# Special test account credentials for Vectara
vectara_corpus_key = "vectara-docs_1"
vectara_api_key = "zqt_UXrBcnI2UXINZkrv4g1tQPhzj02vfdtqYJIDiA"


class TestLLMPackage(unittest.TestCase):

    def test_vectara_openai(self):
        vec_factory = VectaraToolFactory(
            vectara_corpus_key=vectara_corpus_key,
            vectara_api_key=vectara_api_key
        )

        self.assertEqual(vectara_corpus_key, vec_factory.vectara_corpus_key)
        self.assertEqual(vectara_api_key, vec_factory.vectara_api_key)

        query_tool = vec_factory.create_rag_tool(
            tool_name="rag_tool",
            tool_description="""
            Returns a response (str) to the user query based on the data in this corpus.
            """,
        )

        self.assertIsInstance(query_tool, VectaraTool)
        self.assertIsInstance(query_tool, FunctionTool)
        self.assertEqual(query_tool.metadata.tool_type, ToolType.QUERY)

        res = query_tool(query="What is Vectara?")
        self.assertIn("Vectara is an end-to-end platform", str(res))

        query_tool = vec_factory.create_rag_tool(
            tool_name="rag_tool",
            tool_description="""
            Returns a response (str) to the user query based on the data in this corpus.
            """,
            llm_name="gpt-4o-mini",
        )

        self.assertIsInstance(query_tool, VectaraTool)
        self.assertIsInstance(query_tool, FunctionTool)
        self.assertEqual(query_tool.metadata.tool_type, ToolType.QUERY)

        res = query_tool(query="What is Vectara?")
        self.assertIn("Vectara is an end-to-end platform", str(res))

    def test_vectara_mockingbird(self):
        vec_factory = VectaraToolFactory(vectara_corpus_key, vectara_api_key)
        query_tool = vec_factory.create_rag_tool(
            tool_name="rag_tool",
            tool_description="""
            Returns a response (str) to the user query based on the data in this corpus.
            """,
            vectara_summarizer="mockingbird-2.0",
        )
        res = query_tool(query="What is Vectara?")
        self.assertIn("Vectara is an end-to-end platform", str(res))


if __name__ == "__main__":
    unittest.main()
