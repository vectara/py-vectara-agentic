import unittest

from vectara_agentic.agent import Agent, AgentType
from vectara_agentic.agent_config import AgentConfig
from vectara_agentic.tools import ToolsFactory, VectaraToolFactory
from vectara_agentic.types import ModelProvider

import nest_asyncio

nest_asyncio.apply()

vectara_corpus_key = "vectara-docs_1"
vectara_api_key = 'zqt_UXrBcnI2UXINZkrv4g1tQPhzj02vfdtqYJIDiA'

vec_factory = VectaraToolFactory(vectara_api_key=vectara_api_key,
                                 vectara_corpus_key=vectara_corpus_key)
summarizer = 'vectara-summary-table-md-query-ext-jan-2025-gpt-4o'
ask_vectara = vec_factory.create_rag_tool(
    tool_name = "ask_vectara",
    tool_description = "This tool can respond to questions about Vectara.",
    reranker = "multilingual_reranker_v1", rerank_k = 100, rerank_cutoff = 0.1,
    n_sentences_before = 2, n_sentences_after = 2, lambda_val = 0.005,
    summary_num_results = 10,
    vectara_summarizer = summarizer,
    include_citations = True,
    verbose=False,
)


statements = [
    "The sky is blue.",
    "Cats are better than dogs.",
    "Python is a great programming language.",
    "The Earth revolves around the Sun.",
    "Chocolate is the best ice cream flavor.",
]
st_inx = 0
def get_statement() -> str:
    "Generate next statement"
    global st_inx
    st = statements[st_inx]
    st_inx += 1
    return st


fc_config = AgentConfig(
    agent_type=AgentType.FUNCTION_CALLING,
    main_llm_provider=ModelProvider.ANTHROPIC,
    tool_llm_provider=ModelProvider.ANTHROPIC,
)


class TestFCS(unittest.TestCase):

    def test_fcs(self):
        tools = [ToolsFactory().create_tool(get_statement)]
        topic = "statements"
        instructions = (
            f"Call the get_statement tool multiple times to get all {len(statements)} statements."
            f"Respond to the user question based exclusively on the statements you receive - do not use any other knowledge or information."
        )

        agent = Agent(
            tools=tools,
            topic=topic,
            agent_config=fc_config,
            custom_instructions=instructions,
            vectara_api_key=vectara_api_key,
        )

        res = agent.chat("Are cats better than dogs?")
        fcs = res.metadata.get("fcs", None)
        self.assertIsNotNone(fcs, "FCS score should not be None")
        self.assertIsInstance(fcs, float, "FCS score should be a float")
        self.assertGreater(
            fcs, 0.5, "FCS score should be higher than 0.5 for this question"
        )

    def test_vectara_corpus(self):
        tools = [ask_vectara]
        topic = "vectara"
        instructions = "Answer user queries about Vectara."

        query = "What is Vectara and what API endpoints are available of the Vectara platform?"
        agent = Agent(
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
            agent_config=AgentConfig(),
            vectara_api_key=vectara_api_key,
        )

        res = agent.chat(query)
        fcs = res.metadata.get("fcs", None)
        self.assertIn("Vectara", res.response)
        self.assertGreater(fcs, 0.5, "FCS score should be higher than 0.5 for this question")


if __name__ == "__main__":
    unittest.main()
