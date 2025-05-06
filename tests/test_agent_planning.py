import unittest

from vectara_agentic.agent_config import AgentConfig
from vectara_agentic.agent import Agent
from vectara_agentic.tools import VectaraToolFactory

# SETUP special test account credentials for vectara
# It's okay to expose these credentials in the test code
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

class TestAgentPlanningPackage(unittest.TestCase):

    def test_no_planning(self):
        tools = [ask_vectara]
        topic = "vectara"
        instructions = "Answer user queries about Vectara."

        query = "What is Vectara and what demos are available of the Vectara platform?"
        agent = Agent(
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
            agent_config=AgentConfig(),
        )
        res = agent.chat(query)
        self.assertIn("demos", res.response)
        self.assertIn("Vectara", res.response)

    def test_structured_planning(self):
        tools = [ask_vectara]
        topic = "vectara"
        instructions = "Answer user queries about Vectara."

        query = "What is Vectara and what demos are available of the Vectara platform?"
        agent = Agent(
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
            agent_config=AgentConfig(),
            use_structured_planning=True,
        )

        res = agent.chat(query)
        self.assertIn("demos", res.response)
        self.assertIn("Vectara", res.response)


if __name__ == "__main__":
    unittest.main()
