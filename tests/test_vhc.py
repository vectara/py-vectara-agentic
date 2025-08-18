# Suppress external dependency warnings before any other imports
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

import unittest

from vectara_agentic.agent import Agent, AgentType
from vectara_agentic.agent_config import AgentConfig
from vectara_agentic.tools import ToolsFactory
from vectara_agentic.types import ModelProvider

import nest_asyncio

nest_asyncio.apply()

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
    main_llm_provider=ModelProvider.OPENAI,
    tool_llm_provider=ModelProvider.OPENAI,
)

vectara_api_key = "zqt_UXrBcnI2UXINZkrv4g1tQPhzj02vfdtqYJIDiA"


class TestVHC(unittest.TestCase):

    def test_vhc(self):
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

        _ = agent.chat("Are large cats better than small dogs?")
        vhc_res = agent.compute_vhc()
        vhc_corrections = vhc_res.get("corrections", [])
        self.assertTrue(
            len(vhc_corrections) >= 0 and len(vhc_corrections) <= 2,
            "Corrections should be between 0 and 2",
        )


if __name__ == "__main__":
    unittest.main()
