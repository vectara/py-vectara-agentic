import unittest

from pydantic import Field, BaseModel

from vectara_agentic.agent import Agent, AgentType
from vectara_agentic.agent_config import AgentConfig
from vectara_agentic.tools import VectaraToolFactory
from vectara_agentic.types import ModelProvider


import nest_asyncio
nest_asyncio.apply()

tickers = {
    "C": "Citigroup",
    "COF": "Capital One",
    "JPM": "JPMorgan Chase",
    "AAPL": "Apple Computer",
    "GOOG": "Google",
    "AMZN": "Amazon",
    "SNOW": "Snowflake",
    "TEAM": "Atlassian",
    "TSLA": "Tesla",
    "NVDA": "Nvidia",
    "MSFT": "Microsoft",
    "AMD": "Advanced Micro Devices",
    "INTC": "Intel",
    "NFLX": "Netflix",
    "STT": "State Street",
    "BK": "Bank of New York Mellon",
}
years = list(range(2015, 2025))


def mult(x: float, y: float) -> float:
    "Multiply two numbers"
    return x * y


def get_company_info() -> list[str]:
    """
    Returns a dictionary of companies you can query about. Always check this before using any other tool.
    The output is a dictionary of valid ticker symbols mapped to company names.
    You can use this to identify the companies you can query about, and their ticker information.
    """
    return tickers


def get_valid_years() -> list[str]:
    """
    Returns a list of the years for which financial reports are available.
    Always check this before using any other tool.
    """
    return years


fc_config_gemini = AgentConfig(
    agent_type=AgentType.FUNCTION_CALLING,
    main_llm_provider=ModelProvider.GEMINI,
    tool_llm_provider=ModelProvider.GEMINI,
)


class TestGEMINI(unittest.TestCase):

    def test_tool_with_many_arguments(self):

        vectara_corpus_key = "vectara-docs_1"
        vectara_api_key = "zqt_UXrBcnI2UXINZkrv4g1tQPhzj02vfdtqYJIDiA"
        vec_factory = VectaraToolFactory(vectara_corpus_key, vectara_api_key)

        class QueryToolArgs(BaseModel):
            arg1: str = Field(description="the first argument", examples=["val1"])
            arg2: str = Field(description="the second argument", examples=["val2"])
            arg3: str = Field(description="the third argument", examples=["val3"])
            arg4: str = Field(description="the fourth argument", examples=["val4"])
            arg5: str = Field(description="the fifth argument", examples=["val5"])
            arg6: str = Field(description="the sixth argument", examples=["val6"])
            arg7: str = Field(description="the seventh argument", examples=["val7"])
            arg8: str = Field(description="the eighth argument", examples=["val8"])
            arg9: str = Field(description="the ninth argument", examples=["val9"])
            arg10: str = Field(description="the tenth argument", examples=["val10"])
            arg11: str = Field(description="the eleventh argument", examples=["val11"])
            arg12: str = Field(description="the twelfth argument", examples=["val12"])
            arg13: str = Field(
                description="the thirteenth argument", examples=["val13"]
            )
            arg14: str = Field(
                description="the fourteenth argument", examples=["val14"]
            )
            arg15: str = Field(description="the fifteenth argument", examples=["val15"])

        query_tool_1 = vec_factory.create_rag_tool(
            tool_name="rag_tool",
            tool_description="""
            A dummy tool that takes 15 arguments and returns a response (str) to the user query based on the data in this corpus.
            We are using this tool to test the tool factory works and does not crash with OpenAI.
            """,
            tool_args_schema=QueryToolArgs,
        )

        agent = Agent(
            tools=[query_tool_1],
            topic="Sample topic",
            custom_instructions="Call the tool with 15 arguments",
            agent_config=fc_config_gemini,
        )
        res = agent.chat("What is the stock price?")
        self.assertTrue(
            any(sub in str(res) for sub in ["I don't know", "I do not have"])
        )


if __name__ == "__main__":
    unittest.main()
