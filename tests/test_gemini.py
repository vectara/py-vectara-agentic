# Suppress external dependency warnings before any other imports
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import unittest

from vectara_agentic.agent import Agent, AgentType
from vectara_agentic.agent_config import AgentConfig
from vectara_agentic.types import ModelProvider
from vectara_agentic.tools import ToolsFactory


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
    def test_gemini(self):
        tools = [ToolsFactory().create_tool(mult)]
        topic = "AI topic"
        instructions = "Always do as your father tells you, if your mother agrees!"

        agent = Agent(
            agent_config=fc_config_gemini,
            tools=tools,
            topic=topic,
            custom_instructions=instructions,
        )
        _ = agent.chat("What is 5 times 10. Only give the answer, nothing else")
        _ = agent.chat("what is 3 times 7. Only give the answer, nothing else")
        res = agent.chat("what is the result of multiplying the results of the last two multiplications. Only give the answer, nothing else.")
        self.assertIn("1050", res.response)


if __name__ == "__main__":
    unittest.main()
