import unittest

from vectara_agentic.agent import Agent, AgentType
from vectara_agentic.agent_config import AgentConfig
from vectara_agentic.tools import ToolsFactory
from vectara_agentic.types import ModelProvider

from vectara_agentic.sub_query_workflow import SequentialSubQuestionsWorkflow
from vectara_agentic.agent import AgentStatusType

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


def get_valid_years(
    start_year: int | str = None,
    end_year: int | str = None,
) -> list[str]:
    """
    Returns a list of the years for which financial reports are available.
    Always check this before using any other tool.
    """
    if isinstance(start_year, str):
        start_year = int(start_year)
    if isinstance(end_year, str):
        end_year = int(end_year)
    return [
        year
        for year in years
        if (start_year is None or year >= start_year)
        and (end_year is None or year <= end_year)
    ]


config_gemini = AgentConfig(
    agent_type=AgentType.FUNCTION_CALLING,
    main_llm_provider=ModelProvider.GEMINI,
    tool_llm_provider=ModelProvider.GEMINI,
)

import llama_index.core

llama_index.core.set_global_handler("simple")


def agent_progress_callback(status_type: AgentStatusType, msg: str, event_id: str):
    print(f"Agent progress: {status_type} (id: {event_id}) - {msg}")


class TestGeminiMode(unittest.TestCase):

    def test_tools(self):
        tools = [
            ToolsFactory().create_tool(tool)
            for tool in [mult, get_company_info, get_valid_years]
        ]

        financial_bot_instructions = """
        - You are a helpful financial assistant, with expertise in financial reporting, in conversation with a user.
        - Never base your on general industry knowledge, only use information from tool calls.
        - Always check the 'get_company_info' and 'get_valid_years' tools to validate company and year are valid.
        - Respond in a compact format by using appropriate units of measure (e.g., K for thousands, M for millions, B for billions).
        Do not report the same number twice (e.g. $100K and 100,000 USD).
        - Do not include URLs unless they are provided in the output of a tool response and are valid URLs.
        Ignore references or citations in the 'ask_transcripts' tool output if they have an empty URL (for example "[2]()").
        - When querying a tool for a numeric value or KPI, use a concise and non-ambiguous description of what you are looking for.
        - If you calculate a metric, make sure you have all the necessary information to complete the calculation. Don't guess.
        - Your response should not be in markdown format.
        """

        def query_logging(query: str, response: str):
            print(f"Logging query={query}, response={response}")

        agent = Agent(
            tools=tools,
            topic="Financial data, annual reports and 10-K filings",
            custom_instructions=financial_bot_instructions,
            agent_config=config_gemini,
            agent_progress_callback=agent_progress_callback,
            query_logging_callback=query_logging,
            verbose=True,
            workflow_cls=SequentialSubQuestionsWorkflow,
        )

        res = agent.chat("What is the EBITDA of Citigroup in 2022?")
        print(res)


if __name__ == "__main__":
    unittest.main()
