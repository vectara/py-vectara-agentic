# vectara-agentic

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Maintained](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/vectara/py-vectara-agentic/graphs/commit-activity)
[![Twitter](https://img.shields.io/twitter/follow/vectara.svg?style=social&label=Follow%20%40Vectara)](https://twitter.com/vectara)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-blue?style=social&logo=discord)](https://discord.com/invite/GFb8gMz6UH)

## Overview

`vectara-agentic` is a Python library for developing powerful AI assistants using Vectara and Agentic-RAG. It leverages the LlamaIndex Agent framework, customized for use with Vectara.

### Key Features

- Supports `ReAct` and `OpenAIAgent` agent types.
- Includes pre-built tools for various domains (e.g., finance, legal).
- Enables easy creation of custom AI assistants and agents.

## Important Links

- Documentation: [https://vectara.github.io/vectara-agentic-docs/](https://vectara.github.io/vectara-agentic-docs/)

## Prerequisites

- [Vectara account](https://console.vectara.com/signup/?utm_source=github&utm_medium=code&utm_term=DevRel&utm_content=vectara-agentic&utm_campaign=github-code-DevRel-vectara-agentic)
- A Vectara corpus with an [API key](https://docs.vectara.com/docs/api-keys)
- [Python 3.10 or higher](https://www.python.org/downloads/)
- OpenAI API key (or API keys for Anthropic, TOGETHER.AI, Fireworks AI, Cohere, or GROQ)

## Installation

```bash
pip install vectara-agentic
```

## Quick Start

1. **Create a Vectara RAG tool**

```python
import os
from vectara_agentic import VectaraToolFactory

vec_factory = VectaraToolFactory(
    vectara_api_key=os.environ['VECTARA_API_KEY'],
    vectara_customer_id=os.environ['VECTARA_CUSTOMER_ID'],
    vectara_corpus_id=os.environ['VECTARA_CORPUS_ID']
)

class QueryFinancialReportsArgs(BaseModel):
        query: str = Field(..., description="The user query.")
        year: int = Field(..., description=f"The year. An integer between {min(years)} and {max(years)}.")
        ticker: str = Field(..., description=f"The company ticker. Must be a valid ticket symbol from the list {tickers.keys()}.")

query_financial_reports = vec_factory.create_rag_tool(
    tool_name="query_financial_reports",
    tool_description="Query financial reports for a company and year",
    tool_args_schema=QueryFinancialReportsArgs,
)
```

2. **Create other tools (optional)**

In addition to RAG tools, you can generate a lot of other types of tools the agent can use. These could be mathematical tools, tools 
that call other APIs to get more information, or any other type of tool.

See [Tools](#agent-tools) for more information.

3. **Create your agent**

```python
agent = Agent(
    tools = [query_financial_reports],
    topic = topic_of_expertise,
    custom_instructions = financial_bot_instructions,
)
```
- `tools` is the list of tools you want to provide to the agent. In this example it's just a single tool.
- `topic` is a string that defines the expertise you want the agent to specialize in.
- `custom_instructions` is an optional string that defines special instructions to the agent.

For example, for a financial agent we might use:

```python
topic_of_expertise = "10-K financial reports",

financial_bot_instructions = """
    - You are a helpful financial assistant in conversation with a user. Use your financial expertise when crafting a query to the tool, to ensure you get the most accurate information.
    - You can answer questions, provide insights, or summarize any information from financial reports.
    - A user may refer to a company's ticker instead of its full name - consider those the same when a user is asking about a company.
    - When calculating a financial metric, make sure you have all the information from tools to complete the calculation.
    - In many cases you may need to query tools on each sub-metric separately before computing the final metric.
    - When using a tool to obtain financial data, consider the fact that information for a certain year may be reported in the the following year's report.
    - Report financial data in a consistent manner. For example if you report revenue in thousands, always report revenue in thousands.
    """
```

## Configuration

Configure `vectara-agentic` using environment variables:

- `VECTARA_AGENTIC_AGENT_TYPE`: valid values are `REACT`, `LLMCOMPILER` or `OPENAI` (default: `OPENAI`)
- `VECTARA_AGENTIC_MAIN_LLM_PROVIDER`: valid values are `OPENAI`, `ANTHROPIC`, `TOGETHER`, `GROQ`, `COHERE` or `FIREWORKS` (default: `OPENAI`)
- `VECTARA_AGENTIC_MAIN_MODEL_NAME`: agent model name (default depends on provider)
- `VECTARA_AGENTIC_TOOL_LLM_PROVIDER`: tool LLM provider (default: `OPENAI`)
- `VECTARA_AGENTIC_TOOL_MODEL_NAME`: tool model name (default depends on provider)

## Agent Tools

`vectara-agentic` provides a few tools out of the box:
1. Standard tools: 
- `summarize_text`: a tool to summarize a long text into a shorter summary (uses LLM)
- `rephrase_text`: a tool to rephrase a given text, given a set of rephrase instructions (uses LLM)
  
2. Legal tools: a set of tools for the legal vertical, such as:
- `summarize_legal_text`: summarize legal text with a certain point of view
- `critique_as_judge`: critique a legal text as a judge, providing their perspective

3. Financial tools: based on tools from Yahoo Finance:
- tools to understand the financials of a public company like: `balance_sheet`, `income_statement`, `cash_flow`
- `stock_news`: provides news about a company
- `stock_analyst_recommendations`: provides stock analyst recommendations for a company.

4. database_tools: providing a few tools to inspect and query a database
- `list_tables`: list all tables in the database
- `describe_tables`: describe the schema of tables in the database
- `load_data`: returns data based on a SQL query

More tools coming soon.

You can create your own tool directly from a Python function using the `create_tool()` method of the `ToolsFactor` class:

```Python
def mult_func(x, y):
    return x*y

mult_tool = ToolsFactory().create_tool(mult_func)
```

## Agent Diagnostics

The `Agent` class defines a few helpful methods to help you understand the internals of your application. 
* The `report()` method prints out the agent object’s type, the tools, and the LLMs used for the main agent and tool calling.
* The `token_counts()` method tells you how many tokens you have used in the current session for both the main agent and tool calling LLMs. This can be helpful if you want to track spend by token.

## Examples

Check out our example AI assistants:

- [Financial Assistant](https://huggingface.co/spaces/vectara/finance-chat)
- [Justice Harvard Teaching Assistant](https://huggingface.co/spaces/vectara/Justice-Harvard)
- [Legal Assistant](https://huggingface.co/spaces/vectara/legal-agent)


## Contributing

We welcome contributions! Please see our [contributing guide](https://github.com/vectara/py-vectara-agentic/blob/main/CONTRIBUTING.md) for more information.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](https://github.com/vectara/py-vectara-agentic/blob/master/LICENSE) file for details.

## Contact

- Website: [vectara.com](https://vectara.com)
- Twitter: [@vectara](https://twitter.com/vectara)
- GitHub: [@vectara](https://github.com/vectara)
- LinkedIn: [@vectara](https://www.linkedin.com/company/vectara/)
- Discord: [Join our community](https://discord.gg/GFb8gMz6UH)
