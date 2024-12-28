# <img src="https://raw.githubusercontent.com/vectara/py-vectara-agentic/main/.github/assets/Vectara-logo.png" alt="Vectara Logo" width="30" height="30" style="vertical-align: middle;"> vectara-agentic

<p align="center">
  <a href="https://vectara.github.io/vectara-agentic-docs">Documentation</a> ·
  <a href="#examples">Examples</a> ·
  <a href="https://discord.gg/S9dwgCNEFs">Discord</a>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License">
  </a>
  <a href="https://github.com/vectara/py-vectara-agentic/graphs/commit-activity">
    <img src="https://img.shields.io/badge/Maintained%3F-yes-green.svg" alt="Maintained">
  </a>
  <a href="https://twitter.com/vectara">
    <img src="https://img.shields.io/twitter/follow/vectara.svg?style=social&label=Follow%20%40Vectara" alt="Twitter">
  </a>
</p>

## ✨ Overview

`vectara-agentic` is a Python library for developing powerful AI assistants and agents using Vectara and Agentic-RAG. It leverages the LlamaIndex Agent framework, customized for use with Vectara.

<p align="center">
<img src="https://raw.githubusercontent.com/vectara/py-vectara-agentic/main/.github/assets/diagram1.png" alt="Agentic RAG diagram" width="100%" style="vertical-align: middle;">
</p>

###  Features

- Enables easy creation of custom AI assistants and agents.
- Create a Vectara RAG tool with a single line of code.
- Supports `ReAct`, `OpenAIAgent`, `LATS' and `LLMCompiler` agent types.
- Includes pre-built tools for various domains (e.g., finance, legal).
- Integrates with various LLM inference services like OpenAI, Anthropic, Gemini, GROQ, Together.AI, Cohere and Fireworks
- Built-in support for observability with Arize Phoenix

### 📚 Example AI Assistants

Check out our example AI assistants:

- [Financial Assistant](https://huggingface.co/spaces/vectara/finance-chat)
- [Justice Harvard Teaching Assistant](https://huggingface.co/spaces/vectara/Justice-Harvard)
- [Legal Assistant](https://huggingface.co/spaces/vectara/legal-agent)


###  Prerequisites

- [Vectara account](https://console.vectara.com/signup/?utm_source=github&utm_medium=code&utm_term=DevRel&utm_content=vectara-agentic&utm_campaign=github-code-DevRel-vectara-agentic)
- A Vectara corpus with an [API key](https://docs.vectara.com/docs/api-keys)
- [Python 3.10 or higher](https://www.python.org/downloads/)
- OpenAI API key (or API keys for Anthropic, TOGETHER.AI, Fireworks AI, Cohere, GEMINI or GROQ, if you choose to use them)

###  Installation

```bash
pip install vectara-agentic
```

## 🚀 Quick Start

### 1. Create a Vectara RAG tool

```python
import os
from vectara_agentic.tools import VectaraToolFactory
from pydantic import BaseModel, Field

vec_factory = VectaraToolFactory(
    vectara_api_key=os.environ['VECTARA_API_KEY'],
    vectara_customer_id=os.environ['VECTARA_CUSTOMER_ID'],
    vectara_corpus_id=os.environ['VECTARA_CORPUS_ID']
)

years = list(range(2020, 2024))
tickers = {
    "AAPL": "Apple Computer",
    "GOOG": "Google",
    "AMZN": "Amazon",
    "SNOW": "Snowflake",
}

class QueryFinancialReportsArgs(BaseModel):
    query: str = Field(..., description="The user query.")
    year: int | str = Field(..., description=f"The year this query relates to. An integer between {min(years)} and {max(years)} or a string specifying a condition on the year (example: '>2020').")
    ticker: str = Field(..., description=f"The company ticker. Must be a valid ticket symbol from the list {tickers.keys()}.")

query_financial_reports_tool = vec_factory.create_rag_tool(
    tool_name="query_financial_reports",
    tool_description="Query financial reports for a company and year",
    tool_args_schema=QueryFinancialReportsArgs,
)
```

### 2. Create other tools (optional)

In addition to RAG tools, you can generate a lot of other types of tools the agent can use. These could be mathematical tools, tools 
that call other APIs to get more information, or any other type of tool.

See [Agent Tools](#agent-tools) for more information.

### 3. Create your agent

```python
from vectara_agentic import Agent

agent = Agent(
    tools=[query_financial_reports_tool],
    topic="10-K financial reports",
    custom_instructions="""
        - You are a helpful financial assistant in conversation with a user. Use your financial expertise when crafting a query to the tool, to ensure you get the most accurate information.
        - You can answer questions, provide insights, or summarize any information from financial reports.
        - A user may refer to a company's ticker instead of its full name - consider those the same when a user is asking about a company.
        - When calculating a financial metric, make sure you have all the information from tools to complete the calculation.
        - In many cases you may need to query tools on each sub-metric separately before computing the final metric.
        - When using a tool to obtain financial data, consider the fact that information for a certain year may be reported in the following year's report.
        - Report financial data in a consistent manner. For example if you report revenue in thousands, always report revenue in thousands.
    """
)
```

### 4. Run your agent

```python
response = agent.chat("What was the revenue for Apple in 2021?")
print(response)
```

## 🛠️ Agent Tools

`vectara-agentic` provides a few tools out of the box:
1. **Standard tools**: 
- `summarize_text`: a tool to summarize a long text into a shorter summary (uses LLM)
- `rephrase_text`: a tool to rephrase a given text, given a set of rephrase instructions (uses LLM)
  
2. **Legal tools**: a set of tools for the legal vertical, such as:
- `summarize_legal_text`: summarize legal text with a certain point of view
- `critique_as_judge`: critique a legal text as a judge, providing their perspective

3. **Financial tools**: based on tools from Yahoo! Finance:
- tools to understand the financials of a public company like: `balance_sheet`, `income_statement`, `cash_flow`
- `stock_news`: provides news about a company
- `stock_analyst_recommendations`: provides stock analyst recommendations for a company.

4. **Database tools**: providing tools to inspect and query a database
- `list_tables`: list all tables in the database
- `describe_tables`: describe the schema of tables in the database
- `load_data`: returns data based on a SQL query
- `load_sample_data`: returns the first 25 rows of a table
- `load_unique_values`: returns the top unique values for a given column

In addition, we include various other tools from LlamaIndex ToolSpecs:
* Tavily search
* EXA.AI
* arxiv
* neo4j & Kuzu for Graph integration
* Google tools (including gmail, calendar, and search)
* Slack

Note that some of these tools may require API keys as environment variables

You can create your own tool directly from a Python function using the `create_tool()` method of the `ToolsFactory` class:

```python
def mult_func(x, y):
    return x * y

mult_tool = ToolsFactory().create_tool(mult_func)
```

## 🛠️ Configuration

Configure `vectara-agentic` using environment variables:

- `VECTARA_AGENTIC_AGENT_TYPE`: valid values are `REACT`, `LLMCOMPILER`, `LATS` or `OPENAI` (default: `OPENAI`)
- `VECTARA_AGENTIC_MAIN_LLM_PROVIDER`: valid values are `OPENAI`, `ANTHROPIC`, `TOGETHER`, `GROQ`, `COHERE`, `GEMINI` or `FIREWORKS` (default: `OPENAI`)
- `VECTARA_AGENTIC_MAIN_MODEL_NAME`: agent model name (default depends on provider)
- `VECTARA_AGENTIC_TOOL_LLM_PROVIDER`: tool LLM provider (default: `OPENAI`)
- `VECTARA_AGENTIC_TOOL_MODEL_NAME`: tool model name (default depends on provider)
- `VECTARA_AGENTIC_OBSERVER_TYPE`: valid values are `ARIZE_PHOENIX` or `NONE` (default: `NONE`)

When creating a VectaraToolFactory, you can pass in a `vectara_api_key`, `vectara_customer_id`, and `vectara_corpus_id` to the factory. If not passed in, it will be taken from the environment variables. Note that `VECTARA_CORPUS_ID` can be a single ID or a comma-separated list of IDs (if you want to query multiple corpora).

## ℹ️ Additional Information

### About Custom Instructions for your Agent

The custom instructions you provide to the agent guide its behavior.
Here are some guidelines when creating your instructions:
- Write precise and clear instructions, without overcomplicating.
- Consider edge cases and unusual or atypical scenarios.
- Be cautious to not over-specify behavior based on your primary use-case, as it may limit the agent's ability to behave properly in others.

###  Diagnostics

The `Agent` class defines a few helpful methods to help you understand the internals of your application. 
* The `report()` method prints out the agent object’s type, the tools, and the LLMs used for the main agent and tool calling.
* The `token_counts()` method tells you how many tokens you have used in the current session for both the main agent and tool calling LLMs. This can be helpful if you want to track spend by token.

###  Serialization

The `Agent` class supports serialization. Use the `dumps()` to serialize and `loads()` to read back from a serialized stream.

###  Observability

vectara-agentic supports observability via the existing integration of LlamaIndex and Arize Phoenix.
First, set `os["VECTARA_AGENTIC_OBSERVER_TYPE"] = "ARIZE_PHOENIX"`.
Then you can use Arize Phoenix in three ways: 
1. **Locally**. 
   1. If you have a local phoenix server that you've run using e.g. `python -m phoenix.server.main serve`, vectara-agentic will send all traces to it.
   2. If not, vectara-agentic will run a local instance during the agent's lifecycle, and will close it when finished.
   3. In both cases, traces will be sent to the local instance, and you can see the dashboard at `http://localhost:6006`
2. **Hosted Instance**. In this case the traces are sent to the Phoenix instances hosted on Arize.
   1. Go to `https://app.phoenix.arize.com`, setup an account if you don't have one.
   2. create an API key and put it in the `PHOENIX_API_KEY` variable. This variable indicates you want to use the hosted version.
   3. To view the traces go to `https://app.phoenix.arize.com`.

Now when you run your agent, all call traces are sent to Phoenix and recorded. 
In addition, vectara-agentic also records `FCS` (factual consistency score, aka HHEM) values into Arize for every Vectara RAG call. You can see those results in the `Feedback` column of the arize UI.

## 🌐 API Endpoint

`vectara-agentic` can be easily hosted locally or on a remote machine behind an API endpoint, by following theses steps:

### Step 1: Setup your API key
Ensure that you have your API key set up as an environment variable:

```
export VECTARA_AGENTIC_API_KEY=<YOUR-ENDPOINT-API-KEY>
```

if you don't specify an Endpoint API key it uses the default "dev-api-key".

### Step 2: Start the API Server
Initialize the agent and start the FastAPI server by following this example:


```
from vectara_agentic.agent import Agent
from vectara_agentic.agent_endpoint import start_app
agent = Agent(...)            # Initialize your agent with appropriate parameters
start_app(agent)
```

You can customize the host and port by passing them as arguments to `start_app()`:
* Default: host="0.0.0.0" and port=8000.
For example:
```
start_app(agent, host="0.0.0.0", port=8000)
```

### Step 3: Access the API Endpoint
Once the server is running, you can interact with it using curl or any HTTP client. For example:

```
curl -G "http://<remote-server-ip>:8000/chat" \
--data-urlencode "message=What is Vectara?" \
-H "X-API-Key: <YOUR-ENDPOINT-API-KEY>"
```

## 🤝 Contributing

We welcome contributions! Please see our [contributing guide](https://github.com/vectara/py-vectara-agentic/blob/main/CONTRIBUTING.md) for more information.

## 📝 License

This project is licensed under the Apache 2.0 License. See the [LICENSE](https://github.com/vectara/py-vectara-agentic/blob/master/LICENSE) file for details.

## 📞 Contact

- Website: [vectara.com](https://vectara.com)
- Twitter: [@vectara](https://twitter.com/vectara)
- GitHub: [@vectara](https://github.com/vectara)
- LinkedIn: [@vectara](https://www.linkedin.com/company/vectara/)
- Discord: [Join our community](https://discord.gg/GFb8gMz6UH)
