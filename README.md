# <img src="https://raw.githubusercontent.com/vectara/py-vectara-agentic/main/.github/assets/Vectara-logo.png" alt="Vectara Logo" width="30" height="30" style="vertical-align: middle;"> vectara-agentic

<p align="center">
  <a href="https://vectara.github.io/py-vectara-agentic">Documentation</a> ·
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

`vectara-agentic` is a Python library for developing powerful AI assistants and agents using Vectara and Agentic-RAG. It leverages the LlamaIndex Agent framework and provides helper functions to quickly create tools that connect to Vectara corpora.

<p align="center">
<img src="https://raw.githubusercontent.com/vectara/py-vectara-agentic/main/.github/assets/diagram1.png" alt="Agentic RAG diagram" width="100%" style="vertical-align: middle;">
</p>

### Key Features

- **Rapid Tool Creation:**  
  Build Vectara RAG tools or search tools with a single line of code.
- **Agent Flexibility:**  
  Supports multiple agent types including `ReAct`, `OpenAIAgent`, `LATS`, and `LLMCompiler`.
- **Pre-Built Domain Tools:**  
  Tools tailored for finance, legal, and other verticals.
- **Multi-LLM Integration:**  
  Seamless integration with OpenAI, Anthropic, Gemini, GROQ, Together.AI, Cohere, Bedrock, and Fireworks.
- **Observability:**  
  Built-in support with Arize Phoenix for monitoring and feedback.
- **Workflow Support:**  
  Extend your agent’s capabilities by defining custom workflows using the `run()` method.

### 📚 Example AI Assistants

Check out our example AI assistants:

- [Financial Assistant](https://huggingface.co/spaces/vectara/finance-chat)
- [Justice Harvard Teaching Assistant](https://huggingface.co/spaces/vectara/Justice-Harvard)
- [Legal Assistant](https://huggingface.co/spaces/vectara/legal-agent)
- [EV Assistant](https://huggingface.co/spaces/vectara/ev-assistant)

###  Prerequisites

- [Vectara account](https://console.vectara.com/signup/?utm_source=github&utm_medium=code&utm_term=DevRel&utm_content=vectara-agentic&utm_campaign=github-code-DevRel-vectara-agentic)
- A Vectara corpus with an [API key](https://docs.vectara.com/docs/api-keys)
- [Python 3.10 or higher](https://www.python.org/downloads/)
- OpenAI API key (or API keys for Anthropic, TOGETHER.AI, Fireworks AI, Bedrock, Cohere, GEMINI or GROQ, if you choose to use them)

###  Installation

```bash
pip install vectara-agentic
```

## 🚀 Quick Start

### 1. Initialize the Vectara tool factory

```python
import os
from vectara_agentic.tools import VectaraToolFactory

vec_factory = VectaraToolFactory(
    vectara_api_key=os.environ['VECTARA_API_KEY'],
    vectara_customer_id=os.environ['VECTARA_CUSTOMER_ID'],
    vectara_corpus_key=os.environ['VECTARA_CORPUS_KEY']
)
```

### 2. Create a Vectara RAG Tool

A RAG tool calls the full Vectara RAG pipeline to provide summarized responses to queries grounded in data.

```python
from pydantic import BaseModel, Field

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
    lambda_val=0.005,
    summary_num_results=7, 
    # Additional arguments
)
```

See the [docs](https://vectara.github.io/vectara-agentic-docs/) for additional arguments to customize your Vectara RAG tool.

### 3. Create other tools (optional)

In addition to RAG tools, you can generate a lot of other types of tools the agent can use. These could be mathematical tools, tools 
that call other APIs to get more information, or any other type of tool.

See [Agent Tools](#agent-tools) for more information.

### 4. Create your agent

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

See the [docs](https://vectara.github.io/vectara-agentic-docs/) for additional arguments, including `agent_progress_callback` and `query_logging_callback`.

### 5. Run a chat interaction

```python
res = agent.chat("What was the revenue for Apple in 2021?")
print(res.response)
```

Note that:
1. `vectara-agentic` also supports `achat()` and two streaming variants `stream_chat()` and `astream_chat()`.
2. The response types from `chat()` and `achat()` are of type `AgentResponse`. If you just need the actual string
   response it's available as the `response` variable, or just use `str()`. For advanced use-cases you can look 
   at other `AgentResponse` variables [such as `sources`](https://github.com/run-llama/llama_index/blob/659f9faaafbecebb6e6c65f42143c0bf19274a37/llama-index-core/llama_index/core/chat_engine/types.py#L53).

## Advanced Usage: Workflows

In addition to standard chat interactions, `vectara-agentic` supports custom workflows via the `run()` method. 
Workflows allow you to structure multi-step interactions where inputs and outputs are validated using Pydantic models.
To learn more about workflows read [the documentation](https://docs.llamaindex.ai/en/stable/understanding/workflows/basic_flow/)

### Defining a Custom Workflow

Create a workflow by subclassing `llama_index.core.workflow.Workflow` and defining the input/output models:

```python
from pydantic import BaseModel
from llama_index.core.workflow import (
    StartEvent,StopEvent, Workflow, step,
)

class MyWorkflow(Workflow):
    class InputsModel(BaseModel):
        query: str

    class OutputsModel(BaseModel):
        answer: str

    @step
    async def my_step(self, ev: StartEvent) -> StopEvent:
        # do something here
        return StopEvent(result="Hello, world!")
```

When the `run()` method in vectara-agentic is invoked, it calls the workflow with the following variables in the StartEvent:
* `agent`: the agent object used to call `run()` (self)
* `tools`: the tools provided to the agent. Those can be used as needed in the flow.
* `llm`: a pointer to a LlamaIndex llm, so it can be used in the workflow. For example, one of the steps may call `llm.acomplete(prompt)`
* `verbose`: controls whether extra debug information is displayed
* `inputs`: this is the actual inputs to the workflow provided by the call to `run()` and must be of type `InputsModel`

### Using the Workflow with Your Agent

When initializing your agent, pass the workflow class using the `workflow_cls` parameter:

```python
agent = Agent(
    tools=[query_financial_reports_tool],
    topic="10-K financial reports",
    custom_instructions="You are a helpful financial assistant.",
    workflow_cls=MyWorkflow,       # Provide your custom workflow here
    workflow_timeout=120           # Optional: Set a timeout (default is 120 seconds)
)
```

### Running the Workflow

Prepare the inputs using your workflow’s `InputsModel` and execute the workflow using `run()`:

```python
# Create an instance of the workflow's input model
inputs = MyWorkflow.InputsModel(query="What is Vectara?", extra_param=42)

# Run the workflow (ensure you're in an async context or use asyncio.run)
workflow_result = asyncio.run(agent.run(inputs))

# Access the output from the workflow's OutputsModel
print(workflow_result.answer)
```

### Using SubQuestionQueryWorkflow

vectara-agentic already includes one useful workflow you can use right away (it is also useful as an advanced example)
This workflow is called `SubQuestionQueryWorkflow` and it works by breaking a complex query into sub-queries and then
executing each sub-query with the agent until it reaches a good response.

## 🧰 Vectara tools

`vectara-agentic` provides two helper functions to connect with Vectara RAG
* `create_rag_tool()` to create an agent tool that connects with a Vectara corpus for querying. 
* `create_search_tool()` to create a tool to search a Vectara corpus and return a list of matching documents.

See the documentation for the full list of arguments for `create_rag_tool()` and `create_search_tool()`, 
to understand how to configure Vectara query performed by those tools.

### Creating a Vectara RAG tool

A Vectara RAG tool is often the main workhorse for any Agentic RAG application, and enables the agent to query 
one or more Vectara RAG corpora. 

The tool generated always includes the `query` argument, followed by 1 or more optional arguments used for 
metadata filtering, defined by `tool_args_schema`.

For example, in the quickstart example the schema is:

```
class QueryFinancialReportsArgs(BaseModel):
    query: str = Field(..., description="The user query.")
    year: int | str = Field(..., description=f"The year this query relates to. An integer between {min(years)} and {max(years)} or a string specifying a condition on the year (example: '>2020').")
    ticker: str = Field(..., description=f"The company ticker. Must be a valid ticket symbol from the list {tickers.keys()}.")
```

The `query` is required and is always the query string.
The other arguments are optional and will be interpreted as Vectara metadata filters.

For example, in the example above, the agent may call the `query_financial_reports_tool` tool with 
query='what is the revenue?', year=2022 and ticker='AAPL'. Subsequently the RAG tool will issue
a Vectara RAG query with the same query, but with metadata filtering (doc.year=2022 and doc.ticker='AAPL').

There are also additional cool features supported here:
* An argument can be a condition, for example year='>2022' translates to the correct metadata 
  filtering condition doc.year>2022
* if `fixed_filter` is defined in the RAG tool, it provides a constant metadata filtering that is always applied.
  For example, if fixed_filter=`doc.filing_type='10K'` then a query with query='what is the reveue', year=2022
  and ticker='AAPL' would translate into query='what is the revenue' with metadata filtering condition of
  "doc.year=2022 AND doc.ticker='AAPL' and doc.filing_type='10K'"

Note that `tool_args_type` is an optional dictionary that indicates the level at which metadata filtering
is applied for each argument (`doc` or `part`)

### Creating a Vectara search tool

The Vectara search tool allows the agent to list documents that match a query.
This can be helpful to the agent to answer queries like "how many documents discuss the iPhone?" or other
similar queries that require a response in terms of a list of matching documents.

## 🛠️ Agent Tools at a Glance

`vectara-agentic` provides a few tools out of the box (see ToolsCatalog for details):

1. **Standard tools**: 
- `summarize_text`: a tool to summarize a long text into a shorter summary (uses LLM)
- `rephrase_text`: a tool to rephrase a given text, given a set of rephrase instructions (uses LLM)
These tools use an LLM and so would use the `Tools` LLM specified in your `AgentConfig`.
To instantiate them:

```python
from vectara_agentic.tools_catalog import ToolsCatalog
summarize_text = ToolsCatalog(agent_config).summarize_text
```

This ensures the summarize_text tool is configured with the proper LLM provider and model as 
specified in the Agent configuration.

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
* Tavily search and EXA.AI
* arxiv
* neo4j & Kuzu for Graph DB integration
* Google tools (including gmail, calendar, and search)
* Slack

Note that some of these tools may require API keys as environment variables

You can create your own tool directly from a Python function using the `create_tool()` method of the `ToolsFactory` class:

```python
def mult_func(x, y):
    return x * y

mult_tool = ToolsFactory().create_tool(mult_func)
```

Note: When you define your own Python functions as tools, implement them at the top module level,
and not as nested functions. Nested functions are not supported if you use serialization 
(dumps/loads or from_dict/to_dict).

## 🛠️ Configuration

## Configuring Vectara-agentic

The main way to control the behavior of `vectara-agentic` is by passing an `AgentConfig` object to your `Agent` when creating it.
For example:

```python
agent_config = AgentConfig(
    agent_type = AgentType.REACT,
    main_llm_provider = ModelProvider.ANTHROPIC,
    main_llm_model_name = 'claude-3-5-sonnet-20241022',
    tool_llm_provider = ModelProvider.TOGETHER,
    tool_llm_model_name = 'meta-llama/Llama-3.3-70B-Instruct-Turbo'
)

agent = Agent(
    tools=[query_financial_reports_tool],
    topic="10-K financial reports",
    custom_instructions="You are a helpful financial assistant in conversation with a user.",
    agent_config=agent_config
)
```

The `AgentConfig` object may include the following items:
- `agent_type`: the agent type. Valid values are `REACT`, `LLMCOMPILER`, `LATS` or `OPENAI` (default: `OPENAI`).
- `main_llm_provider` and `tool_llm_provider`: the LLM provider for main agent and for the tools. Valid values are `OPENAI`, `ANTHROPIC`, `TOGETHER`, `GROQ`, `COHERE`, `BEDROCK`, `GEMINI` or `FIREWORKS` (default: `OPENAI`).
- `main_llm_model_name` and `tool_llm_model_name`: agent model name for agent and tools (default depends on provider).
- `observer`: the observer type; should be `ARIZE_PHOENIX` or if undefined no observation framework will be used.
- `endpoint_api_key`: a secret key if using the API endpoint option (defaults to `dev-api-key`)
- `max_reasoning_steps`: the maximum number of reasoning steps (iterations for React and function calls for OpenAI agent, respectively). Defaults to 50.

If any of these are not provided, `AgentConfig` first tries to read the values from the OS environment.

## Configuring Vectara tools: rag_tool, or search_tool

When creating a `VectaraToolFactory`, you can pass in a `vectara_api_key`, and `vectara_corpus_key` to the factory. 

If not passed in, it will be taken from the environment variables (`VECTARA_API_KEY` and `VECTARA_CORPUS_KEY`). Note that `VECTARA_CORPUS_KEY` can be a single KEY or a comma-separated list of KEYs (if you want to query multiple corpora).

These values will be used as credentials when creating Vectara tools - in `create_rag_tool()` and `create_search_tool()`.

## Setting up a privately hosted LLM

If you want to setup vectara-agentic to use your own self-hosted LLM endpoint, follow the example below

```python
        config = AgentConfig(
            agent_type=AgentType.REACT,
            main_llm_provider=ModelProvider.PRIVATE,
            main_llm_model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            private_llm_api_base="http://vllm-server.company.com/v1",
            private_llm_api_key="TEST_API_KEY",
        )
        agent = Agent(agent_config=config, tools=tools, topic=topic,
                      custom_instructions=custom_instructions)
```

In this case we specify the Main LLM provider to be privately hosted with Llama-3.1-8B as the model.
- The `ModelProvider.PRIVATE` specifies a privately hosted LLM.
- The `private_llm_api_base` specifies the api endpoint to use, and the `private_llm_api_key`
  specifies the private API key requires to use this service.

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

Note: due to cloudpickle limitations, if a tool contains Python `weakref` objects, serialization won't work and an exception will be raised.

###  Observability

vectara-agentic supports observability via the existing integration of LlamaIndex and Arize Phoenix.
First, set `VECTARA_AGENTIC_OBSERVER_TYPE` to `ARIZE_PHOENIX` in `AgentConfig` (or env variable).

Then you can use Arize Phoenix in three ways: 
1. **Locally**. 
   1. If you have a local phoenix server that you've run using e.g. `python -m phoenix.server.main serve`, vectara-agentic will send all traces to it.
   2. If not, vectara-agentic will run a local instance during the agent's lifecycle, and will close it when finished.
   3. In both cases, traces will be sent to the local instance, and you can see the dashboard at `http://localhost:6006`
2. **Hosted Instance**. In this case the traces are sent to the Phoenix instances hosted on Arize.
   1. Go to `https://app.phoenix.arize.com`, setup an account if you don't have one.
   2. create an API key and put it in the `PHOENIX_API_KEY` environment variable - this indicates you want to use the hosted version.
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
