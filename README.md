# vectara-agentic

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Maintained](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/vectara/py-vectara-agentic/graphs/commit-activity)

[![Twitter](https://img.shields.io/twitter/follow/vectara.svg?style=social&label=Follow%20%40Vectara)](https://twitter.com/vectara)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-blue?style=social&logo=discord)](https://discord.com/invite/GFb8gMz6UH)


The idea of LLM-based agents it to use the LLM for building sophisticated AI assistants:
- The LLM is used for reasoning and coming up with a game-plan for how to respond to the user query.
- There are 1 or more "tools" provided to the agent. These tools can be used by the LLM to execute its plan.

`vectara-agentic` is a Python library that let's you develop powerful AI assistants with Vectara, using Agentic-RAG:
* Based on LlamaIndex Agent framework, customized for use with Vectara.
* Supports the `ReAct` or `OpenAIAgent` agent types.
* Includes many tools out of the box (e.g. for finance, legal and other verticals).

## Getting Started

### Prerequisites
* A [Vectara account](https://console.vectara.com/signup)
* A Vectara corpus with an [API key](https://docs.vectara.com/docs/api-keys)
* [Python 3.10 (or higher)](https://www.python.org/downloads/)
* An OpenAI API key specified in your environment as `OPENAI_API_KEY`

### Install vectara-agentic

- `python -m pip install vectara-agentic`

### Create your AI assistant

Creating an AI assistant with `vectara-agentic` involves the following:

#### Step 1: Create Vectara RAG tool

First, create an instance of the `VectaraToolFactory` class as follows:

```python
vec_factory = VectaraToolFactory(vectara_api_key=os.environ['VECTARA_API_KEY'],
                                 vectara_customer_id=os.environ['VECTARA_CUSTOMER_ID'], 
                                 vectara_corpus_id=os.environ['VECTARA_CORPUS_ID'])
```
The tools factory has a useful helper function called `create_rag_tool` which automates the creation of a 
tool to query Vectara RAG. 

For example if my Vectara corpus includes financial information from company 
10K annual reports for multiple companies and years, I can use the following:

```python

class QueryFinancialReportsArgs(BaseModel):
    query: str = Field(..., description="The user query. Must be a question about the company's financials, and should not include the company name, ticker or year.")
    year: int = Field(..., description=f"The year. an integer.")
    ticker: str = Field(..., description=f"The company ticker. Must be a valid ticket symbol.")
query_financial_reports = vec_factory.create_rag_tool(
    tool_name = "query_financial_reports",
    tool_description = """
    Given a company name and year, 
    returns a response (str) to a user query about the company's financials for that year.
    When using this tool, make sure to provide a valid company ticker and year. 
    Use this tool to get financial information one metric at a time.
    """,
    tool_args_schema = QueryFinancialReportsArgs,
    tool_filter_template = "doc.year = {year} and doc.ticker = '{ticker}'"
)
```
Note how `QueryFinancialReportsArgs` defines the arguments for my tool using pydantic's `Field` class. The `tool_description` 
as well as the description of each argument are important as they provide the LLM with the ability to understand how to use 
this tool in the most effective way.
The `tool_filter_template` provides the template filtering expression the tool should use when calling Vectara.

You can of course create more than one Vectara tool; tools may point at different corpora or may have different parameters for search
or generation. Remember though to think about your tools wisely and from the agent point of view - at the end of the day they are just tools
in the service of the agent, so should be differentiated.

#### Step 2: Create Other Tools, as needed

In addition to RAG tools, you can generate a lot of other types of tools the agent can use. These could be mathematical tools, tools 
that call other APIs to get more information, and much more.

`vectara-agentic` provides a few tools out of the box:
1. Standard tools: 
- `get_current_date`: allows the agent to figure out which date it is.
- `summarize_text`: a tool to summarize a long text into a shorter summary (uses LLM)
- `rephrase_text`: a tool to rephrase a given text, given a set of rephrase instructions (uses LLM)
  
2. Financial tools: a set of tools for financial analysis of public company data:
- `get_company_name`: get company name given its ticker (uses Yahoo Finance)
- `calculate_return_on_equity`, `calculate_return_on_assets`, `calculate_debt_to_equity_ratio` and `calculate_ebitda`

You can create your own tool directly from a Python function using the `create_tool()` method:

```Python
def mult_func(x, y):
    return x*y

mult_tool = ToolsFactory().create_tool(mult_func)
```

3. More tools to be coming soon
 
#### Step 3: Create your agent

```python
agent = Agent(
    tools = tools,
    topic = topic_of_expertise
    custom_instructions = financial_bot_instructions,
    update_func = update_func
)
```
- `tools` is the list of tools you want to provide to the agent
- `topic` is a string that defines the expertise you want the agent to specialize in.
- `custom_instructions` is an optional string that defines special instructions to the agent
- `update_func` is a callback function that will be called by the agent as it performs its task
  The inputs to this function you provide are `status_type` of type AgentStatusType and 
  `msg` which is a string.

Note that the Agent type (`OPENAI` or `REACT`) is defined as an environment variables `VECTARA_AGENTIC_AGENT_TYPE`.

For example, for a financial agent we can use:

```python
topic = "10-K financial reports",

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

`vectara-agentic` is using environment variables for a few global configuration 
- `VECTARA_AGENTIC_AGENT_TYPE`: type of agent - `REACT` or `OPENAI` (default `OPENAI`)
- `VECTARA_AGENTIC_MAIN_LLM_PROVIDER`: agent LLM provider `OPENAI`, `ANTHROPIC`, `TOGETHER`, `GROQ`, or `FIREWORKS` (default `OPENAI`)
- `VECTARA_AGENTIC_MAIN_MODEL_NAME`: agent model name (default depends on provider)
- `VECTARA_AGENTIC_TOOL_LLM_PROVIDER`: tool LLM provider `OPENAI`, `ANTHROPIC`, `TOGETHER`, `GROQ`, or `FIREWORKS` (default `OPENAI`)
- `VECTARA_AGENTIC_TOOL_MODEL_NAME`: tool model name (default depends on provider)

## Examples

We have created a few example AI assistants that you can look at for inspiration and code examples:
- [Financial Assistant](https://huggingface.co/spaces/vectara/finance-chat).
- [Justice Harvard Teaching Assistant](https://huggingface.co/spaces/vectara/Justice-Harvard).
- [Legal Assistant](https://huggingface.co/spaces/vectara/legal-agent).

## Author

👤 **Vectara**

- Website: [vectara.com](https://vectara.com)
- Twitter: [@vectara](https://twitter.com/vectara)
- GitHub: [@vectara](https://github.com/vectara)
- LinkedIn: [@vectara](https://www.linkedin.com/company/vectara/)
- Discord: [@vectara](https://discord.gg/GFb8gMz6UH)

## 🤝 Contributing

Contributions, issues and feature requests are welcome and appreciated!<br />
Feel free to check [issues page](https://github.com/vectara/py-vectara-agentic/issues). You can also take a look at the [contributing guide](https://github.com/vectara/py-vectara-agentic/blob/master/CONTRIBUTING.md).

## Show your support

Give a ⭐️ if this project helped you!

## 📝 License

Copyright © 2024 [Vectara](https://github.com/vectara).<br />
This project is [Apache 2.0](https://github.com/vectara/py-vectara-agentic/blob/master/LICENSE) licensed.