# Usage
Let's walk through a complete example of creating an AI assistant using
vectara-agentic. We will build a finance assistant that can answer
questions about the annual financial reports for Apple Computer, Google,
Amazon, Snowflake, Atlassian, Tesla, Nvidia, Microsoft, Advanced Micro
Devices, Intel, and Netflix between the years 2020 and 2024.

## Import Dependencies
First, we must import some libraries and define some constants for our
demo.

```python
import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import requests
from pydantic import Field

load_dotenv(override=True)
```

We then use the `load_dotenv` function to load our environment variables
from a `.env` file.

## Create Tools
Next, we will create the tools for our agent.

There are three categories of tools you can use with vectara-agentic:

1.  A query tool that connects to Vectara to ask a question about data
    in a Vectara corpus.
2.  Pre-built tools that are available out of the box, or ready to use
    tools from the LlamaIndex [Tools
    Hub](https://llamahub.ai/?tab=tools).
3.  Any other tool that you want to make for your agent, based on custom
    code in Python.

**Vectara RAG Query Tool**
Let's see how to create a Vectara query tool. In order to use this
tool, you need to create a corpus and API key with a [Vectara
account](https://console.vectara.com/signup/?utm_source=github&utm_medium=code&utm_term=DevRel&utm_content=vectara-agentic&utm_campaign=github-code-DevRel-vectara-agentic).
In this example, we will create the `ask_transcripts` tool, which can be
used to perform RAG queries on analyst call transcripts. You can see
this tool in use with our [Finance Assistant
demo](https://huggingface.co/spaces/vectara/finance-assistant).

```python
from pydantic import BaseModel

# define the arguments schema for the tool
class QueryTranscriptsArgs(BaseModel):
    year: int = Field(..., description=f"The year. An integer between {min(years)} and {max(years)}.")
    ticker: str = Field(..., description=f"The company ticker. Must be a valid ticket symbol from the list {tickers.keys()}.")
```

Note that:

- The arguments for this tool are defined using Python's `pydantic` package with the `Field` class. By defining the tool in this
  way, we provide a good description for each argument so that the agent LLM can easily understand the tool's functionality 
  and how to use it properly.
- The `query` argument is added automatically to the RAG tool, and you don't need to specify it here

You can also define an argument to support optional conditional
arguments, for example:

```python
from pydantic import BaseModel

# define the arguments schema for the tool
class QueryTranscriptsArgs(BaseModel):
    year: int | str = Field(
        default=None,
        description=f"The year this query relates to. An integer between {min(years)} and {max(years)} or a string specifying a condition on the year",
        examples=[2020, '>2021', '<2023', '>=2021', '<=2023', '[2021, 2023]', '[2021, 2023)']
    )
    ticker: str = Field(..., description=f"The company ticker. Must be a valid ticket symbol from the list {tickers.keys()}.")
```

With this change for the `year` argument, we are telling the agent that
both an int value (e.g. 2022) or a string value (e.g. '>2022' or
'<2022') are valid inputs for this argument. You can also use range
filters (e.g. '[2021, 2023]') to specify a range of years. If a
string value is provided, `vectara-agentic` knows how to parse it
properly in the backend and set a metadata filter with the right
condition for Vectara.

Now to create the actual tool, we use the `create_rag_tool()` method
from the `VectaraToolFactory` class as follows:

```python
from vectara_agentic.tools import VectaraToolFactory

vec_factory = VectaraToolFactory(vectara_api_key=vectara_api_key,
                                 vectara_corpus_key=vectara_corpus_key)

ask_transcripts = vec_factory.create_rag_tool(
    tool_name = "ask_transcripts",
    tool_description = """
    Given a company name and year,
    returns a response (str) to a user question about a company, based on analyst call transcripts about the company's financial reports for that year.
    You can ask this tool any question about the company including risks, opportunities, financial performance, competitors and more.
    Make sure to provide the a valid company ticker and year.
    """,
    tool_args_schema = QueryTranscriptsArgs,
    tool_args_type = {
      "year": "doc",
      "ticker": "doc"
    },
    reranker = "chain", rerank_k = 100,
    rerank_chain = [
      {
        "type": "slingshot"
      },
      {
        "type": "userfn",
        "user_function": "knee()"
      }
      {
        "type": "mmr",
        "diversity_bias": 0.1
      }
    ],
    n_sentences_before = 2, n_sentences_after = 2, lambda_val = 0.005,
    summary_num_results = 10,
    vectara_summarizer = 'vectara-summary-ext-24-05-med-omni',
    include_citations = False,
    fcs_threshold = 0.2
)
```

In the code above, we did the following:

-   First, we initialized the `VectaraToolFactory` with the Vectara
    corpus key and API key. If you don't want to explicitly pass in
    these arguments, you can specify them in your environment as
    `VECTARA_CORPUS_KEY` and `VECTARA_API_KEY`. Additionally, you can
    also create a single `VectaraToolFactory` that queries multiple
    corpora. This may be helpful if you have related information across
    multiple corpora in Vectara. To do this, create a query API key on
    the
    [Authorization](https://console.vectara.com/console/apiAccess/apiKeys)
    page and give it to access to all the corpora you want for this
    query tool. When specifying your environment variables, set
    `VECTARA_CORPUS_KEY` to a list of corpus IDs separated by commas
    (e.g. `5,6,19`).
-   Then we called `create_rag_tool()`, specifying the tool name,
    description and schema for the tool, followed by various optional
    parameters to control the Vectara RAG query tool. Notice that we
    also specified the type of each additional argument in the schema.
    The type of each argument can be `"doc"` or `"part"`, corresponding
    to whether the metadata argument is document metadata or part
    metadata in the Vectara corpus. See this
    [page](https://docs.vectara.com/docs/learn/metadata-search-filtering/metadata-examples-and-use-cases)
    on metadata for more information.

One important parameter to point out is `fcs_threshold`. This allows you
to specify a minimum factual consistency score (between 0 and 1) for the
response to be considered a "good" response. If the generated response
has an `FCS` below this threshold, the agent will not use the generated
summary (considering it a hallucination). You can think of this as a
hallucination guardrail. The higher you set `fcs_threshold`, the
stricter your guardrail will be.

If your agent continuously rejects all of the generated responses,
consider lowering the threshold.

Another important parameter is `reranker`. In this example, we are using
a chain reranker, which chains together multiple reranking methods to
achieve better control over the reranking and combines the strengths of
various reranking methods. In the example above, we use the
[multilingual](https://docs.vectara.com/docs/learn/vectara-multi-lingual-reranker)
(or slingshot) reranker followed by a user-defined function (the
[knee](https://docs.vectara.com/docs/learn/knee-reranking) reranker),
and finally the [MMR](https://docs.vectara.com/docs/learn/mmr-reranker)
reranker with a diversity bias of 0.1. You can also supply other
parameters to each reranker, such as a `cutoff` parameter, which removes
documents that have scores below this threshold value after applying the
given reranker. Lastly, you can add another [user defined
function](https://docs.vectara.com/docs/learn/user-defined-function-reranker)
reranker as the last reranker in the chain to specify a customized
expression to rerank results in a way that is relevant to your specific
application. If you want to learn more about reranking tips and best
practices, check out our blog posts on [user defined
functions](https://www.vectara.com/blog/rag-with-user-defined-functions-based-reranking)
and [knee
reranking](https://www.vectara.com/blog/introducing-the-knee-reranking-smart-result-filtering-for-better-results)
as well as this [example
notebook](https://github.com/vectara/example-notebooks/blob/main/notebooks/udf-reranking-demo.ipynb)
on user defined functions for some guidance and inspiration.

That's it: now the `ask_transcripts` tool is ready to be added to the
agent.

Notes:

- You can use the `VectaraToolFactory` to generate more than one RAG tool
with different parameters, depending on your needs.
- `create_rag_tool` and `create_search_tool` both support the `vectara_base_url` 
  argument. If specified, it allows you to specify a different base URL for Vectara,
  for example when you have an on-premise installation.
- If you want to specify a Certificate Authority for a local installation,
  you can set "export REQUESTS_CA_BUNDLE=/path/to/custom_ca_bundle.pem".

**Vectara Search Tool**
In most cases, you will likely want to use the Vectara RAG query tool,
which generates a summary to return to the agent along with the source
text and documents used to generate that summary.

In some applications, you may want the tool to only retrieve the actual
text/documents that best match the query rather than summarizing all of
the results. For example, you may ask your agent "How many documents
mention information about tax laws and regulations?". The agent will be
able to get a list of documents from your Vectara corpus and analyze the
results to answer your question.

**Metadata Filtering**
In most cases, you will want to use the `tool_args_schema` to define the
metadata fields used in your Vectara RAG or Search tool. Defining your
parameters in this way allows the agent to interpret the user query and
determine if any of these filters should be applied on that particular
query.

In some instances you may want to have a metadata filter that applies in
every call to a Vectara RAG or search tool. For example, you may want to
enforce that the oldest possible search results are from 2022. In this
case, you can use the `fixed_filter` parameter to the
`create_rag_tool()` or `create_search_tool()` functions.

In our example where we want all results to be from 2022 and later, we
would specify `fixed_filter = "doc.year >= 2022"`.

**Additional Tools**
To generate non-RAG tools, you can use the `ToolsFactory` class, which
provides some out-of-the-box tools that you might find helpful when
building your agents, as well as an easy way to create custom tools.

Currently, we have a few tool groups you may want to consider using:

-   `standard_tools()`: These are basic tools that can be helpful, and
    include the `summarize_text` tool and `rephrase_text` tool.
-   `finance_tools()`: includes a set of financial query tools based on
    Yahoo! finance.
-   `legal_tools()`: These tools are designed to help with legal
    queries, and include `critique_as_judge` and `summarize_legal_text`.
-   `database_tools()`: tools to explore SQL databases and make queries
    based on user prompts.
-   `guardrail_tools()`: These tools are designed to help the agent
    avoid certain topics from its response.

For example, to get access to all the legal tools, you can use the
following:

```python
from vectara_agentic.tools import ToolsFactory

legal_tools = ToolsFactory().legal_tools()
```

For more details about the tools see `Tools <tools>`{.interpreted-text
role="doc"}.

**Create your own tool**
You can also create your own tool directly by defining a Python
function:

```python
import numpy as np

def earnings_per_share(
  net_income: float = Field(description="the net income for the company"),
  number_of_shares: float = Field(description="the number of oustanding shares"),
) -> float:
    """
    This tool returns the EPS (earnings per share).
    """
    return np.round(net_income / number_of_shares,4)

my_tool = tools_factory.create_tool(earnings_per_share)
```

A few important things to note:

1.  A tool may accept any type of argument (e.g. float, int) and return
    any type of value (e.g. float). The `create_tool()` method will
    handle the conversion of the arguments and response into strings
    (which is type the agent expects).
2.  It is important to define a clear and concise docstring for your
    tool. This will help the agent understand what the tool does and how
    to use it.

Here are some functions we will define for our finance assistant
example:

```python
tickers = {
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
}
years = [2020, 2021, 2022, 2023, 2024]

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

# Tool to get the income statement for a given company and year using the FMP API
def get_income_statement(
ticker=Field(description="the ticker symbol of the company."),
year=Field(description="the year for which to get the income statement."),
) -> str:
"""
Get the income statement for a given company and year using the FMP (https://financialmodelingprep.com) API.
Returns a dictionary with the income statement data. All data is in USD, but you can convert it to more compact form like K, M, B.
"""
fmp_api_key = os.environ.get("FMP_API_KEY", None)
if fmp_api_key is None:
   return "FMP_API_KEY environment variable not set. This tool does not work."
url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?apikey={fmp_api_key}"
response = requests.get(url)
if response.status_code == 200:
   data = response.json()
   income_statement = pd.DataFrame(data)
   income_statement["date"] = pd.to_datetime(income_statement["date"])
   income_statement_specific_year = income_statement[
     income_statement["date"].dt.year == int(year)
   ]
   values_dict = income_statement_specific_year.to_dict(orient="records")[0]
   return f"Financial results: {', '.join([f'{key}: {value}' for key, value in values_dict.items() if key not in ['date', 'cik', 'link', 'finalLink']])}"
else:
   return "FMP API returned error. This tool does not work."
```

The `get_income_statement()` tool utilizes the FMP API to get the income
statement for a given company and year. Notice how the tool description
is structured. We describe each of the expected arguments to the
function using pydantic's `Field` class. The function description only
describes to the agent what the function does and how the agent should
use the tool. This function definition follows best practices for
defining tools. You should make this description detailed enough so that
your agent knows when to use each of your tools.

You can define your tool as an individual python function (as shown
above) or as a method in a Python class. It may be helpful to define all
of your tools (Vectara tools, other pre-built tools, and your custom
tools) in a single AgentTools class. Please note that you **cannot**
define a tool as a function within another tool. Each tool must be a
separate Python function.

Your tools should also handle any exceptions gracefully by returning an
`Exception` or a string describing the failure. The agent can interpret
that string and then decide how to deal with the failure (either calling
another tool to accomplish the task or telling the user that their
request was unable to be processed).

Finally, notice that we have used snake_case for all of our function
names. While this is not required, it's a best practice that we
recommend for you to follow.

## Initialize The Agent
Now that we have our tools, let's create the agent, using the following
arguments:

1.  `tools: list[FunctionTool]`: A list of tools that the agent will use
    to interact with information and apply actions. For any tools you
    create yourself, make sure to pass them to the `create_tool()`
    method of your `ToolsFactory` object.
2.  `topic: str = "general"`: This is simply a string (should be a noun)
    that is used to identify the agent's area of expertise. For our
    example we set this to `financial analyst`.
3.  `custom_instructions: str = ""`: This is a set of instructions that
    the agent will follow. These instructions should not tell the agent
    what your tools do (that's what the tool descriptions are for) but
    rather any particular behavior you want your LLM to have, such as
    how to present the information it receives from the tools to the
    user.
4.  `agent_config: Optional[AgentConfig] = None`: the agent configuration
    See below for more details. If unspecified, defaults are used.
5.  `fallback_agent_config: Optional[AgentConfig] = None`: configuration
    for a fallback_agent. If specified, this will get activated if the
    main agent API is not responding (e.g. when inference enpoint is down).
    If unspecified, no fallback agent is assumed.
6.  `agent_progress_callback: Optional[Callable[[AgentStatusType, dict, str], None]] = None`:
    This is an optional callback function that will be called on every
    agent step (see below)
7.  `query_logging_callback: Optional[Callable[[str, str], None]] = None`:
    This is an optional callback function that will be called at the end
    of response generation, with the query and response strings.
8.  `validate_tools: bool = False`: whether to validate tool inconsistency 
    with instructions.


Every agent has its own default set of instructions that it follows to
interpret users' messages and use the necessary tools to complete its
task. However, we can (and often should) define custom instructions (via
the `custom_instructions` argument) for our AI assistant. Here are some
guidelines to follow when creating your instructions:

-   Write precise and clear instructions without overcomplicating the
    agent.
-   Consider edge cases and unusual or atypical scenarios.
-   Be cautious to not over-specify behavior based on your primary use
    case as this may limit the agent's ability to behave properly in
    other situations.

Here are the instructions we are using for our financial AI assistant:

```python
financial_assistant_instructions = """
  - You are a helpful financial assistant, with expertise in financial reporting, in conversation with a user.
  - Never discuss politics, and always respond politely.
  - Respond in a compact format by using appropriate units of measure (e.g., K for thousands, M for millions, B for billions).
  - Do not report the same number twice (e.g. $100K and 100,000 USD).
  - Always check the get_company_info and get_valid_years tools to validate company and year are valid.
  - When querying a tool for a numeric value or KPI, use a concise and non-ambiguous description of what you are looking for.
  - If you calculate a metric, make sure you have all the necessary information to complete the calculation. Don't guess.
"""
```

Notice how these instructions are different from the tool function
descriptions. These instructions are general rules that the agent should
follow. At times, these instructions may refer to specific tools, but in
general, the agent should be able to decide for itself what tools it
should call. This is what makes agents very powerful and makes our job
as coders much simpler.

**agent_progress_callback callback**
The `agent_progress_callback` is an optional `Callable` function that can serve a
variety of purposes for your assistant. It is a callback function that
is managed by the agent, and it will be called anytime the agent is
updated, such as when calling a tool, or when receiving a response from
a tool.

In our example, we will use it to log the actions of our agent so users
can see the steps the agent is taking as it answers their questions.
Since our assistant is using streamlit to display the results, we will
append the log messages to the session state.

```python
from vectara_agentic.agent import AgentStatusType

def agent_progress_callback(status_type: AgentStatusType, payload: dict, event_id: str):
  output = f"{status_type.value} - {msg}"
  st.session_state.log_messages.append(output)
```

**agent_config**
The `agent_config` argument is an optional object that you can use to
explicitly specify the configuration of your agent, including the following:

- `agent_type`: the agent type. Valid values are `REACT`, `LLMCOMPILER`, `LATS`, `FUNCTION_CALLING` or `OPENAI` (default: `OPENAI`).
- `main_llm_provider` and `tool_llm_provider`: the LLM provider for main agent and for the tools. Valid values are `OPENAI`, `ANTHROPIC`, `TOGETHER`, `GROQ`, `COHERE`, `BEDROCK`, `GEMINI` or `FIREWORKS` (default: `OPENAI`).
- `main_llm_model_name` and `tool_llm_model_name`: agent model name for agent and tools (default depends on provider).
- `observer`: the observer type; should be `ARIZE_PHOENIX` or if undefined no observation framework will be used.
- `endpoint_api_key`: a secret key if using the API endpoint option (defaults to `dev-api-key`)
- `max_reasoning_steps`: the maximum number of reasoning steps (iterations for React and function calls for OpenAI agent, respectively). defaults to 50.

By default, each of these parameters will be read from your environment, but you can also
explicitly define them with the `AgentConfig` class.

For example, here is how we can define an `AgentConfig` object to create
a `ReAct` agent using `OPENAI` as the LLM for the agent and `Cohere` as the
LLM for the agent's tools:

```python
from vectara_agentic.agent_config import AgentConfig

config = AgentConfig(
  agent_type="REACT",
  main_llm_provider="OPENAI",
  tool_llm_provider="COHERE"
)
```

**Creating the agent**
Here is how we will instantiate our finance assistant:

```python
from vectara_agentic import Agent

agent = Agent(
     tools=[tools_factory.create_tool(tool, tool_type="query") for tool in
               [
                   get_company_info,
                   get_valid_years,
                   get_income_statement
               ]
           ] +
           tools_factory.standard_tools() +
           tools_factory.financial_tools() +
           tools_factory.guardrail_tools() +
           [ask_transcripts],
     topic="10-K annual financial reports",
     custom_instructions=financial_assistant_instructions,
     agent_progress_callback=agent_progress_callback
)
```

Notice that when we call the `create_tool()` method, we specified a
`tool_type`. This can either be `"query"` (default) or `"action"`. For
our example, all of the tools are query tools, so we can easily add all
of them to our agent with a list comprehension, as shown above.

## Chat with your Assistant
Once you have created your agent, using it is quite simple. All you have
to do is call its `chat()` method, which prompts your agent to answer
the user's query using its available set of tools. It's that easy.

```python
query = "Which 3 companies had the highest revenue in 2022, and how did they do in 2021?"
print(str(agent.chat(query)))
```

The agent returns the response:

> The three companies with the highest revenue in 2022 were:
>
> 1.  **Amazon (AMZN)**: $513.98B
> 2.  **Apple (AAPL)**: $394.33B
> 3.  **Google (GOOG)**: $282.84B
>
> Their revenues in 2021 were:
>
> 1.  **Amazon (AMZN)**: $469.82B
> 2.  **Apple (AAPL)**: $365.82B
> 3.  **Google (GOOG)**: $257.64B

The `chat()` function returns an `AgentResponse` object, which includes
the agent's generated response text and a list of `ToolOutput` objects.
The agent's response text can easily be retrieved `response` member (or
simply by using `str()`). The tool information can be extracted with the
`sources` member of the `AgentResponse` class and will return a list of
tool outputs, including the name of each tool that was called and the
output from that tool that was given to the agent.

To make a full Streamlit app, there is some extra code that is necessary
to configure the demo layout. You can check out the [full
code](https://huggingface.co/spaces/vectara/finance-assistant/tree/main)
and [demo](https://huggingface.co/spaces/vectara/finance-assistant) for
this app on Hugging Face.

## Other Chat Options
The standard `chat()` method will run synchronously, so your application
will wait until the agent finishes generating its response before making
any other function calls. If you would prefer to run your queries
asynchronously with your application, you can use the `achat()` method.

The `chat()` function also returns the response as a single string,
which could be a lengthy text. If you would prefer to stream the
agent's response by chunks, you can use the `stream_chat()` method (or
`astream_chat()` method to run asynchronously). This will return an
`AgentStreamingResponse` object. If you want to directly print out the
response, you can use the `print_response_stream()` method. If you need
to yield the chunks in some other way for your application, you can
obtain the generator object by accessing the `chat_stream` member.

## Using Workflows

vectara-agentic now supports custom workflows via the `run()` method, enabling you to define multi-step interactions with validated inputs and outputs.
To learn more about workflows read [the documentation](https://docs.llamaindex.ai/en/stable/understanding/workflows/basic_flow/)

### Defining a Custom Workflow

To create a workflow, subclass the Workflow class from `llama_index.core.workflow` and define two Pydantic models: `InputsModel` and ``OutputsModel`. 
For example:

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

When the `run()` method in vectara-agentic is invoked, it calls the workflow with the following variables in the `StartEvent`:

- `agent`: the agent object used to call `run()` (self)
- `tools`: the tools provided to the agent. Those can be used as needed in the flow.
- `llm`: a pointer to a LlamaIndex llm, so it can be used in the workflow. For example, one of the steps may call `llm.acomplete(prompt)`
- `verbose`: controls whether extra debug information is displayed
- `inputs`: this is the actual inputs to the workflow provided by the call to `run()` and must be of type `InputsModel`

If you want to use `agent`, `tools`, `llm` or `verbose` in other events (that are not `StartEvent`), you can store them in
the `Context` of the Workflow as follows:

```python
await ctx.set("agent", ev.agent)
```

and then in any other event you can pull that agent object with

```python
agent = await ctx.get("agent")
```

Similarly you can reuse the `llm`, `tools` or `verbose` arguments within other nodes in the workflow.

### Integrating the Workflow with Your Agent

When instantiating your agent, pass your workflow class to the `workflow_cls` parameter (and optionally set a workflow timeout):

```python
agent = Agent(
    tools=[...],  # your list of tools
    topic="10-K annual financial reports",
    custom_instructions=financial_assistant_instructions,
    agent_progress_callback=agent_progress_callback,
    workflow_cls=FinanceWorkflow,   # Provide your workflow class here
    workflow_timeout=120            # Optional timeout in seconds
)
```

### Running the Workflow

To run the workflow, create an instance of your workflow's `InputsModel` with the required parameters and call the agent's `run()` method. For example:

```python
# Create input for the workflow
workflow_inputs = FinanceWorkflow.InputsModel(query="What were the revenue trends for Apple?", analysis_depth=3)

# Execute the workflow asynchronously (ensure you're in an async context or use asyncio.run)
workflow_output = asyncio.run(agent.run(workflow_inputs))

# Access the final answer from the output model
print(workflow_output.answer)
```

The `run()` method executes your workflow’s logic, validates the output against the `OutputsModel`, and returns a structured result.

### Using SubQuestionQueryWorkflow

vectara-agentic already includes one useful workflow you can use right away (it is also useful as an advanced example)
This workflow is called `SubQuestionQueryWorkflow` and it works by breaking a complex query into sub-queries and then
executing each sub-query with the agent until it reaches a good response.


## Additional Information

**Agent Information**
The `Agent` class defines a few helpful methods to help you understand
the internals of your application.

1.  The `report()` method prints out the agent object's type (REACT,
    OPENAI, or LLMCOMPILER), the tools, and the LLMs used for the main
    agent and tool calling.
2.  The `token_counts()` method tells you how many tokens you have used
    in the current session for both the main agent and tool calling
    LLMs. This can be helpful for users who want to track how many
    tokens have been used, which translates to how much money they are
    spending.

If you have any other information that you would like to be accessible
to users, feel free to make a suggestion on our community
[server](https://discord.com/channels/1022303169612611615/1100640116843761685).

**Observability**
You can also setup full observability for your vectara-agentic assistant
or agent using [Arize Phoenix](https://phoenix.arize.com/). This allows
you to view LLM prompt inputs and outputs, the latency of each task and
subtask, and many of the individual function calls performed by the LLM,
as well as FCS scores for each response.

To set up observability for your app, follow these steps:

1.  Set `os["VECTARA_AGENTIC_OBSERVER_TYPE"] = "ARIZE_PHOENIX"` or
    specify `observer = "ARIZE_PHOENIX"` in your `AgentConfig`.
2.  Connect to a local phoenix server:
    1.  If you have a local phoenix server that you've run using e.g.
        `python -m phoenix.server.main serve`, vectara-agentic will send
        all traces to it automatically.
    2.  If not, vectara-agentic will run a local instance during the
        agent's lifecycle, and will close it when finished.
    3.  In both cases, traces will be sent to the local instance, and
        you can see the dashboard at <http://localhost:6006>.
3.  Alternatively, you can connect to a Phoenix instance hosted on
    Arize.
    1.  Go to <https://app.phoenix.arize.com>, and set up an account if
        you don't have one.
    2.  Create an API key and put it in the `PHOENIX_API_KEY` variable.
        This variable indicates you want to use the hosted version.
    3.  To view the traces go to <https://app.phoenix.arize.com>.

In addition to the raw traces, vectara-agentic also records `FCS` values
into Arize for every Vectara RAG call. You can see those results in the
`Feedback` column of the arize UI.

**Query Callback**
You can define a callback function to log query/response pairs in your
agent. This function should be specified in the `query_logging_callback`
argument when you create your agent and should take in two string
arguments. The first argument passed to this function will be the user
query and the second will be the agent's response.

If defined, this function is called every time the agent receives a
query and generates a response.

## Using a Private LLM
vectara-agentic offers a wide variety of LLM options from several
providers to use for the main agent and for tool calling. However, in
some instances, you may be interested in using your own LLM hosted
locally at your company.

If you would like the main agent LLM to be a custom LLM, specify
`VECTARA_AGENTIC_MAIN_LLM_PROVIDER="PRIVATE"` in your environment or
`main_llm_provider="PRIVATE"` in your `AgentConfig` object and
`VECTARA_AGENTIC_MAIN_MODEL_NAME` (or `main_llm_model_name` in
`AgentConfig`) as the model name of your LLM.

If you would like the tool calling LLM to be a custom LLM, specify
`VECTARA_AGENTIC_TOOL_LLM_PROVIDER="PRIVATE"` in your environment or
`tool_llm_provider="PRIVATE"` in your `AgentConfig` object and
`VECTARA_AGENTIC_TOOL_MODEL_NAME` (or `tool_llm_model_name` in
`AgentConfig`) as the model name of your LLM.

Additionally, you should specify `VECTARA_AGENTIC_PRIVATE_LLM_API_BASE`
in your environment (or `private_llm_api_base` in the `AgentConfig`) as
the API endpoint url for your private LLM and
`VECTARA_AGENTIC_PRIVATE_API_KEY` (or `private_llm_api_key`) as the API
key to your LLM.
