# Tools

`vectara-agentic` provides a set of pre-built tools that you can use
out-of-the-box for various purposes.

## Standard Tools

Basic tools for general purposes:

- **summarize_text**: Summarizes text from a specific perspective or expertise level
- **rephrase_text**: Rephrases text according to specified instructions (e.g., for a 5-year-old or in formal tone)

## Finance Tools

`vectara-agentic` includes a few financial tools you can use right away in
your agent, based on the LlamaIndex
[YahooFinanceToolSpec](https://llamahub.ai/l/tools/llama-index-tools-yahoo-finance):

- **balance_sheet**: Returns a company's balance sheet
- **income_statement**: Returns a company's income statement
- **cash_flow**: Returns a company's cash flow statement
- **stock_news**: Returns latest news about a company
- **stock_basic_info**: Returns basic company information including price
- **stock_analyst_recommendations**: Returns analyst recommendations for a company

## Legal Tools

vectara-agentic includes a few tools for the legal space:

- **summarize_legal_text**: Summarizes legal documents
- **critique_as_judge**: Critiques legal text from an expert judge's perspective

## Guardrail Tools

The guardrail tools help you AI assistant or agent to avoid certain
topics or responses that are prohibited by your organization or by law.

The `get_bad_topics` tool returns a list of topics that are prohibited
(politics, religion, violence, hate speech, adult content, illegal
activities). The agent prompt has special instructions to call this tool
if it exists, and avoid these topics.

If you want to create your own set of topics, you can define a new tool
by the **same name** (`get_bad_topics`) that returns a list of different topics, and the
agent will use that list to avoid these topics.

## Database Tools

Database tools are quite useful if your agent requires access to a
combination of RAG tools along with analytics capabilities. For example,
consider the
[EV-assistant](https://huggingface.co/spaces/vectara/ev-assistant) demo,
providing answers about electric vehicles.

We have provided this assistant with the following tools:

1.  `ask_vehicles`: A Vectara RAG tool that answers general questions
    about electric vehicles.
2.  `ask_policies`: A Vectara RAG tool that answers questions about
    electric vehicle policies.
3.  The `database_tools` that can help the agent answer analytics
    queries based on three datasets: EV population data, EV population
    size history by county, and EV title and registration activity.

With the `ask_vehicles` and `ask_policies` tools, the ev-assistant can
answer questions based on text, and it will use the database tools to
answer analytical questions, based on the data.

Here is an example for instantiating the database tools:

```python
# For a single database
database_tools = ToolsFactory().database_tools(
    sql_database=your_database_object,
    tool_name_prefix="ev"
)
```

This creates four tools:

1.  `ev_list_tables`: A tool that lists the tables in the database.
2.  `ev_describe_tables`: A tool that describes the schema of a table.
3.  `ev_load_data`: A tool that loads data from a table.
4.  `ev_load_sample_data` tool which provides a sample of the data from a table.
5.  `ev_load_unique_values` tool which provides unique values for a set of columns in a table.

Together, these 5 tools provide a comprehensive set of capabilities for
an agent to interact with a database. 

For example, an agent can use the
`ev_list_tables` tool to get a list of tables in the database, and then use the `ev_describe_tables` tool to get the schema of a specific table. It will use the `ev_load_sample_data` to get a sample of the data in the table, or the `ev_load_unique_values` to explore the type of values valid for a column. Finally, the agent can use the `ev_load_data` tool to load the data into the agent\'s memory.

**Multiple databases**

In the case of EV-assistant, we use only a single database with 3
tables, and `tool_name_prefix="ev"`

If your use-case includes multiple databases, you can define multiple
database tools: each with a different database connection and a
different `tool_name_prefix`.

## Other Tools

In addition to the tools above, vectara-agentic also supports these
additional tools from the LlamaIndex Tools hub:

1.  `arxiv`: A tool that queries the arXiv respository of papers.
2.  `tavily_research`: A tool that queries the web using Tavily.
3.  `kuzu`: A tool that queries the Kuzu graph database.
4.  `exa.ai`: A tool that uses EXA.AI search.
5.  `brave`: A tool that uses Brave Search.
6.  `neo4j`: A tool that queries a Neo4J graph database.
7.  `google`: A set of tools that interact with Google services,
    including Gmail, Google Calendar, and Google Search.
8.  `slack`: A tool that interacts with Slack.
