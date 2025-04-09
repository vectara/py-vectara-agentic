# Tools

vectara-agentic provides a set of pre-built tools that you can use
out-of-the-box for various purposes.

## Standard Tools

The standard tools includes two tools that can be used for general
purposes:

1.  `summarize_text`: a tool that summarizes text, given a certain
    perspective and expertise. For example, you can use this tool to
    summarize text as a math teacher, a lawyer, or a doctor.
2.  `rephrase_text`: a tool that rephrases text given instructions. For
    example, you can instruct the tool to rephrase a response for a
    5-year-old\'s understanding or to adapt it to a formal tone.

## Finance Tools

vectara-agentic includes a few financial tools you can use right away in
your agent, based on the LlamaIndex
[YahooFinanceToolSpec](https://llamahub.ai/l/tools/llama-index-tools-yahoo-finance):

1.  `balance_sheet`: A tool that returns the balance sheet of a company.
2.  `income_statement`: A tool that returns the income statement of a
    company.
3.  `cash_flow`: A tool that returns the cash flow of a company.
4.  `stock_news`: A tool that returns the latest news about a company.
5.  `stock_basic_info`: A tool that returns basic information about a
    company including price.
6.  `stock_analyst_recommendations`: A tool that returns analyst
    recommendations for a company.

## Legal Tools

vectara-agentic includes a few tools for the legal space:

1.  `summarize_legal_text`: A tool that summarizes legal text.
2.  `critique_as_judge`: A tool that critiques legal text from the
    perspective of an expert judge.

## Guardrail Tools

These specialized tools help you AI assistnat or agent to avoid certain
topics or responses that are prohibited by your organization or by law.

The `get_bad_topics` tool returns a list of topics that are prohibited
(politics, religion, violence, hate speech, adult content, illegal
activities). The agent prompt has special instructions to call this tool
if it exists, and avoid these topics.

If you want to create your own set of topics, you can define a new tool
by the **same name** that returns a list of different topics, and the
agent will use that list to avoid these topics.

## Database Tools

Database tools are quite useful if your agent requires access to a
combination of RAG tools along with analytics capabilities. For example,
consider an
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

**Setting up the Database Tools**

The database tools are based on the LlamaIndex
[DatabaseToolSpec](https://llamahub.ai/l/tools/llama-index-tools-database)
You can [define these database
tools](https://vectara.github.io/vectara-agentic-docs/vectara_agentic.html#vectara_agentic.tools.ToolsFactory.database_tools)
in two ways:

1.  Specify a `sql_database` argument OR
2.  Specify the dbname, host, scheme, port, user and password, and the
    tool will generate the sql_database object for you from those.

This creates four tools:

1.  `list_tables`: A tool that lists the tables in the database.
2.  `describe_tables`: A tool that describes the schema of a table.
3.  `load_data`: A tool that loads data from a table.
4.  `load_sample_data` tool which provides a sample of the data from a
    table.
5.  `load_unique_values` tool which provides unique values for a set of
    columns in a table.

Together, these 4 tools provide a comprehensive set of capabilities for
an agent to interact with a database. For example, an agent can use the
`list_tables` tool to get a list of tables in the database, and then use
the `describe_tables` tool to get the schema of a specific table. It
will use the `load_sample_data` to get a sample of the data in the
table, or the `load_unique_values` to explore the type of values valid
for a column. Finally, the agent can use the `load_data` tool to load
the data into the agent\'s memory.

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
