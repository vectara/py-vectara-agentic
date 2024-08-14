"""
This module contains the ToolsFactory class for creating agent tools.
"""

import inspect
import re
import importlib

from typing import Callable, List, Any, Optional
from pydantic import BaseModel, Field

from llama_index.core.tools import FunctionTool
from llama_index.core.base.response.schema import Response
from llama_index.indices.managed.vectara import VectaraIndex
from llama_index.core.utilities.sql_wrapper import SQLDatabase

from .types import ToolType
from .tools_catalog import (
    # General tools
    summarize_text,
    rephrase_text,
    critique_text,
    # Guardrail tools
    guardrails_no_politics,
    guardrails_be_polite,
)

LI_packages = {
    "yahoo_finance": ToolType.QUERY,
    "arxiv": ToolType.QUERY,
    "tavily_research": ToolType.QUERY,
    "database": ToolType.QUERY,
    "google": {
        "GmailToolSpec": {
            "load_data": ToolType.QUERY,
            "search_messages": ToolType.QUERY,
            "create_draft": ToolType.ACTION,
            "update_draft": ToolType.ACTION,
            "get_draft": ToolType.QUERY,
            "send_draft": ToolType.ACTION,
        },
        "GoogleCalendarToolSpec": {
            "load_data": ToolType.QUERY,
            "create_event": ToolType.ACTION,
            "get_date": ToolType.QUERY,
        },
        "GoogleSearchToolSpec": {"google_search": ToolType.QUERY},
    },
}


class VectaraTool:
    """
    A wrapper of FunctionTool class for Vectara tools, adding the tool_type attribute.
    """

    def __init__(self, function_tool: FunctionTool, tool_type: ToolType) -> None:
        self.function_tool = function_tool
        self.tool_type = tool_type

    def __getattr__(self, name):
        return getattr(self.function_tool, name)

    def __call__(self, *args, **kwargs):
        return self.function_tool(*args, **kwargs)


class VectaraToolFactory:
    """
    A factory class for creating Vectara RAG tools.
    """

    def __init__(
        self,
        vectara_customer_id: str,
        vectara_corpus_id: str,
        vectara_api_key: str,
    ) -> None:
        self.vectara_customer_id = vectara_customer_id
        self.vectara_corpus_id = vectara_corpus_id
        self.vectara_api_key = vectara_api_key

    def create_rag_tool(
        self,
        tool_name: str,
        tool_description: str,
        tool_args_schema: type[BaseModel],
        vectara_summarizer: str = "vectara-summary-ext-24-05-sml",
        summary_num_results: int = 5,
        summary_response_lang: str = "eng",
        n_sentences_before: int = 2,
        n_sentences_after: int = 2,
        lambda_val: float = 0.005,
        reranker: str = "mmr",
        rerank_k: int = 50,
        mmr_diversity_bias: float = 0.2,
        include_citations: bool = True,
    ) -> VectaraTool:
        """
        Creates a RAG (Retrieve and Generate) tool.

        Args:

            tool_name (str): The name of the tool.
            tool_description (str): The description of the tool.
            tool_args_schema (BaseModel): The schema for the tool arguments.
            vectara_summarizer (str): The Vectara summarizer to use.
            summary_num_results (int): The number of summary results.
            summary_response_lang (str): The response language for the summary.
            n_sentences_before (int): Number of sentences before the summary.
            n_sentences_after (int): Number of sentences after the summary.
            lambda_val (float): Lambda value for the Vectara query.
            reranker (str): The reranker mode.
            rerank_k (int): Number of top-k documents for reranking.
            mmr_diversity_bias (float): MMR diversity bias.
            include_citations (bool): Whether to include citations in the response.
                If True, uses MARKDOWN vectara citations that requires the Vectara scale plan.

        Returns:
            VectaraTool: A VectaraTool object.
        """
        vectara = VectaraIndex(
            vectara_api_key=self.vectara_api_key,
            vectara_customer_id=self.vectara_customer_id,
            vectara_corpus_id=self.vectara_corpus_id,
        )

        def build_filter_string(kwargs):
            filter_parts = []
            for key, value in kwargs.items():
                if value:
                    if isinstance(value, str):
                        filter_parts.append(f"doc.{key}='{value}'")
                    else:
                        filter_parts.append(f"doc.{key}={value}")
            return " AND ".join(filter_parts)

        # Dynamically generate the RAG function
        def rag_function(*args, **kwargs) -> dict[str, Any]:
            """
            Dynamically generated function for RAG query with Vectara.
            """
            # Convert args to kwargs using the function signature
            sig = inspect.signature(rag_function)
            bound_args = sig.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()
            kwargs = bound_args.arguments

            query = kwargs.pop("query")
            filter_string = build_filter_string(kwargs)

            vectara_query_engine = vectara.as_query_engine(
                summary_enabled=True,
                summary_num_results=summary_num_results,
                summary_response_lang=summary_response_lang,
                summary_prompt_name=vectara_summarizer,
                vectara_query_mode=reranker,
                rerank_k=rerank_k,
                mmr_diversity_bias=mmr_diversity_bias,
                n_sentence_before=n_sentences_before,
                n_sentence_after=n_sentences_after,
                lambda_val=lambda_val,
                filter=filter_string,
                citations_url_pattern="{doc.url}" if include_citations else None,
            )
            response = vectara_query_engine.query(query)

            if str(response) == "None":
                return Response(
                    response="Tool failed to generate a response.", source_nodes=[]
                )

            # Extract citation metadata
            pattern = r"\[\[(\d+)\]" if include_citations else r"\[(\d+)\]"
            matches = re.findall(pattern, response.response)
            citation_numbers = [int(match) for match in matches]
            citation_metadata: dict = {
                f"metadata for citation {citation_number}": response.source_nodes[
                    citation_number - 1
                ].metadata
                for citation_number in citation_numbers
            }
            res = {
                "response": response.response,
                "citation_metadata": citation_metadata,
                "factual_consistency": (
                    response.metadata["fcs"] if "fcs" in response.metadata else 0.0
                ),
            }
            return res

        # Create signature for rag_function based on tool_args_schema
        parameters = []
        for name, param in tool_args_schema.__fields__.items():
            default = inspect.Parameter.empty
            if param.default is not None:
                default = param.default
            elif param.default_factory is not None:
                default = param.default_factory()

            parameters.append(
                inspect.Parameter(
                    name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=default,
                    annotation=param.type_ if param.required else Optional[param.type_],
                )
            )
        if (
            "query" not in tool_args_schema.__fields__
        ):  # Add 'query' parameter if it's not already in the schema
            parameters.append(
                inspect.Parameter(
                    "query", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str
                )
            )
        new_signature = inspect.Signature(parameters, return_annotation=dict[str, Any])
        setattr(rag_function, "__signature__", new_signature)

        # Set the function name and docstring
        doc_string = f"{tool_description}\n\n"
        doc_string += "Parameters:\n"
        for field_name, field in tool_args_schema.__fields__.items():
            type_name = field.type_.__name__
            if field.allow_none:
                type_name = f"Optional[{type_name}]"

            default_info = ""
            if field.default is not None:
                default_info = f" (default: {field.default})"
            elif field.default_factory is not None:
                default_info = " (default: set by factory)"

            doc_string += f"- {field_name} ({type_name}): {field.field_info.description}{default_info}\n"

        doc_string += "\nReturns: a dict[str, Any] with the following:\n"
        doc_string += (
            "- response: The response string in markdown format with citations.\n"
        )
        doc_string += "- citation_metadata: a dictionary of metadata for each citation included in the response string.\n"
        doc_string += "- response_factual_consistency: a value representing confidence in the response being factually correct (1.0=high, 0.0=low).\n"

        rag_function.__name__ = "_" + re.sub(r"[^A-Za-z0-9_]", "_", tool_name)
        rag_function.__doc__ = doc_string

        # Create the tool
        tool = FunctionTool.from_defaults(
            fn=rag_function,
            name=tool_name,
            description=doc_string,
            fn_schema=tool_args_schema,
        )
        return VectaraTool(tool, ToolType.QUERY)


class ToolsFactory:
    """
    A factory class for creating agent tools.
    """

    def create_tool(
        self, function: Callable, tool_type: ToolType = ToolType.QUERY
    ) -> List[FunctionTool]:
        """
        Create a tool from a function.

        Args:
            function (Callable): a function to convert into a tool.
            tool_type (ToolType): the type of tool.

        Returns:
            List[FunctionTool]: A list of FunctionTool objects.
        """
        return VectaraTool(FunctionTool.from_defaults(function), tool_type)

    def get_llama_index_tools(
        self,
        tool_package_name: str,
        tool_spec_name: str,
        tool_name_prefix: str = "",
        **kwargs: dict,
    ) -> List[FunctionTool]:
        """
        Get a tool from the llama_index hub.

        Parameters:
            tool_package_name (str): The name of the tool package.
            tool_spec_name (str): The name of the tool spec.
            tool_name_prefix (str): The prefix to add to the tool names (added to every tool in the spec).
            kwargs (dict): The keyword arguments to pass to the tool constructor (see Hub for tool specific details).

        Returns:
            list[FunctionTool]: A list of FunctionTool objects.
        """
        # Dynamically install and import the module
        if tool_package_name not in LI_packages.keys():
            raise ValueError(
                f"Tool package {tool_package_name} from LlamaIndex not supported by Vectara-agentic."
            )

        module_name = f"llama_index.tools.{tool_package_name}"
        module = importlib.import_module(module_name)

        # Get the tool spec class or function from the module
        tool_spec = getattr(module, tool_spec_name)

        func_type = LI_packages[tool_package_name]
        tools = tool_spec(**kwargs).to_tool_list()
        vtools = []
        for tool in tools:
            if len(tool_name_prefix) > 0:
                tool._metadata.name = tool_name_prefix + "_" + tool._metadata.name
            if isinstance(func_type, dict):
                if tool_spec_name not in func_type.keys():
                    raise ValueError(
                        f"Tool spec {tool_spec_name} not found in package {tool_package_name}."
                    )
                tool_type = func_type[tool_spec_name]
            else:
                tool_type = func_type
            vtools.append(VectaraTool(tool, tool_type))

        return vtools

    def standard_tools(self) -> List[FunctionTool]:
        """
        Create a list of standard tools.
        """
        return [self.create_tool(tool) for tool in [summarize_text, rephrase_text]]

    def guardrail_tools(self) -> List[FunctionTool]:
        return [
            self.create_tool(tool)
            for tool in [guardrails_no_politics, guardrails_be_polite]
        ]

    def financial_tools(self):
        """
        Create a list of financial tools.
        """
        return self.get_llama_index_tools("yahoo_finance", "YahooFinanceToolSpec")

    def legal_tools(self) -> List[FunctionTool]:
        def summarize_legal_text(
            text: str = Field(description="the original text."),
        ) -> str:
            """
            Use this tool to summarize legal text with no more than summary_max_length characters.
            """
            return summarize_text(text, expertise="law")

        def critique_as_judge(
            text: str = Field(description="the original text."),
        ) -> str:
            """
            Critique the legal document.
            """
            return critique_text(
                text,
                role="judge",
                point_of_view="""
                an experienced judge evaluating a legal document to provide areas of concern
                or that may require further legal scrutiny or legal argument.
                """,
            )

        return [
            self.create_tool(tool) for tool in [summarize_legal_text, critique_as_judge]
        ]

    def database_tools(
        self,
        tool_name_prefix: str = "",
        content_description: Optional[str] = None,
        sql_database: Optional[SQLDatabase] = None,
        scheme: Optional[str] = None,
        host: str = "localhost",
        port: str = "5432",
        user: str = "postgres",
        password: str = "Password",
        dbname: str = "postgres",
    ) -> List[FunctionTool]:
        """
        Returns a list of database tools.

        Args:

            tool_name_prefix (str, optional): The prefix to add to the tool names. Defaults to "".
            content_description (str, optional): The content description for the database. Defaults to None.
            sql_database (SQLDatabase, optional): The SQLDatabase object. Defaults to None.
            scheme (str, optional): The database scheme. Defaults to None.
            host (str, optional): The database host. Defaults to "localhost".
            port (str, optional): The database port. Defaults to "5432".
            user (str, optional): The database user. Defaults to "postgres".
            password (str, optional): The database password. Defaults to "Password".
            dbname (str, optional): The database name. Defaults to "postgres".
               You must specify either the sql_database object or the scheme, host, port, user, password, and dbname.

        Returns:
            List[FunctionTool]: A list of FunctionTool objects.
        """
        if sql_database:
            tools = self.get_llama_index_tools(
                "database",
                "DatabaseToolSpec",
                tool_name_prefix=tool_name_prefix,
                sql_database=sql_database,
            )
        else:
            if scheme in ["postgresql", "mysql", "sqlite", "mssql", "oracle"]:
                tools = self.get_llama_index_tools(
                    "database",
                    "DatabaseToolSpec",
                    tool_name_prefix=tool_name_prefix,
                    scheme=scheme,
                    host=host,
                    port=port,
                    user=user,
                    password=password,
                    dbname=dbname,
                )
            else:
                raise "Please provide a SqlDatabase option or a valid DB scheme type (postgresql, mysql, sqlite, mssql, oracle)."

        # Update tools with description
        for tool in tools:
            if content_description:
                tool._metadata.description = (
                    tool._metadata.description
                    + f"The database tables include data about {content_description}."
                )
        return tools
