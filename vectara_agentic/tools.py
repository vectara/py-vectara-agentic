"""
This module contains the ToolsFactory class for creating agent tools.
"""

import inspect
import re
import importlib
import os
import asyncio

from typing import Callable, List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

from llama_index.core.tools import FunctionTool
from llama_index.indices.managed.vectara import VectaraIndex
from llama_index.core.utilities.sql_wrapper import SQLDatabase
from llama_index.core.tools.types import ToolOutput

from .types import ToolType
from .tools_catalog import ToolsCatalog, get_bad_topics
from .db_tools import DatabaseTools
from .utils import summarize_documents, is_float
from .agent_config import AgentConfig
from .tool_utils import (
    create_tool_from_dynamic_function,
    build_filter_string,
    VectaraTool,
)

LI_packages = {
    "yahoo_finance": ToolType.QUERY,
    "arxiv": ToolType.QUERY,
    "tavily_research": ToolType.QUERY,
    "exa": ToolType.QUERY,
    "brave_search": ToolType.QUERY,
    "bing_search": ToolType.QUERY,
    "neo4j": ToolType.QUERY,
    "kuzu": ToolType.QUERY,
    "wikipedia": ToolType.QUERY,
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
    "slack": {
        "SlackToolSpec": {
            "load_data": ToolType.QUERY,
            "send_message": ToolType.ACTION,
            "fetch_channel": ToolType.QUERY,
        }
    },
}


class VectaraToolFactory:
    """
    A factory class for creating Vectara RAG tools.
    """

    def __init__(
        self,
        vectara_corpus_key: str = str(os.environ.get("VECTARA_CORPUS_KEY", "")),
        vectara_api_key: str = str(os.environ.get("VECTARA_API_KEY", "")),
        compact_docstring: bool = False,
    ) -> None:
        """
        Initialize the VectaraToolFactory
        Args:
            vectara_corpus_key (str): The Vectara corpus key (or comma separated list of keys).
            vectara_api_key (str): The Vectara API key.
            compact_docstring (bool): Whether to use a compact docstring format for tools
              This is useful if OpenAI complains on the 1024 token limit.
        """
        self.vectara_corpus_key = vectara_corpus_key
        self.vectara_api_key = vectara_api_key
        self.num_corpora = len(vectara_corpus_key.split(","))
        self.compact_docstring = compact_docstring

    def create_search_tool(
        self,
        tool_name: str,
        tool_description: str,
        tool_args_schema: type[BaseModel] = None,
        tool_args_type: Dict[str, str] = {},
        summarize_docs: Optional[bool] = None,
        summarize_llm_name: Optional[str] = None,
        fixed_filter: str = "",
        lambda_val: Union[List[float], float] = 0.005,
        semantics: Union[List[str] | str] = "default",
        custom_dimensions: Union[List[Dict], Dict] = {},
        offset: int = 0,
        n_sentences_before: int = 2,
        n_sentences_after: int = 2,
        reranker: str = "slingshot",
        rerank_k: int = 50,
        rerank_limit: Optional[int] = None,
        rerank_cutoff: Optional[float] = None,
        mmr_diversity_bias: float = 0.2,
        udf_expression: str = None,
        rerank_chain: List[Dict] = None,
        return_direct: bool = False,
        save_history: bool = True,
        verbose: bool = False,
        vectara_base_url: str = "https://api.vectara.io",
        vectara_verify_ssl: bool = True,
    ) -> VectaraTool:
        """
        Creates a Vectara search/retrieval tool

        Args:
            tool_name (str): The name of the tool.
            tool_description (str): The description of the tool.
            tool_args_schema (BaseModel, optional): The schema for the tool arguments.
            tool_args_type (Dict[str, str], optional): The type of each argument (doc or part).
            fixed_filter (str, optional): A fixed Vectara filter condition to apply to all queries.
            lambda_val (Union[List[float] | float], optional): Lambda value (or list of values for each corpora)
                for the Vectara query, when using hybrid search.
            semantics (Union[List[str], str], optional): Indicates whether the query is intended as a query or response.
                Include list if using multiple corpora specifying the query type for each corpus.
            custom_dimensions (Union[List[Dict] | Dict], optional): Custom dimensions for the query (for each corpora).
            offset (int, optional): Number of results to skip.
            n_sentences_before (int, optional): Number of sentences before the matching document part.
            n_sentences_after (int, optional): Number of sentences after the matching document part.
            reranker (str, optional): The reranker mode.
            rerank_k (int, optional): Number of top-k documents for reranking.
            rerank_limit (int, optional): Maximum number of results to return after reranking.
            rerank_cutoff (float, optional): Minimum score threshold for results to include after reranking.
            mmr_diversity_bias (float, optional): MMR diversity bias.
            udf_expression (str, optional): the user defined expression for reranking results.
            rerank_chain (List[Dict], optional): A list of rerankers to be applied sequentially.
                Each dictionary should specify the "type" of reranker (mmr, slingshot, udf)
                and any other parameters (e.g. "limit" or "cutoff" for any type,
                "diversity_bias" for mmr, and "user_function" for udf).
                If using slingshot/multilingual_reranker_v1, it must be first in the list.
            save_history (bool, optional): Whether to save the query in history.
            return_direct (bool, optional): Whether the agent should return the tool's response directly.
            verbose (bool, optional): Whether to print verbose output.
            vectara_base_url (str, optional): The base URL for the Vectara API.
            vectara_verify_ssl (bool, optional): Whether to verify SSL certificates for the Vectara API.

        Returns:
            VectaraTool: A VectaraTool object.
        """

        vectara = VectaraIndex(
            vectara_api_key=self.vectara_api_key,
            vectara_corpus_key=self.vectara_corpus_key,
            x_source_str="vectara-agentic",
            vectara_base_url=vectara_base_url,
            vectara_verify_ssl=vectara_verify_ssl,
        )

        # Dynamically generate the search function
        def search_function(*args: Any, **kwargs: Any) -> ToolOutput:
            """
            Dynamically generated function for semantic search Vectara.
            """
            # Convert args to kwargs using the function signature
            sig = inspect.signature(search_function)
            bound_args = sig.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()
            kwargs = bound_args.arguments

            query = kwargs.pop("query")
            top_k = kwargs.pop("top_k", 10)
            summarize = (
                kwargs.pop("summarize", True)
                if summarize_docs is None
                else summarize_docs
            )
            try:
                filter_string = build_filter_string(
                    kwargs, tool_args_type, fixed_filter
                )
            except ValueError as e:
                return ToolOutput(
                    tool_name=search_function.__name__,
                    content=str(e),
                    raw_input={"args": args, "kwargs": kwargs},
                    raw_output={"response": str(e)},
                )

            vectara_retriever = vectara.as_retriever(
                summary_enabled=False,
                similarity_top_k=top_k,
                reranker=reranker,
                rerank_k=(
                    rerank_k
                    if rerank_k * self.num_corpora <= 100
                    else int(100 / self.num_corpora)
                ),
                rerank_limit=rerank_limit,
                rerank_cutoff=rerank_cutoff,
                mmr_diversity_bias=mmr_diversity_bias,
                udf_expression=udf_expression,
                rerank_chain=rerank_chain,
                lambda_val=lambda_val,
                semantics=semantics,
                custom_dimensions=custom_dimensions,
                offset=offset,
                filter=filter_string,
                n_sentences_before=n_sentences_before,
                n_sentences_after=n_sentences_after,
                save_history=save_history,
                x_source_str="vectara-agentic",
                verbose=verbose,
            )
            response = vectara_retriever.retrieve(query)

            if len(response) == 0:
                msg = "Vectara Tool failed to retrieve any results for the query."
                return ToolOutput(
                    tool_name=search_function.__name__,
                    content=msg,
                    raw_input={"args": args, "kwargs": kwargs},
                    raw_output={"response": msg},
                )
            unique_ids = set()
            docs = []
            for doc in response:
                if doc.id_ in unique_ids:
                    continue
                unique_ids.add(doc.id_)
                docs.append((doc.id_, doc.metadata))
            tool_output = "Matching documents:\n"
            if summarize:
                summaries_dict = asyncio.run(
                    summarize_documents(
                        corpus_key=self.vectara_corpus_key,
                        api_key=self.vectara_api_key,
                        llm_name=summarize_llm_name,
                        doc_ids=list(unique_ids),
                    )
                )
                for doc_id, metadata in docs:
                    summary = summaries_dict.get(doc_id, "")
                    tool_output += f"document_id: '{doc_id}'\nmetadata: '{metadata}'\nsummary: '{summary}'\n\n"
            else:
                for doc_id, metadata in docs:
                    tool_output += (
                        f"document_id: '{doc_id}'\nmetadata: '{metadata}'\n\n"
                    )

            out = ToolOutput(
                tool_name=search_function.__name__,
                content=tool_output,
                raw_input={"args": args, "kwargs": kwargs},
                raw_output=response,
            )
            return out

        class SearchToolBaseParams(BaseModel):
            """Model for the base parameters of the search tool."""

            query: str = Field(
                ...,
                description="The search query to perform, in the form of a question.",
            )
            top_k: int = Field(
                default=10, description="The number of top documents to retrieve."
            )
            summarize: bool = Field(
                True,
                description="Whether to summarize the retrieved documents.",
            )

        class SearchToolBaseParamsWithoutSummarize(BaseModel):
            """Model for the base parameters of the search tool."""

            query: str = Field(
                ...,
                description="The search query to perform, in the form of a question.",
            )
            top_k: int = Field(
                10, description="The number of top documents to retrieve."
            )

        search_tool_extra_desc = (
            tool_description
            + "\n"
            + "Use this tool to search for relevant documents, not to ask questions."
        )

        tool = create_tool_from_dynamic_function(
            search_function,
            tool_name,
            search_tool_extra_desc,
            (
                SearchToolBaseParams
                if summarize_docs is None
                else SearchToolBaseParamsWithoutSummarize
            ),
            tool_args_schema,
            compact_docstring=self.compact_docstring,
            return_direct=return_direct,
        )
        return tool

    def create_rag_tool(
        self,
        tool_name: str,
        tool_description: str,
        tool_args_schema: type[BaseModel] = None,
        tool_args_type: Dict[str, dict] = {},
        fixed_filter: str = "",
        vectara_summarizer: str = "vectara-summary-ext-24-05-med-omni",
        vectara_prompt_text: str = None,
        summary_num_results: int = 5,
        summary_response_lang: str = "eng",
        n_sentences_before: int = 2,
        n_sentences_after: int = 2,
        offset: int = 0,
        lambda_val: Union[List[float], float] = 0.005,
        semantics: Union[List[str] | str] = "default",
        custom_dimensions: Union[List[Dict], Dict] = {},
        reranker: str = "slingshot",
        rerank_k: int = 50,
        rerank_limit: Optional[int] = None,
        rerank_cutoff: Optional[float] = None,
        mmr_diversity_bias: float = 0.2,
        udf_expression: str = None,
        rerank_chain: List[Dict] = None,
        max_response_chars: Optional[int] = None,
        max_tokens: Optional[int] = None,
        llm_name: Optional[str] = None,
        temperature: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        include_citations: bool = True,
        save_history: bool = False,
        fcs_threshold: float = 0.0,
        return_direct: bool = False,
        verbose: bool = False,
        vectara_base_url: str = "https://api.vectara.io",
        vectara_verify_ssl: bool = True,
    ) -> VectaraTool:
        """
        Creates a RAG (Retrieve and Generate) tool.

        Args:
            tool_name (str): The name of the tool.
            tool_description (str): The description of the tool.
            tool_args_schema (BaseModel, optional): The schema for any tool arguments for filtering.
            tool_args_type (Dict[str, dict], optional): attributes for each argument where they key is the field name
                and the value is a dictionary with the following keys:
                - 'type': the type of each filter attribute in Vectara (doc or part).
                - 'is_list': whether the filterable attribute is a list.
            fixed_filter (str, optional): A fixed Vectara filter condition to apply to all queries.
            vectara_summarizer (str, optional): The Vectara summarizer to use.
            vectara_prompt_text (str, optional): The prompt text for the Vectara summarizer.
            summary_num_results (int, optional): The number of summary results.
            summary_response_lang (str, optional): The response language for the summary.
            n_sentences_before (int, optional): Number of sentences before the summary.
            n_sentences_after (int, optional): Number of sentences after the summary.
            offset (int, optional): Number of results to skip.
            lambda_val (Union[List[float] | float], optional): Lambda value (or list of values for each corpora)
                for the Vectara query, when using hybrid search.
            semantics (Union[List[str], str], optional): Indicates whether the query is intended as a query or response.
                Include list if using multiple corpora specifying the query type for each corpus.
            custom_dimensions (Union[List[Dict] | Dict], optional): Custom dimensions for the query (for each corpora).
            reranker (str, optional): The reranker mode.
            rerank_k (int, optional): Number of top-k documents for reranking.
            rerank_limit (int, optional): Maximum number of results to return after reranking.
            rerank_cutoff (float, optional): Minimum score threshold for results to include after reranking.
            mmr_diversity_bias (float, optional): MMR diversity bias.
            udf_expression (str, optional): The user defined expression for reranking results.
            rerank_chain (List[Dict], optional): A list of rerankers to be applied sequentially.
                Each dictionary should specify the "type" of reranker (mmr, slingshot, udf)
                and any other parameters (e.g. "limit" or "cutoff" for any type,
                "diversity_bias" for mmr, and "user_function" for udf).
                If using slingshot/multilingual_reranker_v1, it must be first in the list.
            max_response_chars (int, optional): The desired maximum number of characters for the generated summary.
            max_tokens (int, optional): The maximum number of tokens to be returned by the LLM.
            llm_name (str, optional): The name of the LLM to use for generation.
            temperature (float, optional): The sampling temperature; higher values lead to more randomness.
            frequency_penalty (float, optional): How much to penalize repeating tokens in the response,
                higher values reducing likelihood of repeating the same line.
            presence_penalty (float, optional): How much to penalize repeating tokens in the response,
                higher values increasing the diversity of topics.
            include_citations (bool, optional): Whether to include citations in the response.
                If True, uses markdown vectara citations that requires the Vectara scale plan.
            save_history (bool, optional): Whether to save the query in history.
            fcs_threshold (float, optional): A threshold for factual consistency.
                If set above 0, the tool notifies the calling agent that it "cannot respond" if FCS is too low.
            return_direct (bool, optional): Whether the agent should return the tool's response directly.
            verbose (bool, optional): Whether to print verbose output.
            vectara_base_url (str, optional): The base URL for the Vectara API.
            vectara_verify_ssl (bool, optional): Whether to verify SSL certificates for the Vectara API.

        Returns:
            VectaraTool: A VectaraTool object.
        """

        vectara = VectaraIndex(
            vectara_api_key=self.vectara_api_key,
            vectara_corpus_key=self.vectara_corpus_key,
            x_source_str="vectara-agentic",
            vectara_base_url=vectara_base_url,
            vectara_verify_ssl=vectara_verify_ssl,
        )

        # Dynamically generate the RAG function
        def rag_function(*args: Any, **kwargs: Any) -> ToolOutput:
            """
            Dynamically generated function for RAG query with Vectara.
            """
            # Convert args to kwargs using the function signature
            sig = inspect.signature(rag_function)
            bound_args = sig.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()
            kwargs = bound_args.arguments

            query = kwargs.pop("query")
            try:
                filter_string = build_filter_string(
                    kwargs, tool_args_type, fixed_filter
                )
            except ValueError as e:
                return ToolOutput(
                    tool_name=rag_function.__name__,
                    content=str(e),
                    raw_input={"args": args, "kwargs": kwargs},
                    raw_output={"response": str(e)},
                )

            vectara_query_engine = vectara.as_query_engine(
                summary_enabled=True,
                similarity_top_k=summary_num_results,
                summary_num_results=summary_num_results,
                summary_response_lang=summary_response_lang,
                summary_prompt_name=vectara_summarizer,
                prompt_text=vectara_prompt_text,
                reranker=reranker,
                rerank_k=(
                    rerank_k
                    if rerank_k * self.num_corpora <= 100
                    else int(100 / self.num_corpora)
                ),
                rerank_limit=rerank_limit,
                rerank_cutoff=rerank_cutoff,
                mmr_diversity_bias=mmr_diversity_bias,
                udf_expression=udf_expression,
                rerank_chain=rerank_chain,
                n_sentences_before=n_sentences_before,
                n_sentences_after=n_sentences_after,
                offset=offset,
                lambda_val=lambda_val,
                semantics=semantics,
                custom_dimensions=custom_dimensions,
                filter=filter_string,
                max_response_chars=max_response_chars,
                max_tokens=max_tokens,
                llm_name=llm_name,
                temperature=temperature,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                citations_style="markdown" if include_citations else None,
                citations_url_pattern="{doc.url}" if include_citations else None,
                save_history=save_history,
                x_source_str="vectara-agentic",
                verbose=verbose,
            )
            response = vectara_query_engine.query(query)

            if len(response.source_nodes) == 0:
                msg = (
                    "Tool failed to generate a response since no matches were found. "
                    "Please check the arguments and try again."
                )
                return ToolOutput(
                    tool_name=rag_function.__name__,
                    content=msg,
                    raw_input={"args": args, "kwargs": kwargs},
                    raw_output={"response": msg},
                )
            if str(response) == "None":
                msg = "Tool failed to generate a response."
                return ToolOutput(
                    tool_name=rag_function.__name__,
                    content=msg,
                    raw_input={"args": args, "kwargs": kwargs},
                    raw_output={"response": msg},
                )

            # Extract citation metadata
            pattern = r"\[(\d+)\]"
            matches = re.findall(pattern, response.response)
            citation_numbers = sorted(set(int(match) for match in matches))
            citation_metadata = ""
            keys_to_ignore = ["lang", "offset", "len"]
            for citation_number in citation_numbers:
                metadata = response.source_nodes[citation_number - 1].metadata
                citation_metadata += (
                    f"[{citation_number}]: "
                    + "; ".join(
                        [
                            f"{k}='{v}'"
                            for k, v in metadata.items()
                            if k not in keys_to_ignore
                        ]
                    )
                    + ".\n"
                )
            fcs = 0.0
            fcs_str = response.metadata["fcs"] if "fcs" in response.metadata else "0.0"
            if fcs_str and is_float(fcs_str):
                fcs = float(fcs_str)
                if fcs < fcs_threshold:
                    msg = f"Could not answer the query due to suspected hallucination (fcs={fcs})."
                    return ToolOutput(
                        tool_name=rag_function.__name__,
                        content=msg,
                        raw_input={"args": args, "kwargs": kwargs},
                        raw_output={"response": msg},
                    )
            res = {
                "response": response.response,
                "references_metadata": citation_metadata,
                "fcs_score": fcs,
            }
            if len(citation_metadata) > 0:
                tool_output = f"""
                    Response: '''{res['response']}'''
                    fcs_score: {res['fcs_score']:.4f}
                    References:
                    {res['references_metadata']}
                """
            else:
                tool_output = f"Response: '''{res['response']}'''"
            out = ToolOutput(
                tool_name=rag_function.__name__,
                content=tool_output,
                raw_input={"args": args, "kwargs": kwargs},
                raw_output=res,
            )
            return out

        class RagToolBaseParams(BaseModel):
            """Model for the base parameters of the RAG tool."""

            query: str = Field(
                ...,
                description="The search query to perform, in the form of a question",
            )

        tool = create_tool_from_dynamic_function(
            rag_function,
            tool_name,
            tool_description,
            RagToolBaseParams,
            tool_args_schema,
            compact_docstring=self.compact_docstring,
            return_direct=return_direct,
        )
        return tool


class ToolsFactory:
    """
    A factory class for creating agent tools.
    """

    def __init__(self, agent_config: AgentConfig = None) -> None:
        self.agent_config = agent_config

    def create_tool(
        self, function: Callable, tool_type: ToolType = ToolType.QUERY
    ) -> VectaraTool:
        """
        Create a tool from a function.

        Args:
            function (Callable): a function to convert into a tool.
            tool_type (ToolType): the type of tool.

        Returns:
            VectaraTool: A VectaraTool object.
        """
        return VectaraTool.from_defaults(tool_type=tool_type, fn=function)

    def get_llama_index_tools(
        self,
        tool_package_name: str,
        tool_spec_name: str,
        tool_name_prefix: str = "",
        **kwargs: dict,
    ) -> List[VectaraTool]:
        """
        Get a tool from the llama_index hub.

        Args:
            tool_package_name (str): The name of the tool package.
            tool_spec_name (str): The name of the tool spec.
            tool_name_prefix (str, optional): The prefix to add to the tool names (added to every tool in the spec).
            kwargs (dict): The keyword arguments to pass to the tool constructor (see Hub for tool specific details).

        Returns:
            List[VectaraTool]: A list of VectaraTool objects.
        """
        # Dynamically install and import the module
        if tool_package_name not in LI_packages:
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
                tool.metadata.name = tool_name_prefix + "_" + tool.metadata.name
            if isinstance(func_type, dict):
                if tool_spec_name not in func_type.keys():
                    raise ValueError(
                        f"Tool spec {tool_spec_name} not found in package {tool_package_name}."
                    )
                tool_type = func_type[tool_spec_name]
            else:
                tool_type = func_type
            vtool = VectaraTool(
                tool_type=tool_type,
                fn=tool.fn,
                metadata=tool.metadata,
                async_fn=tool.async_fn,
            )
            vtools.append(vtool)
        return vtools

    def standard_tools(self) -> List[FunctionTool]:
        """
        Create a list of standard tools.
        """
        tc = ToolsCatalog(self.agent_config)
        return [
            self.create_tool(tool)
            for tool in [tc.summarize_text, tc.rephrase_text, tc.critique_text]
        ]

    def guardrail_tools(self) -> List[FunctionTool]:
        """
        Create a list of guardrail tools to avoid controversial topics.
        """
        return [self.create_tool(get_bad_topics)]

    def financial_tools(self):
        """
        Create a list of financial tools.
        """
        return self.get_llama_index_tools(
            tool_package_name="yahoo_finance", tool_spec_name="YahooFinanceToolSpec"
        )

    def legal_tools(self) -> List[FunctionTool]:
        """
        Create a list of legal tools.
        """

        def summarize_legal_text(
            text: str = Field(description="the original text."),
        ) -> str:
            """
            Use this tool to summarize legal text with no more than summary_max_length characters.
            """
            tc = ToolsCatalog(self.agent_config)
            return tc.summarize_text(text, expertise="law")

        def critique_as_judge(
            text: str = Field(description="the original text."),
        ) -> str:
            """
            Critique the legal document.
            """
            tc = ToolsCatalog(self.agent_config)
            return tc.critique_text(
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
        max_rows: int = 1000,
    ) -> List[VectaraTool]:
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
            max_rows (int, optional): if specified, instructs the load_data tool to never return more than max_rows
               rows. Defaults to 1000.

        Returns:
            List[VectaraTool]: A list of VectaraTool objects.
        """
        if sql_database:
            dbt = DatabaseTools(
                tool_name_prefix=tool_name_prefix,
                sql_database=sql_database,
                max_rows=max_rows,
            )
        else:
            if scheme in ["postgresql", "mysql", "sqlite", "mssql", "oracle"]:
                dbt = DatabaseTools(
                    tool_name_prefix=tool_name_prefix,
                    scheme=scheme,
                    host=host,
                    port=port,
                    user=user,
                    password=password,
                    dbname=dbname,
                    max_rows=max_rows,
                )
            else:
                raise ValueError(
                    "Please provide a SqlDatabase option or a valid DB scheme type "
                    " (postgresql, mysql, sqlite, mssql, oracle)."
                )

        # Update tools with description
        tools = dbt.to_tool_list()
        vtools = []
        for tool in tools:
            if content_description:
                tool.metadata.description = (
                    tool.metadata.description
                    + f"The database tables include data about {content_description}."
                )
            vtool = VectaraTool(
                tool_type=ToolType.QUERY,
                fn=tool.fn,
                async_fn=tool.async_fn,
                metadata=tool.metadata,
            )
            vtools.append(vtool)
        return vtools
