"""
This module contains the ToolsFactory class for creating agent tools.
"""

import inspect
import re
import importlib
import os

from typing import Callable, List, Dict, Any, Optional, Type
from pydantic import BaseModel, Field

from llama_index.core.tools import FunctionTool
from llama_index.core.tools.function_tool import AsyncCallable
from llama_index.indices.managed.vectara import VectaraIndex, VectaraRetriever, VectaraQueryEngine, VectaraAutoRetriever
from llama_index.core.utilities.sql_wrapper import SQLDatabase
from llama_index.core.tools.types import ToolMetadata, ToolOutput
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo, VectorStoreQuerySpec
from llama_index.core.llms import LLM
from llama_index.core.schema import QueryBundle


from .types import ToolType, LLMRole
from .utils import get_llm
from .tools_catalog import summarize_text, rephrase_text, critique_text, get_bad_topics
from .db_tools import DBLoadSampleData, DBLoadUniqueValues, DBLoadData

LI_packages = {
    "yahoo_finance": ToolType.QUERY,
    "arxiv": ToolType.QUERY,
    "tavily_research": ToolType.QUERY,
    "neo4j": ToolType.QUERY,
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
    "slack": {
        "SlackToolSpec": {
            "load_data": ToolType.QUERY,
            "send_message": ToolType.ACTION,
            "fetch_channel": ToolType.QUERY,
        }
    }
}

class VectaraToolMetadata(ToolMetadata):
    """
    A subclass of ToolMetadata adding the tool_type attribute.
    """
    tool_type: ToolType

    def __init__(self, tool_type: ToolType, **kwargs):
        super().__init__(**kwargs)
        self.tool_type = tool_type

    def __repr__(self) -> str:
        """
        Returns a string representation of the VectaraToolMetadata object, including the tool_type attribute.
        """
        base_repr = super().__repr__()
        return f"{base_repr}, tool_type={self.tool_type}"

class VectaraTool(FunctionTool):
    """
    A subclass of FunctionTool adding the tool_type attribute.
    """

    def __init__(
        self,
        tool_type: ToolType,
        metadata: ToolMetadata,
        fn: Optional[Callable[..., Any]] = None,
        async_fn: Optional[AsyncCallable] = None,
    ) -> None:
        metadata_dict = metadata.dict() if hasattr(metadata, 'dict') else metadata.__dict__
        vm = VectaraToolMetadata(tool_type=tool_type, **metadata_dict)
        super().__init__(fn, vm, async_fn)

    @classmethod
    def from_defaults(
        cls,
        fn: Optional[Callable[..., Any]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        return_direct: bool = False,
        fn_schema: Optional[Type[BaseModel]] = None,
        async_fn: Optional[AsyncCallable] = None,
        tool_metadata: Optional[ToolMetadata] = None,
        tool_type: ToolType = ToolType.QUERY,
    ) -> "VectaraTool":
        tool = FunctionTool.from_defaults(fn, name, description, return_direct, fn_schema, async_fn, tool_metadata)
        vectara_tool = cls(tool_type=tool_type, fn=tool.fn, metadata=tool.metadata, async_fn=tool.async_fn)
        return vectara_tool

    def __eq__(self, other):
        if self.metadata.tool_type != other.metadata.tool_type:
            return False

        # Check if fn_schema is an instance of a BaseModel or a class itself (metaclass)
        self_schema_dict = self.metadata.fn_schema.model_fields
        other_schema_dict = other.metadata.fn_schema.model_fields
        is_equal = True
        for key in self_schema_dict.keys():
            if key not in other_schema_dict:
                is_equal = False
                break
            if (
                self_schema_dict[key].annotation != other_schema_dict[key].annotation
                or self_schema_dict[key].description != other_schema_dict[key].description
                or self_schema_dict[key].is_required() != other_schema_dict[key].is_required()
            ):
                is_equal = False
                break
        return is_equal

def _build_auto_retriever(
    query_string: str,
    index: VectaraIndex,
    vector_store_info: VectorStoreInfo,
    llm: LLM,
    **kwargs: Any,
) ->  VectaraQueryEngine:
    auto_retriever = VectaraAutoRetriever(
        index=index,
        vector_store_info = vector_store_info,
        llm=get_llm(LLMRole.TOOL),
        **kwargs
    )
    query_bundle = QueryBundle(query_str=query_string)
    spec = auto_retriever.generate_retrieval_spec(query_bundle)
    vectara_retriever, new_query = auto_retriever._build_retriever_from_spec(
        VectorStoreQuerySpec(
            query=spec.query, filters=spec.filters
        )
    )

    return VectaraQueryEngine.from_args(retriever=vectara_retriever)    

class VectaraToolFactory:
    """
    A factory class for creating Vectara RAG tools.
    """

    def __init__(
        self,
        vectara_customer_id: str = str(os.environ.get("VECTARA_CUSTOMER_ID", "")),
        vectara_corpus_id: str = str(os.environ.get("VECTARA_CORPUS_ID", "")),
        vectara_api_key: str = str(os.environ.get("VECTARA_API_KEY", "")),
    ) -> None:
        """
        Initialize the VectaraToolFactory
        Args:
            vectara_customer_id (str): The Vectara customer ID.
            vectara_corpus_id (str): The Vectara corpus ID (or comma separated list of IDs).
            vectara_api_key (str): The Vectara API key.
        """
        self.vectara_customer_id = vectara_customer_id
        self.vectara_corpus_id = vectara_corpus_id
        self.vectara_api_key = vectara_api_key
        self.num_corpora = len(vectara_corpus_id.split(","))

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
        udf_expression: str = None,
        rerank_chain: List[Dict] = None,
        include_citations: bool = True,
        fcs_threshold: float = 0.0,
    ) -> VectaraTool:
        """
        Creates a RAG (Retrieve and Generate) tool.

        Args:
            tool_name (str): The name of the tool.
            tool_description (str): The description of the tool.
            tool_args_schema (BaseModel): The schema for the tool arguments.
            vectara_summarizer (str, optional): The Vectara summarizer to use.
            summary_num_results (int, optional): The number of summary results.
            summary_response_lang (str, optional): The response language for the summary.
            n_sentences_before (int, optional): Number of sentences before the summary.
            n_sentences_after (int, optional): Number of sentences after the summary.
            lambda_val (float, optional): Lambda value for the Vectara query.
            reranker (str, optional): The reranker mode.
            rerank_k (int, optional): Number of top-k documents for reranking.
            mmr_diversity_bias (float, optional): MMR diversity bias.
            udf_expression (str, optional): the user defined expression for reranking results.
            rerank_chain (List[Dict], optional): A list of rerankers to be applied sequentially.
                Each dictionary should specify the "type" of reranker (mmr, slingshot, udf)
                and any other parameters (e.g. "limit" or "cutoff" for any type,
                "diversity_bias" for mmr, and "user_function" for udf).
            If using slingshot/multilingual_reranker_v1, it must be first in the list.
            include_citations (bool, optional): Whether to include citations in the response.
                If True, uses markdown vectara citations that requires the Vectara scale plan.
            fcs_threshold (float, optional): a threshold for factual consistency.
                If set above 0, the tool notifies the calling agent that it "cannot respond" if FCS is too low.

        Returns:
            VectaraTool: A VectaraTool object.
        """
        vectara = VectaraIndex(
            vectara_api_key=self.vectara_api_key,
            vectara_customer_id=self.vectara_customer_id,
            vectara_corpus_id=self.vectara_corpus_id,
            x_source_str="vectara-agentic",
        )

        # Set up Metadata Info
        fields = tool_args_schema.model_fields
        vector_store_info = VectorStoreInfo(
            content_info=tool_description,
            metadata_info = [MetadataInfo(
                name=field_info.alias if field_info.alias else name,
                description=field_info.description,
                type=str(field_info.annotation)
            ) for name, field_info in fields.items()],
        )

        # Dynamically generate the RAG function
        def rag_function(query: str) -> ToolOutput:
            """
            Dynamically generated function for RAG query with Vectara.
            """

            # Set up query engine
            vectara_query_engine = _build_auto_retriever(
                query,
                vectara,
                vector_store_info,
                get_llm(LLMRole.TOOL),
                summary_enabled=True,
                summary_num_results=summary_num_results,
                summary_response_lang=summary_response_lang,
                summary_prompt_name=vectara_summarizer,
                reranker=reranker,
                rerank_k=rerank_k if rerank_k * self.num_corpora <= 100 else int(100 / self.num_corpora),
                mmr_diversity_bias=mmr_diversity_bias,
                udf_expression=udf_expression,
                rerank_chain=rerank_chain,
                n_sentence_before=n_sentences_before,
                n_sentence_after=n_sentences_after,
                lambda_val=lambda_val,
                citations_style="MARKDOWN" if include_citations else None,
                citations_url_pattern="{doc.url}" if include_citations else None,
                x_source_str="vectara-agentic",
            )

            response = vectara_query_engine.query(query)

            if str(response) == "None":
                msg = "Tool failed to generate a response due to internal error."
                return ToolOutput(
                    tool_name=rag_function.__name__,
                    content=msg,
                    raw_input={"query": query},
                    raw_output={"response": msg},
                )
            if len(response.source_nodes) == 0:
                msg = "Tool failed to generate a response since no matches were found."
                return ToolOutput(
                    tool_name=rag_function.__name__,
                    content=msg,
                    raw_input={"query": query},
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
            fcs = response.metadata["fcs"] if "fcs" in response.metadata else 0.0
            if fcs < fcs_threshold:
                msg = f"Could not answer the query due to suspected hallucination (fcs={fcs})."
                return ToolOutput(
                    tool_name=rag_function.__name__,
                    content=msg,
                    raw_input={"query": query},
                    raw_output={"response": msg},
                )
            res = {
                "response": response.response,
                "references_metadata": citation_metadata,
            }
            if len(citation_metadata) > 0:
                tool_output = f"""
                    Response: '''{res['response']}'''
                    References:
                    {res['references_metadata']}
                """
            else:
                tool_output = f"Response: '''{res['response']}'''"
            out = ToolOutput(
                tool_name=rag_function.__name__,
                content=tool_output,
                raw_input={"query": query},
                raw_output=res,
            )
            return out

        rag_function.__annotations__["return"] = dict[str, Any]
        rag_function.__name__ = "_" + re.sub(r"[^A-Za-z0-9_]", "_", tool_name)

        # Create the tool
        tool = VectaraTool.from_defaults(
            fn=rag_function,
            name=tool_name,
            description=tool_description,
            tool_type=ToolType.QUERY,
        )
        return tool


class ToolsFactory:
    """
    A factory class for creating agent tools.
    """

    def create_tool(self, function: Callable, tool_type: ToolType = ToolType.QUERY) -> VectaraTool:
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
            raise ValueError(f"Tool package {tool_package_name} from LlamaIndex not supported by Vectara-agentic.")

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
                    raise ValueError(f"Tool spec {tool_spec_name} not found in package {tool_package_name}.")
                tool_type = func_type[tool_spec_name]
            else:
                tool_type = func_type
            vtool = VectaraTool(tool_type=tool_type, fn=tool.fn, metadata=tool.metadata, async_fn=tool.async_fn)
            vtools.append(vtool)
        return vtools

    def standard_tools(self) -> List[FunctionTool]:
        """
        Create a list of standard tools.
        """
        return [self.create_tool(tool) for tool in [summarize_text, rephrase_text]]

    def guardrail_tools(self) -> List[FunctionTool]:
        """
        Create a list of guardrail tools to avoid controversial topics.
        """
        return [self.create_tool(get_bad_topics)]

    def financial_tools(self):
        """
        Create a list of financial tools.
        """
        return self.get_llama_index_tools(tool_package_name="yahoo_finance", tool_spec_name="YahooFinanceToolSpec")

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

        return [self.create_tool(tool) for tool in [summarize_legal_text, critique_as_judge]]

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
        max_rows: int = 500,
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
            max_rows (int, optional): if specified, instructs the load_data tool to never return more than max_rows rows.
               Defaults to 500.

        Returns:
            List[VectaraTool]: A list of VectaraTool objects.
        """
        if sql_database:
            tools = self.get_llama_index_tools(
                tool_package_name="database",
                tool_spec_name="DatabaseToolSpec",
                tool_name_prefix=tool_name_prefix,
                sql_database=sql_database,
            )
        else:
            if scheme in ["postgresql", "mysql", "sqlite", "mssql", "oracle"]:
                tools = self.get_llama_index_tools(
                    tool_package_name="database",
                    tool_spec_name="DatabaseToolSpec",
                    tool_name_prefix=tool_name_prefix,
                    scheme=scheme,
                    host=host,
                    port=port,
                    user=user,
                    password=password,
                    dbname=dbname,
                )
            else:
                raise ValueError(
                    "Please provide a SqlDatabase option or a valid DB scheme type "
                    " (postgresql, mysql, sqlite, mssql, oracle)."
                )

        # Update tools with description
        for tool in tools:
            if content_description:
                tool.metadata.description = (
                    tool.metadata.description + f"The database tables include data about {content_description}."
                )

        # Add two new tools: load_sample_data and load_unique_values
        load_data_tool_index = next(i for i, t in enumerate(tools) if t.metadata.name.endswith("load_data"))
        load_data_fn_original = tools[load_data_tool_index].fn

        load_data_fn = DBLoadData(load_data_fn_original, max_rows=max_rows)
        load_data_fn.__name__ = f"{tool_name_prefix}_load_data"
        load_data_tool = self.create_tool(load_data_fn, ToolType.QUERY)

        sample_data_fn = DBLoadSampleData(load_data_fn_original)
        sample_data_fn.__name__ = f"{tool_name_prefix}_load_sample_data"
        sample_data_tool = self.create_tool(sample_data_fn, ToolType.QUERY)

        load_unique_values_fn = DBLoadUniqueValues(load_data_fn_original)
        load_unique_values_fn.__name__ = f"{tool_name_prefix}_load_unique_values"
        load_unique_values_tool = self.create_tool(load_unique_values_fn, ToolType.QUERY)

        tools[load_data_tool_index] = load_data_tool
        tools.extend([sample_data_tool, load_unique_values_tool])
        return tools
