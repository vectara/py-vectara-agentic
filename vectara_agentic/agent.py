"""
This module contains the Agent class for handling different types of agents and their interactions.
"""
from typing import List, Callable, Optional, Dict, Any, Union, Tuple
import os
import re
from datetime import date
import time
import json
import logging
import traceback
import asyncio

from collections import Counter

import cloudpickle as pickle

from dotenv import load_dotenv

from retrying import retry
from pydantic import Field, create_model

from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.agent.llm_compiler import LLMCompilerAgentWorker
from llama_index.agent.lats import LATSAgentWorker
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.agent.openai import OpenAIAgent


from .types import AgentType, AgentStatusType, LLMRole, ToolType, AgentResponse, AgentStreamingResponse
from .utils import get_llm, get_tokenizer_for_model
from ._prompts import REACT_PROMPT_TEMPLATE, GENERAL_PROMPT_TEMPLATE, GENERAL_INSTRUCTIONS
from ._callback import AgentCallbackHandler
from ._observability import setup_observer, eval_fcs
from .tools import VectaraToolFactory, VectaraTool, ToolsFactory
from .tools_catalog import get_current_date
from .agent_config import AgentConfig

class IgnoreUnpickleableAttributeFilter(logging.Filter):
    '''
    Filter to ignore log messages that contain certain strings
    '''
    def filter(self, record):
        msgs_to_ignore = [
            "Removing unpickleable private attribute _chunking_tokenizer_fn",
            "Removing unpickleable private attribute _split_fns",
            "Removing unpickleable private attribute _sub_sentence_split_fns",
        ]
        return all(msg not in record.getMessage() for msg in msgs_to_ignore)


logging.getLogger().addFilter(IgnoreUnpickleableAttributeFilter())

logger = logging.getLogger("opentelemetry.exporter.otlp.proto.http.trace_exporter")
logger.setLevel(logging.CRITICAL)

load_dotenv(override=True)

def _get_prompt(prompt_template: str, topic: str, custom_instructions: str):
    """
    Generate a prompt by replacing placeholders with topic and date.

    Args:
        prompt_template (str): The template for the prompt.
        topic (str): The topic to be included in the prompt.
        custom_instructions(str): The custom instructions to be included in the prompt.

    Returns:
        str: The formatted prompt.
    """
    return (
        prompt_template.replace("{chat_topic}", topic)
        .replace("{today}", date.today().strftime("%A, %B %d, %Y"))
        .replace("{custom_instructions}", custom_instructions)
    )


def _get_llm_compiler_prompt(prompt: str, topic: str, custom_instructions: str) -> str:
    """
    Add custom instructions to the prompt.

    Args:
        prompt (str): The prompt to which custom instructions should be added.

    Returns:
        str: The prompt with custom instructions added.
    """
    prompt += "\nAdditional Instructions:\n"
    prompt += f"You have experise in {topic}.\n"
    prompt += GENERAL_INSTRUCTIONS
    prompt += custom_instructions
    prompt += f"Today is {date.today().strftime('%A, %B %d, %Y')}"
    return prompt

def _retry_if_exception(exception):
    # Define the condition to retry on certain exceptions
    return isinstance(exception, (TimeoutError))


def get_field_type(field_schema: dict) -> Any:
    """
    Convert a JSON schema field definition to a Python type.
    Handles 'type' and 'anyOf' cases.
    """
    json_type_to_python = {
        "string": str,
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
        "number": float,
    }
    if "anyOf" in field_schema:
        types = []
        for option in field_schema["anyOf"]:
            # If the option has a type, convert it; otherwise, use Any.
            if "type" in option:
                types.append(json_type_to_python.get(option["type"], Any))
            else:
                types.append(Any)
        # Return a Union of the types. For example, Union[str, int]
        return Union[tuple(types)]
    elif "type" in field_schema:
        return json_type_to_python.get(field_schema["type"], Any)
    else:
        return Any

class Agent:
    """
    Agent class for handling different types of agents and their interactions.
    """

    def __init__(
        self,
        tools: list[FunctionTool],
        topic: str = "general",
        custom_instructions: str = "",
        verbose: bool = True,
        update_func: Optional[Callable[[AgentStatusType, str], None]] = None,
        agent_progress_callback: Optional[Callable[[AgentStatusType, str], None]] = None,
        query_logging_callback: Optional[Callable[[str, str], None]] = None,
        agent_config: Optional[AgentConfig] = None,
        chat_history: Optional[list[Tuple[str, str]]] = None,
        validate_tools: bool = False,
    ) -> None:
        """
        Initialize the agent with the specified type, tools, topic, and system message.

        Args:

            tools (list[FunctionTool]): A list of tools to be used by the agent.
            topic (str, optional): The topic for the agent. Defaults to 'general'.
            custom_instructions (str, optional): Custom instructions for the agent. Defaults to ''.
            verbose (bool, optional): Whether the agent should print its steps. Defaults to True.
            agent_progress_callback (Callable): A callback function the code calls on any agent updates.
                update_func (Callable): old name for agent_progress_callback. Will be deprecated in future.
            query_logging_callback (Callable): A callback function the code calls upon completion of a query
            agent_config (AgentConfig, optional): The configuration of the agent.
                Defaults to AgentConfig(), which reads from environment variables.
            chat_history (Tuple[str, str], optional): A list of user/agent chat pairs to initialize the agent memory.
            validate_tools (bool, optional): Whether to validate tool inconsistency with instructions.
                Defaults to False.
        """
        self.agent_config = agent_config or AgentConfig()
        self.agent_type = self.agent_config.agent_type
        self.tools = tools
        if not any(tool.metadata.name == 'get_current_date' for tool in self.tools):
            self.tools += [ToolsFactory().create_tool(get_current_date)]
        self.llm = get_llm(LLMRole.MAIN, config=self.agent_config)
        self._custom_instructions = custom_instructions
        self._topic = topic
        self.agent_progress_callback = agent_progress_callback if agent_progress_callback else update_func
        self.query_logging_callback = query_logging_callback

        # Validate tools
        # Check for:
        # 1. multiple copies of the same tool
        # 2. Instructions for using tools that do not exist
        tool_names = [tool.metadata.name for tool in self.tools]
        duplicates = [tool for tool, count in Counter(tool_names).items() if count > 1]
        if duplicates:
            raise ValueError(f"Duplicate tools detected: {', '.join(duplicates)}")

        if validate_tools:
            prompt = f'''
            Given the following instructions, and a list of tool names,
            Please identify tools mentioned in the instructions that do not exist in the list.
            Instructions:
            {self._custom_instructions}
            Tool names: {', '.join(tool_names)}
            Your response should include a comma separated list of tool names that do not exist in the list.
            Your response should be an empty string if all tools mentioned in the instructions are in the list.
            '''
            llm = get_llm(LLMRole.MAIN, config=self.agent_config)
            bad_tools = llm.complete(prompt).text.split(", ")
            if bad_tools:
                raise ValueError(f"The Agent custom instructions mention these invalid tools: {', '.join(bad_tools)}")

        # Create token counters for the main and tool LLMs
        main_tok = get_tokenizer_for_model(role=LLMRole.MAIN)
        self.main_token_counter = TokenCountingHandler(tokenizer=main_tok) if main_tok else None
        tool_tok = get_tokenizer_for_model(role=LLMRole.TOOL)
        self.tool_token_counter = TokenCountingHandler(tokenizer=tool_tok) if tool_tok else None

        # Setup callback manager
        callbacks: list[BaseCallbackHandler] = [AgentCallbackHandler(self.agent_progress_callback)]
        if self.main_token_counter:
            callbacks.append(self.main_token_counter)
        if self.tool_token_counter:
            callbacks.append(self.tool_token_counter)
        callback_manager = CallbackManager(callbacks)  # type: ignore
        self.llm.callback_manager = callback_manager
        self.verbose = verbose

        if chat_history:
            msg_history = []
            for text_pairs in chat_history:
                msg_history.append(ChatMessage.from_str(content=text_pairs[0], role=MessageRole.USER))
                msg_history.append(ChatMessage.from_str(content=text_pairs[1], role=MessageRole.ASSISTANT))
            self.memory = ChatMemoryBuffer.from_defaults(token_limit=128000, chat_history=msg_history)
        else:
            self.memory = ChatMemoryBuffer.from_defaults(token_limit=128000)
        if self.agent_type == AgentType.REACT:
            prompt = _get_prompt(REACT_PROMPT_TEMPLATE, topic, custom_instructions)
            self.agent = ReActAgent.from_tools(
                tools=self.tools,
                llm=self.llm,
                memory=self.memory,
                verbose=verbose,
                react_chat_formatter=ReActChatFormatter(system_header=prompt),
                max_iterations=self.agent_config.max_reasoning_steps,
                callable_manager=callback_manager,
            )
        elif self.agent_type == AgentType.OPENAI:
            prompt = _get_prompt(GENERAL_PROMPT_TEMPLATE, topic, custom_instructions)
            self.agent = OpenAIAgent.from_tools(
                tools=self.tools,
                llm=self.llm,
                memory=self.memory,
                verbose=verbose,
                callable_manager=callback_manager,
                max_function_calls=self.agent_config.max_reasoning_steps,
                system_prompt=prompt,
            )
        elif self.agent_type == AgentType.LLMCOMPILER:
            agent_worker = LLMCompilerAgentWorker.from_tools(
                tools=self.tools,
                llm=self.llm,
                verbose=verbose,
                callable_manager=callback_manager,
            )
            agent_worker.system_prompt = _get_prompt(
                _get_llm_compiler_prompt(agent_worker.system_prompt, topic, custom_instructions),
                topic, custom_instructions
            )
            agent_worker.system_prompt_replan = _get_prompt(
                _get_llm_compiler_prompt(agent_worker.system_prompt_replan, topic, custom_instructions),
                topic, custom_instructions
            )
            self.agent = agent_worker.as_agent()
        elif self.agent_type == AgentType.LATS:
            agent_worker = LATSAgentWorker.from_tools(
                tools=self.tools,
                llm=self.llm,
                num_expansions=3,
                max_rollouts=-1,
                verbose=verbose,
                callable_manager=callback_manager,
            )
            prompt = _get_prompt(REACT_PROMPT_TEMPLATE, topic, custom_instructions)
            agent_worker.chat_formatter = ReActChatFormatter(system_header=prompt)
            self.agent = agent_worker.as_agent()
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

        try:
            self.observability_enabled = setup_observer(self.agent_config)
        except Exception as e:
            print(f"Failed to set up observer ({e}), ignoring")
            self.observability_enabled = False

    def clear_memory(self) -> None:
        """
        Clear the agent's memory.
        """
        self.agent.memory.reset()

    def __eq__(self, other):
        if not isinstance(other, Agent):
            print(f"Comparison failed: other is not an instance of Agent. (self: {type(self)}, other: {type(other)})")
            return False

        # Compare agent_type
        if self.agent_type != other.agent_type:
            print(
                f"Comparison failed: agent_type differs. (self.agent_type: {self.agent_type}, "
                f"other.agent_type: {other.agent_type})"
            )
            return False

        # Compare tools
        if self.tools != other.tools:
            print(
                "Comparison failed: tools differ."
                f"(self.tools: {[t.metadata.name for t in self.tools]}, "
                f"other.tools: {[t.metadata.name for t in other.tools]})")
            return False

        # Compare topic
        if self._topic != other._topic:
            print(f"Comparison failed: topic differs. (self.topic: {self._topic}, other.topic: {other._topic})")
            return False

        # Compare custom_instructions
        if self._custom_instructions != other._custom_instructions:
            print(
                "Comparison failed: custom_instructions differ. (self.custom_instructions: "
                f"{self._custom_instructions}, other.custom_instructions: {other._custom_instructions})"
            )
            return False

        # Compare verbose
        if self.verbose != other.verbose:
            print(f"Comparison failed: verbose differs. (self.verbose: {self.verbose}, other.verbose: {other.verbose})")
            return False

        # Compare agent
        if self.agent.memory.chat_store != other.agent.memory.chat_store:
            print(
                f"Comparison failed: agent memory differs. (self.agent: {repr(self.agent.memory.chat_store)}, "
                f"other.agent: {repr(other.agent.memory.chat_store)})"
            )
            return False

        # If all comparisons pass
        print("All comparisons passed. Objects are equal.")
        return True

    @classmethod
    def from_tools(
        cls,
        tools: List[FunctionTool],
        topic: str = "general",
        custom_instructions: str = "",
        verbose: bool = True,
        update_func: Optional[Callable[[AgentStatusType, str], None]] = None,
        agent_progress_callback: Optional[Callable[[AgentStatusType, str], None]] = None,
        query_logging_callback: Optional[Callable[[str, str], None]] = None,
        agent_config: AgentConfig = AgentConfig(),
        chat_history: Optional[list[Tuple[str, str]]] = None,
    ) -> "Agent":
        """
        Create an agent from tools, agent type, and language model.

        Args:

            tools (list[FunctionTool]): A list of tools to be used by the agent.
            topic (str, optional): The topic for the agent. Defaults to 'general'.
            custom_instructions (str, optional): custom instructions for the agent. Defaults to ''.
            verbose (bool, optional): Whether the agent should print its steps. Defaults to True.
            agent_progress_callback (Callable): A callback function the code calls on any agent updates.
                update_func (Callable): old name for agent_progress_callback. Will be deprecated in future.
            query_logging_callback (Callable): A callback function the code calls upon completion of a query
            agent_config (AgentConfig, optional): The configuration of the agent.
            chat_history (Tuple[str, str], optional): A list of user/agent chat pairs to initialize the agent memory.

        Returns:
            Agent: An instance of the Agent class.
        """
        return cls(
            tools=tools, topic=topic, custom_instructions=custom_instructions,
            verbose=verbose, agent_progress_callback=agent_progress_callback,
            query_logging_callback=query_logging_callback,
            update_func=update_func, agent_config=agent_config,
            chat_history=chat_history,
        )

    @classmethod
    def from_corpus(
        cls,
        tool_name: str,
        data_description: str,
        assistant_specialty: str,
        vectara_corpus_key: str = str(os.environ.get("VECTARA_CORPUS_KEY", "")),
        vectara_api_key: str = str(os.environ.get("VECTARA_API_KEY", "")),
        agent_progress_callback: Optional[Callable[[AgentStatusType, str], None]] = None,
        query_logging_callback: Optional[Callable[[str, str], None]] = None,
        verbose: bool = False,
        vectara_filter_fields: list[dict] = [],
        vectara_offset: int = 0,
        vectara_lambda_val: float = 0.005,
        vectara_semantics: str = "default",
        vectara_custom_dimensions: Dict = {},
        vectara_reranker: str = "slingshot",
        vectara_rerank_k: int = 50,
        vectara_rerank_limit: Optional[int] = None,
        vectara_rerank_cutoff: Optional[float] = None,
        vectara_diversity_bias: float = 0.2,
        vectara_udf_expression: str = None,
        vectara_rerank_chain: List[Dict] = None,
        vectara_n_sentences_before: int = 2,
        vectara_n_sentences_after: int = 2,
        vectara_summary_num_results: int = 10,
        vectara_summarizer: str = "vectara-summary-ext-24-05-med-omni",
        vectara_summary_response_language: str = "eng",
        vectara_summary_prompt_text: Optional[str] = None,
        vectara_max_response_chars: Optional[int] = None,
        vectara_max_tokens: Optional[int] = None,
        vectara_temperature: Optional[float] = None,
        vectara_frequency_penalty: Optional[float] = None,
        vectara_presence_penalty: Optional[float] = None,
        vectara_save_history: bool = True,
    ) -> "Agent":
        """
        Create an agent from a single Vectara corpus

        Args:
            tool_name (str): The name of Vectara tool used by the agent
            vectara_corpus_key (str): The Vectara corpus key (or comma separated list of keys).
            vectara_api_key (str): The Vectara API key.
            agent_progress_callback (Callable): A callback function the code calls on any agent updates.
            query_logging_callback (Callable): A callback function the code calls upon completion of a query
            data_description (str): The description of the data.
            assistant_specialty (str): The specialty of the assistant.
            verbose (bool, optional): Whether to print verbose output.
            vectara_filter_fields (List[dict], optional): The filterable attributes
                (each dict maps field name to Tuple[type, description]).
            vectara_offset (int, optional): Number of results to skip.
            vectara_lambda_val (float, optional): Lambda value for Vectara hybrid search.
            vectara_semantics: (str, optional): Indicates whether the query is intended as a query or response.
            vectara_custom_dimensions: (Dict, optional): Custom dimensions for the query.
            vectara_reranker (str, optional): The Vectara reranker name (default "slingshot")
            vectara_rerank_k (int, optional): The number of results to use with reranking.
            vectara_rerank_limit: (int, optional): The maximum number of results to return after reranking.
            vectara_rerank_cutoff: (float, optional): The minimum score threshold for results to include after
                reranking.
            vectara_diversity_bias (float, optional): The MMR diversity bias.
            vectara_udf_expression (str, optional): The user defined expression for reranking results.
            vectara_rerank_chain (List[Dict], optional): A list of Vectara rerankers to be applied sequentially.
            vectara_n_sentences_before (int, optional): The number of sentences before the matching text
            vectara_n_sentences_after (int, optional): The number of sentences after the matching text.
            vectara_summary_num_results (int, optional): The number of results to use in summarization.
            vectara_summarizer (str, optional): The Vectara summarizer name.
            vectara_summary_response_language (str, optional): The response language for the Vectara summary.
            vectara_summary_prompt_text (str, optional): The custom prompt, using appropriate prompt variables and
                functions.
            vectara_max_response_chars (int, optional): The desired maximum number of characters for the generated
                summary.
            vectara_max_tokens (int, optional): The maximum number of tokens to be returned by the LLM.
            vectara_temperature (float, optional): The sampling temperature; higher values lead to more randomness.
            vectara_frequency_penalty (float, optional): How much to penalize repeating tokens in the response,
                higher values reducing likelihood of repeating the same line.
            vectara_presence_penalty (float, optional): How much to penalize repeating tokens in the response,
                higher values increasing the diversity of topics.
            vectara_save_history (bool, optional): Whether to save the query in history.

        Returns:
            Agent: An instance of the Agent class.
        """
        vec_factory = VectaraToolFactory(
            vectara_api_key=vectara_api_key,
            vectara_corpus_key=vectara_corpus_key,
        )
        field_definitions = {}
        field_definitions["query"] = (str, Field(description="The user query"))  # type: ignore
        for field in vectara_filter_fields:
            field_definitions[field["name"]] = (
                eval(field["type"]),
                Field(description=field["description"]),
            )  # type: ignore
        query_args = create_model("QueryArgs", **field_definitions)  # type: ignore

        # tool name must be valid Python function name
        if tool_name:
            tool_name = re.sub(r"[^A-Za-z0-9_]", "_", tool_name)

        vectara_tool = vec_factory.create_rag_tool(
            tool_name=tool_name or f"vectara_{vectara_corpus_key}",
            tool_description=f"""
            Given a user query,
            returns a response (str) to a user question about {data_description}.
            """,
            tool_args_schema=query_args,
            reranker=vectara_reranker,
            rerank_k=vectara_rerank_k,
            rerank_limit=vectara_rerank_limit,
            rerank_cutoff=vectara_rerank_cutoff,
            mmr_diversity_bias=vectara_diversity_bias,
            udf_expression=vectara_udf_expression,
            rerank_chain=vectara_rerank_chain,
            n_sentences_before=vectara_n_sentences_before,
            n_sentences_after=vectara_n_sentences_after,
            offset=vectara_offset,
            lambda_val=vectara_lambda_val,
            semantics=vectara_semantics,
            custom_dimensions=vectara_custom_dimensions,
            summary_num_results=vectara_summary_num_results,
            vectara_summarizer=vectara_summarizer,
            summary_response_lang=vectara_summary_response_language,
            vectara_prompt_text=vectara_summary_prompt_text,
            max_response_chars=vectara_max_response_chars,
            max_tokens=vectara_max_tokens,
            temperature=vectara_temperature,
            frequency_penalty=vectara_frequency_penalty,
            presence_penalty=vectara_presence_penalty,
            save_history=vectara_save_history,
            include_citations=True,
            verbose=verbose,
        )

        assistant_instructions = f"""
        - You are a helpful {assistant_specialty} assistant.
        - You can answer questions about {data_description}.
        - Never discuss politics, and always respond politely.
        """

        return cls(
            tools=[vectara_tool],
            topic=assistant_specialty,
            custom_instructions=assistant_instructions,
            verbose=verbose,
            agent_progress_callback=agent_progress_callback,
            query_logging_callback=query_logging_callback,
        )

    def report(self) -> None:
        """
        Get a report from the agent.

        Returns:
            str: The report from the agent.
        """
        print("Vectara agentic Report:")
        print(f"Agent Type = {self.agent_type}")
        print(f"Topic = {self._topic}")
        print("Tools:")
        for tool in self.tools:
            if hasattr(tool, 'metadata'):
                print(f"- {tool.metadata.name}")
            else:
                print("- tool without metadata")
        print(f"Agent LLM = {get_llm(LLMRole.MAIN, config=self.agent_config).metadata.model_name}")
        print(f"Tool LLM = {get_llm(LLMRole.TOOL, config=self.agent_config).metadata.model_name}")

    def token_counts(self) -> dict:
        """
        Get the token counts for the agent and tools.

        Returns:
            dict: The token counts for the agent and tools.
        """
        return {
            "main token count": self.main_token_counter.total_llm_token_count if self.main_token_counter else -1,
            "tool token count": self.tool_token_counter.total_llm_token_count if self.tool_token_counter else -1,
        }

    async def _aformat_for_lats(self, prompt, agent_response):
        llm_prompt = f"""
        Given the question '{prompt}', and agent response '{agent_response.response}',
        Please provide a well formatted final response to the query.
        final response:
        """
        agent_response.response = str(self.llm.acomplete(llm_prompt))

    def chat(self, prompt: str) -> AgentResponse:           # type: ignore
        """
        Interact with the agent using a chat prompt.

        Args:
            prompt (str): The chat prompt.

        Returns:
            AgentResponse: The response from the agent.
        """
        return asyncio.run(self.achat(prompt))

    @retry(
        retry_on_exception=_retry_if_exception,
        stop_max_attempt_number=3,
        wait_fixed=2000,
    )
    async def achat(self, prompt: str) -> AgentResponse:    # type: ignore
        """
        Interact with the agent using a chat prompt.

        Args:
            prompt (str): The chat prompt.

        Returns:
            AgentResponse: The response from the agent.
        """

        try:
            st = time.time()
            agent_response = await self.agent.achat(prompt)
            if self.agent_type == AgentType.LATS:
                await self._aformat_for_lats(prompt, agent_response)
            if self.verbose:
                print(f"Time taken: {time.time() - st}")
            if self.observability_enabled:
                eval_fcs()
            if self.query_logging_callback:
                self.query_logging_callback(prompt, agent_response.response)
            return agent_response
        except Exception as e:
            return AgentResponse(
                response = (
                    f"Vectara Agentic: encountered an exception ({e}) at ({traceback.format_exc()})"
                    ", and can't respond."
                )
            )

    def stream_chat(self, prompt: str) -> AgentStreamingResponse:    # type: ignore
        """
        Interact with the agent using a chat prompt with streaming.
        Args:
            prompt (str): The chat prompt.
        Returns:
            AgentStreamingResponse: The streaming response from the agent.
        """
        return asyncio.run(self.astream_chat(prompt))

    @retry(
        retry_on_exception=_retry_if_exception,
        stop_max_attempt_number=3,
        wait_fixed=2000,
    )
    async def astream_chat(self, prompt: str) -> AgentStreamingResponse:    # type: ignore
        """
        Interact with the agent using a chat prompt asynchronously with streaming.
        Args:
            prompt (str): The chat prompt.
        Returns:
            AgentStreamingResponse: The streaming response from the agent.
        """
        try:
            agent_response = await self.agent.astream_chat(prompt)
            original_async_response_gen = agent_response.async_response_gen

            # Wrap async_response_gen
            async def _stream_response_wrapper():
                async for token in original_async_response_gen():
                    yield token  # Yield async token to keep streaming behavior

                # After streaming completes, execute additional logic
                if self.agent_type == AgentType.LATS:
                    await self._aformat_for_lats(prompt, agent_response)
                if self.query_logging_callback:
                    self.query_logging_callback(prompt, agent_response.response)
                if self.observability_enabled:
                    eval_fcs()

            agent_response.async_response_gen = _stream_response_wrapper  # Override method
            return agent_response
        except Exception as e:
            raise ValueError(
                f"Vectara Agentic: encountered an exception ({e}) at ({traceback.format_exc()}), and can't respond."
            ) from e

    #
    # Serialization methods
    #
    def dumps(self) -> str:
        """Serialize the Agent instance to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def loads(cls, data: str) -> "Agent":
        """Create an Agent instance from a JSON string."""
        return cls.from_dict(json.loads(data))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the Agent instance to a dictionary."""
        tool_info = []

        for tool in self.tools:
            # Serialize each tool's metadata, function, and dynamic model schema (QueryArgs)
            tool_dict = {
                "tool_type": tool.metadata.tool_type.value,
                "name": tool.metadata.name,
                "description": tool.metadata.description,
                "fn": pickle.dumps(tool.fn).decode("latin-1") if tool.fn else None,  # Serialize fn
                "async_fn": pickle.dumps(tool.async_fn).decode("latin-1")
                if tool.async_fn
                else None,  # Serialize async_fn
                "fn_schema": tool.metadata.fn_schema.model_json_schema()
                if hasattr(tool.metadata, "fn_schema")
                else None,  # Serialize schema if available
            }
            tool_info.append(tool_dict)

        return {
            "agent_type": self.agent_type.value,
            "memory": pickle.dumps(self.agent.memory).decode("latin-1"),
            "tools": tool_info,
            "topic": self._topic,
            "custom_instructions": self._custom_instructions,
            "verbose": self.verbose,
            "agent_config": self.agent_config.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Agent":
        """Create an Agent instance from a dictionary."""
        agent_config = AgentConfig.from_dict(data["agent_config"])
        tools = []

        for tool_data in data["tools"]:
            # Recreate the dynamic model using the schema info
            if tool_data.get("fn_schema"):
                field_definitions = {}
                for field, values in tool_data["fn_schema"]["properties"].items():
                    # Instead of checking for 'type', use the helper:
                    field_type = get_field_type(values)
                    # If there's a default value, include it.
                    if "default" in values:
                        field_definitions[field] = (
                            field_type,
                            Field(description=values.get("description", ""), default=values["default"]),
                        )
                    else:
                        field_definitions[field] = (
                            field_type,
                            Field(description=values.get("description", "")),
                        )
                query_args_model = create_model("QueryArgs", **field_definitions)  # type: ignore
            else:
                query_args_model = create_model("QueryArgs")

            fn = pickle.loads(tool_data["fn"].encode("latin-1")) if tool_data["fn"] else None
            async_fn = pickle.loads(tool_data["async_fn"].encode("latin-1")) if tool_data["async_fn"] else None

            tool = VectaraTool.from_defaults(
                name=tool_data["name"],
                description=tool_data["description"],
                fn=fn,
                async_fn=async_fn,
                fn_schema=query_args_model,  # Re-assign the recreated dynamic model
                tool_type=ToolType(tool_data["tool_type"]),
            )
            tools.append(tool)

        agent = cls(
            tools=tools,
            agent_config=agent_config,
            topic=data["topic"],
            custom_instructions=data["custom_instructions"],
            verbose=data["verbose"],
        )
        memory = pickle.loads(data["memory"].encode("latin-1")) if data.get("memory") else None
        if memory:
            agent.agent.memory = memory
        return agent
