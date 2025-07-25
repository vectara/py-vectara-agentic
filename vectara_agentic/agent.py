"""
This module contains the Agent class for handling different types of agents and their interactions.
"""

import warnings

# Suppress external Pydantic v2 deprecation warnings to reduce noise
# Global suppression for all deprecation warnings since they come from external dependencies
warnings.simplefilter("ignore", DeprecationWarning)

# pylint: disable=wrong-import-position
from typing import List, Callable, Optional, Dict, Any, Union, Tuple, TYPE_CHECKING
import os
from datetime import date
import json
import logging
import asyncio
from collections import Counter
import uuid

from pydantic import ValidationError
from pydantic_core import PydanticUndefined

from dotenv import load_dotenv

# Runtime imports for components used at module level
from llama_index.core.llms import MessageRole

# Heavy llama_index imports moved to TYPE_CHECKING for lazy loading
if TYPE_CHECKING:
    from llama_index.core.tools import FunctionTool
    from llama_index.core.workflow import Workflow
    from llama_index.core.callbacks import CallbackManager
    from llama_index.core.agent.runner.base import AgentRunner
    from llama_index.core.agent.types import BaseAgent

from .types import (
    AgentType,
    AgentStatusType,
    LLMRole,
    ModelProvider,
    AgentResponse,
    AgentStreamingResponse,
    AgentConfigType,
)
from .llm_utils import get_llm
from .agent_core.prompts import GENERAL_INSTRUCTIONS
from ._callback import AgentCallbackHandler
from ._observability import setup_observer
from .tools import ToolsFactory
from .tool_utils import _is_human_readable_output
from .tools_catalog import get_current_date
from .agent_config import AgentConfig
from .hhem import HHEM

# Import utilities from agent core modules
from .agent_core.streaming import (
    FunctionCallingStreamHandler,
    execute_post_stream_processing,
)
from .agent_core.factory import create_agent_from_config, create_agent_from_corpus
from .agent_core.serialization import (
    serialize_agent_to_dict,
    deserialize_agent_from_dict,
)
from .agent_core.utils import (
    sanitize_tools_for_gemini,
    setup_agent_logging,
)

setup_agent_logging()

logger = logging.getLogger("opentelemetry.exporter.otlp.proto.http.trace_exporter")
logger.setLevel(logging.CRITICAL)

load_dotenv(override=True)


class Agent:
    """
    Agent class for handling different types of agents and their interactions.
    """

    def __init__(
        self,
        tools: List["FunctionTool"],
        topic: str = "general",
        custom_instructions: str = "",
        general_instructions: str = GENERAL_INSTRUCTIONS,
        verbose: bool = False,
        agent_progress_callback: Optional[
            Callable[[AgentStatusType, dict, str], None]
        ] = None,
        query_logging_callback: Optional[Callable[[str, str], None]] = None,
        agent_config: Optional[AgentConfig] = None,
        fallback_agent_config: Optional[AgentConfig] = None,
        chat_history: Optional[list[Tuple[str, str]]] = None,
        validate_tools: bool = False,
        workflow_cls: Optional["Workflow"] = None,
        workflow_timeout: int = 120,
        vectara_api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize the agent with the specified type, tools, topic, and system message.

        Args:

            tools (list[FunctionTool]): A list of tools to be used by the agent.
            topic (str, optional): The topic for the agent. Defaults to 'general'.
            custom_instructions (str, optional): Custom instructions for the agent. Defaults to ''.
            general_instructions (str, optional): General instructions for the agent.
                The Agent has a default set of instructions that are crafted to help it operate effectively.
                This allows you to customize the agent's behavior and personality, but use with caution.
            verbose (bool, optional): Whether the agent should print its steps. Defaults to True.
            agent_progress_callback (Callable): A callback function the code calls on any agent updates.
            query_logging_callback (Callable): A callback function the code calls upon completion of a query
            agent_config (AgentConfig, optional): The configuration of the agent.
                Defaults to AgentConfig(), which reads from environment variables.
            fallback_agent_config (AgentConfig, optional): The fallback configuration of the agent.
                This config is used when the main agent config fails multiple times.
            chat_history (Tuple[str, str], optional): A list of user/agent chat pairs to initialize the agent memory.
            validate_tools (bool, optional): Whether to validate tool inconsistency with instructions.
                Defaults to False.
            workflow_cls (Workflow, optional): The workflow class to be used with run(). Defaults to None.
            workflow_timeout (int, optional): The timeout for the workflow in seconds. Defaults to 120.
            vectara_api_key (str, optional): The Vectara API key for FCS evaluation. Defaults to None.
        """
        self.agent_config = agent_config or AgentConfig()
        self.agent_config_type = AgentConfigType.DEFAULT
        self.tools = tools
        if not any(tool.metadata.name == "get_current_date" for tool in self.tools):
            self.tools += [ToolsFactory().create_tool(get_current_date)]
        self.agent_type = self.agent_config.agent_type
        self._llm = None  # Lazy loading
        self._custom_instructions = custom_instructions
        self._general_instructions = general_instructions
        self._topic = topic
        self.agent_progress_callback = agent_progress_callback

        self.query_logging_callback = query_logging_callback
        self.workflow_cls = workflow_cls
        self.workflow_timeout = workflow_timeout
        self.vectara_api_key = vectara_api_key or os.environ.get("VECTARA_API_KEY", "")

        # Sanitize tools for Gemini if needed
        if self.agent_config.main_llm_provider == ModelProvider.GEMINI:
            self.tools = sanitize_tools_for_gemini(self.tools)

        # Validate tools
        # Check for:
        # 1. multiple copies of the same tool
        # 2. Instructions for using tools that do not exist
        tool_names = [tool.metadata.name for tool in self.tools]
        duplicates = [tool for tool, count in Counter(tool_names).items() if count > 1]
        if duplicates:
            raise ValueError(f"Duplicate tools detected: {', '.join(duplicates)}")

        if validate_tools:
            # pylint: disable=duplicate-code
            prompt = f"""
            You are provided these tools:
            <tools>{','.join(tool_names)}</tools>
            And these instructions:
            <instructions>
            {self._custom_instructions}
            </instructions>
            Your task is to identify invalid tools.
            A tool is invalid if it is mentioned in the instructions but not in the tools list.
            A tool's name must have at least two characters.
            Your response should be a comma-separated list of the invalid tools.
            If no invalid tools exist, respond with "<OKAY>" (and nothing else).
            """
            llm = get_llm(LLMRole.MAIN, config=self.agent_config)
            bad_tools_str = llm.complete(prompt).text.strip("\n")
            if bad_tools_str and bad_tools_str != "<OKAY>":
                bad_tools = [tool.strip() for tool in bad_tools_str.split(",")]
                numbered = ", ".join(
                    f"({i}) {tool}" for i, tool in enumerate(bad_tools, 1)
                )
                raise ValueError(
                    f"The Agent custom instructions mention these invalid tools: {numbered}"
                )

        # Setup callback manager
        from llama_index.core.callbacks.base_handler import BaseCallbackHandler
        from llama_index.core.callbacks import CallbackManager

        callbacks: list[BaseCallbackHandler] = [
            AgentCallbackHandler(self.agent_progress_callback)
        ]
        self.callback_manager = CallbackManager(callbacks)  # type: ignore
        self.verbose = verbose

        self.session_id = (
            getattr(self, "session_id", None) or f"{topic}:{date.today().isoformat()}"
        )
        from llama_index.core.memory import Memory

        self.memory = Memory.from_defaults(
            session_id=self.session_id, token_limit=65536
        )
        if chat_history:
            from llama_index.core.llms import ChatMessage

            msgs = []
            for u, a in chat_history:
                msgs.append(ChatMessage.from_str(u, role=MessageRole.USER))
                msgs.append(ChatMessage.from_str(a, role=MessageRole.ASSISTANT))
            self.memory.put_messages(msgs)

        # Set up main agent and fallback agent
        self._agent = None  # Lazy loading
        self.fallback_agent_config = fallback_agent_config
        self._fallback_agent = None  # Lazy loading

        # Setup observability
        try:
            self.observability_enabled = setup_observer(self.agent_config, self.verbose)
        except Exception as e:
            print(f"Failed to set up observer ({e}), ignoring")
            self.observability_enabled = False

    @property
    def llm(self):
        """Lazy-loads the LLM."""
        if self._llm is None:
            self._llm = get_llm(LLMRole.MAIN, config=self.agent_config)
        return self._llm

    @property
    def agent(self):
        """Lazy-loads the agent."""
        if self._agent is None:
            self._agent = self._create_agent(self.agent_config, self.callback_manager)
        return self._agent

    @property
    def fallback_agent(self):
        """Lazy-loads the fallback agent."""
        if self._fallback_agent is None and self.fallback_agent_config:
            self._fallback_agent = self._create_agent(
                self.fallback_agent_config, self.callback_manager
            )
        return self._fallback_agent

    def _create_agent(
        self, config: AgentConfig, llm_callback_manager: "CallbackManager"
    ) -> Union["BaseAgent", "AgentRunner"]:
        """
        Creates the agent based on the configuration object.

        Args:
            config: The configuration of the agent.
            llm_callback_manager: The callback manager for the agent's llm.

        Returns:
            Union[BaseAgent, AgentRunner]: The configured agent object.
        """
        # Use the same LLM instance for consistency
        llm = (
            self.llm
            if config == self.agent_config
            else get_llm(LLMRole.MAIN, config=config)
        )
        llm.callback_manager = llm_callback_manager

        return create_agent_from_config(
            tools=self.tools,
            llm=llm,
            memory=self.memory,
            config=config,
            callback_manager=llm_callback_manager,
            general_instructions=self._general_instructions,
            topic=self._topic,
            custom_instructions=self._custom_instructions,
            verbose=self.verbose,
        )

    def clear_memory(self) -> None:
        """Clear the agent's memory."""
        self.memory.reset()
        if getattr(self, "_agent", None):
            self._agent.memory = self.memory
        if getattr(self, "_fallback_agent", None):
            self._fallback_agent.memory = self.memory

    def __eq__(self, other):
        if not isinstance(other, Agent):
            print(
                f"Comparison failed: other is not an instance of Agent. (self: {type(self)}, other: {type(other)})"
            )
            return False

        # Compare agent_type
        if self.agent_config.agent_type != other.agent_config.agent_type:
            print(
                f"Comparison failed: agent_type differs. (self.agent_config.agent_type: {self.agent_config.agent_type},"
                f" other.agent_config.agent_type: {other.agent_config.agent_type})"
            )
            return False

        # Compare tools
        if self.tools != other.tools:
            print(
                "Comparison failed: tools differ."
                f"(self.tools: {[t.metadata.name for t in self.tools]}, "
                f"other.tools: {[t.metadata.name for t in other.tools]})"
            )
            return False

        # Compare topic
        if self._topic != other._topic:
            print(
                f"Comparison failed: topic differs. (self.topic: {self._topic}, other.topic: {other._topic})"
            )
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
            print(
                f"Comparison failed: verbose differs. (self.verbose: {self.verbose}, other.verbose: {other.verbose})"
            )
            return False

        # Compare agent memory
        if self.memory.get() != other.memory.get():
            print("Comparison failed: agent memory differs.")
            return False

        # If all comparisons pass
        print("All comparisons passed. Objects are equal.")
        return True

    @classmethod
    def from_tools(
        cls,
        tools: List["FunctionTool"],
        topic: str = "general",
        custom_instructions: str = "",
        verbose: bool = True,
        agent_progress_callback: Optional[
            Callable[[AgentStatusType, dict, str], None]
        ] = None,
        query_logging_callback: Optional[Callable[[str, str], None]] = None,
        agent_config: AgentConfig = AgentConfig(),
        validate_tools: bool = False,
        fallback_agent_config: Optional[AgentConfig] = None,
        chat_history: Optional[list[Tuple[str, str]]] = None,
        workflow_cls: Optional["Workflow"] = None,
        workflow_timeout: int = 120,
    ) -> "Agent":
        """
        Create an agent from tools, agent type, and language model.

        Args:

            tools (list[FunctionTool]): A list of tools to be used by the agent.
            topic (str, optional): The topic for the agent. Defaults to 'general'.
            custom_instructions (str, optional): custom instructions for the agent. Defaults to ''.
            verbose (bool, optional): Whether the agent should print its steps. Defaults to True.
            agent_progress_callback (Callable): A callback function the code calls on any agent updates.
            query_logging_callback (Callable): A callback function the code calls upon completion of a query
            agent_config (AgentConfig, optional): The configuration of the agent.
            fallback_agent_config (AgentConfig, optional): The fallback configuration of the agent.
            chat_history (Tuple[str, str], optional): A list of user/agent chat pairs to initialize the agent memory.
            validate_tools (bool, optional): Whether to validate tool inconsistency with instructions.
                Defaults to False.
            workflow_cls (Workflow, optional): The workflow class to be used with run(). Defaults to None.
            workflow_timeout (int, optional): The timeout for the workflow in seconds. Defaults to 120.

        Returns:
            Agent: An instance of the Agent class.
        """
        return cls(
            tools=tools,
            topic=topic,
            custom_instructions=custom_instructions,
            verbose=verbose,
            agent_progress_callback=agent_progress_callback,
            query_logging_callback=query_logging_callback,
            agent_config=agent_config,
            chat_history=chat_history,
            validate_tools=validate_tools,
            fallback_agent_config=fallback_agent_config,
            workflow_cls=workflow_cls,
            workflow_timeout=workflow_timeout,
        )

    @classmethod
    def from_corpus(
        cls,
        tool_name: str,
        data_description: str,
        assistant_specialty: str,
        general_instructions: str = GENERAL_INSTRUCTIONS,
        vectara_corpus_key: str = str(os.environ.get("VECTARA_CORPUS_KEY", "")),
        vectara_api_key: str = str(os.environ.get("VECTARA_API_KEY", "")),
        agent_progress_callback: Optional[
            Callable[[AgentStatusType, dict, str], None]
        ] = None,
        query_logging_callback: Optional[Callable[[str, str], None]] = None,
        agent_config: AgentConfig = AgentConfig(),
        fallback_agent_config: Optional[AgentConfig] = None,
        chat_history: Optional[list[Tuple[str, str]]] = None,
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
        vectara_udf_expression: Optional[str] = None,
        vectara_rerank_chain: Optional[List[Dict]] = None,
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
        return_direct: bool = False,
    ) -> "Agent":
        """Create an agent from a single Vectara corpus using the factory function."""
        # Use the factory function to avoid code duplication
        config = create_agent_from_corpus(
            tool_name=tool_name,
            data_description=data_description,
            assistant_specialty=assistant_specialty,
            general_instructions=general_instructions,
            vectara_corpus_key=vectara_corpus_key,
            vectara_api_key=vectara_api_key,
            agent_config=agent_config,
            fallback_agent_config=fallback_agent_config,
            verbose=verbose,
            vectara_filter_fields=vectara_filter_fields,
            vectara_offset=vectara_offset,
            vectara_lambda_val=vectara_lambda_val,
            vectara_semantics=vectara_semantics,
            vectara_custom_dimensions=vectara_custom_dimensions,
            vectara_reranker=vectara_reranker,
            vectara_rerank_k=vectara_rerank_k,
            vectara_rerank_limit=vectara_rerank_limit,
            vectara_rerank_cutoff=vectara_rerank_cutoff,
            vectara_diversity_bias=vectara_diversity_bias,
            vectara_udf_expression=vectara_udf_expression,
            vectara_rerank_chain=vectara_rerank_chain,
            vectara_n_sentences_before=vectara_n_sentences_before,
            vectara_n_sentences_after=vectara_n_sentences_after,
            vectara_summary_num_results=vectara_summary_num_results,
            vectara_summarizer=vectara_summarizer,
            vectara_summary_response_language=vectara_summary_response_language,
            vectara_summary_prompt_text=vectara_summary_prompt_text,
            vectara_max_response_chars=vectara_max_response_chars,
            vectara_max_tokens=vectara_max_tokens,
            vectara_temperature=vectara_temperature,
            vectara_frequency_penalty=vectara_frequency_penalty,
            vectara_presence_penalty=vectara_presence_penalty,
            vectara_save_history=vectara_save_history,
            return_direct=return_direct,
        )

        return cls(
            chat_history=chat_history,
            agent_progress_callback=agent_progress_callback,
            query_logging_callback=query_logging_callback,
            **config,
        )

    def _switch_agent_config(self) -> None:
        """
        Switch the configuration type of the agent.
        This function is called automatically to switch the agent configuration if the current configuration fails.
        """
        if self.agent_config_type == AgentConfigType.DEFAULT:
            self.agent_config_type = AgentConfigType.FALLBACK
        else:
            self.agent_config_type = AgentConfigType.DEFAULT

    def report(self, detailed: bool = False) -> None:
        """
        Get a report from the agent.

        Args:
            detailed (bool, optional): Whether to include detailed information. Defaults to False.

        Returns:
            str: The report from the agent.
        """
        print("Vectara agentic Report:")
        print(f"Agent Type = {self.agent_config.agent_type}")
        print(f"Topic = {self._topic}")
        print("Tools:")
        for tool in self.tools:
            if hasattr(tool, "metadata"):
                if detailed:
                    print(f"- {tool.metadata.description}")
                else:
                    print(f"- {tool.metadata.name}")
            else:
                print("- tool without metadata")
        print(
            f"Agent LLM = {get_llm(LLMRole.MAIN, config=self.agent_config).metadata.model_name}"
        )
        print(
            f"Tool LLM = {get_llm(LLMRole.TOOL, config=self.agent_config).metadata.model_name}"
        )

    def _get_current_agent(self):
        return (
            self.agent
            if self.agent_config_type == AgentConfigType.DEFAULT
            else self.fallback_agent
        )

    def _get_current_agent_type(self):
        return (
            self.agent_config.agent_type
            if self.agent_config_type == AgentConfigType.DEFAULT
            or not self.fallback_agent_config
            else self.fallback_agent_config.agent_type
        )

    async def _aformat_for_lats(self, prompt, agent_response):
        llm_prompt = f"""
        Given the question '{prompt}', and agent response '{agent_response.response}',
        Please provide a well formatted final response to the query.
        final response:
        """
        agent_type = self._get_current_agent_type()
        if agent_type != AgentType.LATS:
            return

        agent = self._get_current_agent()
        agent_response.response = (await agent.llm.acomplete(llm_prompt)).text

    def _calc_fcs(self, agent_response: str) -> float | None:
        """
        Calculate the Factual consistency score for the agent response.
        """
        if not self.vectara_api_key:
            logging.debug("FCS calculation skipped: 'vectara_api_key' is missing.")
            return None  # can't calculate FCS without Vectara API key

        import time

        hhem_start = time.time()
        chat_history = self.memory.get()
        context = []
        num_tool_calls = 0
        for msg in chat_history:
            if msg.role == MessageRole.TOOL:
                num_tool_calls += 1
                content = msg.content
                if _is_human_readable_output(content):
                    try:
                        content = content.to_human_readable()
                    except Exception as e:
                        logging.debug(
                            f"Failed to get human-readable format for FCS calculation: {e}"
                        )
                        # Fall back to string representation of the object
                        content = str(content)

                context.append(content)
            elif msg.role in [MessageRole.USER, MessageRole.ASSISTANT] and msg.content:
                context.append(msg.content)

        if not context or num_tool_calls == 0:
            return None

        context_str = "\n".join(context)
        try:
            score = HHEM(self.vectara_api_key).compute(context_str, agent_response)
            if self.verbose:
                logging.info(
                    f"FCS calculated: {score} (took {time.time() - hhem_start:.2f} seconds)"
                )
            return score
        except Exception as e:
            logging.error(f"Failed to calculate FCS: {e}")
            return None

    def chat(self, prompt: str) -> AgentResponse:
        """
        Interact with the agent using a chat prompt.

        Args:
            prompt (str): The chat prompt.

        Returns:
            AgentResponse: The response from the agent.
        """
        try:
            _ = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.achat(prompt))

        # We are inside a running loop (Jupyter, uvicorn, etc.)
        raise RuntimeError(
            "Use `await agent.achat(...)` inside an event loop (e.g. Jupyter)."
        )

    async def achat(self, prompt: str) -> AgentResponse:  # type: ignore
        """
        Interact with the agent using a chat prompt.

        Args:
            prompt (str): The chat prompt.

        Returns:
            AgentResponse: The response from the agent.
        """
        if not prompt:
            return AgentResponse(response="")

        max_attempts = 4 if self.fallback_agent_config else 2
        attempt = 0
        orig_llm = self.llm.metadata.model_name
        last_error = None
        while attempt < max_attempts:
            try:
                current_agent = self._get_current_agent()

                # Deal with workflow-based agent types (Function Calling and ReAct)
                if self._get_current_agent_type() in [
                    AgentType.FUNCTION_CALLING,
                    AgentType.REACT,
                ]:
                    from llama_index.core.workflow import Context

                    ctx = Context(current_agent)
                    handler = current_agent.run(
                        user_msg=prompt, ctx=ctx, memory=self.memory
                    )

                    # Listen to workflow events if progress callback is set
                    if self.agent_progress_callback:
                        async for event in handler.stream_events():
                            event_id = str(uuid.uuid4())

                            # Handle different types of workflow events using same logic as FunctionCallingStreamHandler
                            from llama_index.core.agent.workflow import (
                                ToolCall,
                                ToolCallResult,
                                AgentInput,
                                AgentOutput,
                            )

                            if isinstance(event, ToolCall):
                                self.agent_progress_callback(
                                    status_type=AgentStatusType.TOOL_CALL,
                                    msg={
                                        "tool_name": event.tool_name,
                                        "arguments": json.dumps(event.tool_kwargs),
                                    },
                                    event_id=event_id,
                                )
                            elif isinstance(event, ToolCallResult):
                                self.agent_progress_callback(
                                    status_type=AgentStatusType.TOOL_OUTPUT,
                                    msg={
                                        "tool_name": event.tool_name,
                                        "content": str(event.tool_output),
                                    },
                                    event_id=event_id,
                                )
                            elif isinstance(event, AgentInput):
                                self.agent_progress_callback(
                                    status_type=AgentStatusType.AGENT_UPDATE,
                                    msg={"content": f"Agent input: {event.input}"},
                                    event_id=event_id,
                                )
                            elif isinstance(event, AgentOutput):
                                self.agent_progress_callback(
                                    status_type=AgentStatusType.AGENT_UPDATE,
                                    msg={"content": f"Agent output: {event.response}"},
                                    event_id=event_id,
                                )

                    result = await handler

                    # Ensure we have an AgentResponse object with a string response
                    if hasattr(result, "response"):
                        response_text = result.response
                    else:
                        response_text = str(result)

                    # Handle case where response is a ChatMessage object
                    if hasattr(response_text, "content"):
                        response_text = response_text.content
                    elif not isinstance(response_text, str):
                        response_text = str(response_text)

                    if response_text is None or response_text == "None":
                        # Try to find tool outputs in the result object
                        response_text = None

                        # Check various possible locations for tool outputs
                        if hasattr(result, "tool_outputs") and result.tool_outputs:
                            # Get the latest tool output
                            latest_output = (
                                result.tool_outputs[-1]
                                if isinstance(result.tool_outputs, list)
                                else result.tool_outputs
                            )
                            response_text = str(latest_output)

                        # Check if there are tool_calls with results
                        elif hasattr(result, "tool_calls") and result.tool_calls:
                            # Tool calls might contain the outputs - let's try to extract them
                            for tool_call in result.tool_calls:
                                if (
                                    hasattr(tool_call, "tool_output")
                                    and tool_call.tool_output is not None
                                ):
                                    response_text = str(tool_call.tool_output)
                                    break

                        elif hasattr(result, "sources") or hasattr(
                            result, "source_nodes"
                        ):
                            sources = getattr(
                                result, "sources", getattr(result, "source_nodes", [])
                            )
                            if (
                                sources
                                and len(sources) > 0
                                and hasattr(sources[0], "text")
                            ):
                                response_text = sources[0].text

                        # Check for workflow context or chat history that might contain tool results
                        elif hasattr(result, "chat_history"):
                            # Look for the most recent assistant message that might contain tool results
                            chat_history = result.chat_history
                            if chat_history and len(chat_history) > 0:
                                for msg in reversed(chat_history):
                                    if (
                                        msg.role == MessageRole.TOOL
                                        and msg.content
                                        and str(msg.content).strip()
                                    ):
                                        response_text = msg.content
                                        break
                                    if (
                                        hasattr(msg, "content")
                                        and msg.content
                                        and str(msg.content).strip()
                                    ):
                                        response_text = msg.content
                                        break

                        # If we still don't have a response, provide a fallback
                        if response_text is None or response_text == "None":
                            response_text = "Response completed."

                    agent_response = AgentResponse(
                        response=response_text, metadata=getattr(result, "metadata", {})
                    )

                # Standard chat interaction for other agent types
                else:
                    agent_response = await current_agent.achat(prompt)

                # Post processing after response is generated
                agent_response.metadata = agent_response.metadata or {}
                user_metadata = agent_response.metadata
                agent_response = await execute_post_stream_processing(
                    agent_response, prompt, self, user_metadata
                )
                return agent_response

            except Exception as e:
                last_error = e
                if self.verbose:
                    print(f"LLM call failed on attempt {attempt}. " f"Error: {e}.")
                if attempt >= 2 and self.fallback_agent_config:
                    if self.verbose:
                        print(
                            f"LLM call failed on attempt {attempt}. Switching agent configuration."
                        )
                    self._switch_agent_config()
                await asyncio.sleep(1)
                attempt += 1

        return AgentResponse(
            response=(
                f"For {orig_llm} LLM - failure can't be resolved after "
                f"{max_attempts} attempts ({last_error})."
            )
        )

    def stream_chat(self, prompt: str) -> AgentStreamingResponse:
        """
        Interact with the agent using a chat prompt with streaming.
        Args:
            prompt (str): The chat prompt.
        Returns:
            AgentStreamingResponse: The streaming response from the agent.
        """
        try:
            _ = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.astream_chat(prompt))
        raise RuntimeError(
            "Use `await agent.astream_chat(...)` inside an event loop (e.g. Jupyter)."
        )

    async def astream_chat(self, prompt: str) -> AgentStreamingResponse:  # type: ignore
        """
        Interact with the agent using a chat prompt asynchronously with streaming.
        Args:
            prompt (str): The chat prompt.
        Returns:
            AgentStreamingResponse: The streaming response from the agent.
        """
        max_attempts = 4 if self.fallback_agent_config else 2
        attempt = 0
        orig_llm = self.llm.metadata.model_name
        last_error = None
        while attempt < max_attempts:
            try:
                current_agent = self._get_current_agent()
                user_meta: Dict[str, Any] = {}

                # Deal with Function Calling agent type
                if self._get_current_agent_type() == AgentType.FUNCTION_CALLING:
                    from llama_index.core.workflow import Context

                    ctx = Context(current_agent)
                    handler = current_agent.run(
                        user_msg=prompt, ctx=ctx, memory=self.memory
                    )

                    # Use the dedicated FunctionCallingStreamHandler
                    stream_handler = FunctionCallingStreamHandler(self, handler, prompt)
                    streaming_adapter = stream_handler.create_streaming_response(
                        user_meta
                    )

                    return AgentStreamingResponse(
                        base=streaming_adapter, metadata=user_meta
                    )

                #
                # For other agent types, use the standard async chat method
                #
                li_stream = await current_agent.astream_chat(prompt)
                orig_async = li_stream.async_response_gen

                # Define a wrapper to preserve streaming behavior while executing post-stream logic.
                async def _stream_response_wrapper():
                    async for tok in orig_async():
                        yield tok

                    # Use shared post-processing function
                    await execute_post_stream_processing(
                        li_stream, prompt, self, user_meta
                    )

                li_stream.async_response_gen = _stream_response_wrapper
                return AgentStreamingResponse(base=li_stream, metadata=user_meta)

            except Exception as e:
                last_error = e
                if attempt >= 2 and self.fallback_agent_config:
                    if self.verbose:
                        print(
                            f"LLM call failed on attempt {attempt}. Switching agent configuration."
                        )
                    self._switch_agent_config()
                await asyncio.sleep(1)
                attempt += 1

        return AgentStreamingResponse.from_error(
            f"For {orig_llm} LLM - failure can't be resolved after "
            f"{max_attempts} attempts ({last_error})."
        )

    #
    # run() method for running a workflow
    # workflow will always get these arguments in the StartEvent: agent, tools, llm, verbose
    # the inputs argument comes from the call to run()
    #
    async def run(
        self,
        inputs: Any,
        verbose: bool = False,
    ) -> Any:
        """
        Run a workflow using the agent.
        workflow class must be provided in the agent constructor.
        Args:
            inputs (Any): The inputs to the workflow.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
        Returns:
            Any: The output or context of the workflow.
        """
        # Create workflow
        if self.workflow_cls:
            workflow = self.workflow_cls(timeout=self.workflow_timeout, verbose=verbose)
        else:
            raise ValueError("Workflow is not defined.")

        # Validate inputs is in the form of workflow.InputsModel
        if not isinstance(inputs, self.workflow_cls.InputsModel):
            raise ValueError(f"Inputs must be an instance of {workflow.InputsModel}.")

        outputs_model_on_fail_cls = getattr(
            workflow.__class__, "OutputModelOnFail", None
        )
        if outputs_model_on_fail_cls:
            fields_without_default = []
            for name, field_info in outputs_model_on_fail_cls.model_fields.items():
                if field_info.default_factory is PydanticUndefined:
                    fields_without_default.append(name)
            if fields_without_default:
                raise ValueError(
                    f"Fields without default values: {fields_without_default}"
                )

        from llama_index.core.workflow import Context

        workflow_context = Context(workflow=workflow)
        try:
            # run workflow
            result = await workflow.run(
                ctx=workflow_context,
                agent=self,
                tools=self.tools,
                llm=self.llm,
                verbose=verbose,
                inputs=inputs,
            )

            # return output in the form of workflow.OutputsModel(BaseModel)
            try:
                output = workflow.OutputsModel.model_validate(result)
            except ValidationError as e:
                raise ValueError(f"Failed to map workflow output to model: {e}") from e

        except Exception as e:
            _missing = object()
            if outputs_model_on_fail_cls:
                model_fields = outputs_model_on_fail_cls.model_fields
                input_dict = {}
                for key in model_fields:
                    value = await workflow_context.get(key, default=_missing)
                    if value is not _missing:
                        input_dict[key] = value
                output = outputs_model_on_fail_cls.model_validate(input_dict)
            else:
                print(f"Vectara Agentic: Workflow failed with unexpected error: {e}")
                raise type(e)(str(e)).with_traceback(e.__traceback__)

        return output

    #
    # Serialization methods
    #
    def dumps(self) -> str:
        """Serialize the Agent instance to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def loads(
        cls,
        data: str,
        agent_progress_callback: Optional[
            Callable[[AgentStatusType, dict, str], None]
        ] = None,
        query_logging_callback: Optional[Callable[[str, str], None]] = None,
    ) -> "Agent":
        """Create an Agent instance from a JSON string."""
        return cls.from_dict(
            json.loads(data), agent_progress_callback, query_logging_callback
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the Agent instance to a dictionary."""
        return serialize_agent_to_dict(self)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        agent_progress_callback: Optional[Callable] = None,
        query_logging_callback: Optional[Callable] = None,
    ) -> "Agent":
        """Create an Agent instance from a dictionary."""
        return deserialize_agent_from_dict(
            cls, data, agent_progress_callback, query_logging_callback
        )
