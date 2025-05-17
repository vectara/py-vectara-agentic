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
import asyncio
import importlib
from collections import Counter
import inspect
from inspect import Signature, Parameter, ismethod
from pydantic import Field, create_model, ValidationError, BaseModel
import cloudpickle as pickle

from dotenv import load_dotenv

from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import (
    ReActAgent,
    StructuredPlannerAgent,
    FunctionCallingAgent,
)
from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.agent.llm_compiler import LLMCompilerAgentWorker
from llama_index.agent.lats import LATSAgentWorker
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.agent.runner.base import AgentRunner
from llama_index.core.agent.types import BaseAgent
from llama_index.core.workflow import Workflow, Context

from .types import (
    AgentType,
    AgentStatusType,
    LLMRole,
    ToolType,
    ModelProvider,
    AgentResponse,
    AgentStreamingResponse,
    AgentConfigType,
)
from .llm_utils import get_llm, get_tokenizer_for_model
from ._prompts import (
    REACT_PROMPT_TEMPLATE,
    GENERAL_PROMPT_TEMPLATE,
    GENERAL_INSTRUCTIONS,
    STRUCTURED_PLANNER_PLAN_REFINE_PROMPT,
    STRUCTURED_PLANNER_INITIAL_PLAN_PROMPT,
)
from ._callback import AgentCallbackHandler
from ._observability import setup_observer, eval_fcs
from .tools import VectaraToolFactory, VectaraTool, ToolsFactory
from .tools_catalog import get_current_date
from .agent_config import AgentConfig


class IgnoreUnpickleableAttributeFilter(logging.Filter):
    """
    Filter to ignore log messages that contain certain strings
    """

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


def _get_prompt(
    prompt_template: str,
    general_instructions: str,
    topic: str,
    custom_instructions: str,
):
    """
    Generate a prompt by replacing placeholders with topic and date.

    Args:
        prompt_template (str): The template for the prompt.
        general_instructions (str): General instructions to be included in the prompt.
        topic (str): The topic to be included in the prompt.
        custom_instructions(str): The custom instructions to be included in the prompt.

    Returns:
        str: The formatted prompt.
    """
    return (
        prompt_template.replace("{chat_topic}", topic)
        .replace("{today}", date.today().strftime("%A, %B %d, %Y"))
        .replace("{custom_instructions}", custom_instructions)
        .replace("{INSTRUCTIONS}", general_instructions)
    )


def _get_llm_compiler_prompt(
    prompt: str, general_instructions: str, topic: str, custom_instructions: str
) -> str:
    """
    Add custom instructions to the prompt.

    Args:
        prompt (str): The prompt to which custom instructions should be added.

    Returns:
        str: The prompt with custom instructions added.
    """
    prompt += "\nAdditional Instructions:\n"
    prompt += f"You have experise in {topic}.\n"
    prompt += general_instructions
    prompt += custom_instructions
    prompt += f"Today is {date.today().strftime('%A, %B %d, %Y')}"
    return prompt


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
        "null": type(None),
    }
    if not field_schema:  # Handles empty schema {}
        return Any

    if "anyOf" in field_schema:
        types = []
        for option_schema in field_schema["anyOf"]:
            types.append(get_field_type(option_schema))  # Recursive call
        if not types:
            return Any
        return Union[tuple(types)]

    if "type" in field_schema and isinstance(field_schema["type"], list):
        types = []
        for type_name in field_schema["type"]:
            if type_name == "array":
                item_schema = field_schema.get("items", {})
                types.append(List[get_field_type(item_schema)])
            elif type_name in json_type_to_python:
                types.append(json_type_to_python[type_name])
            else:
                types.append(Any)  # Fallback for unknown types in the list
        if not types:
            return Any
        return Union[tuple(types)]  # type: ignore

    if "type" in field_schema:
        schema_type_name = field_schema["type"]
        if schema_type_name == "array":
            item_schema = field_schema.get(
                "items", {}
            )  # Default to Any if "items" is missing
            return List[get_field_type(item_schema)]

        return json_type_to_python.get(schema_type_name, Any)

    # If only "items" is present (implies array by some conventions, but less standard)
    # Or if it's a schema with other keywords like 'properties' (implying object)
    # For simplicity, if no "type" or "anyOf" at this point, default to Any or add more specific handling.
    # If 'properties' in field_schema or 'additionalProperties' in field_schema, it's likely an object.
    if "properties" in field_schema or "additionalProperties" in field_schema:
        # This path might need to reconstruct a nested Pydantic model if you encounter such schemas.
        # For now, treating as 'dict' or 'Any' might be a simpler placeholder.
        return dict  # Or Any, or more sophisticated object reconstruction.

    return Any


class Agent:
    """
    Agent class for handling different types of agents and their interactions.
    """

    def __init__(
        self,
        tools: List[FunctionTool],
        topic: str = "general",
        custom_instructions: str = "",
        general_instructions: str = GENERAL_INSTRUCTIONS,
        verbose: bool = True,
        use_structured_planning: bool = False,
        update_func: Optional[Callable[[AgentStatusType, str], None]] = None,
        agent_progress_callback: Optional[
            Callable[[AgentStatusType, str], None]
        ] = None,
        query_logging_callback: Optional[Callable[[str, str], None]] = None,
        agent_config: Optional[AgentConfig] = None,
        fallback_agent_config: Optional[AgentConfig] = None,
        chat_history: Optional[list[Tuple[str, str]]] = None,
        validate_tools: bool = False,
        workflow_cls: Optional[Workflow] = None,
        workflow_timeout: int = 120,
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
            use_structured_planning (bool, optional)
                Whether or not we want to wrap the agent with LlamaIndex StructuredPlannerAgent.
            agent_progress_callback (Callable): A callback function the code calls on any agent updates.
                update_func (Callable): old name for agent_progress_callback. Will be deprecated in future.
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
        """
        self.agent_config = agent_config or AgentConfig()
        self.agent_config_type = AgentConfigType.DEFAULT
        self.tools = tools
        if not any(tool.metadata.name == "get_current_date" for tool in self.tools):
            self.tools += [ToolsFactory().create_tool(get_current_date)]
        self.agent_type = self.agent_config.agent_type
        self.use_structured_planning = use_structured_planning
        self.llm = get_llm(LLMRole.MAIN, config=self.agent_config)
        self._custom_instructions = custom_instructions
        self._general_instructions = general_instructions
        self._topic = topic
        self.agent_progress_callback = (
            agent_progress_callback if agent_progress_callback else update_func
        )
        self.query_logging_callback = query_logging_callback

        self.workflow_cls = workflow_cls
        self.workflow_timeout = workflow_timeout

        # Sanitize tools for Gemini if needed
        if self.agent_config.main_llm_provider == ModelProvider.GEMINI:
            self.tools = self._sanitize_tools_for_gemini(self.tools)

        # Validate tools
        # Check for:
        # 1. multiple copies of the same tool
        # 2. Instructions for using tools that do not exist
        tool_names = [tool.metadata.name for tool in self.tools]
        duplicates = [tool for tool, count in Counter(tool_names).items() if count > 1]
        if duplicates:
            raise ValueError(f"Duplicate tools detected: {', '.join(duplicates)}")

        if validate_tools:
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
            bad_tools_str = llm.complete(prompt).text.strip('\n')
            if bad_tools_str and bad_tools_str != "<OKAY>":
                bad_tools = [tool.strip() for tool in bad_tools_str.split(",")]
                numbered = ", ".join(
                    f"({i}) {tool}" for i, tool in enumerate(bad_tools, 1)
                )
                raise ValueError(
                    f"The Agent custom instructions mention these invalid tools: {numbered}"
                )

        # Create token counters for the main and tool LLMs
        main_tok = get_tokenizer_for_model(role=LLMRole.MAIN)
        self.main_token_counter = (
            TokenCountingHandler(tokenizer=main_tok) if main_tok else None
        )
        tool_tok = get_tokenizer_for_model(role=LLMRole.TOOL)
        self.tool_token_counter = (
            TokenCountingHandler(tokenizer=tool_tok) if tool_tok else None
        )

        # Setup callback manager
        callbacks: list[BaseCallbackHandler] = [
            AgentCallbackHandler(self.agent_progress_callback)
        ]
        if self.main_token_counter:
            callbacks.append(self.main_token_counter)
        if self.tool_token_counter:
            callbacks.append(self.tool_token_counter)
        callback_manager = CallbackManager(callbacks)  # type: ignore
        self.verbose = verbose

        if chat_history:
            msg_history = []
            for text_pairs in chat_history:
                msg_history.append(
                    ChatMessage.from_str(content=text_pairs[0], role=MessageRole.USER)
                )
                msg_history.append(
                    ChatMessage.from_str(
                        content=text_pairs[1], role=MessageRole.ASSISTANT
                    )
                )
            self.memory = ChatMemoryBuffer.from_defaults(
                token_limit=128000, chat_history=msg_history
            )
        else:
            self.memory = ChatMemoryBuffer.from_defaults(token_limit=128000)

        # Set up main agent and fallback agent
        self.agent = self._create_agent(self.agent_config, callback_manager)
        self.fallback_agent_config = fallback_agent_config
        if self.fallback_agent_config:
            self.fallback_agent = self._create_agent(
                self.fallback_agent_config, callback_manager
            )
        else:
            self.fallback_agent_config = None

        # Setup observability
        try:
            self.observability_enabled = setup_observer(self.agent_config, self.verbose)
        except Exception as e:
            print(f"Failed to set up observer ({e}), ignoring")
            self.observability_enabled = False

    def _sanitize_tools_for_gemini(
        self, tools: list[FunctionTool]
    ) -> list[FunctionTool]:
        """
        Strip all default values from:
        - tool.fn
        - tool.async_fn
        - tool.metadata.fn_schema
        so Gemini sees *only* required parameters, no defaults.
        """
        for tool in tools:
            # 1) strip defaults off the actual callables
            for func in (tool.fn, tool.async_fn):
                if not func:
                    continue
                orig_sig = inspect.signature(func)
                new_params = [
                    p.replace(default=Parameter.empty)
                    for p in orig_sig.parameters.values()
                ]
                new_sig = Signature(
                    new_params, return_annotation=orig_sig.return_annotation
                )
                if ismethod(func):
                    func.__func__.__signature__ = new_sig
                else:
                    func.__signature__ = new_sig

            # 2) rebuild the Pydantic schema so that *every* field is required
            schema_cls = getattr(tool.metadata, "fn_schema", None)
            if schema_cls and hasattr(schema_cls, "model_fields"):
                # collect (name → (type, Field(...))) for all fields
                new_fields: dict[str, tuple[type, Any]] = {}
                for name, mf in schema_cls.model_fields.items():
                    typ = mf.annotation
                    desc = getattr(mf, "description", "")
                    # force required (no default) with Field(...)
                    new_fields[name] = (typ, Field(..., description=desc))

                # make a brand-new schema class where every field is required
                no_default_schema = create_model(
                    f"{schema_cls.__name__}",  # new class name
                    **new_fields,  # type: ignore
                )

                # give it a clean __signature__ so inspect.signature sees no defaults
                params = [
                    Parameter(n, Parameter.POSITIONAL_OR_KEYWORD, annotation=typ)
                    for n, (typ, _) in new_fields.items()
                ]
                no_default_schema.__signature__ = Signature(params)

                # swap it back onto the tool
                tool.metadata.fn_schema = no_default_schema

        return tools

    def _create_agent(
        self, config: AgentConfig, llm_callback_manager: CallbackManager
    ) -> Union[BaseAgent, AgentRunner]:
        """
        Creates the agent based on the configuration object.

        Args:

            config: The configuration of the agent.
            llm_callback_manager: The callback manager for the agent's llm.

        Returns:
            Union[BaseAgent, AgentRunner]: The configured agent object.
        """
        agent_type = config.agent_type
        llm = get_llm(LLMRole.MAIN, config=config)
        llm.callback_manager = llm_callback_manager

        if agent_type == AgentType.FUNCTION_CALLING:
            if config.tool_llm_provider == ModelProvider.OPENAI:
                raise ValueError(
                    "Vectara-agentic: Function calling agent type is not supported with the OpenAI LLM."
                )
            prompt = _get_prompt(
                GENERAL_PROMPT_TEMPLATE,
                self._general_instructions,
                self._topic,
                self._custom_instructions,
            )
            agent = FunctionCallingAgent.from_tools(
                tools=self.tools,
                llm=llm,
                memory=self.memory,
                verbose=self.verbose,
                max_function_calls=config.max_reasoning_steps,
                callback_manager=llm_callback_manager,
                system_prompt=prompt,
                allow_parallel_tool_calls=True,
            )
        elif agent_type == AgentType.REACT:
            prompt = _get_prompt(
                REACT_PROMPT_TEMPLATE,
                self._general_instructions,
                self._topic,
                self._custom_instructions,
            )
            agent = ReActAgent.from_tools(
                tools=self.tools,
                llm=llm,
                memory=self.memory,
                verbose=self.verbose,
                react_chat_formatter=ReActChatFormatter(system_header=prompt),
                max_iterations=config.max_reasoning_steps,
                callable_manager=llm_callback_manager,
            )
        elif agent_type == AgentType.OPENAI:
            if config.tool_llm_provider != ModelProvider.OPENAI:
                raise ValueError(
                    "Vectara-agentic: OPENAI agent type requires the OpenAI LLM."
                )
            prompt = _get_prompt(
                GENERAL_PROMPT_TEMPLATE,
                self._general_instructions,
                self._topic,
                self._custom_instructions,
            )
            agent = OpenAIAgent.from_tools(
                tools=self.tools,
                llm=llm,
                memory=self.memory,
                verbose=self.verbose,
                callback_manager=llm_callback_manager,
                max_function_calls=config.max_reasoning_steps,
                system_prompt=prompt,
            )
        elif agent_type == AgentType.LLMCOMPILER:
            agent_worker = LLMCompilerAgentWorker.from_tools(
                tools=self.tools,
                llm=llm,
                verbose=self.verbose,
                callback_manager=llm_callback_manager,
            )
            agent_worker.system_prompt = _get_prompt(
                prompt_template=_get_llm_compiler_prompt(
                    prompt=agent_worker.system_prompt,
                    general_instructions=self._general_instructions,
                    topic=self._topic,
                    custom_instructions=self._custom_instructions,
                ),
                general_instructions=self._general_instructions,
                topic=self._topic,
                custom_instructions=self._custom_instructions,
            )
            agent_worker.system_prompt_replan = _get_prompt(
                prompt_template=_get_llm_compiler_prompt(
                    prompt=agent_worker.system_prompt_replan,
                    general_instructions=GENERAL_INSTRUCTIONS,
                    topic=self._topic,
                    custom_instructions=self._custom_instructions,
                ),
                general_instructions=GENERAL_INSTRUCTIONS,
                topic=self._topic,
                custom_instructions=self._custom_instructions,
            )
            agent = agent_worker.as_agent()
        elif agent_type == AgentType.LATS:
            agent_worker = LATSAgentWorker.from_tools(
                tools=self.tools,
                llm=llm,
                num_expansions=3,
                max_rollouts=-1,
                verbose=self.verbose,
                callback_manager=llm_callback_manager,
            )
            prompt = _get_prompt(
                REACT_PROMPT_TEMPLATE,
                self._general_instructions,
                self._topic,
                self._custom_instructions,
            )
            agent_worker.chat_formatter = ReActChatFormatter(system_header=prompt)
            agent = agent_worker.as_agent()
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # Set up structured planner if needed
        if self.use_structured_planning or self.agent_type in [
            AgentType.LLMCOMPILER,
            AgentType.LATS,
        ]:
            planner_llm = get_llm(LLMRole.TOOL, config=config)
            agent = StructuredPlannerAgent(
                agent_worker=agent.agent_worker,
                tools=self.tools,
                llm=planner_llm,
                memory=self.memory,
                verbose=self.verbose,
                initial_plan_prompt=STRUCTURED_PLANNER_INITIAL_PLAN_PROMPT,
                plan_refine_prompt=STRUCTURED_PLANNER_PLAN_REFINE_PROMPT,
            )

        return agent

    def clear_memory(self) -> None:
        """
        Clear the agent's memory.
        """
        if self.agent_config_type == AgentConfigType.DEFAULT:
            self.agent.memory.reset()
        elif (
            self.agent_config_type == AgentConfigType.FALLBACK
            and self.fallback_agent_config
        ):
            self.fallback_agent.memory.reset()
        else:
            raise ValueError(f"Invalid agent config type {self.agent_config_type}")

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
        agent_progress_callback: Optional[
            Callable[[AgentStatusType, str], None]
        ] = None,
        query_logging_callback: Optional[Callable[[str, str], None]] = None,
        agent_config: AgentConfig = AgentConfig(),
        validate_tools: bool = False,
        fallback_agent_config: Optional[AgentConfig] = None,
        chat_history: Optional[list[Tuple[str, str]]] = None,
        workflow_cls: Optional[Workflow] = None,
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
                update_func (Callable): old name for agent_progress_callback. Will be deprecated in future.
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
            update_func=update_func,
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
            Callable[[AgentStatusType, str], None]
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
        """
        Create an agent from a single Vectara corpus

        Args:
            tool_name (str): The name of Vectara tool used by the agent
            vectara_corpus_key (str): The Vectara corpus key (or comma separated list of keys).
            vectara_api_key (str): The Vectara API key.
            agent_progress_callback (Callable): A callback function the code calls on any agent updates.
            query_logging_callback (Callable): A callback function the code calls upon completion of a query
            agent_config (AgentConfig, optional): The configuration of the agent.
            fallback_agent_config (AgentConfig, optional): The fallback configuration of the agent.
            chat_history (Tuple[str, str], optional): A list of user/agent chat pairs to initialize the agent memory.
            data_description (str): The description of the data.
            assistant_specialty (str): The specialty of the assistant.
            general_instructions (str, optional): General instructions for the agent.
                The Agent has a default set of instructions that are crafted to help it operate effectively.
                This allows you to customize the agent's behavior and personality, but use with caution.
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
            return_direct (bool, optional): Whether the agent should return the tool's response directly.

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
            return_direct=return_direct,
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
            general_instructions=general_instructions,
            verbose=verbose,
            agent_progress_callback=agent_progress_callback,
            query_logging_callback=query_logging_callback,
            agent_config=agent_config,
            fallback_agent_config=fallback_agent_config,
            chat_history=chat_history,
        )

    def _switch_agent_config(self) -> None:
        """ "
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

    def token_counts(self) -> dict:
        """
        Get the token counts for the agent and tools.

        Returns:
            dict: The token counts for the agent and tools.
        """
        return {
            "main token count": (
                self.main_token_counter.total_llm_token_count
                if self.main_token_counter
                else -1
            ),
            "tool token count": (
                self.tool_token_counter.total_llm_token_count
                if self.tool_token_counter
                else -1
            ),
        }

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
        agent_response.response = str(agent.llm.acomplete(llm_prompt))

    def chat(self, prompt: str) -> AgentResponse:  # type: ignore
        """
        Interact with the agent using a chat prompt.

        Args:
            prompt (str): The chat prompt.

        Returns:
            AgentResponse: The response from the agent.
        """
        return asyncio.run(self.achat(prompt))

    async def achat(self, prompt: str) -> AgentResponse:  # type: ignore
        """
        Interact with the agent using a chat prompt.

        Args:
            prompt (str): The chat prompt.

        Returns:
            AgentResponse: The response from the agent.
        """
        max_attempts = 4 if self.fallback_agent_config else 2
        attempt = 0
        orig_llm = self.llm.metadata.model_name
        last_error = None
        while attempt < max_attempts:
            try:
                current_agent = self._get_current_agent()
                agent_response = await current_agent.achat(prompt)
                await self._aformat_for_lats(prompt, agent_response)
                if self.observability_enabled:
                    eval_fcs()
                if self.query_logging_callback:
                    self.query_logging_callback(prompt, agent_response.response)
                return agent_response

            except Exception as e:
                last_error = e
                if attempt >= 2:
                    if self.verbose:
                        print(
                            f"LLM call failed on attempt {attempt}. Switching agent configuration."
                        )
                    self._switch_agent_config()
                time.sleep(1)
                attempt += 1

        return AgentResponse(
            response=(
                f"For {orig_llm} LLM - failure can't be resolved after "
                f"{max_attempts} attempts ({last_error}."
            )
        )

    def stream_chat(self, prompt: str) -> AgentStreamingResponse:  # type: ignore
        """
        Interact with the agent using a chat prompt with streaming.
        Args:
            prompt (str): The chat prompt.
        Returns:
            AgentStreamingResponse: The streaming response from the agent.
        """
        return asyncio.run(self.astream_chat(prompt))

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
        while attempt < max_attempts:
            try:
                current_agent = self._get_current_agent()
                agent_response = await current_agent.astream_chat(prompt)
                original_async_response_gen = agent_response.async_response_gen

                # Define a wrapper to preserve streaming behavior while executing post-stream logic.
                async def _stream_response_wrapper():
                    async for token in original_async_response_gen():
                        yield token  # Yield tokens as they are generated
                    # Post-streaming additional logic:
                    await self._aformat_for_lats(prompt, agent_response)
                    if self.query_logging_callback:
                        self.query_logging_callback(prompt, agent_response.response)
                    if self.observability_enabled:
                        eval_fcs()

                agent_response.async_response_gen = (
                    _stream_response_wrapper  # Override the generator
                )
                return agent_response

            except Exception as e:
                last_error = e
                if attempt >= 2:
                    if self.verbose:
                        print(
                            f"LLM call failed on attempt {attempt}. Switching agent configuration."
                        )
                    self._switch_agent_config()
                time.sleep(1)
                attempt += 1

        return AgentStreamingResponse(
            response=(
                f"For {orig_llm} LLM - failure can't be resolved after "
                f"{max_attempts} attempts ({last_error})."
            )
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
            outputs_model_on_fail_cls = getattr(workflow.__class__, "OutputModelOnFail", None)
            if outputs_model_on_fail_cls:
                model_fields = outputs_model_on_fail_cls.model_fields
                input_dict = {
                    key: await workflow_context.get(key, None)
                    for key in model_fields
                }

                # return output in the form of workflow.OutputModelOnFail(BaseModel)
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
    def loads(cls, data: str) -> "Agent":
        """Create an Agent instance from a JSON string."""
        return cls.from_dict(json.loads(data))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the Agent instance to a dictionary."""
        tool_info = []
        for tool in self.tools:
            if hasattr(tool.metadata, "fn_schema"):
                fn_schema_cls = tool.metadata.fn_schema
                fn_schema_serialized = {
                    "schema": (
                        fn_schema_cls.model_json_schema()
                        if fn_schema_cls and hasattr(fn_schema_cls, "model_json_schema")
                        else None
                    ),
                    "metadata": {
                        "module": fn_schema_cls.__module__ if fn_schema_cls else None,
                        "class": fn_schema_cls.__name__ if fn_schema_cls else None,
                    },
                }
            else:
                fn_schema_serialized = None

            tool_dict = {
                "tool_type": tool.metadata.tool_type.value,
                "name": tool.metadata.name,
                "description": tool.metadata.description,
                "fn": (
                    pickle.dumps(getattr(tool, "fn", None)).decode("latin-1")
                    if getattr(tool, "fn", None)
                    else None
                ),
                "async_fn": (
                    pickle.dumps(getattr(tool, "async_fn", None)).decode("latin-1")
                    if getattr(tool, "async_fn", None)
                    else None
                ),
                "fn_schema": fn_schema_serialized,
            }
            tool_info.append(tool_dict)

        return {
            "agent_type": self.agent_config.agent_type.value,
            "memory": pickle.dumps(self.agent.memory).decode("latin-1"),
            "tools": tool_info,
            "topic": self._topic,
            "custom_instructions": self._custom_instructions,
            "verbose": self.verbose,
            "agent_config": self.agent_config.to_dict(),
            "fallback_agent": (
                self.fallback_agent_config.to_dict()
                if self.fallback_agent_config
                else None
            ),
            "workflow_cls": self.workflow_cls if self.workflow_cls else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Agent":
        """Create an Agent instance from a dictionary."""
        agent_config = AgentConfig.from_dict(data["agent_config"])
        fallback_agent_config = (
            AgentConfig.from_dict(data["fallback_agent_config"])
            if data.get("fallback_agent_config")
            else None
        )
        tools: list[FunctionTool] = []

        for tool_data in data["tools"]:
            query_args_model = None
            if tool_data.get("fn_schema"):
                schema_info = tool_data["fn_schema"]
                try:
                    module_name = schema_info["metadata"]["module"]
                    class_name = schema_info["metadata"]["class"]
                    mod = importlib.import_module(module_name)
                    candidate_cls = getattr(mod, class_name)
                    if inspect.isclass(candidate_cls) and issubclass(
                        candidate_cls, BaseModel
                    ):
                        query_args_model = candidate_cls
                    else:
                        # It's not the Pydantic model class we expected (e.g., it's the function itself)
                        # Force fallback to JSON schema reconstruction by raising an error.
                        raise ImportError(
                            f"Retrieved '{class_name}' from '{module_name}' is not a Pydantic BaseModel class. "
                            "Falling back to JSON schema reconstruction."
                        )
                except Exception:
                    # Fallback: rebuild using the JSON schema
                    field_definitions = {}
                    json_schema_to_rebuild = schema_info.get("schema")
                    if json_schema_to_rebuild and isinstance(
                        json_schema_to_rebuild, dict
                    ):
                        for field, values in json_schema_to_rebuild.get(
                            "properties", {}
                        ).items():
                            field_type = get_field_type(values)
                            field_description = values.get(
                                "description"
                            )  # Defaults to None
                            if "default" in values:
                                field_definitions[field] = (
                                    field_type,
                                    Field(
                                        description=field_description,
                                        default=values["default"],
                                    ),
                                )
                            else:
                                field_definitions[field] = (
                                    field_type,
                                    Field(description=field_description),
                                )
                        query_args_model = create_model(
                            json_schema_to_rebuild.get(
                                "title", f"{tool_data['name']}_QueryArgs"
                            ),
                            **field_definitions,
                        )
                    else:  # If schema part is missing or not a dict, create a default empty model
                        query_args_model = create_model(
                            f"{tool_data['name']}_QueryArgs"
                        )

            # If fn_schema was not in tool_data or reconstruction failed badly, default to empty pydantic model
            if query_args_model is None:
                query_args_model = create_model(f"{tool_data['name']}_QueryArgs")

            fn = (
                pickle.loads(tool_data["fn"].encode("latin-1"))
                if tool_data["fn"]
                else None
            )
            async_fn = (
                pickle.loads(tool_data["async_fn"].encode("latin-1"))
                if tool_data["async_fn"]
                else None
            )

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
            fallback_agent_config=fallback_agent_config,
            workflow_cls=data["workflow_cls"],
        )
        memory = (
            pickle.loads(data["memory"].encode("latin-1"))
            if data.get("memory")
            else None
        )
        if memory:
            agent.agent.memory = memory
        return agent
