"""
This module contains the Agent class for handling different types of agents and their interactions.
"""

from typing import List, Callable, Optional
import os
from datetime import date

from retrying import retry
from pydantic import Field, create_model


from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.memory import ChatMemoryBuffer

from dotenv import load_dotenv

from .types import AgentType, AgentStatusType, LLMRole
from .utils import get_llm, get_tokenizer_for_model
from ._prompts import REACT_PROMPT_TEMPLATE, GENERAL_PROMPT_TEMPLATE
from ._callback import AgentCallbackHandler
from .tools import VectaraToolFactory

load_dotenv(override=True)


def _get_prompt(prompt_template: str, topic: str, custom_instructions: str):
    """
    Generate a prompt by replacing placeholders with topic and date.

    Args:

        prompt_template (str): The template for the prompt.
        topic (str): The topic to be included in the prompt.

    Returns:
        str: The formatted prompt.
    """
    return (
        prompt_template.replace("{chat_topic}", topic)
        .replace("{today}", date.today().strftime("%A, %B %d, %Y"))
        .replace("{custom_instructions}", custom_instructions)
    )


def _retry_if_exception(exception):
    # Define the condition to retry on certain exceptions
    return isinstance(
        exception, (TimeoutError)
    )  # Replace SomeOtherException with other exceptions you want to catch


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
    ):
        """
        Initialize the agent with the specified type, tools, topic, and system message.

        Args:

            tools (list[FunctionTool]): A list of tools to be used by the agent.
            topic (str, optional): The topic for the agent. Defaults to 'general'.
            custom_instructions (str, optional): custom instructions for the agent. Defaults to ''.
            update_func (Callable): a callback function the code calls on any agent updates.
        """
        self.agent_type = AgentType(os.getenv("VECTARA_AGENTIC_AGENT_TYPE", "OPENAI"))
        self.tools = tools
        self.llm = get_llm(LLMRole.MAIN)
        self._custom_instructions = custom_instructions
        self._topic = topic

        main_tok = get_tokenizer_for_model(role=LLMRole.MAIN)
        self.main_token_counter = TokenCountingHandler(tokenizer = main_tok) if main_tok else None
        tool_tok = get_tokenizer_for_model(role=LLMRole.TOOL)
        self.tool_token_counter = TokenCountingHandler(tokenizer = tool_tok) if tool_tok else None
        
        callbacks = [AgentCallbackHandler(update_func)]
        if self.main_token_counter:
            callbacks.append(self.main_token_counter)
        if self.tool_token_counter:
            callbacks.append(self.tool_token_counter)
        callback_manager = CallbackManager(callbacks)   # type: ignore
        self.llm.callback_manager = callback_manager

        memory = ChatMemoryBuffer.from_defaults(token_limit=128000)
        if self.agent_type == AgentType.REACT:
            prompt = _get_prompt(REACT_PROMPT_TEMPLATE, topic, custom_instructions)
            self.agent = ReActAgent.from_tools(
                tools=tools,
                llm=self.llm,
                memory=memory,
                verbose=verbose,
                react_chat_formatter=ReActChatFormatter(system_header=prompt),
                max_iterations=20,
                callable_manager=callback_manager,
            )
        elif self.agent_type == AgentType.OPENAI:
            prompt = _get_prompt(GENERAL_PROMPT_TEMPLATE, topic, custom_instructions)
            self.agent = OpenAIAgent.from_tools(
                tools=tools,
                llm=self.llm,
                memory=memory,
                verbose=verbose,
                callable_manager=callback_manager,
                max_function_calls=10,
                system_prompt=prompt,
            )
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

    @classmethod
    def from_tools(
        cls,
        tools: List[FunctionTool],
        topic: str = "general",
        custom_instructions: str = "",
        verbose: bool = True,
        update_func: Optional[Callable[[AgentStatusType, str], None]] = None,
    ) -> "Agent":
        """
        Create an agent from tools, agent type, and language model.

        Args:

            tools (list[FunctionTool]): A list of tools to be used by the agent.
            topic (str, optional): The topic for the agent. Defaults to 'general'.
            custom_instructions (str, optional): custom instructions for the agent. Defaults to ''.
            llm (LLM): The language model to be used by the agent.

        Returns:
            Agent: An instance of the Agent class.
        """
        return cls(tools, topic, custom_instructions, verbose, update_func)


    @classmethod
    def from_corpus(
        cls,
        vectara_customer_id: str,
        vectara_corpus_id: str,
        vectara_api_key: str,
        data_description: str,
        assistant_specialty: str,
        verbose: bool = False,
        vectara_filter_fields: list[dict] = [],
        vectara_lambda_val: float = 0.005,
        vectara_reranker: str = "mmr",
        vectara_rerank_k: int = 50,
        vectara_n_sentences_before: int = 2,
        vectara_n_sentences_after: int = 2,
        vectara_summary_num_results: int = 10,
        vectara_summarizer: str = "vectara-summary-ext-24-05-sml",
    ) -> "Agent":
        """
        Create an agent from a single Vectara corpus

        Args:
            name (str): The name .
            vectara_customer_id (str): The Vectara customer ID.
            vectara_corpus_id (str): The Vectara corpus ID.
            vectara_api_key (str): The Vectara API key.
            data_description (str): The description of the data.
            assistant_specialty (str): The specialty of the assistant.
            verbose (bool): Whether to print verbose output.
            vectara_filter_fields (List[dict]): The filterable attributes (each dict includes name, type, and description).
            vectara_lambda_val (float): The lambda value for Vectara hybrid search.
            vectara_reranker (str): The Vectara reranker name (default "mmr")
            vectara_rerank_k (int): The number of results to use with reranking.
            vectara_n_sentences_before (int): The number of sentences before the matching text
            vectara_n_sentences_after (int): The number of sentences after the matching text.
            vectara_summary_num_results (int): The number of results to use in summarization.
            vectara_summarizer (str): The Vectara summarizer name.

        Returns:
            Agent: An instance of the Agent class.
        """
        vec_factory = VectaraToolFactory(vectara_api_key=vectara_api_key, 
                                         vectara_customer_id=vectara_customer_id,
                                         vectara_corpus_id=vectara_corpus_id)        
        QueryArgs = create_model(
            "QueryArgs",
            query=(str, Field(description="The user query")),
            **{
                field['name']: (field['type'], Field(description=field['description'], default=None))
                for field in vectara_filter_fields
            }
        )

        vectara_tool = vec_factory.create_rag_tool(
            tool_name = f"vectara_{vectara_corpus_id}",
            tool_description = f"""
            Given a user query,
            returns a response (str) to a user question about {data_description}.
            """,
            tool_args_schema = QueryArgs,
            reranker = vectara_reranker, rerank_k = vectara_rerank_k, 
            n_sentences_before = vectara_n_sentences_before, 
            n_sentences_after = vectara_n_sentences_after, 
            lambda_val = vectara_lambda_val,
            summary_num_results = vectara_summary_num_results,
            vectara_summarizer = vectara_summarizer,
            include_citations = False,
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
            update_func=None
        )

    def report(self) -> str:
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
            print(f"- {tool._metadata.name}")
        print(f"Agent LLM = {get_llm(LLMRole.MAIN).model}")
        print(f"Tool LLM = {get_llm(LLMRole.TOOL).model}")

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

    @retry(
        retry_on_exception=_retry_if_exception,
        stop_max_attempt_number=3,
        wait_fixed=2000,
    )
    def chat(self, prompt: str) -> str:
        """
        Interact with the agent using a chat prompt.

        Args:
            prompt (str): The chat prompt.

        Returns:
            str: The response from the agent.
        """

        try:
            agent_response = self.agent.chat(prompt)
            return agent_response.response
        except Exception as e:
            import traceback

            return f"Vectara Agentic: encountered an exception ({e}) at ({traceback.format_exc()}), and can't respond."
