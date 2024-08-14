"""
This module contains the Agent class for handling different types of agents and their interactions.
"""

from typing import List, Callable, Optional
import os
from datetime import date

from retrying import retry

from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.core.callbacks import CallbackManager
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.memory import ChatMemoryBuffer

from dotenv import load_dotenv

from .types import AgentType, AgentStatusType, LLMRole
from .utils import get_llm
from ._prompts import REACT_PROMPT_TEMPLATE, GENERAL_PROMPT_TEMPLATE
from ._callback import AgentCallbackHandler

load_dotenv(override=True)


def get_prompt(prompt_template: str, topic: str, custom_instructions: str):
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


def retry_if_exception(exception):
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

        callback_manager = CallbackManager([AgentCallbackHandler(update_func)])  # type: ignore
        self.llm.callback_manager = callback_manager

        memory = ChatMemoryBuffer.from_defaults(token_limit=128000)
        if self.agent_type == AgentType.REACT:
            prompt = get_prompt(REACT_PROMPT_TEMPLATE, topic, custom_instructions)
            self.agent = ReActAgent.from_tools(
                tools=tools,
                llm=self.llm,
                memory=memory,
                verbose=True,
                react_chat_formatter=ReActChatFormatter(system_header=prompt),
                max_iterations=20,
                callable_manager=callback_manager,
            )
        elif self.agent_type == AgentType.OPENAI:
            prompt = get_prompt(GENERAL_PROMPT_TEMPLATE, topic, custom_instructions)
            self.agent = OpenAIAgent.from_tools(
                tools=tools,
                llm=self.llm,
                memory=memory,
                verbose=True,
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
        return cls(tools, topic, custom_instructions, update_func)

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

    @retry(
        retry_on_exception=retry_if_exception,
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
