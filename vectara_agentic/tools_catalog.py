"""
This module contains the tools catalog for the Vectara Agentic.
"""
from typing import List
from functools import lru_cache, wraps
from datetime import date

from inspect import signature
import requests

from .types import LLMRole
from .agent_config import AgentConfig
from .utils import get_llm

req_session = requests.Session()

get_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:98.0) Gecko/20100101 Firefox/98.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

def get_current_date() -> str:
    """
    Returns: the current date.
    """
    return date.today().strftime("%A, %B %d, %Y")


def remove_self_from_signature(func):
    """Decorator to remove 'self' from a method's signature for introspection."""
    sig = signature(func)
    params = list(sig.parameters.values())
    # Remove the first parameter if it is named 'self'
    if params and params[0].name == "self":
        params = params[1:]
    new_sig = sig.replace(parameters=params)
    func.__signature__ = new_sig
    return func

class ToolsCatalog:
    def __init__(self, agent_config: AgentConfig):
        self.agent_config = agent_config

    @remove_self_from_signature
    def summarize_text(self, text: str, expertise: str) -> str:
        if not isinstance(expertise, str):
            return "Please provide a valid string for expertise."
        if not isinstance(text, str):
            return "Please provide a valid string for text."
        expertise = "general" if len(expertise) < 3 else expertise.lower()
        prompt = (
            f"As an expert in {expertise}, summarize the provided text "
            "into a concise summary.\n"
            f"Original text: {text}\nSummary:"
        )
        llm = get_llm(LLMRole.TOOL, config=self.agent_config)
        response = llm.complete(prompt)
        return response.text

    @remove_self_from_signature
    def rephrase_text(self, text: str, instructions: str) -> str:
        prompt = (
            f"Rephrase the provided text according to the following instructions: {instructions}.\n"
            "If the input is Markdown, keep the output in Markdown as well.\n"
            f"Original text: {text}\nRephrased text:"
        )
        llm = get_llm(LLMRole.TOOL, config=self.agent_config)
        response = llm.complete(prompt)
        return response.text

    @remove_self_from_signature
    def critique_text(self, text: str, role: str, point_of_view: str) -> str:
        if role:
            prompt = f"As a {role}, critique the provided text from the point of view of {point_of_view}."
        else:
            prompt = f"Critique the provided text from the point of view of {point_of_view}."
        prompt += "\nStructure the critique as bullet points.\n"
        prompt += f"Original text: {text}\nCritique:"
        llm = get_llm(LLMRole.TOOL, config=self.agent_config)
        response = llm.complete(prompt)
        return response.text

#
# Guardrails tool: returns list of topics to avoid
#
def get_bad_topics() -> List[str]:
    """
    Get the list of topics to avoid in the response.
    """
    return [
        "politics",
        "religion",
        "violence",
        "hate speech",
        "adult content",
        "illegal activities",
    ]
