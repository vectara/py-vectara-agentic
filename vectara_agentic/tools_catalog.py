"""
This module contains the tools catalog for the Vectara Agentic.
"""

from typing import Optional
from pydantic import Field
import requests

from .types import LLMRole
from .utils import get_llm

req_session = requests.Session()

get_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:98.0) Gecko/20100101 Firefox/98.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

#
# Standard Tools
#

def summarize_text(
    text: str = Field(description="the original text."),
    expertise: str = Field(
        description="the expertise to apply to the summarization.",
    ),
) -> str:
    """
    This is a helper tool. It does not provide new information.
    Use this tool to summarize text with no more than summary_max_length 
    characters.
    """
    expertise = "general" if len(expertise) < 3 else expertise.lower()
    prompt = f"As an expert in {expertise}, summarize the provided text"
    prompt += " into a concise summary."
    prompt += f"\noriginal text: {text}\nsummary:"
    llm = get_llm(LLMRole.TOOL)
    response = llm.complete(prompt)
    return response.text


def rephrase_text(
    text: str = Field(description="the original text."),
    instructions: str = Field(
        description="the specific instructions for how to rephrase the text."
    ),
) -> str:
    """
    This is a helper tool. It does not provide new information.
    Use this tool to rephrase the text according to the provided instructions.
    For example, instructions could be "as a 5 year old would say it."
    """
    prompt = f"""
    Rephrase the provided text according to the following instructions: {instructions}.
    If the input is Markdown, keep the output in Markdown as well.
    original text: {text}
    rephrased text:
    """
    llm = get_llm(LLMRole.TOOL)
    response = llm.complete(prompt)
    return response.text


def critique_text(
    text: str = Field(description="the original text."),
    role: Optional[str] = Field(
        None, description="the role of the person providing critique."
    ),
    point_of_view: Optional[str] = Field(
        None, description="the point of view with which to provide critique."
    ),
) -> str:
    """
    This is a helper tool. It does not provide new information.
    Critique the text from the specified point of view.
    """
    if role:
        prompt = f"As a {role}, critique the provided text from the point of view of {point_of_view}."
    else:
        prompt = (
            f"Critique the provided text from the point of view of {point_of_view}."
        )
    prompt += "Structure the critique as bullet points.\n"
    prompt += f"Original text: {text}\nCritique:"
    llm = get_llm(LLMRole.TOOL)
    response = llm.complete(prompt)
    return response.text


#
# Guardrails tools
#


def guardrails_no_politics(text: str = Field(description="the original text.")) -> str:
    """
    A guardrails tool.
    Can be used to rephrase text so that it does not have political content.
    """
    return rephrase_text(text, "avoid any specific political content.")


def guardrails_be_polite(text: str = Field(description="the original text.")) -> str:
    """
    A guardrails tool.
    Can be used to rephrase the text so that the response is in a polite tone.
    """
    return rephrase_text(text, "Ensure the response is super polite.")
