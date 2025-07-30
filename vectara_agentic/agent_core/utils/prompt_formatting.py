"""
Prompt formatting and templating utilities.

This module handles prompt template processing, placeholder replacement,
and LLM-specific prompt formatting for different agent types.
"""

from datetime import date

def format_prompt(
    prompt_template: str,
    general_instructions: str,
    topic: str,
    custom_instructions: str,
) -> str:
    """
    Generate a prompt by replacing placeholders with topic and date.

    Args:
        prompt_template: The template for the prompt
        general_instructions: General instructions to be included in the prompt
        topic: The topic to be included in the prompt
        custom_instructions: The custom instructions to be included in the prompt

    Returns:
        str: The formatted prompt
    """
    return (
        prompt_template.replace("{chat_topic}", topic)
        .replace("{today}", date.today().strftime("%A, %B %d, %Y"))
        .replace("{custom_instructions}", custom_instructions)
        .replace("{INSTRUCTIONS}", general_instructions)
    )


def format_llm_compiler_prompt(
    prompt: str, general_instructions: str, topic: str, custom_instructions: str
) -> str:
    """
    Add custom instructions to the prompt for LLM compiler agents.

    Args:
        prompt: The base prompt to which custom instructions should be added
        general_instructions: General instructions for the agent
        topic: Topic expertise for the agent
        custom_instructions: Custom user instructions

    Returns:
        str: The prompt with custom instructions added
    """
    prompt += "\nAdditional Instructions:\n"
    prompt += f"You have expertise in {topic}.\n"
    prompt += general_instructions
    prompt += custom_instructions
    prompt += f"Today is {date.today().strftime('%A, %B %d, %Y')}"
    return prompt
