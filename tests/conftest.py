# Suppress external dependency warnings before any other imports
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

"""
Common test utilities, configurations, and fixtures for the vectara-agentic test suite.
"""

import unittest
from contextlib import contextmanager
from typing import Any

from vectara_agentic.agent_config import AgentConfig
from vectara_agentic.types import AgentType, ModelProvider


# ========================================
# Common Test Functions
# ========================================


def mult(x: float, y: float) -> float:
    """Multiply two numbers - common test function used across multiple test files."""
    return x * y


def add(x: float, y: float) -> float:
    """Add two numbers - common test function used in workflow tests."""
    return x + y


# ========================================
# Common Test Data
# ========================================

# Standard test topic used across most tests
STANDARD_TEST_TOPIC = "AI topic"

# Standard test instructions used across most tests
STANDARD_TEST_INSTRUCTIONS = (
    "Always do as your father tells you, if your mother agrees!"
)

# Alternative instructions for specific tests
WORKFLOW_TEST_INSTRUCTIONS = "You are a helpful AI assistant."
MATH_AGENT_INSTRUCTIONS = "you are an agent specializing in math, assisting a user."


# ========================================
# Agent Configuration Objects
# ========================================

# Default configurations
default_config = AgentConfig()

# Function Calling configurations for all providers
fc_config_anthropic = AgentConfig(
    agent_type=AgentType.FUNCTION_CALLING,
    main_llm_provider=ModelProvider.ANTHROPIC,
    tool_llm_provider=ModelProvider.ANTHROPIC,
)

fc_config_gemini = AgentConfig(
    agent_type=AgentType.FUNCTION_CALLING,
    main_llm_provider=ModelProvider.GEMINI,
    tool_llm_provider=ModelProvider.GEMINI,
)

fc_config_together = AgentConfig(
    agent_type=AgentType.FUNCTION_CALLING,
    main_llm_provider=ModelProvider.TOGETHER,
    tool_llm_provider=ModelProvider.TOGETHER,
)

fc_config_openai = AgentConfig(
    agent_type=AgentType.FUNCTION_CALLING,
    main_llm_provider=ModelProvider.OPENAI,
    tool_llm_provider=ModelProvider.OPENAI,
)

fc_config_groq = AgentConfig(
    agent_type=AgentType.FUNCTION_CALLING,
    main_llm_provider=ModelProvider.GROQ,
    tool_llm_provider=ModelProvider.GROQ,
)

fc_config_bedrock = AgentConfig(
    agent_type=AgentType.FUNCTION_CALLING,
    main_llm_provider=ModelProvider.BEDROCK,
    tool_llm_provider=ModelProvider.BEDROCK,
)

# ReAct configurations for all providers
react_config_anthropic = AgentConfig(
    agent_type=AgentType.REACT,
    main_llm_provider=ModelProvider.ANTHROPIC,
    tool_llm_provider=ModelProvider.ANTHROPIC,
)

react_config_gemini = AgentConfig(
    agent_type=AgentType.REACT,
    main_llm_provider=ModelProvider.GEMINI,
    main_llm_model_name="models/gemini-2.5-flash-lite",
    tool_llm_provider=ModelProvider.GEMINI,
    tool_llm_model_name="models/gemini-2.5-flash-lite",
)

react_config_together = AgentConfig(
    agent_type=AgentType.REACT,
    main_llm_provider=ModelProvider.TOGETHER,
    tool_llm_provider=ModelProvider.TOGETHER,
)

react_config_openai = AgentConfig(
    agent_type=AgentType.REACT,
    main_llm_provider=ModelProvider.OPENAI,
    tool_llm_provider=ModelProvider.OPENAI,
)

react_config_groq = AgentConfig(
    agent_type=AgentType.REACT,
    main_llm_provider=ModelProvider.GROQ,
    tool_llm_provider=ModelProvider.GROQ,
)


# Private LLM configurations
private_llm_react_config = AgentConfig(
    agent_type=AgentType.REACT,
    main_llm_provider=ModelProvider.PRIVATE,
    main_llm_model_name="gpt-4.1-mini",
    private_llm_api_base="http://localhost:8000/v1",
    tool_llm_provider=ModelProvider.PRIVATE,
    tool_llm_model_name="gpt-4.1-mini",
)

private_llm_fc_config = AgentConfig(
    agent_type=AgentType.FUNCTION_CALLING,
    main_llm_provider=ModelProvider.PRIVATE,
    main_llm_model_name="gpt-4.1-mini",
    private_llm_api_base="http://localhost:8000/v1",
    tool_llm_provider=ModelProvider.PRIVATE,
    tool_llm_model_name="gpt-4.1-mini",
)


# ========================================
# Error Detection and Testing Utilities
# ========================================


def is_rate_limited(response_text: str) -> bool:
    """
    Check if a response indicates a rate limit error from any LLM provider.

    Args:
        response_text: The response text from the agent

    Returns:
        bool: True if the response indicates rate limiting
    """
    rate_limit_indicators = [
        # Generic indicators
        "Error code: 429",
        "rate_limit_exceeded",
        "Rate limit reached",
        "rate limit",
        "quota exceeded",
        "usage limit",
        # OpenAI-specific
        "requests per minute",
        "RPM",
        "tokens per minute",
        "TPM",
        # Anthropic-specific
        "overloaded_error",
        "Overloaded",
        "APIStatusError",
        "anthropic.APIStatusError",
        "usage_limit_exceeded",
        # General API limit indicators
        "try again in",
        "Please wait",
        "Too many requests",
        "throttled",
        # Additional rate limit patterns
        "Limit.*Used.*Requested",
        "Need more tokens",
        # Provider failure patterns
        "failure can't be resolved after",
        "Got empty message",
    ]

    response_lower = response_text.lower()
    return any(
        indicator.lower() in response_lower for indicator in rate_limit_indicators
    )


def is_api_key_error(response_text: str) -> bool:
    """
    Check if a response indicates an API key authentication error.

    Args:
        response_text: The response text from the agent

    Returns:
        bool: True if the response indicates API key issues
    """
    api_key_indicators = [
        "Error code: 401",
        "Invalid API Key",
        "authentication",
        "unauthorized",
        "invalid_api_key",
        "missing api key",
        "api key not found",
    ]

    response_lower = response_text.lower()
    return any(indicator.lower() in response_lower for indicator in api_key_indicators)


def skip_if_rate_limited(
    test_instance: unittest.TestCase, response_text: str, provider: str = "LLM"
) -> None:
    """
    Skip a test if the response indicates rate limiting.

    Args:
        test_instance: The test case instance
        response_text: The response text to check
        provider: The name of the provider (for clearer skip messages)
    """
    if is_rate_limited(response_text):
        test_instance.skipTest(f"{provider} rate limit reached - skipping test")


def skip_if_api_key_error(
    test_instance: unittest.TestCase, response_text: str, provider: str = "LLM"
) -> None:
    """
    Skip a test if the response indicates API key issues.

    Args:
        test_instance: The test case instance
        response_text: The response text to check
        provider: The name of the provider (for clearer skip messages)
    """
    if is_api_key_error(response_text):
        test_instance.skipTest(f"{provider} API key invalid/missing - skipping test")


def skip_if_provider_error(
    test_instance: unittest.TestCase, response_text: str, provider: str = "LLM"
) -> None:
    """
    Skip a test if the response indicates common provider errors (rate limiting or API key issues).

    Args:
        test_instance: The test case instance
        response_text: The response text to check
        provider: The name of the provider (for clearer skip messages)
    """
    skip_if_rate_limited(test_instance, response_text, provider)
    skip_if_api_key_error(test_instance, response_text, provider)


class AgentTestMixin:
    """
    Mixin class providing utility methods for agent testing.
    """

    @contextmanager
    def with_provider_fallback(self, provider: str = "LLM"):
        """
        Context manager that catches and handles provider errors from any agent method.

        Args:
            provider: Provider name for error messages

        Usage:
            with self.with_provider_fallback("OpenAI"):
                response = agent.chat("test")

            with self.with_provider_fallback("OpenAI"):
                async for chunk in agent.astream_chat("test"):
                    pass

        Raises:
            unittest.SkipTest: If rate limiting or API key errors occur
        """
        try:
            yield
        except Exception as e:
            error_text = str(e)
            if is_rate_limited(error_text) or is_api_key_error(error_text):
                self.skipTest(f"{provider} error: {error_text}")
            raise

    def check_response_and_skip(self, response: Any, provider: str = "LLM") -> Any:
        """
        Check response content and skip test if provider errors are detected.

        Args:
            response: The response object from agent method
            provider: Provider name for error messages

        Returns:
            The response object if no errors detected

        Raises:
            unittest.SkipTest: If rate limiting or API key errors detected in response
        """
        response_text = getattr(response, "response", str(response))
        skip_if_provider_error(self, response_text, provider)
        return response
