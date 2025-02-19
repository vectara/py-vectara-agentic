import os
from typing import Any, Optional

from llama_index.llms.openai_like import OpenAILike

class PrivateLLM(OpenAILike):
    """
    Custom LLM call.
    """
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: str = "http://llm-endpoint.company.com:8000/v1",
        is_chat_model: bool = True,
        is_function_calling_model: bool = True,
        **kwargs: Any,
    ) -> None:
        
        super().__init__(
            model=model,
            api_key=api_key,
            api_base=api_base,
            is_chat_model=is_chat_model,
            is_function_calling_model=is_function_calling_model,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "PrivateLLM"
    
