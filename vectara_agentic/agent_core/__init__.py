"""
Agent core module containing essential components for agent functionality.

This module organizes core agent functionality into focused components:
- factory: Agent creation and configuration
- streaming: Streaming response handling and adapters
- serialization: Agent persistence and restoration
- evaluation: Response evaluation and post-processing
- prompts: Core prompt templates and instructions
- utils: Shared utilities for prompts, schemas, tools, and logging
"""

# Import main utilities that should be available at agent module level
from .streaming import (
    StreamingResponseAdapter,
    FunctionCallingStreamHandler,
    StandardStreamHandler,
    extract_response_text_from_chat_message,
    create_stream_post_processing_task,
)

__all__ = [
    "StreamingResponseAdapter",
    "FunctionCallingStreamHandler",
    "StandardStreamHandler",
    "extract_response_text_from_chat_message",
    "create_stream_post_processing_task",
]
