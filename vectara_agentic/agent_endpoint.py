"""
agent_endpoint.py
"""

import logging
import time
import uuid
from typing import Any, List, Literal, Optional, Union

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
import uvicorn

from .agent import Agent
from .agent_config import AgentConfig


class ChatRequest(BaseModel):
    """Request schema for the /chat endpoint."""

    message: str


class CompletionRequest(BaseModel):
    """Request schema for the /v1/completions endpoint."""

    model: str
    prompt: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(16, ge=1)
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(1, ge=1)
    stop: Optional[Union[str, List[str]]] = None


class Choice(BaseModel):
    """Choice schema returned in CompletionResponse."""

    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Literal["stop", "length", "error", None]


class CompletionUsage(BaseModel):
    """Token usage details in CompletionResponse."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    """Response schema for the /v1/completions endpoint."""

    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[Choice]
    usage: CompletionUsage


class ChatMessage(BaseModel):
    """Schema for individual chat messages in ChatCompletionRequest."""
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """Request schema for the /v1/chat endpoint."""
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(1, ge=1)


class ChatCompletionChoice(BaseModel):
    """Choice schema returned in ChatCompletionResponse."""
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "error", None]


class ChatCompletionResponse(BaseModel):
    """Response schema for the /v1/chat endpoint."""
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: CompletionUsage


def create_app(agent: Agent, config: AgentConfig) -> FastAPI:
    """
    Create and configure the FastAPI app.

    Args:
        agent (Agent): The agent instance to handle chat/completion.
        config (AgentConfig): Configuration containing the API key.

    Returns:
        FastAPI: Configured FastAPI application.
    """
    app = FastAPI()
    logger = logging.getLogger("uvicorn.error")
    logging.basicConfig(level=logging.INFO)

    api_key_header = APIKeyHeader(name="X-API-Key")

    async def _verify_api_key(api_key: str = Depends(api_key_header)):
        """
        Dependency that verifies the X-API-Key header.

        Raises:
            HTTPException(403): If the provided key does not match.

        Returns:
            bool: True if key is valid.
        """
        if api_key != config.endpoint_api_key:
            raise HTTPException(status_code=403, detail="Unauthorized")
        return True

    @app.get(
        "/chat", summary="Chat with the agent", dependencies=[Depends(_verify_api_key)]
    )
    async def chat(message: str):
        """
        Handle GET /chat requests.

        Args:
            message (str): The user's message to the agent.

        Returns:
            dict: Contains the agent's response under 'response'.

        Raises:
            HTTPException(400): If message is empty.
            HTTPException(500): On internal errors.
        """
        if not message:
            raise HTTPException(status_code=400, detail="No message provided")
        try:
            res = agent.chat(message)
            return {"response": res}
        except Exception as e:
            raise HTTPException(status_code=500, detail="Internal server error") from e

    @app.post(
        "/v1/completions",
        response_model=CompletionResponse,
        dependencies=[Depends(_verify_api_key)],
    )
    async def completions(req: CompletionRequest):
        """
        Handle POST /v1/completions requests.

        Args:
            req (CompletionRequest): The completion request payload.

        Returns:
            CompletionResponse: The generated completion and usage stats.

        Raises:
            HTTPException(400): If prompt is missing.
            HTTPException(500): On internal errors.
        """
        if not req.prompt:
            raise HTTPException(status_code=400, detail="`prompt` is required")
        raw = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
        try:
            start = time.time()
            text = agent.chat(raw)
            logger.info(f"Agent returned in {time.time()-start:.2f}s")
        except Exception as e:
            raise HTTPException(status_code=500, detail="Internal server error") from e

        p_tokens = len(raw.split())
        c_tokens = len(text.split())

        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4()}",
            object="text_completion",
            created=int(time.time()),
            model=req.model,
            choices=[Choice(text=text, index=0, logprobs=None, finish_reason="stop")],
            usage=CompletionUsage(
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                total_tokens=p_tokens + c_tokens,
            ),
        )

    @app.post(
        "/v1/chat",
        response_model=ChatCompletionResponse,
        dependencies=[Depends(_verify_api_key)],
    )
    async def chat_completion(req: ChatCompletionRequest):
        if not req.messages:
            raise HTTPException(status_code=400, detail="`messages` is required")

        # concatenate all user messages into a single prompt
        raw = " ".join(m.content for m in req.messages if m.role == "user")

        try:
            start = time.time()
            text = agent.chat(raw)
            logger.info(f"Agent returned in {time.time()-start:.2f}s")
        except Exception as e:
            raise HTTPException(status_code=500, detail="Internal server error") from e

        p_tokens = len(raw.split())
        c_tokens = len(text.split())

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            object="chat.completion",
            created=int(time.time()),
            model=req.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                total_tokens=p_tokens + c_tokens,
            ),
        )

    return app


def start_app(agent: Agent, host="0.0.0.0", port=8000):
    """
    Launch the FastAPI application using Uvicorn.

    Args:
        agent (Agent): The agent instance for request handling.
        host (str, optional): Host interface. Defaults to "0.0.0.0".
        port (int, optional): Port number. Defaults to 8000.
    """
    app = create_app(agent, config=AgentConfig())
    uvicorn.run(app, host=host, port=port)
