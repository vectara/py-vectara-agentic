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
    message: str

class CompletionRequest(BaseModel):
    model: str
    prompt: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(16, ge=1)
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(1, ge=1)
    stop: Optional[Union[str, List[str]]] = None

class Choice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Literal["stop", "length", "error", None]

class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[Choice]
    usage: CompletionUsage


def create_app(agent: Agent, config: AgentConfig) -> FastAPI:
    app = FastAPI()
    logger = logging.getLogger("uvicorn.error")
    logging.basicConfig(level=logging.INFO)

    # ← define the header and the verifier *inside* create_app,
    # so it closes over the config you passed.
    api_key_header = APIKeyHeader(name="X-API-Key")
    async def _verify_api_key(api_key: str = Depends(api_key_header)):
        if api_key != config.endpoint_api_key:
            raise HTTPException(status_code=403, detail="Unauthorized")
        return True

    @app.get(
        "/chat",
        summary="Chat with the agent",
        dependencies=[Depends(_verify_api_key)]
    )
    async def chat(message: str):
        if not message:
            raise HTTPException(status_code=400, detail="No message provided")
        try:
            res = agent.chat(message)
            return {"response": res}
        except Exception:
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post(
        "/v1/completions",
        response_model=CompletionResponse,
        dependencies=[Depends(_verify_api_key)]
    )
    async def completions(req: CompletionRequest):
        if not req.prompt:
            raise HTTPException(status_code=400, detail="`prompt` is required")
        raw = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
        try:
            start = time.time()
            text = agent.chat(raw)
            logger.info(f"Agent returned in {time.time()-start:.2f}s")
        except Exception:
            raise HTTPException(status_code=500, detail="Internal server error")

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

    return app


def start_app(agent: Agent, host="0.0.0.0", port=8000):
    app = create_app(agent, config=AgentConfig())
    uvicorn.run(app, host=host, port=port)
