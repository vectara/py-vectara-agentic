from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from agent import Agent  # Assuming agent.py is in the same directory

class ChatRequest(BaseModel):
    message: str

class AgentEndpoint:
    def __init__(self, agent: Agent):
        self.agent = agent
        self.app = FastAPI()

        @self.app.get("/chat")
        async def chat(message: str):
            if not message:
                raise HTTPException(status_code=400, detail="No message provided")

            response = self.agent.chat(message)
            return {"response": response}

    def start(self, host='127.0.0.1', port=8001):
        """
        Start the FastAPI server.

        Args:
            host (str, optional): The host address for the API. Defaults to '127.0.0.1'.
            port (int, optional): The port for the API. Defaults to 8000.
        """
        uvicorn.run(self.app, host=host, port=port)

# Example usage:
#     from agent import Agent
#     agent = Agent(tools=tools, topic="general")
#     endpoint = AgentEndpoint(agent)
#     endpoint.start()
#
# Then:
#     curl -G "http://127.0.0.1:8000/chat" --data-urlencode "message=Hello, Agent!"