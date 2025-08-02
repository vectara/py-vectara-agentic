# API Endpoint

It's super easy to host your vectara-agentic assistant or agent behind
an API endpoint:

`vectara-agentic` can be easily hosted locally or on a remote machine
behind an API endpoint, by following these steps:

1\. **Setup your API key** Ensure that you have your API key set up as
an environment variable:

``` python
export VECTARA_AGENTIC_API_KEY=<YOUR-ENDPOINT-API-KEY>
```

2\. **Start the API Server** Initialize the agent and start the FastAPI
server by following this example:

``` python
from agent import Agent
from agent_endpoint import start_app
agent = Agent(...)      # Initialize your agent with appropriate parameters
start_app(agent)
```

You can customize the host and port by passing them as arguments to
start_app().

For example:

``` python
start_app(agent, host="0.0.0.0", port=8000)
```

3\. **Access the API Endpoint** Once the server is running, you can
interact with it using curl or any HTTP client. For example:

``` python
curl -G "http://<remote-server-ip>:8000/chat" \
--data-urlencode "message=What is Vectara?" \
-H "X-API-Key: <YOUR-API-KEY>"
```
