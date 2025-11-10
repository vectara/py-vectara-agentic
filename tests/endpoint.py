#!/usr/bin/env python3
import json
import logging
import os
from functools import wraps
from flask import Flask, request, Response, jsonify
from openai import OpenAI

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)
werkzeug_log = logging.getLogger("werkzeug")
werkzeug_log.setLevel(logging.ERROR)

# Load expected API key from environment (fallback for testing)
EXPECTED_API_KEY = "TEST_API_KEY"

# Together.AI models that should be routed to Together.AI
TOGETHER_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-V3.1",
    "deepseek-ai/DeepSeek-R1",
]


# Authentication decorator
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Unauthorized"}), 401
        api_key = auth_header.split(" ", 1)[1]
        if api_key != EXPECTED_API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)

    return decorated_function


@app.before_request
def log_request_info():
    app.logger.info("%s %s", request.method, request.path)


@app.route("/v1/chat/completions", methods=["POST"])
@require_api_key
def chat_completions():
    """
    Proxy endpoint for OpenAI-compatible Chat Completions.
    Routes to different backends based on model name.
    Supports both streaming and non-streaming modes.
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON payload"}), 400

    # Determine which client to use based on model name
    model_name = data.get("model", "")
    if model_name in TOGETHER_MODELS:
        # Route to Together.AI
        together_api_key = os.getenv("TOGETHER_API_KEY")
        if not together_api_key:
            return jsonify({"error": "TOGETHER_API_KEY environment variable not set"}), 500
        client = OpenAI(
            api_key=together_api_key,
            base_url="https://api.together.xyz/v1"
        )
    else:
        # Default to OpenAI
        client = OpenAI()

    is_stream = data.get("stream", False)

    if is_stream:
        # Stream each chunk to the client as Server-Sent Events
        def generate():
            try:
                for chunk in client.chat.completions.create(**data):
                    # Convert chunk to dict and then JSON
                    event = chunk.model_dump()
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as e:
                # On error, send an SSE event with error info
                error_msg = json.dumps({"error": str(e)})
                yield f"data: {error_msg}\n\n"

        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
        return Response(generate(), headers=headers)

    # Non-streaming path
    try:
        completion = client.chat.completions.create(**data)
        result = completion.model_dump()
        app.logger.info(f"Non-stream response: {result}")
        return jsonify(result), 200
    except Exception as e:
        app.logger.error(f"Error during completion: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Bind to all interfaces on port 5000
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
