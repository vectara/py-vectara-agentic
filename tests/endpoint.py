from openai import OpenAI
from flask import Flask, request, jsonify
import logging
from functools import wraps

app = Flask(__name__)
app.config['TESTING'] = True

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Set your OpenAI API key (ensure you've set this in your environment)

EXPECTED_API_KEY = "TEST_API_KEY"

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get("Authorization").split("Bearer ")[-1]
        if not api_key or api_key != EXPECTED_API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.before_request
def log_request_info():
    app.logger.info("Request received: %s %s", request.method, request.path)

@app.route("/v1/chat/completions", methods=["POST"])
@require_api_key
def chat_completions():
    app.logger.info("Received request on /v1/chat/completions")
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    client = OpenAI()
    try:
        completion = client.chat.completions.create(**data)
        return jsonify(completion.model_dump()), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # Run on port 5000 by default; adjust as needed.
    app.run(debug=True, port=5000, use_reloader=False)
