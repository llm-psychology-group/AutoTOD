import json
from pathlib import Path
import random
from flask import Flask, request, jsonify

from user import User

app = Flask(__name__)

model_name = None
user_agent = None

BASE_PATH = Path(__file__).resolve().parent
DATA_PATH = BASE_PATH / "data/mwoz/origin/data.json"
with open(DATA_PATH) as f:
    dialogue_data = json.load(f)

# Note: due to the chatbot agent not being able to handle multiple sessions, the current server will only be able to handle one session at a time.


@app.route("/init-session", methods=["POST"])
def init_session():
    request_data = request.json
    dialogue_id = request_data.get("dialogue_id")
    if not dialogue_id:
        dialogue_id = random.choice(list(dialogue_data.keys()))
    dialogue = dialogue_data.get(dialogue_id)
    if not dialogue:
        return jsonify({"error": "Invalid dialogue_id"}), 400
    model_name = request_data.get("model_name", "gpt-4o")
    global user_agent
    user_agent = User(dialogue, model=model_name)
    return jsonify({"message": "Session initialized", "dialogue_id": dialogue_id})


@app.route("/get-answer", methods=["POST"])
def get_answer():
    request_data = request.json
    user_message = request_data.get("chatbot_message")
    if user_message is None:
        # Raise bad request error
        return jsonify({"error": "Bad request. chatbot_message missing"}), 400
    response = user_agent(user_message)
    is_end = False
    if "Dialogue Ends" in response:
        is_end = True
        response = response.replace("Dialogue Ends", "").strip()
    return jsonify({"user_answer": response, "is_end": is_end})


if __name__ == "__main__":
    app.run(debug=True, port=8083)
