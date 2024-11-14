from flask import Flask, request, jsonify

from agent import Agent

app = Flask(__name__)

model_name = "gpt-3.5-turbo"
chatbot_agent = Agent(model=model_name)

# Note: due to the chatbot agent not being able to handle multiple sessions, the current server will only be able to handle one session at a time.


@app.route("/init-session", methods=["POST"])
def init_session():
    chatbot_agent.reset()
    return jsonify({"message": "Session initialized"})


@app.route("/get-answer", methods=["POST"])
def get_answer():
    data = request.json
    user_message = data["user_message"]
    response = chatbot_agent(user_message)
    return jsonify({"chatbot_answer": response})


if __name__ == "__main__":
    app.run(debug=True, port=8401)
