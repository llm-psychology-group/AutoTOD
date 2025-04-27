from flask import Flask, request, jsonify

from agent import Agent

app = Flask(__name__)

model_name = None
chatbot_agent = None



@app.route("/init-session", methods=["POST"])
def init_session():
    data = request.json
    model_name = data["model_name"]
    global chatbot_agent
    chatbot_agent = Agent(model=model_name)
    print(f"Session initialized with model: {model_name}")
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
