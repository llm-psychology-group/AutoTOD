import json

from flask import Flask, request, jsonify

from user import User

app = Flask(__name__)

model_name = 'gpt-3.5-turbo'
user_agent = None

with open('./data/mwoz/origin/data.json') as f:
    dialogue_data = json.load(f)

# Note: due to the chatbot agent not being able to handle multiple sessions, the current server will only be able to handle one session at a time.

@app.route('/init-session', methods=['POST'])
def init_session():
    request_data = request.json
    dialogue_id = request_data['dialogue_id']
    dialogue = dialogue_data[dialogue_id]
    global user_agent
    user_agent = User(dialogue, model=model_name)
    first_user_message = user_agent.fisrt_user_utter
    return jsonify({'first_user_message': first_user_message})


@app.route('/get-answer', methods=['POST'])
def get_answer():
    request_data = request.json
    user_message = request_data['chatbot_message']
    response = user_agent(user_message)
    is_end = False
    if 'Dialogue Ends' in response:
        is_end = True
    return jsonify({'user_answer': response, 'is_end': is_end})


if __name__ == '__main__':
    app.run(debug=True, port=8083)
