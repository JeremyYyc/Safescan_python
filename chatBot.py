import os
import json
import openai

from openai import OpenAI
from flask import Flask, request, jsonify

app = Flask(__name__)
client = OpenAI(api_key="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")

def format_questions(questions):
    if len(questions) == 1:
        memory = 'NO QUESTIONS'
        new_question = questions[0]
    else:
        memory = ''
        for i in range(len(questions) - 1):
            memory += f"Q{i + 1}: {questions[i]}\n"
        new_question = questions[-1]
    return memory.strip(), new_question
    
def prompt(memory):
    prompt = f"""You are a chatbot in a home safety analysis app. Based on the user's previous questions (if 'NO QUESTIONS' are shown, it is their first question), answer their new question. Please avoid making the response too lengthy or too summarized. Your primary tasks are to:
1. If the user asks how to address personal safety hazards or mental health issues, provide solutions from different perspectives (e.g., simple methods, cost-effective options, etc.).
2. Help tenants identify potential personal safety/mental health issues in their homes.
3. Provide safety guidelines for emergency situations, such as responding to fires and other unexpected incidents, and offer corresponding prevention advice.
4. Provide suggestions for maintaining a safe and comfortable indoor environment.
5. Explain why indoor lighting and color schemes affect mental health.
6. If the user identifies as belonging to any specific group, provide targeted safety suggestions accordingly.
7. Advise tenants on how to discuss personal safety/mental health or concerns with their landlords.
* For other unrelated questions, politely decline to answer.

Previous user questions:
{memory}
    """
    return prompt.strip()

@app.route('/processChat', methods=['POST'])
def process_chat_endpoint():
    questions_json = request.form.get('user_input')
    try:
        questions_dict = json.loads(questions_json)
        if 'questions' in questions_dict and isinstance(questions_dict['questions'], list) and questions_dict['questions']:
            questions = questions_dict['questions']
        else:
            return jsonify({"error": "'questions' key is missing or not a list or empty!"}), 400
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid user_input format!"}), 400
    
    memory, new_question = format_questions(questions)
    messages = [
        {"role": "system", "content": prompt(memory)},
        {"role": "user", "content": new_question}
    ]

    reply = ""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500
        )
        reply = response.choices[0].message.content
        reply = reply.replace("*", "").replace("-", "")
    except Exception as e:
        reply = "Chatbot is not available now!"
        print(f"API failed: {str(e)}")

    print(reply)
    response_data = {
        "reply": reply
    }

    return jsonify(response_data), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, threaded=True)