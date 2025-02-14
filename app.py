from flask import Flask, render_template, request
import os
from huggingface_hub import InferenceClient

app = Flask(__name__)

# Initialize the client with your API key
client = InferenceClient(api_key=os.getenv("HFToken"))

# Initialize the conversation history
conversation_history = []

@app.route("/", methods=["GET", "POST"])
def index():
    global conversation_history
    if request.method == "POST":
        user_input = request.form["user_input"]
        conversation_history.append({"role": "user", "content": user_input})
        completion = client.chat.completions.create(
            model="Qwen/Qwen2.5-Coder-32B-Instruct", 
            messages=conversation_history, 
            temperature=0.5,
            max_tokens=2048,
            top_p=0.7
        )
        ai_response = completion.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": ai_response})
        return render_template("index.html", conversation_history=conversation_history)
    return render_template("index.html", conversation_history=conversation_history)
