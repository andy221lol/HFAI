from flask import Flask, render_template, request, jsonify
import os
from huggingface_hub import InferenceClient

app = Flask(__name__)
token = os.getenv("HFToken")
# Initialize the client with your API key
client = InferenceClient(api_key=token)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        # Initialize conversation history as empty if no session exists
        conversation_history = []
        return render_template("index.html", conversation_history=conversation_history)
    
    elif request.method == "POST":
        data = request.get_json()
        user_input = data['message']
        conversation_history = data.get('conversationHistory', [])

        # Append user's message to the conversation history
        conversation_history.append({"role": "user", "content": user_input})
        
        # Generate AI response
        completion = client.chat.completions.create(
            model="Qwen/Qwen2.5-Coder-32B-Instruct", 
            messages=conversation_history, 
            temperature=0.5,
            max_tokens=2048,
            top_p=0.7
        )
        
        ai_response = completion.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": ai_response})

        # Return updated conversation history
        return jsonify({'conversationHistory': conversation_history})

if __name__ == "__main__":
    app.run(debug=True)
