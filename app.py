import os
from huggingface_hub import InferenceClient
# Initialize the client with your API key
client = InferenceClient(api_key=os.getenv("HFToken"))

# Initialize the conversation history
conversation_history = []

while True:
    # Get user input
    user_input = input("User: ")

    # Add the user's message to the conversation history
    conversation_history.append({"role": "user", "content": user_input})

    # Create a completion request
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-Coder-32B-Instruct", 
        messages=conversation_history, 
        temperature=0.5,
        max_tokens=2048,
        top_p=0.7
    )

    # Get the AI's response
    ai_response = completion.choices[0].message.content

    # Print the AI's response
    print(f"\nAI: {ai_response}\n")

    # Add the AI's response to the conversation history
    conversation_history.append({"role": "assistant", "content": ai_response})
