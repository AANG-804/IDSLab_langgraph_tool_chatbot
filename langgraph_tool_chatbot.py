import os
import json
from flask import Flask, render_template, request, jsonify, Response
from graph import build_graph

app = Flask(__name__)

# Initialize the LangGraph chatbot
chatbot_graph = build_graph()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({'error': 'Empty message'}), 400

        def generate():
            try:
                # Create initial state with user message using proper message format
                from langchain_core.messages import HumanMessage
                from graph import State

                initial_state: State = {
                    "messages": [HumanMessage(content=user_message)]
                }

                thread_id = "1234567890"
                config = {"configurable": {"thread_id": thread_id}}

                # Use invoke instead of stream for direct response
                result = chatbot_graph.invoke(initial_state, config)
                
                # Get the bot response from the result
                if "messages" in result:
                    latest_message = result["messages"][-1]
                    if hasattr(latest_message, 'content'):
                        content = latest_message.content
                    else:
                        content = latest_message.get('content', '')
                    
                    # Send the complete response as chunks for streaming effect
                    # Split into smaller chunks to simulate streaming
                    words = content.split()
                    for i in range(0, len(words), 3):  # Send 3 words at a time
                        chunk = ' '.join(words[i:i + 3])
                        if i + 3 < len(words):
                            chunk += ' '
                        yield f"data: {json.dumps({'content': chunk})}\n\n"

                # Send end signal
                yield f"data: {json.dumps({'done': True})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(generate(), mimetype='text/plain')

    except Exception as e:
        return jsonify({'error': f'Failed to get response: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8540, debug=False)
