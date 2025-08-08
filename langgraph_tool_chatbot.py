import os
import json
import logging
import threading
import time
import uuid
from flask import Flask, render_template, request, jsonify, Response, session
from graph import build_graph

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET",
                                "dev-secret-key-change-in-production")

# Global variables for tool loading state
tools_loaded = False
chatbot_graph = None
loading_status = "Initializing tools..."


def initialize_tools():
    """Initialize all tools in a separate thread"""
    global tools_loaded, chatbot_graph, loading_status

    try:
        loading_status = "Loading Wikipedia tool..."
        logger.info("Starting tool initialization...")

        loading_status = "Connecting to SQL database..."
        time.sleep(1)  # Brief pause for UI feedback

        loading_status = "Checking for PDF documents in data/ folder..."
        time.sleep(1)  # Brief pause for UI feedback

        loading_status = "Building vector database (this may take a while)..."
        time.sleep(1)  # Brief pause for UI feedback

        # Initialize the LangGraph chatbot with all tools
        chatbot_graph = build_graph()

        loading_status = "Tools loaded successfully!"
        tools_loaded = True
        logger.info("All tools initialized successfully!")

    except Exception as e:
        loading_status = f"Error loading tools: {str(e)}"
        logger.error(f"Failed to initialize tools: {e}")


# Start tool initialization in background
init_thread = threading.Thread(target=initialize_tools)
init_thread.daemon = True
init_thread.start()


@app.route('/')
def index():
    # 새로고침할 때마다 새로운 세션 ID 생성
    new_thread_id = str(uuid.uuid4())
    session['thread_id'] = new_thread_id
    logger.info(f"새로운 세션 시작: {new_thread_id}")
    return render_template('index.html')


@app.route('/status')
def status():
    """Check if tools are loaded"""
    return jsonify({'loaded': tools_loaded, 'status': loading_status})


@app.route('/chat', methods=['POST'])
def chat():
    global chatbot_graph

    if not tools_loaded:
        return jsonify({'error': 'Tools are still loading. Please wait.'}), 503

    if chatbot_graph is None:
        return jsonify({'error': 'Chatbot is not initialized.'}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({'error': 'Empty message'}), 400

        # 세션에서 thread_id 가져오기 (generator 함수 밖에서 미리 가져옴)
        thread_id = session.get('thread_id', str(uuid.uuid4()))
        logger.info(f"메시지 처리 중 - 세션 ID: {thread_id}")

        def generate():
            try:
                # Create initial state with user message using proper message format
                from langchain_core.messages import HumanMessage
                from graph import State
                from langgraph.managed.is_last_step import RemainingSteps

                initial_state: State = {
                    "messages": [HumanMessage(content=user_message)],
                    "remaining_steps": 15
                }

                from langchain_core.runnables import RunnableConfig
                
                config: RunnableConfig = {
                    "configurable": {
                        "thread_id": thread_id
                    },
                    "recursion_limit": 30
                }

                # Use invoke instead of stream for direct response
                if chatbot_graph is not None:
                    result = chatbot_graph.invoke(initial_state, config)
                else:
                    raise Exception("Chatbot graph is not initialized")

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
                logger.error(f"Error in chat generation: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(generate(), mimetype='text/plain')

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': f'Failed to get response: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
