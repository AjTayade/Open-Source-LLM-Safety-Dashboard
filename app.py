import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from a .env file (for local testing)
load_dotenv()

# Initialize the Flask app
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) to allow requests from the frontend
CORS(app) 

# Get your Hugging Face API token from environment variables
# This is the secure way. Render will provide this value.
HF_API_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
API_BASE_URL = "https://api-inference.huggingface.co/models/"

@app.route('/generate', methods=['POST'])
def generate_text():
    """
    Endpoint to proxy requests to the Hugging Face Inference API.
    It takes a 'prompt' and 'model_name' from the client,
    securely adds the API key, and forwards the request.
    """
    
    # Get data from the frontend request
    data = request.get_json()
    prompt = data.get('prompt')
    model_name = data.get('model_name') # e.g., "deepseek-ai/DeepSeek-V3.2-Exp"

    if not all([prompt, model_name]):
        return jsonify({"error": "Invalid request. 'prompt' and 'model_name' are required."}), 400

    if not HF_API_TOKEN:
        return jsonify({"error": "Hugging Face API token is not configured on the server."}), 500

    # Prepare the request for the Hugging Face API
    api_url = f"{API_BASE_URL}{model_name}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    # Set model parameters. 'wait_for_model' is important for models that aren't pre-loaded.
    payload = {
        "inputs": prompt,
        "options": {
            "wait_for_model": True
        }
    }

    try:
        # Make the request to Hugging Face
        response = requests.post(api_url, headers=headers, json=payload)
        
        # Check for model loading errors or other issues
        if response.status_code == 503:
            return jsonify({"error": "Model is loading on Hugging Face, please try again in a moment."}), 503
        
        # Raise an exception for other bad status codes (4xx or 5xx)
        response.raise_for_status() 
        
        result = response.json()

        # The response format can vary, but for text-generation it's often a list
        if isinstance(result, list) and result:
            generated_text = result[0].get('generated_text', "Error: Could not parse model output.")
        else:
            generated_text = result.get('generated_text', "Error: Unexpected API response format.")
        
        # We only want the newly generated text, not the input prompt
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        return jsonify({"generated_text": generated_text})

    except requests.exceptions.RequestException as e:
        # Handle network errors, timeouts, etc.
        print(f"Error calling Hugging Face API: {e}")
        error_message = f"An error occurred: {str(e)}"
        if e.response:
            error_message = e.response.json().get("error", str(e))
        return jsonify({"error": error_message}), 500

if __name__ == '__main__':
    # This block runs only when you execute `python app.py` locally
    # Render will use Gunicorn and ignore this
    print("Starting Flask server for local development...")
    app.run(host='0.0.0.0', port=5000, debug=True)

