from flask import Flask, request, jsonify
from flask_cors import CORS  # Already imported and used for CORS handling
from chatbot.chatbot import generate_response, initialize_bot
from typing import Dict
import pandas as pd
import sys
from pathlib import Path
from flask import Flask, request, jsonify
# In app.py
from budgetplanning.bd_pridiction import predict_budget

# Add the project root directory to Python path
sys.path.append(str(Path(__file__).parent))

app = Flask(__name__)

# Enable CORS for the Flask app
CORS(app)  # You don't need to add middleware for CORS here

bot_data: Dict = initialize_bot()

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        user_message = data['message']
        response = generate_response(user_message, bot_data)
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500



@app.route('/predict_budget', methods=['POST'])
def predict_budget_api():
    try:
        # Get data from POST request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['from_place', 'to_place', 'trip_mode', 'number_of_travelers']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Validate trip_mode
        valid_modes = ['Car', 'Bus', 'Train', 'Flight']
        if data['trip_mode'] not in valid_modes:
            return jsonify({
                'error': f'Invalid trip_mode. Must be one of: {", ".join(valid_modes)}'
            }), 400
        
        # Validate number_of_travelers
        try:
            data['number_of_travelers'] = int(data['number_of_travelers'])
            if data['number_of_travelers'] <= 0:
                return jsonify({'error': 'number_of_travelers must be greater than 0'}), 400
        except ValueError:
            return jsonify({'error': 'number_of_travelers must be a valid integer'}), 400
        
        # Make prediction
        predicted_budget = predict_budget(data)
        
        # Return prediction result
        return jsonify({
            'status': 'success',
            'predicted_budget': round(float(predicted_budget), 2),
            # 'input_data': data
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Service is running'})

if __name__ == '__main__':
    app.run(debug=True, host="192.168.56.1", port=5000)
