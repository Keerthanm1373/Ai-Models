import torch
import pickle
import numpy as np
import os
import google.generativeai as genai
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from typing import Dict, List, Optional, Union, Any

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set. Please check your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

# Define default greetings
DEFAULT_GREETINGS = {
    'hello': 'Hello! I am Triplo, your Karnataka travel guide. How may I assist you today?',
    'hi': 'Hi! I am Triplo, ready to help you explore Karnataka!',
    'hey': 'Hey there! I am Triplo, your Karnataka travel companion!',
    'good morning': 'Good morning! I am Triplo, ready to plan your Karnataka adventure!',
    'good afternoon': 'Good afternoon! This is Triplo, your Karnataka travel expert!',
    'good evening': 'Good evening! Triplo here to help with your Karnataka travel plans!',
    'namaste': 'Namaste! I am Triplo, your guide to Karnataka\'s wonders!'
}

# Identity-related keywords
IDENTITY_KEYWORDS = [
    'who are you', 'what are you', 'your name', 'who created you', 
    'what do you do', 'introduce yourself', 'tell me about yourself'
]

IDENTITY_RESPONSE = "I am Triplo, a travel guide assistant powered by Triple, specialized in Karnataka tourism."

# Define prompt templates
PROMPT_TEMPLATES = {
    "general": "Tell me about travel destinations in Karnataka, focusing on {location}.",
    "itinerary": "Create an itinerary for {location} in Karnataka for {duration} days.",
    "budget": "What's the budget needed for {location} in Karnataka?",
    "transportation": "How to travel around {location} in Karnataka?",
    "accommodation": "Where to stay in {location}, Karnataka?"
}

def initialize_bot() -> Dict[str, Any]:
    """Initialize the chatbot by loading the model and required data"""
    try:
        with open('karnataka_travel_model.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error initializing bot: {str(e)}")
        return {}

def generate_greeting(user_input: str) -> Optional[str]:
    """Generate a contextual greeting using Gemini"""
    prompt = """As Triplo, a Karnataka travel guide assistant, generate a single, friendly greeting line (maximum 15 words) in response to: {user_input}. 
    The response should be welcoming but brief."""
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt.format(user_input=user_input))
        if hasattr(response, 'text'):
            greeting = response.text.strip().split('\n')[0]
            return greeting[:100] if len(greeting) > 100 else greeting
    except Exception:
        return None

def is_identity_question(text: str) -> bool:
    """Check if the input is asking about the bot's identity"""
    return any(keyword in text.lower() for keyword in IDENTITY_KEYWORDS)

def handle_user_input(user_input: str) -> Optional[str]:
    """Handle initial user input processing for greetings and identity"""
    user_input_lower = user_input.lower().strip()
    
    if is_identity_question(user_input_lower):
        return IDENTITY_RESPONSE
    
    if user_input_lower in DEFAULT_GREETINGS:
        return DEFAULT_GREETINGS[user_input_lower]
    
    if any(user_input_lower.startswith(greeting) for greeting in DEFAULT_GREETINGS.keys()):
        generated_greeting = generate_greeting(user_input)
        if generated_greeting:
            return generated_greeting
    
    return None

def encode_text(text: str, tokenizer: BertTokenizer, model: BertModel) -> np.ndarray:
    """Encode text using BERT model"""
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings / np.linalg.norm(embeddings)

def get_best_match(user_query: str, data: Dict[str, Any]) -> Optional[str]:
    """Find the best matching response from the trained model"""
    try:
        if not data or 'tokenizer' not in data or 'model' not in data or 'embeddings' not in data:
            return None
            
        query_embedding = encode_text(user_query.strip().lower(), data['tokenizer'], data['model'])
        similarities = cosine_similarity(query_embedding, data['embeddings'])[0]
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        threshold = 0.80
        
        if best_score > threshold and 'answers' in data:
            return data['answers'][best_idx]
        return None
    except Exception as e:
        print(f"Error in get_best_match: {str(e)}")
        return None

def format_gemini_prompt(user_query: str, prompt_type: str = "general") -> str:
    """Format the prompt for Gemini API"""
    base_prompt = """You are Triplo, an AI travel guide specialized in Karnataka tourism. 
    Provide information ONLY about Karnataka tourist places or locations.
    Format the response as follows:
    - Basic introduction about the place in 1 line
    - Purpose of Visit in 2 lines
    - Historical Significance in 2 lines
    - Summary in 2-4 lines
    
    If the query is not about Karnataka tourism, politely decline to answer.
    """
    
    template = PROMPT_TEMPLATES.get(prompt_type, PROMPT_TEMPLATES["general"])
    return f"{base_prompt}\n\n{template}\n\nQuery: {user_query}"

def get_gemini_response(user_query: str, prompt_type: str = "general") -> str:
    """Get response from Gemini API"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = format_gemini_prompt(user_query, prompt_type)
        response = model.generate_content(prompt)
        
        if hasattr(response, 'text'):
            return response.text.strip()
        return "I apologize, but I can only provide information about tourist places in Karnataka. Please try asking about Karnataka tourism."
    except Exception as e:
        return f"Error fetching response: {str(e)}"

def detect_prompt_type(user_query: str) -> str:
    """Detect the type of travel query"""
    query_lower = user_query.lower()
    if any(word in query_lower for word in ['itinerary', 'schedule', 'plan']):
        return 'itinerary'
    elif any(word in query_lower for word in ['budget', 'cost', 'price', 'expensive']):
        return 'budget'
    elif any(word in query_lower for word in ['transport', 'travel', 'get around']):
        return 'transportation'
    elif any(word in query_lower for word in ['hotel', 'stay', 'accommodation']):
        return 'accommodation'
    return 'general'

def generate_response(user_input: str, bot_data: Dict[str, Any]) -> str:
    """Generate response for API endpoint"""
    try:
        # Check for greetings and identity questions first
        initial_response = handle_user_input(user_input)
        if initial_response:
            return initial_response
        
        prompt_type = detect_prompt_type(user_input)
        response = get_best_match(user_input, bot_data)
        
        if response is None:
            response = get_gemini_response(user_input, prompt_type)
        
        return response
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}"