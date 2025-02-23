from flask import Flask, request, jsonify
import json
import os
import numpy as np
import random
import nltk
import tensorflow_hub as hub
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allows the API to be accessed from different devices

# Load necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load trained AI model
try:
    use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    model_path = "chatbot_model.keras"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found!")
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Prevent crashing if model fails to load

# Load intents
try:
    json_path = "ok11.json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Intent file '{json_path}' not found!")
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    all_tags = [intent['tag'] for intent in data['intents']]
except Exception as e:
    print(f"Error loading intents: {e}")
    data = {"intents": []}
    all_tags = []

# Prepare helper functions
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    words = word_tokenize(text.lower())
    return " ".join([lemmatizer.lemmatize(w) for w in words if w.isalnum() and w not in stop_words])

def predict_intent(text):
    if model is None:
        return "Chatbot is currently unavailable."

    processed_text = preprocess_text(text)
    X_input = use_model([processed_text]).numpy()
    prediction = model.predict(X_input)
    tag_index = np.argmax(prediction)
    confidence = np.max(prediction)

    if confidence < 0.7:
        return "I'm not sure I understand. Can you please rephrase?"

    tag = all_tags[tag_index]
    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    
    return "Sorry, I don't understand that."

# Define API endpoint
@app.route("/chatbot", methods=["POST"])
def chatbot():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"response": "Invalid input"}), 400
    response = predict_intent(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Railway-assigned port
    app.run(host="0.0.0.0", port=port, debug=True)
