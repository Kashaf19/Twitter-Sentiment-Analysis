from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and tokenizer
MODEL_PATH = "TwiterrSentimentAnalysis.h5"
model = load_model(MODEL_PATH)

TOKENIZER_PATH = "tokenizer.pkl"
with open(TOKENIZER_PATH, "rb") as file:
    tokenizer = pickle.load(file)

# Set the maximum length for padding
MAX_LEN = 500

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the user
    input_text = request.form['text']
    
    # Preprocess the input text
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_padded = pad_sequences(input_sequence, maxlen=MAX_LEN)
    
    # Predict the sentiment
    prediction = model.predict(input_padded)
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    probability = float(prediction[0][0])

    return render_template('result.html', 
                           sentiment=sentiment, 
                           probability=probability, 
                           text=input_text)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    # Get JSON input
    data = request.get_json()
    input_text = data['text']

    # Preprocess the input text
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_padded = pad_sequences(input_sequence, maxlen=MAX_LEN)

    # Predict the sentiment
    prediction = model.predict(input_padded)
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    probability = float(prediction[0][0])

    return jsonify({
        "sentiment": sentiment,
        "probability": probability
    })

if __name__ == '__main__':
    app.run(debug=True)