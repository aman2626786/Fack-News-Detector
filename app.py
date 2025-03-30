import os
import sys
import logging
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import re

# Try to import colorama, but don't fail if it's not available
try:
    from colorama import init, Fore, Style
    init()  # Initialize colorama
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    # Create dummy color functions for when colorama is not available
    class DummyColors:
        def __init__(self):
            self.GREEN = ''
            self.RED = ''
            self.YELLOW = ''
            self.RESET_ALL = ''
            self.BOLD = ''
    
    Fore = DummyColors()
    Style = DummyColors()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_project_root():
    """Get the absolute path to the project root directory."""
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle
        return os.path.dirname(sys.executable)
    else:
        # If the application is run from a Python interpreter
        return os.path.dirname(os.path.abspath(__file__))

# Get the project root directory
project_root = get_project_root()

# Create necessary directories if they don't exist
models_dir = os.path.join(project_root, 'models')
templates_dir = os.path.join(project_root, 'templates')
os.makedirs(models_dir, exist_ok=True)
os.makedirs(templates_dir, exist_ok=True)

# Initialize Flask app with the correct template folder
app = Flask(__name__, template_folder=templates_dir)

# Global variables for model and vectorizer
model = None
vectorizer = None

def load_model():
    """Load the trained model and vectorizer."""
    global model, vectorizer
    try:
        model_path = os.path.join(models_dir, 'fake_news_model.joblib')
        vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
        
        logger.info(f"Looking for model files in: {models_dir}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Vectorizer path: {vectorizer_path}")
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            logger.error(f"Model files not found at {model_path} or {vectorizer_path}")
            return False
            
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        logger.info("Model and vectorizer loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

# Load model at startup
if not load_model():
    logger.error("Failed to load model at startup")

@app.route('/')
def home():
    """Render the home page."""
    try:
        template_path = os.path.join(templates_dir, 'index.html')
        logger.info(f"Looking for template at: {template_path}")
        
        if not os.path.exists(template_path):
            logger.error(f"Template file not found at {template_path}")
            return "Error: Template file not found", 500
            
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering home page: {str(e)}")
        return "Error: Failed to render home page", 500

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        if model is None or vectorizer is None:
            logger.error("Model or vectorizer not loaded")
            return jsonify({
                'error': 'Model not loaded',
                'status': 'error'
            }), 500

        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({
                'error': 'No text provided',
                'status': 'error'
            }), 400

        # Log the prediction process
        logger.info(f"Processing prediction for text: {text[:100]}...")
        
        # Transform the input text
        text_vectorized = vectorizer.transform([text])
        
        # Get prediction and probability
        prediction = model.predict(text_vectorized)[0]
        probability = model.predict_proba(text_vectorized)[0]
        
        # Determine confidence level
        confidence = float(probability[1] if prediction == 1 else probability[0])
        
        # Categorize certainty
        if confidence >= 0.8:
            certainty = "High"
        elif confidence >= 0.6:
            certainty = "Moderate"
        else:
            certainty = "Low"
        
        # Log the results
        logger.info(f"Prediction: {'Real' if prediction == 1 else 'Fake'}")
        logger.info(f"Confidence: {confidence:.2f}")
        logger.info(f"Certainty: {certainty}")
        
        return jsonify({
            'prediction': 'Real News' if prediction == 1 else 'Fake News',
            'confidence': confidence,
            'certainty': certainty,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 