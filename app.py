from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from colorama import init, Fore, Style
import os
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize colorama for colored output
init()

def get_project_root():
    """Get the absolute path to the project root directory."""
    try:
        if getattr(sys, 'frozen', False):
            return Path(sys._MEIPASS)
        return Path(__file__).resolve().parent
    except Exception as e:
        logger.error(f"Error getting project root: {str(e)}")
        raise

def ensure_directory_exists(directory):
    """Ensure a directory exists, create if it doesn't."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory exists or was created: {directory}")
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory}: {str(e)}")
        return False

def load_model_files():
    """Load model and vectorizer files with error handling."""
    try:
        model_path = models_dir / 'fake_news_model.joblib'
        vectorizer_path = models_dir / 'tfidf_vectorizer.joblib'
        
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Loading vectorizer from: {vectorizer_path}")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer file not found at: {vectorizer_path}")
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        logger.info(f"{Fore.GREEN}Model and vectorizer loaded successfully!{Style.RESET_ALL}")
        return model, vectorizer
    except Exception as e:
        logger.error(f"{Fore.RED}Error loading model: {str(e)}{Style.RESET_ALL}")
        return None, None

# Get the absolute path to the project root
project_root = get_project_root()

# Create necessary directories if they don't exist
models_dir = project_root / 'models'
templates_dir = project_root / 'templates'

# Ensure directories exist
if not ensure_directory_exists(models_dir) or not ensure_directory_exists(templates_dir):
    raise RuntimeError("Failed to create necessary directories")

# Create Flask app with template folder path
app = Flask(__name__, template_folder=str(templates_dir))

# Load the model and vectorizer
model, vectorizer = load_model_files()

def get_certainty(confidence):
    """Determine certainty level based on confidence score."""
    if confidence >= 70:
        return "high"
    elif confidence >= 40:
        return "moderate"
    return "low"

@app.route('/')
def home():
    """Render the home page."""
    try:
        logger.info(f"Attempting to render template from: {templates_dir}")
        template_path = templates_dir / 'index.html'
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found at: {template_path}")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}")
        return "Error loading template", 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions on the input text."""
    try:
        if model is None or vectorizer is None:
            return jsonify({
                'success': False,
                'error': "Model not loaded. Please try again later."
            }), 500

        text = request.json.get('text')
        if not text:
            return jsonify({
                'success': False,
                'error': "No text provided"
            }), 400

        logger.info(f"{Fore.CYAN}Analyzing text: {text[:100]}...{Style.RESET_ALL}")
        
        # Transform the input text
        text_vectorized = vectorizer.transform([text])
        logger.info(f"{Fore.GREEN}Text vectorization successful{Style.RESET_ALL}")
        
        # Make prediction
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        confidence = float(max(probabilities) * 100)
        
        logger.info(f"{Fore.YELLOW}Prediction: {prediction}")
        logger.info(f"Confidence: {confidence:.2f}%")
        logger.info(f"Probabilities: {probabilities}{Style.RESET_ALL}")
        
        certainty = get_certainty(confidence)
        
        return jsonify({
            'success': True,
            'prediction': "Real News" if prediction == 1 else "Fake News",
            'confidence': confidence,
            'certainty': certainty
        })
    except Exception as e:
        logger.error(f"{Fore.RED}Error during prediction: {str(e)}{Style.RESET_ALL}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 