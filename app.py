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
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle
        return Path(sys._MEIPASS)
    else:
        # If the application is run from a Python interpreter
        return Path(__file__).resolve().parent

# Get the absolute path to the project root
project_root = get_project_root()

# Create Flask app with template folder path
template_dir = project_root / 'templates'
app = Flask(__name__, template_folder=str(template_dir))

# Get the absolute path to the models directory
models_dir = project_root / 'models'

# Load the model and vectorizer
try:
    model_path = models_dir / 'fake_news_model.joblib'
    vectorizer_path = models_dir / 'tfidf_vectorizer.joblib'
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Template directory: {template_dir}")
    logger.info(f"Models directory: {models_dir}")
    logger.info(f"Loading model from: {model_path}")
    logger.info(f"Loading vectorizer from: {vectorizer_path}")
    
    # Check if directories exist
    if not template_dir.exists():
        raise FileNotFoundError(f"Template directory not found at: {template_dir}")
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found at: {models_dir}")
    
    # Check if files exist
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer file not found at: {vectorizer_path}")
    
    # Check if template file exists
    template_file = template_dir / 'index.html'
    if not template_file.exists():
        raise FileNotFoundError(f"Template file not found at: {template_file}")
    
    # Load model and vectorizer
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    logger.info(f"{Fore.GREEN}Model and vectorizer loaded successfully!{Style.RESET_ALL}")
except Exception as e:
    logger.error(f"{Fore.RED}Error loading model: {str(e)}{Style.RESET_ALL}")
    model = None
    vectorizer = None

def get_certainty(confidence):
    if confidence >= 70:
        return "high"
    elif confidence >= 40:
        return "moderate"
    else:
        return "low"

@app.route('/')
def home():
    try:
        logger.info(f"Attempting to render template from: {template_dir}")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}")
        return "Error loading template", 500

@app.route('/predict', methods=['POST'])
def predict():
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