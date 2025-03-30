import os
import sys
import logging
import json
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Get the absolute path to the project root directory
def get_project_root():
    """Get the absolute path to the project root directory."""
    try:
        # Try to get the directory containing app.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Current directory: {current_dir}")
        return current_dir
    except Exception as e:
        logger.error(f"Error getting project root: {str(e)}")
        return os.getcwd()

app = Flask(__name__, 
            template_folder=os.path.join(get_project_root(), 'templates'),
            static_folder=os.path.join(get_project_root(), 'static'))

# Global variables for model and vectorizer
model = None
vectorizer = None

def load_model():
    """Load the model and vectorizer from files."""
    global model, vectorizer
    try:
        # Get the absolute path to the models directory
        current_dir = get_project_root()
        models_dir = os.path.join(current_dir, 'models')
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = os.path.join(models_dir, 'fake_news_model.joblib')
        vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
        
        logger.info(f"Looking for model files in: {models_dir}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Vectorizer path: {vectorizer_path}")
        
        # Check if model files exist
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            logger.error(f"Model files not found at {model_path} or {vectorizer_path}")
            return False
            
        # Load the model and vectorizer
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        logger.info("Model and vectorizer loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def get_certainty(confidence):
    """Determine the certainty level based on confidence score."""
    if confidence >= 0.8:
        return "High"
    elif confidence >= 0.6:
        return "Moderate"
    else:
        return "Low"

@app.route('/')
def home():
    """Render the home page."""
    try:
        template_path = os.path.join(app.template_folder, 'index.html')
        logger.info(f"Looking for template at: {template_path}")
        
        if not os.path.exists(template_path):
            logger.error(f"Template file not found at {template_path}")
            return "Error: Template file not found", 500
            
        logger.info("Template file found, rendering...")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering home page: {str(e)}")
        return "Error: Failed to render home page", 500

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        if model is None or vectorizer is None:
            if not load_model():
                return jsonify({
                    'status': 'error',
                    'error': 'Model not loaded. Please try again later.'
                }), 500
        
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'status': 'error',
                'error': 'No text provided'
            }), 400
            
        text = data['text']
        if not text.strip():
            return jsonify({
                'status': 'error',
                'error': 'Empty text provided'
            }), 400
            
        # Transform the input text
        text_tfidf = vectorizer.transform([text])
        
        # Make prediction
        prediction = model.predict(text_tfidf)[0]
        confidence = model.predict_proba(text_tfidf)[0][1]  # Probability of being real news
        
        # Determine certainty
        certainty = get_certainty(confidence)
        
        return jsonify({
            'status': 'success',
            'prediction': 'Real News' if prediction == 1 else 'Fake News',
            'confidence': float(confidence),
            'certainty': certainty
        })
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Load model at startup
    if not load_model():
        logger.error("Failed to load model at startup")
        sys.exit(1)
    
    # Create templates directory if it doesn't exist
    os.makedirs(app.template_folder, exist_ok=True)
    
    # Run the app
    app.run(debug=True) 