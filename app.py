import os
import sys
import logging
import json
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import create_template

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

# Create template files on startup
logger.info("Creating template files...")
create_template.create_template()

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

# Create Flask app with explicit template folder - try multiple possible locations
template_folders = [
    '/opt/render/project/src/templates',
    os.path.join(get_project_root(), 'templates'),
    os.path.join(os.getcwd(), 'templates')
]

# Find the first template folder that exists
template_folder = None
for folder in template_folders:
    if os.path.exists(folder):
        template_folder = folder
        logger.info(f"Using template folder: {folder}")
        break

if not template_folder:
    # Default to the first option and create it
    template_folder = template_folders[0]
    os.makedirs(template_folder, exist_ok=True)
    logger.info(f"Created template folder: {template_folder}")

# Initialize Flask app
app = Flask(__name__, 
            template_folder=template_folder,
            static_folder=os.path.join(get_project_root(), 'static'))

# Global variables for model and vectorizer
model = None
vectorizer = None

def load_model():
    """Load the model and vectorizer from files."""
    global model, vectorizer
    try:
        # Try multiple possible model locations
        model_paths = [
            os.path.join(get_project_root(), 'models', 'fake_news_model.joblib'),
            os.path.join('/opt/render/project/src', 'models', 'fake_news_model.joblib'),
            os.path.join(os.getcwd(), 'models', 'fake_news_model.joblib')
        ]
        
        vectorizer_paths = [
            os.path.join(get_project_root(), 'models', 'tfidf_vectorizer.joblib'),
            os.path.join('/opt/render/project/src', 'models', 'tfidf_vectorizer.joblib'),
            os.path.join(os.getcwd(), 'models', 'tfidf_vectorizer.joblib')
        ]
        
        # Try to load from each path until successful
        model_loaded = False
        vectorizer_loaded = False
        model_path_used = None
        vectorizer_path_used = None
        
        # Look for model file
        for path in model_paths:
            logger.info(f"Looking for model at: {path}")
            if os.path.exists(path):
                try:
                    model = joblib.load(path)
                    model_loaded = True
                    model_path_used = path
                    logger.info(f"Model loaded from: {path}")
                    break
                except Exception as e:
                    logger.error(f"Error loading model from {path}: {str(e)}")
        
        # Look for vectorizer file
        for path in vectorizer_paths:
            logger.info(f"Looking for vectorizer at: {path}")
            if os.path.exists(path):
                try:
                    vectorizer = joblib.load(path)
                    vectorizer_loaded = True
                    vectorizer_path_used = path
                    logger.info(f"Vectorizer loaded from: {path}")
                    break
                except Exception as e:
                    logger.error(f"Error loading vectorizer from {path}: {str(e)}")
        
        if not model_loaded or not vectorizer_loaded:
            logger.error("Failed to load model or vectorizer from any location")
            return False
        
        # Test the model works with the vectorizer
        try:
            test_text = "This is a test article for the fake news detector."
            test_vector = vectorizer.transform([test_text])
            prediction = model.predict(test_vector)
            probability = model.predict_proba(test_vector)
            logger.info(f"Model test successful. Prediction: {prediction}, Probability: {probability}")
        except Exception as e:
            logger.error(f"Model test failed: {str(e)}")
            return False
            
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
        # Try to ensure the template exists before rendering
        create_template.create_template()
        
        # Get all potential template paths
        template_paths = []
        for folder in template_folders:
            path = os.path.join(folder, 'index.html')
            template_paths.append(path)
            logger.info(f"Potential template path: {path} (exists: {os.path.exists(path)})")
        
        # Verify the template exists in the configured location
        template_path = os.path.join(app.template_folder, 'index.html')
        logger.info(f"Looking for template at: {template_path}")
        
        if not os.path.exists(template_path):
            logger.error(f"Template file not found at {template_path}, trying to recreate it")
            
            # Create the template directory
            os.makedirs(app.template_folder, exist_ok=True)
            
            # Try to copy from other locations if they exist
            copied = False
            for path in template_paths:
                if os.path.exists(path) and path != template_path:
                    logger.info(f"Copying template from {path} to {template_path}")
                    with open(path, 'r') as src, open(template_path, 'w') as dest:
                        dest.write(src.read())
                    copied = True
                    break
            
            # Create a new template if we couldn't copy
            if not copied:
                logger.info(f"Creating new template at {template_path}")
                with open(template_path, 'w') as f:
                    f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fake News Detector</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f8f9fa;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .container {
            max-width: 800px;
            margin: 2rem auto;
            padding: 2rem;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .result-card {
            display: none;
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 8px;
            background-color: #f8f9fa;
        }
        .progress {
            height: 25px;
            margin: 1rem 0;
        }
        .certainty-badge {
            font-size: 1.1rem;
            padding: 0.5rem 1rem;
            border-radius: 20px;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 2rem 0;
        }
        .spinner-border {
            width: 3rem;
            height: 3rem;
        }
        .error-message {
            display: none;
            color: #dc3545;
            margin-top: 1rem;
            padding: 1rem;
            border-radius: 4px;
            background-color: #f8d7da;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center mb-4">Fake News Detector</h1>
        <p class="text-center text-muted mb-4">Paste your news article below to analyze if it's real or fake</p>
        
        <div class="form-group">
            <textarea id="newsText" class="form-control" rows="6" placeholder="Paste your news article here..."></textarea>
        </div>
        
        <div class="text-center mt-3">
            <button id="analyzeBtn" class="btn btn-primary btn-lg">Analyze</button>
        </div>
        
        <div class="error-message" id="errorMessage"></div>
        
        <div class="loading">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-2">Analyzing your article...</p>
        </div>
        
        <div class="result-card">
            <h3 class="text-center mb-3">Analysis Result</h3>
            <div class="text-center">
                <h4 id="prediction" class="mb-3"></h4>
                <div class="progress">
                    <div id="confidenceBar" class="progress-bar" role="progressbar" style="width: 0%"></div>
                </div>
                <p class="mt-2">Confidence: <span id="confidenceValue">0%</span></p>
                <span id="certainty" class="badge certainty-badge"></span>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('analyzeBtn').addEventListener('click', async () => {
            const text = document.getElementById('newsText').value.trim();
            if (!text) {
                showError('Please enter some text to analyze');
                return;
            }

            // Reset UI
            document.querySelector('.error-message').style.display = 'none';
            document.querySelector('.loading').style.display = 'block';
            document.querySelector('.result-card').style.display = 'none';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text }),
                });

                const data = await response.json();

                if (data.status === 'error') {
                    throw new Error(data.error);
                }

                // Update UI with results
                document.getElementById('prediction').textContent = data.prediction;
                document.getElementById('prediction').className = `text-${data.prediction === 'Real News' ? 'success' : 'danger'}`;
                
                const confidence = Math.round(data.confidence * 100);
                document.getElementById('confidenceValue').textContent = `${confidence}%`;
                document.getElementById('confidenceBar').style.width = `${confidence}%`;
                document.getElementById('confidenceBar').className = `progress-bar bg-${data.prediction === 'Real News' ? 'success' : 'danger'}`;
                
                document.getElementById('certainty').textContent = data.certainty;
                document.getElementById('certainty').className = `badge certainty-badge bg-${data.certainty === 'High' ? 'success' : data.certainty === 'Moderate' ? 'warning' : 'danger'}`;
                
                document.querySelector('.result-card').style.display = 'block';
            } catch (error) {
                showError(error.message);
            } finally {
                document.querySelector('.loading').style.display = 'none';
            }
        });

        function showError(message) {
            const errorElement = document.getElementById('errorMessage');
            errorElement.textContent = message;
            errorElement.style.display = 'block';
        }
    </script>
</body>
</html>''')
            
            # Set appropriate permissions
            os.chmod(template_path, 0o644)
            logger.info(f"Set permissions for {template_path}")
            
            # Verify the file was created
            if not os.path.exists(template_path):
                logger.error(f"Failed to create template at {template_path}")
                return "Error: Could not create template file", 500
        
        logger.info("Template file found, rendering...")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering home page: {str(e)}")
        return f"Error: Failed to render home page - {str(e)}", 500

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        logger.info("Received prediction request")
        
        if model is None or vectorizer is None:
            logger.info("Model or vectorizer not loaded, attempting to load...")
            if not load_model():
                logger.error("Failed to load model for prediction")
                return jsonify({
                    'status': 'error',
                    'error': 'Model not loaded. Please try again later.'
                }), 500
        
        data = request.get_json()
        logger.info(f"Received data: {data}")
        
        if not data or 'text' not in data:
            logger.error("No text provided in request")
            return jsonify({
                'status': 'error',
                'error': 'No text provided'
            }), 400
            
        text = data['text']
        if not text.strip():
            logger.error("Empty text provided in request")
            return jsonify({
                'status': 'error',
                'error': 'Empty text provided'
            }), 400
        
        logger.info(f"Analyzing text: {text[:100]}...")
            
        # Transform the input text
        text_tfidf = vectorizer.transform([text])
        
        # Make prediction
        prediction = model.predict(text_tfidf)[0]
        # For logistic regression, class 1 probability is at index 1
        confidence = float(model.predict_proba(text_tfidf)[0][int(prediction)])
        
        # Determine certainty
        certainty = get_certainty(confidence)
        
        result = {
            'status': 'success',
            'prediction': 'Real News' if prediction == 1 else 'Fake News',
            'confidence': confidence,
            'certainty': certainty
        }
        
        logger.info(f"Prediction result: {result}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Load model at startup
    logger.info("Loading model at application startup...")
    
    # Try to load the model a few times
    max_attempts = 3
    attempts = 0
    model_loaded = False
    
    while attempts < max_attempts and not model_loaded:
        attempts += 1
        logger.info(f"Attempt {attempts} to load model")
        model_loaded = load_model()
        if not model_loaded and attempts < max_attempts:
            logger.warning(f"Failed to load model on attempt {attempts}, trying again...")
    
    if not model_loaded:
        logger.error(f"Failed to load model after {max_attempts} attempts")
    
    # Create templates directory if it doesn't exist
    os.makedirs(app.template_folder, exist_ok=True)
    
    # Run the app
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000))) 