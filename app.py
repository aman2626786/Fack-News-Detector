import os
import sys
import logging
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs("templates", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("static", exist_ok=True)

def get_project_root():
    """Get the project root directory"""
    # Try different possible root paths
    possible_roots = [
        os.path.abspath(os.path.dirname(__file__)),  # Current directory
        os.getcwd(),  # Current working directory
        "/opt/render/project/src/",  # Render.com absolute path
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Parent directory
    ]
    
    for root in possible_roots:
        logger.info(f"Checking possible root: {root}")
        if os.path.exists(root):
            logger.info(f"Found valid root: {root}")
            return root
    
    # Default to current directory if none found
    logger.warning("No valid project root found, using current directory")
    return os.path.abspath(os.path.dirname(__file__))

def create_template():
    """Create template directory and index.html file if they don't exist"""
    try:
        # Create templates directory if it doesn't exist
        os.makedirs("templates", exist_ok=True)
        
        # Path for the index.html file
        template_path = os.path.join("templates", "index.html")
        
        # HTML content for the template
        html_content = """<!DOCTYPE html>
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
</html>"""
        
        # Write the HTML content to the file
        with open(template_path, "w") as file:
            file.write(html_content)
        
        # Create static directory and copy the template there as well
        os.makedirs("static", exist_ok=True)
        static_path = os.path.join("static", "index.html")
        with open(static_path, "w") as file:
            file.write(html_content)
            
        logger.info(f"Template created successfully at {template_path} and {static_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating template: {str(e)}")
        return False

def create_quick_model():
    """Create a simple model directly in case loading fails"""
    logger.info("Creating a quick fallback model directly")
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        
        # Define a simple dataset
        texts = [
            "Clinton's illegal emails scandal", 
            "Breaking: Obama born in Kenya",
            "Trump secretly working with Russia",
            "Pope endorses Donald Trump",
            "CDC admits vaccines cause autism",
            "Actor found dead in suspicious circumstances",
            "Government hiding alien evidence",
            "Miracle cure for all diseases suppressed",
            "NASA confirms climate change data",
            "Study shows economic growth in Q4",
            "Scientists discover new species in Amazon",
            "New treatment shows promise for cancer patients",
            "Sports team wins championship game",
            "Company releases quarterly earnings report",
            "Historical museum opens new exhibit",
            "Technology firm announces product update"
        ]
        
        # First 8 are fake (0), last 8 are real (1)
        labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        
        # Create and fit the vectorizer
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X = vectorizer.fit_transform(texts)
        
        # Create and fit the model
        model = LogisticRegression(class_weight='balanced', max_iter=1000)
        model.fit(X, labels)
        
        # Test the model
        test_text = "Breaking news: politician caught in scandal"
        test_vector = vectorizer.transform([test_text])
        prediction = model.predict(test_vector)[0]
        logger.info(f"Quick model test prediction: {prediction}")
        
        # Try to save this model
        try:
            root_dir = get_project_root()
            os.makedirs(os.path.join(root_dir, "models"), exist_ok=True)
            model_path = os.path.join(root_dir, "models", "fake_news_model.joblib")
            vectorizer_path = os.path.join(root_dir, "models", "tfidf_vectorizer.joblib")
            
            joblib.dump(model, model_path)
            joblib.dump(vectorizer, vectorizer_path)
            logger.info(f"Saved quick model to {model_path} and vectorizer to {vectorizer_path}")
        except Exception as e:
            logger.error(f"Failed to save quick model: {str(e)}")
        
        return model, vectorizer
    except Exception as e:
        logger.error(f"Error creating quick model: {str(e)}")
        return None, None

def load_model():
    """Load the trained model and vectorizer from multiple possible locations"""
    model = None
    vectorizer = None
    
    # Project root directory
    root_dir = get_project_root()
    logger.info(f"Project root directory: {root_dir}")
    
    # Define potential paths
    model_paths = [
        os.path.join(root_dir, "models", "fake_news_model.joblib"),
        os.path.join(root_dir, "fake_news_model.joblib"),
        "/opt/render/project/src/models/fake_news_model.joblib",
        os.path.join(os.getcwd(), "models", "fake_news_model.joblib")
    ]
    
    vectorizer_paths = [
        os.path.join(root_dir, "models", "tfidf_vectorizer.joblib"),
        os.path.join(root_dir, "tfidf_vectorizer.joblib"),
        "/opt/render/project/src/models/tfidf_vectorizer.joblib",
        os.path.join(os.getcwd(), "models", "tfidf_vectorizer.joblib")
    ]
    
    # List directory contents to debug
    try:
        logger.info(f"Contents of current directory: {os.listdir('.')}")
        if os.path.exists("models"):
            logger.info(f"Contents of models directory: {os.listdir('models')}")
    except Exception as e:
        logger.error(f"Error listing directories: {str(e)}")
    
    # Try to load model from each path
    for model_path in model_paths:
        try:
            logger.info(f"Attempting to load model from: {model_path}")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                logger.info(f"Successfully loaded model from {model_path}")
                break
            else:
                logger.warning(f"Model path does not exist: {model_path}")
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
    
    # Try to load vectorizer from each path
    for vectorizer_path in vectorizer_paths:
        try:
            logger.info(f"Attempting to load vectorizer from: {vectorizer_path}")
            if os.path.exists(vectorizer_path):
                vectorizer = joblib.load(vectorizer_path)
                logger.info(f"Successfully loaded vectorizer from {vectorizer_path}")
                break
            else:
                logger.warning(f"Vectorizer path does not exist: {vectorizer_path}")
        except Exception as e:
            logger.error(f"Error loading vectorizer from {vectorizer_path}: {str(e)}")
    
    # Final fallback: create a model in memory
    if model is None or vectorizer is None:
        logger.warning("Could not load model or vectorizer, creating in-memory model")
        model, vectorizer = create_quick_model()
    
    return model, vectorizer

def keyword_analysis(text):
    """Simple keyword-based analysis as a fallback"""
    text = text.lower()
    
    # Define keywords that might indicate fake news
    fake_keywords = ['shocking', 'secret', 'conspiracy', 'hoax', 'exposed', 'they don\'t want you to know',
                    'scandal', 'banned', 'censored', 'revealed', 'cover-up', 'what they aren\'t telling you',
                    'government is hiding', 'media won\'t report', 'illuminati']
    
    # Define keywords that might indicate real news
    real_keywords = ['according to', 'study shows', 'research', 'experts say', 'evidence', 'analysis',
                    'investigation', 'data', 'statistics', 'survey', 'report', 'official', 'spokesman',
                    'professor', 'scientist', 'sources confirm']
    
    # Count occurrences
    fake_count = sum(1 for keyword in fake_keywords if keyword in text)
    real_count = sum(1 for keyword in real_keywords if keyword in text)
    
    # Determine result
    if fake_count > real_count:
        prediction = 0
        confidence = min(0.5 + (fake_count - real_count) * 0.05, 0.8)  # Cap at 80% confidence
    else:
        prediction = 1
        confidence = min(0.5 + (real_count - fake_count) * 0.05, 0.8)  # Cap at 80% confidence
    
    return prediction, confidence

# Create template if it doesn't exist
create_template()

# Initialize Flask app
app = Flask(__name__, 
            template_folder=os.path.join(get_project_root(), "templates"),
            static_folder=os.path.join(get_project_root(), "static"))

# Load model and vectorizer
logger.info("Loading model and vectorizer at startup")
model, vectorizer = load_model()
logger.info("Model and vectorizer loaded successfully")

@app.route('/')
def home():
    """Serve the home page"""
    try:
        logger.info("Rendering home template")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}")
        # Fallback to static file if template doesn't work
        try:
            with open(os.path.join(get_project_root(), 'static', 'index.html'), 'r') as f:
                logger.info("Serving static index.html as fallback")
                return f.read()
        except Exception as static_e:
            logger.error(f"Error serving static file: {str(static_e)}")
            return """
            <html>
            <head><title>Fake News Detector</title></head>
            <body>
                <h1>Fake News Detector</h1>
                <p>Welcome to the Fake News Detector. Please use the /predict endpoint with POST request.</p>
                <form action="/predict" method="post">
                    <textarea name="text" rows="10" cols="50"></textarea>
                    <br>
                    <input type="submit" value="Analyze">
                </form>
            </body>
            </html>
            """

@app.route('/predict', methods=['POST'])
def predict():
    """Predict if a news article is fake or real"""
    try:
        # Get data from request
        if request.is_json:
            data = request.json
            logger.info("Received JSON data for prediction")
        else:
            data = request.form
            logger.info("Received form data for prediction")
        
        logger.info(f"Received prediction request with data keys: {data.keys()}")
        
        if not data or ('text' not in data and not request.data):
            # Try to get raw data if JSON parsing failed
            if request.data:
                logger.info("Trying to parse raw request data")
                try:
                    import json
                    data = json.loads(request.data)
                except:
                    logger.warning("Failed to parse raw data as JSON")
            
            if not data or 'text' not in data:
                logger.warning("Invalid request: missing text")
                return jsonify({
                    'status': 'error',
                    'error': 'Please provide text to analyze'
                }), 400
        
        text = data.get('text', '')
        
        if not text or len(text) < 10:
            logger.warning(f"Text too short for analysis: '{text}'")
            return jsonify({
                'status': 'error',
                'error': 'Text is too short for analysis'
            }), 400
        
        # Global model/vectorizer reference
        global model, vectorizer
        
        # Check if model is loaded
        if model is None or vectorizer is None:
            logger.warning("Model or vectorizer not loaded, trying to reload")
            model, vectorizer = load_model()
            
            # If still not loaded, use keyword analysis
            if model is None or vectorizer is None:
                logger.warning("Could not load model, using keyword analysis")
                prediction, confidence = keyword_analysis(text)
            else:
                # Attempt model prediction
                try:
                    # Vectorize input
                    vector = vectorizer.transform([text])
                    
                    # Make prediction
                    prediction = model.predict(vector)[0]
                    
                    # Get confidence
                    probabilities = model.predict_proba(vector)[0]
                    confidence = probabilities[prediction]
                except Exception as e:
                    logger.error(f"Error in model prediction: {str(e)}")
                    prediction, confidence = keyword_analysis(text)
        else:
            # Normal prediction path
            try:
                # Vectorize input
                vector = vectorizer.transform([text])
                
                # Make prediction
                prediction = model.predict(vector)[0]
                
                # Get confidence
                probabilities = model.predict_proba(vector)[0]
                confidence = probabilities[prediction]
                
                logger.info(f"Model prediction: {prediction}, confidence: {confidence:.4f}")
            except Exception as e:
                logger.error(f"Error in model prediction: {str(e)}")
                prediction, confidence = keyword_analysis(text)
        
        # Determine certainty level
        if confidence >= 0.8:
            certainty = "High"
        elif confidence >= 0.6:
            certainty = "Moderate"
        else:
            certainty = "Low"
        
        # Format prediction text
        prediction_text = "Real News" if prediction == 1 else "Fake News"
        
        # Return result
        logger.info(f"Final prediction: {prediction_text}, Confidence: {confidence:.2f}, Certainty: {certainty}")
        return jsonify({
            'status': 'success',
            'prediction': prediction_text,
            'confidence': float(confidence) if isinstance(confidence, np.float64) else confidence,
            'certainty': certainty
        })
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': 'An error occurred during analysis. Please try again.'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting server on port {port}")
    app.run(debug=True, host='0.0.0.0', port=port) 