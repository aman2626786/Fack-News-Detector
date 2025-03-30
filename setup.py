import os
import sys
import logging
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('setup.log')
    ]
)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories with proper permissions."""
    directories = ['models', 'templates']
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            os.chmod(directory, 0o755)
            logger.info(f"Created/verified directory: {directory}")
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {str(e)}")
            raise

def create_sample_data():
    """Create sample data for testing."""
    logger.info("Creating sample data...")
    real_news = [
        "Scientists discover new species of deep-sea creatures in the Pacific Ocean.",
        "New study shows benefits of regular exercise for mental health.",
        "Global renewable energy capacity reaches record high in 2023.",
        "Researchers develop new method for recycling plastic waste.",
        "World Health Organization reports decline in global malaria cases."
    ]
    
    fake_news = [
        "Aliens build secret base on the dark side of the moon.",
        "Scientists discover that drinking coffee makes you immortal.",
        "Government reveals secret time travel program.",
        "Scientists find that the Earth is actually flat.",
        "New study shows that the moon is made of cheese."
    ]
    
    texts = real_news + fake_news
    labels = [1] * len(real_news) + [0] * len(fake_news)
    logger.info(f"Created {len(texts)} sample texts")
    return texts, labels

def generate_model():
    """Generate and save the model files."""
    try:
        # Create sample data
        texts, labels = create_sample_data()
        
        # Create and fit the vectorizer
        logger.info("Creating and fitting TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(texts)
        y = np.array(labels)
        
        # Train the model
        logger.info("Training LogisticRegression model...")
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        # Save the model and vectorizer
        model_path = os.path.join('models', 'fake_news_model.joblib')
        vectorizer_path = os.path.join('models', 'tfidf_vectorizer.joblib')
        
        logger.info(f"Saving model to: {model_path}")
        joblib.dump(model, model_path)
        os.chmod(model_path, 0o644)
        
        logger.info(f"Saving vectorizer to: {vectorizer_path}")
        joblib.dump(vectorizer, vectorizer_path)
        os.chmod(vectorizer_path, 0o644)
        
        # Verify files were created
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError("Model files were not created successfully")
            
        logger.info("Model files created successfully!")
        logger.info(f"Model file size: {os.path.getsize(model_path)} bytes")
        logger.info(f"Vectorizer file size: {os.path.getsize(vectorizer_path)} bytes")
            
    except Exception as e:
        logger.error(f"Error generating model: {str(e)}")
        raise

def create_template():
    """Create the index.html template file."""
    try:
        template_path = os.path.join('templates', 'index.html')
        logger.info(f"Creating template at: {template_path}")
        
        # Template content
        template_content = '''<!DOCTYPE html>
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
                alert('Please enter some text to analyze');
                return;
            }

            // Show loading spinner
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
                alert('Error analyzing text: ' + error.message);
            } finally {
                document.querySelector('.loading').style.display = 'none';
            }
        });
    </script>
</body>
</html>'''
        
        # Write the template file
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(template_content)
        
        # Set proper permissions
        os.chmod(template_path, 0o644)
        
        # Verify file was created
        if not os.path.exists(template_path):
            raise FileNotFoundError("Template file was not created successfully")
            
        logger.info("Template file created successfully!")
        logger.info(f"Template file size: {os.path.getsize(template_path)} bytes")
            
    except Exception as e:
        logger.error(f"Error creating template: {str(e)}")
        raise

def verify_setup():
    """Verify all files and directories are created correctly."""
    required_files = [
        os.path.join('models', 'fake_news_model.joblib'),
        os.path.join('models', 'tfidf_vectorizer.joblib'),
        os.path.join('templates', 'index.html')
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"File not readable: {file_path}")
        logger.info(f"Verified file: {file_path}")

def main():
    """Main setup function."""
    try:
        logger.info("Starting setup process...")
        
        # Create directories
        create_directories()
        
        # Generate model files
        generate_model()
        
        # Create template
        create_template()
        
        # Verify everything
        verify_setup()
        
        logger.info("Setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 