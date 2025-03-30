import os
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('create_template.log')
    ]
)
logger = logging.getLogger(__name__)

def create_template():
    """Create the index.html template file."""
    try:
        # Define the template directories to check
        template_dirs = [
            '/opt/render/project/src/templates',
            os.path.join(os.getcwd(), 'templates')
        ]
        
        # Create template directories if they don't exist
        for template_dir in template_dirs:
            if not os.path.exists(template_dir):
                logger.info(f"Creating template directory: {template_dir}")
                os.makedirs(template_dir, exist_ok=True)
            else:
                logger.info(f"Template directory already exists: {template_dir}")
                
            # Create the index.html file
            template_path = os.path.join(template_dir, 'index.html')
            logger.info(f"Checking template file at: {template_path}")
            
            if not os.path.exists(template_path):
                logger.info(f"Creating template file at: {template_path}")
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
                
                # Set permissions
                os.chmod(template_path, 0o644)
                logger.info(f"Set permissions for {template_path}")
            else:
                logger.info(f"Template file already exists at: {template_path}")
            
            # Verify file size
            if os.path.exists(template_path):
                size = os.path.getsize(template_path)
                logger.info(f"Template file size: {size} bytes")
                
                if size > 0:
                    logger.info(f"Template file at {template_path} is valid")
                else:
                    logger.warning(f"Template file at {template_path} exists but is empty")
                    
        logger.info("Template creation process completed")
        return True
    except Exception as e:
        logger.error(f"Error creating template: {str(e)}")
        return False

if __name__ == "__main__":
    create_template() 