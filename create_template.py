import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("template_creation.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def create_template():
    """Create template directory and index.html file if they don't exist"""
    try:
        # Create templates directory if it doesn't exist
        templates_dir = "templates"
        if not os.path.exists(templates_dir):
            logger.info(f"Creating templates directory at {templates_dir}")
            os.makedirs(templates_dir, exist_ok=True)
        else:
            logger.info(f"Templates directory already exists at {templates_dir}")
        
        # Path for the index.html file
        template_path = os.path.join(templates_dir, "index.html")
        
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
        
        # Write the HTML content to the file if it doesn't exist or is different
        if not os.path.exists(template_path):
            logger.info(f"Creating template file at {template_path}")
            with open(template_path, "w") as file:
                file.write(html_content)
            logger.info(f"Template file created successfully at {template_path}")
        else:
            with open(template_path, "r") as file:
                existing_content = file.read()
            
            if existing_content != html_content:
                logger.info(f"Updating existing template file at {template_path}")
                with open(template_path, "w") as file:
                    file.write(html_content)
                logger.info(f"Template file updated successfully at {template_path}")
            else:
                logger.info(f"Template file already exists with correct content at {template_path}")
        
        # Also create in static directory for fallback
        static_dir = "static"
        if not os.path.exists(static_dir):
            logger.info(f"Creating static directory at {static_dir}")
            os.makedirs(static_dir, exist_ok=True)
        
        static_path = os.path.join(static_dir, "index.html")
        logger.info(f"Creating/updating static template file at {static_path}")
        with open(static_path, "w") as file:
            file.write(html_content)
        logger.info(f"Static template file created/updated successfully at {static_path}")
        
        # Try creating in absolute path if in deployment environment
        try:
            if os.path.exists("/opt/render/project/src"):
                abs_templates_dir = "/opt/render/project/src/templates"
                abs_static_dir = "/opt/render/project/src/static"
                
                logger.info(f"Deployment environment detected, creating templates at {abs_templates_dir}")
                os.makedirs(abs_templates_dir, exist_ok=True)
                os.makedirs(abs_static_dir, exist_ok=True)
                
                abs_template_path = os.path.join(abs_templates_dir, "index.html")
                abs_static_path = os.path.join(abs_static_dir, "index.html")
                
                with open(abs_template_path, "w") as file:
                    file.write(html_content)
                with open(abs_static_path, "w") as file:
                    file.write(html_content)
                
                logger.info(f"Templates created in absolute paths successfully")
        except Exception as e:
            logger.error(f"Error creating templates in absolute paths: {str(e)}")
        
        return True
    except Exception as e:
        logger.error(f"Error creating template: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Running template creation script")
    success = create_template()
    logger.info(f"Template creation {'successful' if success else 'failed'}") 