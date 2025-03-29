from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from colorama import init, Fore, Style
import os

# Initialize colorama for colored output
init()

# Create Flask app with template folder path
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
app = Flask(__name__, template_folder=template_dir)

# Get the absolute path to the models directory
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))

# Load the model and vectorizer
try:
    model_path = os.path.join(models_dir, 'fake_news_model.joblib')
    vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print(f"{Fore.GREEN}Model and vectorizer loaded successfully!{Style.RESET_ALL}")
except Exception as e:
    print(f"{Fore.RED}Error loading model: {str(e)}{Style.RESET_ALL}")
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
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.json['text']
        print(f"{Fore.CYAN}Analyzing text: {text[:100]}...{Style.RESET_ALL}")
        
        # Transform the input text
        text_vectorized = vectorizer.transform([text])
        print(f"{Fore.GREEN}Text vectorization successful{Style.RESET_ALL}")
        
        # Make prediction
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        confidence = max(probabilities) * 100
        
        print(f"{Fore.YELLOW}Prediction: {prediction}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Probabilities: {probabilities}{Style.RESET_ALL}")
        
        certainty = get_certainty(confidence)
        
        return jsonify({
            'success': True,
            'prediction': "Real News" if prediction == 1 else "Fake News",
            'confidence': float(confidence),
            'certainty': certainty
        })
    except Exception as e:
        print(f"{Fore.RED}Error during prediction: {str(e)}{Style.RESET_ALL}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True) 