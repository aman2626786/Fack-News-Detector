import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('train.log')
    ]
)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample data for testing."""
    logger.info("Creating sample data...")
    
    # Real news examples
    real_news = [
        "Scientists discover new species of deep-sea creatures in the Pacific Ocean.",
        "New study shows benefits of regular exercise for mental health.",
        "Global renewable energy capacity reaches record high in 2023.",
        "Researchers develop new method for recycling plastic waste.",
        "World Health Organization reports decline in global malaria cases.",
        "NASA successfully launches new Mars rover mission.",
        "Breakthrough in cancer research leads to new treatment options.",
        "Global temperatures show slight decrease in 2023.",
        "New archaeological discovery reveals ancient civilization.",
        "Scientists develop new method for carbon capture."
    ]
    
    # Fake news examples
    fake_news = [
        "Aliens build secret base on the dark side of the moon.",
        "Scientists discover that drinking coffee makes you immortal.",
        "Government reveals secret time travel program.",
        "Scientists find that the Earth is actually flat.",
        "New study shows that the moon is made of cheese.",
        "Scientists discover that plants can talk to humans.",
        "Government confirms existence of Bigfoot.",
        "New study shows that the sun is actually cold.",
        "Scientists find that gravity is just a theory.",
        "Researchers discover that dinosaurs never existed."
    ]
    
    texts = real_news + fake_news
    labels = [1] * len(real_news) + [0] * len(fake_news)
    
    logger.info(f"Created {len(texts)} sample texts")
    return texts, labels

def train_model():
    """Train and save the model."""
    try:
        # Create models directory if it doesn't exist
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Get sample data
        texts, labels = create_sample_data()
        
        # Create and fit the vectorizer
        logger.info("Creating and fitting TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(texts)
        y = np.array(labels)
        
        # Train the model
        logger.info("Training LogisticRegression model...")
        model = LogisticRegression(C=0.1, class_weight={0: 1.2, 1: 0.8}, random_state=42)
        model.fit(X, y)
        
        # Save the model and vectorizer
        model_path = os.path.join(models_dir, 'fake_news_model.joblib')
        vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
        
        logger.info(f"Saving model to: {model_path}")
        joblib.dump(model, model_path)
        
        logger.info(f"Saving vectorizer to: {vectorizer_path}")
        joblib.dump(vectorizer, vectorizer_path)
        
        # Verify files were created
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError("Model files were not created successfully")
            
        logger.info("Model files created successfully!")
        logger.info(f"Model file size: {os.path.getsize(model_path)} bytes")
        logger.info(f"Vectorizer file size: {os.path.getsize(vectorizer_path)} bytes")
        
        # Test the model
        test_text = "Scientists discover new species of deep-sea creatures in the Pacific Ocean."
        test_tfidf = vectorizer.transform([test_text])
        prediction = model.predict(test_tfidf)[0]
        confidence = model.predict_proba(test_tfidf)[0][1]
        
        logger.info(f"Test prediction: {'Real News' if prediction == 1 else 'Fake News'}")
        logger.info(f"Test confidence: {confidence:.2f}")
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

if __name__ == '__main__':
    train_model() 