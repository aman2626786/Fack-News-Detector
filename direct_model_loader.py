import os
import sys
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("direct_model.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def create_and_save_direct_model():
    """Create a simple model directly and save it to disk"""
    logger.info("Starting direct model creation process")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    logger.info("Created or verified models directory")
    
    # Define paths for saving model and vectorizer
    model_path = os.path.join("models", "fake_news_model.joblib")
    vectorizer_path = os.path.join("models", "tfidf_vectorizer.joblib")
    
    # Define a simple dataset with well-known fake and real news examples
    texts = [
        "Clinton's illegal emails scandal exposed by FBI", 
        "Breaking: Obama born in Kenya, new evidence shows",
        "Trump secretly working with Russia to rig election",
        "Pope endorses Donald Trump for president",
        "CDC admits vaccines cause autism in confidential document",
        "Famous actor found dead in suspicious circumstances",
        "Government hiding alien evidence from Area 51",
        "Miracle cure for all diseases suppressed by pharmaceutical companies",
        "NASA confirms climate change data shows warming trend",
        "Study shows economic growth in Q4 surpassed expectations",
        "Scientists discover new species in Amazon rainforest",
        "New treatment shows promise for cancer patients in clinical trial",
        "Sports team wins championship game in overtime",
        "Company releases quarterly earnings report showing profit",
        "Historical museum opens new exhibit on ancient civilizations",
        "Technology firm announces product update with security fixes"
    ]
    
    # First 8 are fake (0), last 8 are real (1)
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    
    # Create and fit the vectorizer
    logger.info("Creating and fitting TF-IDF vectorizer")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(texts)
    logger.info(f"Vectorizer created with {len(vectorizer.get_feature_names_out())} features")
    
    # Create and fit the model
    logger.info("Creating and fitting Logistic Regression model")
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X, labels)
    logger.info("Model training completed")
    
    # Test the model
    test_text = "Breaking news: politician caught in scandal"
    test_vector = vectorizer.transform([test_text])
    prediction = model.predict(test_vector)[0]
    probabilities = model.predict_proba(test_vector)[0]
    logger.info(f"Model test prediction: {prediction} with probability: {probabilities[prediction]:.4f}")
    
    # Save the model and vectorizer
    try:
        logger.info(f"Saving model to {model_path}")
        joblib.dump(model, model_path)
        logger.info(f"Model saved successfully. File size: {os.path.getsize(model_path) / 1024:.2f} KB")
        
        logger.info(f"Saving vectorizer to {vectorizer_path}")
        joblib.dump(vectorizer, vectorizer_path)
        logger.info(f"Vectorizer saved successfully. File size: {os.path.getsize(vectorizer_path) / 1024:.2f} KB")
        
        # Verify the files exist
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            logger.info("Verified that model and vectorizer files exist")
        else:
            logger.error("Failed to verify model or vectorizer files")
    except Exception as e:
        logger.error(f"Error saving model or vectorizer: {str(e)}")
    
    # Try to save to an absolute path as well (for deployment environments)
    try:
        abs_model_dir = "/opt/render/project/src/models"
        if os.path.exists("/opt/render/project/src"):
            logger.info(f"Deployment environment detected, saving to {abs_model_dir}")
            os.makedirs(abs_model_dir, exist_ok=True)
            
            abs_model_path = os.path.join(abs_model_dir, "fake_news_model.joblib")
            abs_vectorizer_path = os.path.join(abs_model_dir, "tfidf_vectorizer.joblib")
            
            joblib.dump(model, abs_model_path)
            joblib.dump(vectorizer, abs_vectorizer_path)
            
            logger.info(f"Model and vectorizer saved to absolute paths successfully")
        else:
            logger.info("Not in deployment environment, skipping absolute path save")
    except Exception as e:
        logger.error(f"Error saving to absolute path: {str(e)}")
    
    return model, vectorizer

# Run the model creation when this script is executed directly
if __name__ == "__main__":
    logger.info("Running direct model creation script")
    create_and_save_direct_model()
    logger.info("Direct model creation completed") 