import os
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_creation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def create_data_from_notebook():
    """Create data similar to the notebook examples"""
    logger.info("Creating dataset for model training")
    
    # Sample fake news articles
    fake_news = [
        "BREAKING: Hillary Clinton Sold Weapons to ISIS, New Wikileaks Documents Reveal",
        "FBI Agent Suspected in Hillary Email Leaks Found Dead of Apparent Murder-Suicide",
        "BREAKING: Obama Has Just Begun Martial Law By Signing This SECRET Executive Order!",
        "Pope Francis Shocks World, Endorses Donald Trump for President",
        "BREAKING: CDC Confirms – Zika Virus Is Completely Spread by Vaccines",
        "BREAKING: Scientists Link Fluoride In Water To The Rise In ADHD Cases",
        "You Won't Believe What This Celebrity Said About Donald Trump!",
        "Doctors Don't Want You To Know This Simple Cure for Cancer",
        "BREAKING: Supreme Court Revokes Obama's Law License",
        "CONFIRMED: The Government Is Controlling Your Mind With Chemtrails",
        "Secret Document Exposes Government's Plan to Control the Internet",
        "BREAKING: Russia Hacks US Election, Changes Vote Tallies",
        "REVEALED: Hillary Clinton Has Terminal Illness, Months to Live",
        "SHOCKING: Top Doctor Reveals Vaccines Cause Autism, CDC Covers It Up",
        "BREAKING: North Korea Launches Nuclear Attack on US Base"
    ]
    
    # Sample real news articles
    real_news = [
        "Senate Passes Infrastructure Bill With Bipartisan Support",
        "NASA's Perseverance Rover Successfully Lands on Mars",
        "Federal Reserve Announces Interest Rate Decision After Meeting",
        "Supreme Court Issues Ruling on Environmental Regulations",
        "President Signs Executive Order on Climate Change Policy",
        "Study Shows Economic Growth Exceeded Expectations in Q2",
        "Scientists Publish New Research on COVID-19 Variants",
        "Treasury Department Releases Monthly Budget Report",
        "State Department Issues Travel Advisory for European Countries",
        "Congressional Committee Approves Budget Resolution",
        "International Diplomats Reach Agreement on Trade Policy",
        "Stock Market Closes Higher Following Positive Economic Data",
        "Census Bureau Releases New Population Statistics",
        "Health Officials Report Decrease in COVID-19 Cases",
        "Education Department Announces New Student Loan Policy"
    ]
    
    # Create DataFrame
    texts = fake_news + real_news
    labels = [0] * len(fake_news) + [1] * len(real_news)  # 0 for fake, 1 for real
    
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    logger.info(f"Created dataset with {len(df)} samples: {len(fake_news)} fake news, {len(real_news)} real news")
    return df

def create_and_save_model():
    """Train a model on the dataset and save it"""
    try:
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        logger.info("Created or verified models directory")
        
        # Create dataset
        df = create_data_from_notebook()
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
        )
        
        logger.info(f"Split dataset: {len(X_train)} training samples, {len(X_test)} test samples")
        
        # Create TF-IDF vectorizer
        logger.info("Creating and fitting TF-IDF vectorizer")
        vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train logistic regression model
        logger.info("Training logistic regression model")
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            C=1.0,
            solver='liblinear',
            random_state=42
        )
        model.fit(X_train_vec, y_train)
        
        # Evaluate model
        train_score = model.score(X_train_vec, y_train)
        test_score = model.score(X_test_vec, y_test)
        logger.info(f"Model performance: Training accuracy = {train_score:.4f}, Test accuracy = {test_score:.4f}")
        
        # Save model and vectorizer
        model_path = os.path.join("models", "fake_news_model.joblib")
        vectorizer_path = os.path.join("models", "tfidf_vectorizer.joblib")
        
        logger.info(f"Saving model to {model_path}")
        joblib.dump(model, model_path)
        logger.info(f"Model saved: {os.path.getsize(model_path) / 1024:.2f} KB")
        
        logger.info(f"Saving vectorizer to {vectorizer_path}")
        joblib.dump(vectorizer, vectorizer_path)
        logger.info(f"Vectorizer saved: {os.path.getsize(vectorizer_path) / 1024:.2f} KB")
        
        # Verify files exist
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            logger.info("Successfully verified saved model and vectorizer files")
            
            # Test the model
            test_texts = [
                "BREAKING: Secret government conspiracy exposed by anonymous source",
                "Senate approves new bill with bipartisan support, president expected to sign"
            ]
            
            test_vectors = vectorizer.transform(test_texts)
            predictions = model.predict(test_vectors)
            probabilities = model.predict_proba(test_vectors)
            
            for i, text in enumerate(test_texts):
                pred = predictions[i]
                prob = probabilities[i][pred]
                logger.info(f"Test prediction for '{text[:30]}...': {'Real' if pred == 1 else 'Fake'} news with {prob:.4f} confidence")
            
            # Try to save to absolute path if in deployment environment
            try:
                if os.path.exists("/opt/render/project/src"):
                    abs_model_dir = "/opt/render/project/src/models"
                    logger.info(f"Deployment environment detected, also saving to {abs_model_dir}")
                    
                    os.makedirs(abs_model_dir, exist_ok=True)
                    abs_model_path = os.path.join(abs_model_dir, "fake_news_model.joblib")
                    abs_vectorizer_path = os.path.join(abs_model_dir, "tfidf_vectorizer.joblib")
                    
                    joblib.dump(model, abs_model_path)
                    joblib.dump(vectorizer, abs_vectorizer_path)
                    logger.info(f"Successfully saved model files to absolute paths")
            except Exception as e:
                logger.error(f"Error saving to absolute path: {str(e)}")
            
            return True
        else:
            logger.error("Failed to verify saved model files")
            return False
    except Exception as e:
        logger.error(f"Error in model creation: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting model creation process")
    success = create_and_save_model()
    logger.info(f"Model creation {'completed successfully' if success else 'failed'}") 