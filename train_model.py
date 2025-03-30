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
        logging.FileHandler("train.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def create_sample_dataset():
    """Create a sample dataset for training if real data is not available"""
    logger.info("Creating sample dataset for training")
    
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
    
    # Combine datasets
    texts = fake_news + real_news
    labels = [0] * len(fake_news) + [1] * len(real_news)  # 0 for fake, 1 for real
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    logger.info(f"Created sample dataset with {len(df)} examples: {len(fake_news)} fake, {len(real_news)} real")
    return df

def train_and_save_model():
    """Train a model on the dataset and save it"""
    try:
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Try to use real data if available, otherwise use sample data
        try:
            # Check if we can load real data (implement this if you have a specific dataset)
            logger.info("Attempting to load real dataset")
            # df = pd.read_csv("path_to_your_dataset.csv")
            # If no dataset is available, raise an exception to use the sample data
            raise FileNotFoundError("No real dataset available")
        except:
            logger.info("Could not load real dataset, using sample data instead")
            df = create_sample_dataset()
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
        )
        
        logger.info(f"Training data size: {len(X_train)}, Test data size: {len(X_test)}")
        
        # Create and fit TF-IDF vectorizer
        logger.info("Training TF-IDF vectorizer")
        vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        logger.info(f"Vectorizer created with {len(vectorizer.get_feature_names_out())} features")
        
        # Train logistic regression model
        logger.info("Training logistic regression model")
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            C=1.0,
            solver='liblinear'
        )
        model.fit(X_train_vec, y_train)
        
        # Evaluate model
        train_score = model.score(X_train_vec, y_train)
        test_score = model.score(X_test_vec, y_test)
        logger.info(f"Model trained. Training accuracy: {train_score:.4f}, Test accuracy: {test_score:.4f}")
        
        # Save model and vectorizer
        model_path = os.path.join("models", "fake_news_model.joblib")
        vectorizer_path = os.path.join("models", "tfidf_vectorizer.joblib")
        
        logger.info(f"Saving model to {model_path}")
        joblib.dump(model, model_path)
        logger.info(f"Model saved. File size: {os.path.getsize(model_path) / 1024:.2f} KB")
        
        logger.info(f"Saving vectorizer to {vectorizer_path}")
        joblib.dump(vectorizer, vectorizer_path)
        logger.info(f"Vectorizer saved. File size: {os.path.getsize(vectorizer_path) / 1024:.2f} KB")
        
        # Test prediction
        sample_text = "Senate passes new legislation with bipartisan support"
        sample_vec = vectorizer.transform([sample_text])
        prediction = model.predict(sample_vec)[0]
        probability = model.predict_proba(sample_vec)[0][prediction]
        logger.info(f"Test prediction on '{sample_text}': {'Real' if prediction == 1 else 'Fake'} news with {probability:.4f} confidence")
        
        return True
    except Exception as e:
        logger.error(f"Error in training model: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting model training")
    success = train_and_save_model()
    logger.info(f"Model training {'completed successfully' if success else 'failed'}") 