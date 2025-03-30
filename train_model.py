import os
import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample data if real data is not available."""
    logger.info("Creating sample fake news data...")
    
    # Create fake news examples
    fake_texts = [
        "BREAKING: President secretly signed deal with aliens last night.",
        "Scientists discover that chocolate cures all diseases, big pharma hiding the truth.",
        "Celebrities gathering to announce the earth is actually flat, press conference tomorrow.",
        "Government admits to mind control through television signals, whistleblower reveals.",
        "Doctor discovers miracle cure for all ailments, medical industry trying to silence him.",
        "New study shows that vaccines contain microscopic tracking devices.",
        "Anonymous source reveals politicians are actually reptilians in disguise.",
        "Secret technology exists that can generate unlimited free energy, oil companies suppressing it.",
        "Moon landing was filmed in Hollywood, NASA employee confirms in deathbed confession.",
        "Breaking: Famous celebrity caught eating children in restaurant basement."
    ]
    
    # Create real news examples
    real_texts = [
        "President signs new infrastructure bill aimed at improving roads and bridges nationwide.",
        "Scientists publish study showing modest health benefits from dark chocolate consumption.",
        "Celebrities gather for annual charity event, raise millions for children's hospital.",
        "Government announces new transparency measures for public data access.",
        "Doctors recommend new treatment protocol for common illness based on clinical trials.",
        "New study confirms vaccine safety profile consistent with previous research.",
        "Political leaders from both parties reach compromise on budget legislation.",
        "Renewable energy sector shows growth as installation costs decline, report says.",
        "NASA releases new high-definition images from Mars rover expedition.",
        "Celebrity donates significant portion of movie earnings to disaster relief."
    ]
    
    # Create dataframes for fake and real news
    fake_df = pd.DataFrame({
        'text': fake_texts,
        'label': 0  # 0 for fake news
    })
    
    real_df = pd.DataFrame({
        'text': real_texts,
        'label': 1  # 1 for real news
    })
    
    # Combine datasets
    df = pd.concat([fake_df, real_df], ignore_index=True)
    
    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    logger.info(f"Created sample dataset with {len(df)} examples")
    return df

def train_model():
    """Train a fake news detection model."""
    try:
        logger.info("Starting model training process...")
        
        # Create directories if they don't exist
        os.makedirs('models', exist_ok=True)
        
        # Create sample data
        df = create_sample_data()
        
        # Split features and target
        X = df['text']
        y = df['label']
        
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples")
        
        # Create and fit TF-IDF vectorizer - use max_features=10000 as in the notebook
        logger.info("Creating TF-IDF vectorizer...")
        tfidf = TfidfVectorizer(max_features=10000, stop_words="english")
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)
        
        # Train logistic regression model - use the same parameters as in the notebook
        logger.info("Training logistic regression model...")
        model = LogisticRegression(C=0.1, class_weight={0: 1.2, 1: 0.8}, max_iter=1000)
        model.fit(X_train_tfidf, y_train)
        
        # Evaluate model
        accuracy = model.score(X_test_tfidf, y_test)
        logger.info(f"Model accuracy: {accuracy:.4f}")
        
        # Save model and vectorizer
        model_path = os.path.join('models', 'fake_news_model.joblib')
        vectorizer_path = os.path.join('models', 'tfidf_vectorizer.joblib')
        
        logger.info(f"Saving model to {model_path}")
        joblib.dump(model, model_path)
        
        logger.info(f"Saving vectorizer to {vectorizer_path}")
        joblib.dump(tfidf, vectorizer_path)
        
        # Verify files exist
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            logger.info("Model and vectorizer successfully saved")
            model_size = os.path.getsize(model_path)
            vectorizer_size = os.path.getsize(vectorizer_path)
            logger.info(f"Model size: {model_size} bytes")
            logger.info(f"Vectorizer size: {vectorizer_size} bytes")
            return True
        else:
            logger.error("Failed to save model and vectorizer")
            return False
            
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return False

if __name__ == "__main__":
    train_model() 