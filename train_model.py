import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os
from pathlib import Path

def create_sample_data():
    """Create sample data for testing."""
    real_news = [
        "Scientists discover new species of deep-sea creatures in the Pacific Ocean.",
        "Global temperatures continue to rise, breaking previous records.",
        "New study shows benefits of regular exercise on mental health.",
        "Tech company announces breakthrough in renewable energy storage.",
        "Medical researchers develop new treatment for common disease."
    ]
    
    fake_news = [
        "Aliens secretly control world governments, says anonymous source.",
        "Scientists discover that the Earth is actually flat.",
        "Secret society of lizard people revealed to be true.",
        "Time travel machine invented in basement, government covers it up.",
        "Moon landing was filmed in Hollywood studio, claims conspiracy theorist."
    ]
    
    data = pd.DataFrame({
        'text': real_news + fake_news,
        'label': [1] * len(real_news) + [0] * len(fake_news)
    })
    return data

def train_model():
    """Train the model and save it."""
    # Create sample data
    data = create_sample_data()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['label'], test_size=0.2, random_state=42
    )
    
    # Create and fit the vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    
    # Train the model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_vectorized, y_train)
    
    # Create models directory if it doesn't exist
    models_dir = Path(__file__).parent / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # Save the model and vectorizer
    joblib.dump(model, models_dir / 'fake_news_model.joblib')
    joblib.dump(vectorizer, models_dir / 'tfidf_vectorizer.joblib')
    
    # Print model performance
    X_test_vectorized = vectorizer.transform(X_test)
    accuracy = model.score(X_test_vectorized, y_test)
    print(f"Model accuracy: {accuracy:.2f}")

if __name__ == '__main__':
    train_model() 