import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

def create_sample_data():
    """Create sample data for testing."""
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
    
    return real_news + fake_news, [1] * len(real_news) + [0] * len(fake_news)

def generate_model():
    """Generate and save the model files."""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Create sample data
    texts, labels = create_sample_data()
    
    # Create and fit the vectorizer
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)
    
    # Train the model
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    
    # Save the model and vectorizer
    joblib.dump(model, 'models/fake_news_model.joblib')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
    
    print("Model and vectorizer generated successfully!")

if __name__ == '__main__':
    generate_model() 