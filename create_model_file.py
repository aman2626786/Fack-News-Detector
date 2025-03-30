import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('create_model.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def create_data_from_notebook():
    """Create sample data mimicking the notebook's dataset."""
    logger.info("Creating dataset based on notebook example...")
    
    # Sample fake news from the notebook
    fake_news = [
        "Donald Trump Sends Out Embarrassing New Year's Eve Message, Experts Are Worried.",
        "Drunk Bragging Trump Staffer Started Russian Collusion Investigation.",
        "Sheriff David Clarke Becomes An Internet Joke For The Way He Dressed Like At Trump's Inauguration.",
        "Trump Is So Obsessed He Even Has Obama's Name Coded Into His Website.",
        "Pope Francis Just Called Out Donald Trump During His Christmas Speech.",
        "Trump's Director of Social Media Tweeted A Fake News Story About Immigrants Causing California Wildfires.",
        "BREAKING: The FBI Just Confirmed That Hillary Clinton's Email Server Was Actually Hacked By Foreign Agents.",
        "BREAKING: Elizabeth Warren Is In Big Trouble After Being Caught Doing This To A Native American Family.",
        "BREAKING: Trump Just Ended Obama's Vacation Scam And Sent Him A Bill You Won't Believe.",
        "BREAKING: Black Lives Matter Leader Just Admitted The DISGUSTING Thing They're Going To Do To 'Whitey'."
    ]
    
    # Sample real news from the notebook
    real_news = [
        "WASHINGTON (Reuters) - The top Democrat on the House Intelligence Committee said on Thursday that he has grave concerns about the Republican's choice to lead a Russia probe.",
        "HAVANA (Reuters) - Hurricane Irma seriously damaged Cuba's already dilapidated sugar industry infrastructure.",
        "WASHINGTON (Reuters) - U.S. Vice President Mike Pence and other top White House officials discussed Obamacare replacement legislation.",
        "WASHINGTON (Reuters) - President Donald Trump's son told Senate investigators that Wikileaks contacts him during the 2016 presidential campaign.",
        "MANILA (Reuters) - U.S. President Donald Trump said on Monday he had made significant progress on trade issues during a fruitful trip.",
        "WASHINGTON (Reuters) - U.S. Director of National Intelligence James Clapper said on Thursday he has submitted his letter of resignation.",
        "WASHINGTON (Reuters) - President Donald Trump is expected to unveil a new U.S. policy toward Cuba as early as next Friday.",
        "(Reuters) - The state of Hawaii said in a court filing that it intends to seek a temporary restraining order on Wednesday.",
        "LONDON (Reuters) - LexisNexis, a provider of legal, regulatory and business information, said on Tuesday it had withdrawn two products from the Chinese market.",
        "MOSCOW (Reuters) - Vatican Secretary of State Cardinal Pietro Parolin said on Tuesday that there was a great interest in papal travel to Moscow."
    ]
    
    # Create dataframe
    fake_df = pd.DataFrame({
        'text': fake_news,
        'label': [0] * len(fake_news)  # 0 for fake news
    })
    
    real_df = pd.DataFrame({
        'text': real_news,
        'label': [1] * len(real_news)  # 1 for real news
    })
    
    # Combine and shuffle
    df = pd.concat([fake_df, real_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Created sample dataset with {len(df)} rows")
    return df

def create_and_save_model():
    """Train model using the notebook approach and save it."""
    try:
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Get data
        df = create_data_from_notebook()
        
        # Split into features and target
        X = df['text']
        y = df['label']
        
        # Create and train the TF-IDF vectorizer with the same parameters
        logger.info("Creating TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
        X_tfidf = vectorizer.fit_transform(X)
        
        # Create and train logistic regression model with the same parameters
        logger.info("Training logistic regression model...")
        model = LogisticRegression(C=0.1, class_weight={0: 1.2, 1: 0.8}, max_iter=1000, random_state=42)
        model.fit(X_tfidf, y)
        
        # Save model and vectorizer
        model_path = os.path.join('models', 'fake_news_model.joblib')
        vectorizer_path = os.path.join('models', 'tfidf_vectorizer.joblib')
        
        logger.info(f"Saving model to {model_path}")
        joblib.dump(model, model_path)
        
        logger.info(f"Saving vectorizer to {vectorizer_path}")
        joblib.dump(vectorizer, vectorizer_path)
        
        # Verify files exist
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            logger.info("Model and vectorizer successfully saved")
            
            # Test the model
            test_text = "MOSCOW (Reuters) - Vatican Secretary of State Cardinal Pietro Parolin said on Tuesday that there was a great interest in papal travel to Moscow."
            test_vector = vectorizer.transform([test_text])
            prediction = model.predict(test_vector)[0]
            confidence = model.predict_proba(test_vector)[0][1]  # Probability of class 1
            
            logger.info(f"Test prediction: {'Real News' if prediction == 1 else 'Fake News'}, Confidence: {confidence:.4f}")
            
            # Check file sizes
            model_size = os.path.getsize(model_path)
            vectorizer_size = os.path.getsize(vectorizer_path)
            logger.info(f"Model size: {model_size} bytes")
            logger.info(f"Vectorizer size: {vectorizer_size} bytes")
            
            return True
        else:
            logger.error("Model or vectorizer file not found after saving")
            return False
            
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        return False
        
if __name__ == "__main__":
    create_and_save_model() 