import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Load the dataset
df1 = pd.read_csv("data/Fake.csv")
df2 = pd.read_csv("data/True.csv")

# Add labels
df1["label"] = 0  # Fake news
df2["label"] = 1  # Real news

# Combine datasets
df = pd.concat([df1, df2], ignore_index=True)

# Prepare data
X = df['text']
y = df['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

# Create and fit TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=10000, stop_words="english")
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Create and train the model
model = LogisticRegression(C=0.1, class_weight={0: 1.2, 1: 0.8})
model.fit(X_train_tfidf, y_train)

# Save the model and vectorizer
joblib.dump(model, 'models/fake_news_model.joblib')
joblib.dump(tfidf, 'models/tfidf_vectorizer.joblib')

print("Model and vectorizer saved successfully!")

# Print model performance
y_pred = model.predict(X_test_tfidf)
accuracy = (y_pred == y_test).mean()
print(f"\nModel Accuracy: {accuracy:.4f}")

# Test the model with a sample
test_text = "MEXICO CITY (Reuters) - Mexico's foreign minister Luis Videgaray will travel to Washington on Tuesday to meet with senior U.S. officials and to attend a meeting at the Organisation of American States (OAS), his ministry said in a statement."
test_vectorized = tfidf.transform([test_text])
prediction = model.predict(test_vectorized)[0]
print(f"\nSample Test Prediction: {'Real News' if prediction == 1 else 'Fake News'}")

if __name__ == "__main__":
    train_and_save_model() 