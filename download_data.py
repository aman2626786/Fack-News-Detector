import os
import requests
import pandas as pd
import numpy as np

def generate_sample_data():
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Generate fake news samples
    fake_news_templates = [
        "BREAKING: {subject} has been discovered to be a complete hoax! Experts reveal shocking truth.",
        "EXCLUSIVE: {subject} conspiracy finally exposed! What they don't want you to know.",
        "SHOCKING: {subject} is actually a government cover-up! Insider reveals all.",
        "ALERT: {subject} is fake news! Here's the real story they're hiding from you.",
        "REVEALED: The truth about {subject} that mainstream media won't tell you!"
    ]
    
    # Generate real news samples
    real_news_templates = [
        "Reuters: {subject} has been confirmed by multiple sources.",
        "Official Report: {subject} findings released by government agency.",
        "Study Shows: {subject} according to peer-reviewed research.",
        "Confirmed: {subject} as reported by verified sources.",
        "Fact Check: {subject} verified by independent experts."
    ]
    
    subjects = [
        "Global warming", "COVID-19 vaccine", "Moon landing", "Earth shape",
        "Climate change", "Evolution theory", "Vaccination safety", "Space exploration",
        "Scientific research", "Medical discoveries"
    ]
    
    # Generate fake news
    fake_news = []
    for _ in range(1000):
        template = np.random.choice(fake_news_templates)
        subject = np.random.choice(subjects)
        fake_news.append({
            'title': template.format(subject=subject),
            'text': template.format(subject=subject) + " " + " ".join([np.random.choice(subjects) for _ in range(5)]),
            'subject': 'News',
            'date': '2024-03-30'
        })
    
    # Generate real news
    real_news = []
    for _ in range(1000):
        template = np.random.choice(real_news_templates)
        subject = np.random.choice(subjects)
        real_news.append({
            'title': template.format(subject=subject),
            'text': template.format(subject=subject) + " " + " ".join([np.random.choice(subjects) for _ in range(5)]),
            'subject': 'News',
            'date': '2024-03-30'
        })
    
    # Create DataFrames and save to CSV
    df_fake = pd.DataFrame(fake_news)
    df_real = pd.DataFrame(real_news)
    
    df_fake.to_csv('data/Fake.csv', index=False)
    df_real.to_csv('data/True.csv', index=False)
    
    print("Sample dataset generated successfully!")

if __name__ == "__main__":
    generate_sample_data() 