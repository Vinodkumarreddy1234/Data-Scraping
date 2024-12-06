# Data-Scraping


import praw
import pandas as pd

# Initialize Reddit API client
reddit = praw.Reddit(client_id='your_client_id', 
                     client_secret='your_client_secret',
                     user_agent='your_user_agent')

# Choose the subreddit and keyword
subreddit = reddit.subreddit('wallstreetbets')

# Scrape posts from the last 30 days
posts = []
for submission in subreddit.new(limit=100):  # Limit to 100 posts for the demo
    posts.append([submission.title, submission.selftext, submission.score, submission.created_utc])

# Create a DataFrame
df = pd.DataFrame(posts, columns=['Title', 'Text', 'Score', 'Timestamp'])

# Save data
df.to_csv('reddit_stock_posts.csv', index=False)

**Data Preprocessing:**

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.tokenize import word_tokenize
import re

def preprocess_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]  # Stop word removal
    return ' '.join(tokens)

df['cleaned_text'] = df['Text'].apply(preprocess_text)


**Prediction Model**

import yfinance as yf

# Get historical stock price data
stock_data = yf.download('AAPL', start='2023-01-01', end='2024-01-01')

# Merge Reddit data with stock data by matching timestamps
# This will require time-based alignment (e.g., aggregate Reddit posts by day)


**Example model using Random Forest:**

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Assuming you have features like 'sentiment' and 'post_volume' and target as 'stock_movement'
X = df[['sentiment', 'post_volume']]  # Features
y = df['stock_movement']  # Target: 'up' or 'down' based on stock price movement

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print(classification_report(y_test, y_pred))
