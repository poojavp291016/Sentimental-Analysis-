#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas textblob


# In[2]:


import pandas as pd
import string
from textblob import TextBlob

# Step 1: Load the Dataset
file_path = r"C:\Users\pooja\Downloads\user_review.csv"
df = pd.read_csv(file_path)
print("Dataset Loaded Successfully!")

# Step 2: Data Cleaning
df = df.dropna(subset=['review'])  # Remove rows where 'review' is null

# Keep only 'id' and 'review' columns
df = df[['id', 'review']]
print("Data Cleaned Successfully!")

# Step 3: Text Preprocessing
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

df['review'] = df['review'].apply(preprocess_text)
print("Text Preprocessing Done!")

# Step 4: Sentiment Analysis
def analyze_sentiment(review):
    analysis = TextBlob(review)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['review'].apply(analyze_sentiment)
print("Sentiment Analysis Completed!")

# Step 5: Generate Summary Report
sentiment_distribution = df['sentiment'].value_counts()
print("Sentiment Distribution Report:")
print(sentiment_distribution)


# In[ ]:




