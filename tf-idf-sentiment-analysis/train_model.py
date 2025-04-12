import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load both datasets
print("Loading datasets...")
reddit_df = pd.read_csv("Reddit_Data.csv", encoding='utf-8')
twitter_df = pd.read_csv("Twitter_Data.csv", encoding='utf-8')

# Rename columns to match
reddit_df = reddit_df.rename(columns={'clean_comment': 'clean_text'})

# Combine datasets
combined_df = pd.concat([reddit_df, twitter_df], ignore_index=True)

# Handle any NaN values
combined_df['clean_text'] = combined_df['clean_text'].fillna('')
combined_df = combined_df.dropna(subset=['category'])  # Remove rows with NaN categories

# Convert category to int
combined_df['category'] = combined_df['category'].astype(int)

# Print dataset statistics
print("\nDataset Statistics:")
print(f"Total samples: {len(combined_df)}")
print("\nCategory distribution:")
print(combined_df['category'].value_counts().sort_index())

# Prepare the data
X = combined_df['clean_text']
y = combined_df['category']

# Split the data (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create and train the vectorizer
print("\nTraining TF-IDF vectorizer...")
tfidf = TfidfVectorizer(max_features=5000, decode_error='ignore')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train the model
print("Training the model...")
model = LogisticRegression(max_iter=2000)  # Increased max_iter for better convergence
model.fit(X_train_tfidf, y_train)

# Evaluate the model
train_score = model.score(X_train_tfidf, y_train)
test_score = model.score(X_test_tfidf, y_test)

print(f"\nModel Performance:")
print(f"Training Accuracy: {train_score:.4f}")
print(f"Testing Accuracy: {test_score:.4f}")

# Save the model and vectorizer
print("\nSaving model and vectorizer...")
joblib.dump(model, 'model.joblib')
joblib.dump(tfidf, 'vectorizer.joblib')
print("Done! The model and vectorizer have been saved.")
