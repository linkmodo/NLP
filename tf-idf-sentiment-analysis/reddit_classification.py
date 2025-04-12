import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Import and load the data with UTF-8 encoding
df = pd.read_csv("Reddit_Data.csv", encoding='utf-8')

# Handle any NaN values
df['clean_comment'] = df['clean_comment'].fillna('')

# The data is already cleaned, so we'll use it directly
X = df['clean_comment']
y = df['category']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform text to vectors using TF-IDF with unicode handling
tfidf = TfidfVectorizer(max_features=5000, decode_error='ignore')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train the model (using Logistic Regression)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Reddit Comment Classification - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.close()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Function to predict new text
def predict_text(text):
    # Transform to TF-IDF
    text_tfidf = tfidf.transform([text])
    # Predict
    prediction = model.predict(text_tfidf)
    return prediction[0]
