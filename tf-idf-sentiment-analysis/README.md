# Social Media Sentiment Classifier

A machine learning application that classifies social media comments (Reddit and Twitter) into positive, neutral, or negative sentiments using TF-IDF vectorization and Logistic Regression.

## Demo

[Live Demo](YOUR_STREAMLIT_CLOUD_URL)

![App Screenshot](screenshot.png)

## Features

- Real-time sentiment prediction
- Confidence scores for each sentiment class
- Clean and intuitive user interface
- Pre-trained model using both Reddit and Twitter data
- Cross-platform sentiment analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/social-media-sentiment-classifier.git
cd social-media-sentiment-classifier
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Train the model (first time only):
```bash
python train_model.py
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

## Dataset

The model is trained on a combined dataset of Reddit and Twitter comments with the following sentiment labels:
- -1: Negative sentiment
- 0: Neutral sentiment
- 1: Positive sentiment

The data is split into:
- 75% training data
- 25% testing data

## Model Details

- Text Vectorization: TF-IDF with 5000 features
- Algorithm: Logistic Regression
- Training/Testing Split: 75/25

## Technologies Used

- Python
- Streamlit
- scikit-learn
- pandas
- numpy
- joblib

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
