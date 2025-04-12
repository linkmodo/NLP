# Social Media Sentiment Analyzer

A Streamlit web application that analyzes sentiment in social media comments using a pretrained RoBERTa model fine-tuned on Twitter data.

## Features

- Real-time sentiment analysis of text input
- Three sentiment categories: Positive, Neutral, and Negative
- Confidence scores for predictions
- Modern, user-friendly interface

## Model

The application uses the `cardiffnlp/twitter-roberta-base-sentiment-latest` model, which is:
- Based on the RoBERTa architecture
- Fine-tuned specifically for social media sentiment analysis
- Optimized for Twitter-style text

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/social-media-sentiment-analyzer.git
cd social-media-sentiment-analyzer
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and go to `http://localhost:8501`

3. Enter text in the input box and see the sentiment analysis results!

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Project Structure

```
social-media-sentiment-analyzer/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
└── .gitignore         # Git ignore file
```

## License

MIT License

## Contributing

Feel free to open issues or submit pull requests if you find any bugs or have suggestions for improvements.
