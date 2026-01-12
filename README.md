# Sentiment Analysis Dashboard

A comprehensive sentiment and emotion analysis tool built with Streamlit and Transformer models.

## Features

- 📊 **Sentiment Analysis**: Classify text as positive, neutral, or negative
- 🎭 **Emotion Detection**: Analyze 6 core emotions (joy, sadness, anger, fear, surprise, disgust)
- 📈 **Interactive Visualizations**: Charts, graphs, and heatmaps
- 🔍 **Advanced Filtering**: Search and filter results by sentiment, confidence, and keywords
- 📥 **Multiple Export Formats**: CSV, Excel, and JSON
- ⚡ **Batch Processing**: Analyze hundreds of responses with progress tracking
- 🎯 **Divergence Metrics**: Identify mixed or uncertain emotional responses

## Installation

1. Clone the repository:
```bash
git clone https://github.com/martinvc154/Sentiment-Analysis-Claude.git
cd Sentiment-Analysis-Claude
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Upload your CSV or Excel file containing text data
3. Select the column containing text to analyze
4. Click "Run Analysis" to begin

## Data Format

Your input file should contain at least one column with text data. Example:

| Response ID | Text | Date |
|-------------|------|------|
| 1 | I love this product! | 2024-01-01 |
| 2 | Not sure how I feel... | 2024-01-02 |
| 3 | Terrible experience! | 2024-01-03 |

## Output

The analysis provides:
- **sentiment**: positive/neutral/negative classification
- **sentiment_score**: -1 to 1 scale
- **confidence**: model confidence (0-1)
- **Emotion scores**: joy, sadness, anger, fear, surprise, disgust
- **dominant_emotion**: primary emotion detected
- **divergence**: measure of emotional uncertainty (0-1)

## Models Used

- **Sentiment**: DistilBERT fine-tuned on SST-2
- **Emotions**: DistilRoBERTa fine-tuned on GoEmotions

## Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies

## License

MIT License