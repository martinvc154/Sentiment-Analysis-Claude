# Sentiment-Analysis-Claude

A sophisticated sentiment analysis application powered by Claude AI that analyzes text for sentiment, emotions, and detects potential preference falsification through sentiment-emotion divergence.

## Features

### 📊 Dashboard Tab
- **Metric Cards**: Display total responses, unique organizations, average sentiment (color-coded), and high divergence count
- **Sentiment Distribution**: Interactive donut chart showing positive/neutral/negative breakdown
- **Emotion Prevalence**: Horizontal bar chart showing average scores for 6 emotions (joy, sadness, anger, fear, surprise, disgust)
- **Divergence Alert**: Warning box for responses with high sentiment-emotion divergence (>0.4)
- **Organization Comparison**: Grouped bar chart comparing sentiment across organizations

### 📁 Upload Tab
- Upload CSV files with text responses
- Map columns for text and organization
- Batch analyze responses using Claude API
- Real-time progress tracking

### 📋 Results Tab
- View detailed analysis for each response
- Filter by sentiment category and organization
- View divergent responses
- Expandable cards with full emotion breakdowns

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

3. Get your Claude API key from [Anthropic Console](https://console.anthropic.com/)

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. In the **Upload** tab:
   - Enter your Claude API key
   - Upload a CSV file with text responses
   - Map the text column and optionally the organization column
   - Click "Analyze Sentiment"

3. View results in the **Dashboard** tab:
   - Overview metrics
   - Visual sentiment distribution
   - Emotion prevalence analysis
   - Organization comparisons

4. Explore details in the **Results** tab:
   - Filter and search responses
   - View individual emotion scores
   - Identify divergent responses

## Sample Data

A sample CSV file (`sample_data.csv`) is included for testing.

## How It Works

The application uses Claude AI to analyze each text response for:

1. **Sentiment**: A score from -1 (very negative) to 1 (very positive)
2. **Emotions**: Six emotion scores (joy, sadness, anger, fear, surprise, disgust), each from 0 to 1
3. **Divergence**: Measures mismatch between sentiment and emotional content, potentially indicating preference falsification

## Dashboard Components

### Row 1 - Metrics
- Total Responses: Count of analyzed texts
- Organizations Analyzed: Number of unique organizations (or N/A)
- Average Sentiment: Color-coded (green >0.1, red <-0.1, gray otherwise)
- High Divergence Count: Responses with divergence >0.4

### Row 2 - Charts
- **Left**: Sentiment distribution donut chart with custom colors
- **Right**: Emotion prevalence horizontal bar chart, sorted by average score

### Row 3 - Alert
- Displays warning when high divergence responses are detected
- Button to filter and view divergent responses

### Row 4 - Organization Analysis
- Only shown when organization column is mapped
- Grouped bar chart comparing sentiment percentages across organizations

## Color Scheme

- **Sentiments**: Positive (#55a868), Neutral (#8c8c8c), Negative (#c44e52)
- **Emotions**: Joy (#f4b942), Sadness (#5c7fa3), Anger (#c44e52), Fear (#8b5cf6), Surprise (#06b6d4), Disgust (#84735a)

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- Plotly
- Anthropic API

## License

MIT License
