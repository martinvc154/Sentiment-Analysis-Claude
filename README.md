# Sentiment Analysis with Emotion Profiles & Clustering

A comprehensive sentiment analysis application featuring emotion detection, radar chart visualizations, and advanced K-means clustering with PCA projections.

## Features

### 🎭 Emotion Analysis
- Detects 6 core emotions: joy, sadness, anger, fear, surprise, disgust
- Powered by `emotion-english-distilroberta-base` transformer model
- Returns normalized emotion scores (0-1 scale)

### 📊 Emotion Profiles Visualization
- **Radar/Spider Charts**: Interactive 6-axis emotion profiles
- **Organization Mapping**: Separate charts for each organization (2x2 or 3x3 grid layout)
- **Aggregate View**: Single chart when no organizations specified
- **Semi-transparent fills** with color-coded axes
- Export functionality (PNG, PDF, JSON)

### 🔬 Cluster Analysis
- **K-means Clustering**: Automatically tests k=2,3,4,5
- **Silhouette Score Optimization**: Selects optimal number of clusters
- **7-dimensional Feature Space**: sentiment_score + 6 emotions
- **2D PCA Visualization**: Projects high-dimensional data for visualization
- **Confidence Ellipses**: 95% confidence regions around each cluster
- **Centroid Markers**: Larger star markers showing cluster centers
- **Intelligent Labeling**: Automatically suggests cluster labels based on profile:
  - **Aligned Positive**: High positive sentiment + joy
  - **Open Critics**: Negative sentiment + low divergence
  - **Strategic Adapters**: Neutral + high divergence + fear
  - **Preference Falsifiers**: Positive sentiment masking negative emotions

### 📈 Cluster Characteristics Table
- Cluster ID with color coding
- Sample count per cluster
- Average sentiment score
- Dominant emotion
- Suggested label with description
- Export as CSV

### 💾 Export Capabilities
- **Emotion Profiles**: PNG, PDF, JSON
- **Cluster Visualizations**: PNG, PDF, JSON
- **Cluster Table**: CSV format

## Technology Stack

### Backend
- **FastAPI**: High-performance Python web framework
- **Transformers**: Hugging Face models for NLP
  - `distilbert-base-uncased-finetuned-sst-2-english` for sentiment
  - `j-hartmann/emotion-english-distilroberta-base` for emotions
- **scikit-learn**: K-means clustering, PCA, silhouette scoring
- **NumPy & Pandas**: Data processing
- **SciPy**: Statistical calculations

### Frontend
- **React 18**: Modern UI framework
- **Recharts**: Beautiful, responsive charts
- **Axios**: HTTP client
- **html2canvas & jsPDF**: Export functionality

## Project Structure

```
Sentiment-Analysis-Claude/
├── backend/
│   ├── main.py              # FastAPI application
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── components/
│   │   │   ├── EmotionProfiles.js
│   │   │   ├── EmotionProfiles.css
│   │   │   ├── ClusterVisualization.js
│   │   │   └── ClusterVisualization.css
│   │   ├── App.js
│   │   ├── App.css
│   │   ├── index.js
│   │   └── index.css
│   └── package.json
└── README.md
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

```bash
cd backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py
```

The backend will start on `http://localhost:8000`

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

The frontend will start on `http://localhost:3000`

## Usage

### 1. Input Texts
- Enter texts to analyze (with optional organization labels)
- Use "Load Sample Data" button for demo data
- Add multiple texts using the "+ Add Another Text" button

### 2. Analyze
- Click "Analyze Texts" to perform sentiment and emotion analysis
- View results summary

### 3. Visualize

#### Emotion Profiles
- Click "Emotion Profiles" to view radar charts
- If organizations are present: displays grid of charts (one per organization)
- If no organizations: displays single aggregate chart
- Export options: PNG, PDF, JSON

#### Cluster Analysis
- Click "Cluster Analysis" to perform clustering (requires ≥5 samples)
- View optimal cluster count and silhouette score
- Explore 2D PCA scatter plot with:
  - Color-coded clusters
  - Centroid markers (stars)
  - 95% confidence ellipses
- Review cluster characteristics table
- Export visualization or table data

## API Endpoints

### POST `/analyze`
Analyzes sentiment and emotions for multiple texts.

**Request Body:**
```json
{
  "texts": [
    {
      "text": "I love this product!",
      "organization": "Company A"
    }
  ]
}
```

**Response:**
```json
[
  {
    "text": "I love this product!",
    "organization": "Company A",
    "sentiment_label": "POSITIVE",
    "sentiment_score": 0.95,
    "emotions": {
      "joy": 0.85,
      "sadness": 0.02,
      "anger": 0.01,
      "fear": 0.03,
      "surprise": 0.05,
      "disgust": 0.01
    }
  }
]
```

### POST `/cluster`
Performs K-means clustering on analysis results.

**Request Body:** Array of analysis results from `/analyze`

**Response:**
```json
{
  "optimal_k": 3,
  "silhouette_score": 0.65,
  "cluster_assignments": [0, 1, 2, 0, 1],
  "pca_coordinates": [[0.5, 0.3], ...],
  "centroids": [[0.6, 0.4], ...],
  "confidence_ellipses": [...],
  "cluster_info": [
    {
      "cluster_id": 0,
      "n": 25,
      "avg_sentiment": 0.75,
      "dominant_emotion": "joy",
      "suggested_label": "Aligned Positive"
    }
  ]
}
```

### GET `/health`
Health check endpoint.

## Cluster Label Interpretation

- **Aligned Positive**: Authentic positive sentiment with high joy - likely genuine satisfaction
- **Open Critics**: Clear negative sentiment with consistency - direct, honest criticism
- **Strategic Adapters**: Neutral sentiment with fear and high variance - cautious, calculating responses
- **Preference Falsifiers**: Positive words masking negative emotions - potential social desirability bias

## Development

### Running Tests
```bash
# Backend
cd backend
pytest

# Frontend
cd frontend
npm test
```

### Building for Production
```bash
# Frontend
cd frontend
npm run build
```

## Performance Considerations

- **First Run**: Model downloads may take time (~500MB total)
- **Batch Processing**: Recommended for >100 texts
- **Clustering**: Requires minimum 5 samples, optimal with 20+

## Troubleshooting

### Backend won't start
- Ensure Python 3.8+ is installed
- Check if port 8000 is available
- Verify all dependencies are installed

### Frontend can't connect to backend
- Confirm backend is running on port 8000
- Check CORS settings if deploying to different domains
- Set `REACT_APP_API_URL` environment variable if needed

### Model loading errors
- Ensure stable internet connection for first run
- Check available disk space (need ~1GB free)
- Verify transformers library version compatibility

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Acknowledgments

- Hugging Face for transformer models
- scikit-learn for clustering algorithms
- Recharts for visualization components