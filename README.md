# 📊 Sentiment Analysis Dashboard

A comprehensive sentiment and emotion analysis tool powered by state-of-the-art transformer models. Analyze text data to uncover sentiment patterns, emotional nuances, and complex emotional states with beautiful visualizations.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- **Multi-Model Analysis**: Combines sentiment analysis and emotion detection for comprehensive insights
- **Divergence Scoring**: Identifies emotional complexity and mixed emotional states
- **Intelligent Clustering**: Groups texts with similar emotional patterns using K-means clustering
- **Interactive Visualizations**: Beautiful Plotly charts for exploring your data
- **Batch Processing**: Analyze thousands of texts at once with progress tracking
- **Export Functionality**: Download results in CSV or Excel format
- **User-Friendly Interface**: Clean, modern UI with helpful tooltips and explanations

## 🎯 Use Cases

- **Customer Feedback Analysis**: Understand customer sentiment and emotions at scale
- **Social Media Monitoring**: Analyze tweets, comments, and posts for brand insights
- **Survey Analysis**: Extract emotional patterns from open-ended survey responses
- **Content Analysis**: Evaluate emotional tone of articles, reviews, or user-generated content
- **Research**: Study emotional patterns in text data for academic or business research

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Sentiment-Analysis-Claude.git
cd Sentiment-Analysis-Claude
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the app**
```bash
streamlit run app.py
```

The app will open automatically in your default browser at `http://localhost:8501`

### First-Time Setup

On first run, the app will download the required AI models (~500MB total):
- DistilBERT for sentiment analysis
- DistilRoBERTa for emotion detection

This is a one-time download and may take a few minutes depending on your internet connection.

## 📖 Usage Guide

### 1. Prepare Your Data

The app accepts three file formats:

**CSV/Excel Format:**
```csv
text
"I absolutely love this product! It exceeded my expectations."
"The service was terrible and the staff was rude."
"It's okay, nothing special but does the job."
```

**Text Format:**
- One text entry per line
- Empty lines are automatically skipped

### 2. Upload and Configure

1. Click **"Browse files"** to upload your data (CSV, XLSX, or TXT)
2. Select the column containing text to analyze
3. Configure analysis settings in the sidebar:
   - Enable/disable clustering
   - Adjust number of clusters (2-10)

### 3. Analyze

Click **"🚀 Start Analysis"** to begin processing. The app will:
- Analyze sentiment (positive/negative/neutral)
- Detect 7 different emotions
- Calculate divergence scores
- Perform clustering (if enabled)
- Generate interactive visualizations

### 4. Explore Results

**Key Metrics Dashboard:**
- Positive sentiment percentage
- Average divergence score
- Dominant emotion
- Number of clusters found

**Interactive Visualizations:**
- **Sentiment Tab**: Distribution of positive, negative, and neutral sentiments
- **Emotions Tab**: Heatmap and breakdown of 7 emotions (joy, sadness, anger, fear, disgust, surprise, neutral)
- **Divergence Tab**: Scatter plot showing emotional complexity
- **Clusters Tab**: PCA visualization of emotional pattern groups

### 5. Export

Download your analyzed data with all scores and classifications in CSV or Excel format.

## 🧠 Understanding the Analysis

### Sentiment Analysis

Uses DistilBERT fine-tuned on the Stanford Sentiment Treebank (SST-2) to classify text as:
- **Positive**: Optimistic, favorable, or approving language
- **Negative**: Pessimistic, critical, or disapproving language
- **Neutral**: Balanced or factual language without clear sentiment

### Emotion Detection

Uses DistilRoBERTa trained on emotion datasets to detect 7 emotions:
- **Joy**: Happiness, pleasure, contentment
- **Sadness**: Sorrow, disappointment, grief
- **Anger**: Frustration, irritation, rage
- **Fear**: Anxiety, worry, apprehension
- **Disgust**: Revulsion, distaste, repugnance
- **Surprise**: Astonishment, amazement, unexpectedness
- **Neutral**: Absence of strong emotions

### Divergence Score

Measures the complexity of emotional state by calculating how much the detected emotions differ from what we'd expect based on the overall sentiment.

**Interpretation:**
- **0.0 - 0.3 (Low)**: Emotions align well with sentiment (e.g., positive sentiment with high joy)
- **0.3 - 0.6 (Medium)**: Some emotional complexity or nuance present
- **0.6 - 1.0 (High)**: Strong mixed emotions or unexpected emotional patterns

**Example:** A text might be overall positive (high sentiment score) but also contain surprise or fear (high divergence score), indicating complex emotional content like "I'm thrilled but nervous about this opportunity."

### Clustering

Uses K-means clustering on emotion features to identify groups of texts with similar emotional patterns. This helps discover:
- Common emotional themes in your data
- Distinct customer segments based on emotional responses
- Patterns that might not be obvious from sentiment alone

## 🛠️ Technical Details

### Models

| Component | Model | Parameters |
|-----------|-------|------------|
| Sentiment Analysis | `distilbert-base-uncased-finetuned-sst-2-english` | 67M |
| Emotion Detection | `j-hartmann/emotion-english-distilroberta-base` | 82M |

### Architecture

- **Frontend**: Streamlit with custom CSS styling
- **ML Framework**: Hugging Face Transformers with PyTorch
- **Visualization**: Plotly for interactive charts
- **Clustering**: Scikit-learn for K-means and PCA
- **Data Processing**: Pandas and NumPy

### Performance

- Processes ~10-20 texts per second (CPU)
- ~100-200 texts per second on GPU
- Supports batch analysis of thousands of texts
- Models are cached after first load for faster subsequent runs

## 📊 Example Output

After analysis, you'll get:

```
Sentiment Distribution:
- 45% Positive
- 30% Negative
- 25% Neutral

Top Emotions:
- Joy: 0.35
- Sadness: 0.22
- Neutral: 0.20

Average Divergence Score: 0.42 (Medium complexity)

3 Clusters Identified:
- Cluster 0: Positive with high joy (150 texts)
- Cluster 1: Negative with sadness and anger (120 texts)
- Cluster 2: Mixed emotions with high divergence (80 texts)
```

## 🎨 Customization

### Theme Configuration

Edit `.streamlit/config.toml` to customize colors:

```toml
[theme]
primaryColor = "#1e3a5f"        # Accent color
backgroundColor = "#fafbfc"      # Main background
secondaryBackgroundColor = "#ffffff"  # Card background
textColor = "#1a1a2e"           # Text color
```

### Adding Custom Emotions

To add custom emotion models:

1. Replace the model in `load_emotion_model()`:
```python
return pipeline("text-classification", model="your-model-name", return_all_scores=True)
```

2. Update emotion labels in visualization functions

## 🐛 Troubleshooting

### Common Issues

**Issue**: Models fail to download
- **Solution**: Check internet connection, try using a VPN, or manually download models from Hugging Face

**Issue**: Out of memory errors
- **Solution**: Process data in smaller batches, reduce `n_clusters`, or use a machine with more RAM

**Issue**: Slow performance
- **Solution**: Install PyTorch with GPU support for 10-20x speedup

**Issue**: "Column not found" error
- **Solution**: Ensure your CSV/Excel has a valid text column and select it from the dropdown

## 📚 Citation

If you use this tool in your research or project, please cite:

```bibtex
@software{sentiment_analysis_dashboard,
  title = {Sentiment Analysis Dashboard},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/Sentiment-Analysis-Claude}
}
```

### Model Citations

**DistilBERT Sentiment:**
```bibtex
@article{sanh2019distilbert,
  title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},
  journal={arXiv preprint arXiv:1910.01108},
  year={2019}
}
```

**Emotion Detection:**
```bibtex
@article{hartmann2022emotionenglish,
  title={Emotion English DistilRoBERTa-base},
  author={Hartmann, Jochen},
  journal={HuggingFace Model Hub},
  year={2022}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for transformer models
- [Streamlit](https://streamlit.io/) for the amazing web framework
- [Plotly](https://plotly.com/) for interactive visualizations

## 📧 Contact

For questions, issues, or suggestions:
- Open an issue on GitHub
- Email: your.email@example.com
- Twitter: [@yourusername](https://twitter.com/yourusername)

---

**Built with ❤️ using Claude Code and state-of-the-art AI models**
