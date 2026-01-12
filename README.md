# Sentiment Analysis Dashboard

A powerful interactive dashboard for visualizing sentiment analysis data with hierarchical clustering and publication-quality exports.

## Features

### 📁 Data Upload
- Upload CSV or Excel files containing sentiment data
- Real-time data preview and statistics
- Support for various data formats

### 🔗 Column Mapping
- Flexible column mapping for organizations, questions, and sentiment scores
- Automatic detection of unique values and data ranges
- Visual mapping summary

### 📈 Visualizations

#### Sentiment Heatmap
- **Interactive heatmap** showing average sentiment scores by organization and question
- **Hierarchical clustering** on both axes using Ward's method
- **Color-coded visualization** using RdYlGn scale (red=-1, yellow=0, green=1)
- **Dendrograms** showing clustering relationships
- **Publication-quality exports** in PDF and PNG formats

#### Coming Soon
- Divergence Plot
- Emotion Profiles
- Cluster Analysis

### 📤 Export Features
- **Multiple DPI options**: 150, 300, 600
- **Multiple size options**: 7x5, 10x7, 14x10 inches
- **Times New Roman font** for publication styling
- **Professional formatting** with thicker lines and larger labels
- **PDF and PNG** export formats

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

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

3. Follow the workflow:
   - **Step 1**: Upload your data file in the "Data Upload" tab
   - **Step 2**: Map your columns in the "Column Mapping" tab
   - **Step 3**: Visualize and export in the "Visualizations" tab

## Data Format

Your data should include at least three columns:
- **Organization/Category column**: Names or IDs of organizations
- **Question column**: Survey questions or topics
- **Sentiment column**: Numeric sentiment scores (typically -1 to 1)

### Sample Data
A sample dataset (`sample_data.csv`) is included for testing:
```csv
Organization,Question,Sentiment
Acme Corp,How satisfied are you with our product?,-0.2
Acme Corp,Would you recommend us to others?,0.5
...
```

## Requirements

- Python 3.8+
- streamlit >= 1.28.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- plotly >= 5.17.0
- scipy >= 1.11.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- openpyxl >= 3.1.0

## Project Structure

```
Sentiment-Analysis-Claude/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── sample_data.csv       # Sample dataset for testing
├── exports/              # Export directory (created automatically)
│   ├── sentiment_heatmap.pdf
│   └── sentiment_heatmap.png
└── README.md             # This file
```

## How It Works

### Hierarchical Clustering
The heatmap uses Ward's method for hierarchical clustering:
1. Calculates Euclidean distances between organizations and questions
2. Performs agglomerative clustering using Ward's linkage
3. Reorders rows and columns based on cluster dendrograms
4. Groups similar patterns together for easier interpretation

### Color Scale
- **Green (+1)**: Positive sentiment
- **Yellow (0)**: Neutral sentiment
- **Red (-1)**: Negative sentiment

### Export Process
1. Select your preferred DPI and size
2. Click "Export as PDF" or "Export as PNG"
3. The chart is recreated using matplotlib with publication styling
4. File is saved to the `exports/` folder
5. Download button appears for easy file retrieval

## Tips

- **For best results**: Ensure sentiment values are normalized between -1 and 1
- **Clustering**: Works best with at least 3 organizations and 3 questions
- **Export quality**: Use 300 DPI for most publications, 600 DPI for high-quality prints
- **File size**: Higher DPI and larger sizes create bigger files

## License

MIT License - feel free to use and modify for your projects

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues or questions, please open an issue on GitHub.