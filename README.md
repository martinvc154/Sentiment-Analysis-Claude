# 🎭 Sentiment Analysis with Claude

A powerful sentiment analysis application built with Streamlit and powered by Anthropic's Claude AI. Analyze text sentiment with detailed insights including confidence scores, emotions, and reasoning.

## Features

- **Single Text Analysis**: Analyze individual texts with detailed sentiment breakdown
- **Batch Processing**: Upload CSV files or paste multiple texts for bulk analysis
- **Interactive Visualizations**: Beautiful charts showing sentiment distribution
- **Detailed Insights**: Get confidence scores, detected emotions, and reasoning for each analysis
- **Export Options**: Download results in CSV or JSON format
- **Analysis History**: Track and review all your previous analyses
- **Multiple Claude Models**: Choose from Sonnet, Opus, or Haiku models

## Live Demo

🔗 [Try it on Streamlit Cloud](https://your-app-url.streamlit.app) (Update after deployment)

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Anthropic API key ([Get one here](https://console.anthropic.com/))

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Sentiment-Analysis-Claude.git
cd Sentiment-Analysis-Claude
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Enter your Anthropic API key in the sidebar when the app opens

## Usage

### Single Text Analysis

1. Navigate to the "Single Analysis" tab
2. Enter your Anthropic API key in the sidebar
3. Type or paste your text in the text area
4. Click "Analyze Sentiment"
5. View the results including sentiment, confidence, emotions, and reasoning

### Batch Analysis

1. Navigate to the "Batch Analysis" tab
2. Choose your input method:
   - Upload a CSV file with a text column
   - Paste multiple texts (one per line)
3. Click "Start Batch Analysis"
4. View summary statistics and sentiment distribution chart
5. Download results in CSV or JSON format

### Analysis History

- View all previous analyses in the "History" tab
- Export your complete analysis history as CSV

## Project Structure

```
Sentiment-Analysis-Claude/
├── app.py                 # Main Streamlit application
├── utils/
│   ├── __init__.py
│   └── analysis.py        # Analysis helper functions
├── .streamlit/
│   └── config.toml        # Streamlit configuration
├── data/                  # Data storage (gitignored)
├── exports/               # Export files (gitignored)
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

## Configuration

### API Key

You can set your API key in two ways:

1. **In the app**: Enter it in the sidebar (recommended for testing)
2. **Environment variable**: Set `ANTHROPIC_API_KEY` in your environment

### Streamlit Configuration

Customize the app appearance by editing `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

## Technologies Used

- **Streamlit**: Web application framework
- **Anthropic Claude**: AI model for sentiment analysis
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **Python**: Core programming language

## Requirements

See `requirements.txt` for full list of dependencies:

- streamlit==1.31.1
- anthropic==0.18.1
- pandas==2.2.0
- numpy==1.26.4
- plotly==5.18.0
- python-dotenv==1.0.1

## Deployment to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository and branch
6. Set main file path to `app.py`
7. Add your `ANTHROPIC_API_KEY` in the "Secrets" section
8. Click "Deploy"

## API Usage

The app uses the Anthropic Messages API. Costs depend on the model and usage:

- **Claude 3 Haiku**: Most cost-effective
- **Claude 3.5 Sonnet**: Balanced performance and cost
- **Claude 3 Opus**: Highest quality

See [Anthropic Pricing](https://www.anthropic.com/pricing) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Anthropic Claude](https://www.anthropic.com/)
- Icons from [Streamlit Emoji Shortcodes](https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/)

## Support

If you encounter any issues or have questions:

- Open an issue on GitHub
- Check the [Anthropic Documentation](https://docs.anthropic.com/)
- Review [Streamlit Documentation](https://docs.streamlit.io/)

## Roadmap

- [ ] Add support for file uploads (PDF, TXT, DOCX)
- [ ] Implement sentiment trend analysis over time
- [ ] Add multilingual support
- [ ] Create API endpoint for programmatic access
- [ ] Add more visualization options
- [ ] Implement caching for improved performance

---

Made with ❤️ using Claude AI
