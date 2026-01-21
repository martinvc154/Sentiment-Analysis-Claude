# RoBert - Sentiment Analysis Tool



## Features

- **Single Text Analysis**: Analyze individual texts with detailed sentiment breakdown
- **Batch Processing**: Upload CSV files or paste multiple texts for bulk analysis
- **Interactive Visualizations**: Beautiful charts showing sentiment distribution
- **Detailed Insights**: Get confidence scores, detected emotions, and reasoning for each analysis
- **Export Options**: Download results in CSV or JSON format
- **Analysis History**: Track and review all your previous analyses
- **Multiple Claude Models**: Choose from the latest Claude models

---

## Complete Beginner's Guide

If you've never used a terminal (command line) before, don't worry! This guide will walk you through everything step by step.

### What You'll Need

1. **A computer** (Windows, Mac, or Linux)
2. **An internet connection**
3. **An Anthropic API key** (we'll show you how to get one)

---

### Step 1: Get Your Anthropic API Key

Before you can use RoBert, you need an API key from Anthropic (the company that makes Claude AI).

1. Go to [console.anthropic.com](https://console.anthropic.com/)
2. Click "Sign Up" if you don't have an account, or "Log In" if you do
3. Once logged in, look for "API Keys" in the menu
4. Click "Create Key" and give it a name (like "RoBert")
5. **Important**: Copy this key and save it somewhere safe! You'll need it later.

---

### Step 2: Install Python

Python is the programming language that runs this app. Here's how to install it:

#### On Windows:
1. Go to [python.org/downloads](https://www.python.org/downloads/)
2. Click the big yellow "Download Python" button
3. Run the downloaded file
4. **IMPORTANT**: Check the box that says "Add Python to PATH" before clicking Install
5. Click "Install Now"

#### On Mac:
1. Open the "Terminal" app (search for it using Spotlight: press Cmd + Space, type "Terminal")
2. Copy and paste this command, then press Enter:
   ```
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
3. After that finishes, type this and press Enter:
   ```
   brew install python
   ```

#### On Linux (Ubuntu/Debian):
1. Open Terminal
2. Type this and press Enter:
   ```
   sudo apt update && sudo apt install python3 python3-pip python3-venv
   ```

---

### Step 3: Open Your Terminal

The terminal is where you'll type commands to run the app.

#### On Windows:
1. Press the Windows key on your keyboard
2. Type "cmd" or "Command Prompt"
3. Click on "Command Prompt" to open it

#### On Mac:
1. Press Cmd + Space to open Spotlight
2. Type "Terminal"
3. Press Enter

#### On Linux:
1. Press Ctrl + Alt + T
   (Or search for "Terminal" in your applications)

---

### Step 4: Download This Project

Now we'll download the RoBert project to your computer.

1. In your terminal, type this command and press Enter:
   ```
   git clone https://github.com/martinvc154/Sentiment-Analysis-Claude.git
   ```

   **If you see "git is not recognized"**, you need to install Git first:
   - Windows: Download from [git-scm.com](https://git-scm.com/download/win) and install
   - Mac: Type `xcode-select --install` and press Enter
   - Linux: Type `sudo apt install git` and press Enter

   Then try the clone command again.

2. Move into the project folder by typing:
   ```
   cd Sentiment-Analysis-Claude
   ```

---

### Step 5: Set Up the Project

Now we'll install all the necessary components.

1. **Create a virtual environment** (this keeps the project's files separate):

   On Windows:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

   On Mac/Linux:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

   You should see `(venv)` appear at the start of your command line. This means you're inside the virtual environment.

2. **Install the required packages**:
   ```
   pip install -r requirements.txt
   ```

   This might take a minute or two. Wait until you see your command prompt again.

---

### Step 6: Run the App

You're almost there! Now let's start the app.

1. In your terminal, type:
   ```
   streamlit run app.py
   ```

2. Your web browser should automatically open with the app. If it doesn't, look in your terminal for a line that says something like:
   ```
   Local URL: http://localhost:8501
   ```
   Copy that URL and paste it into your web browser.

3. **You should now see the RoBert app in your browser!**

---

### Step 7: Using the App

1. **Enter your API key**: Look at the left sidebar and paste your Anthropic API key into the "Anthropic API Key" field

2. **Analyze text**:
   - Type or paste any text into the big text box
   - Click "Analyze Sentiment"
   - See the results: positive, negative, neutral, or mixed!

3. **Batch analysis**:
   - Click the "Batch Analysis" tab
   - Upload a CSV file or paste multiple texts (one per line)
   - Analyze them all at once!

---

### Step 8: Stopping the App

When you're done using the app:

1. Go back to your terminal
2. Press `Ctrl + C` on your keyboard
3. The app will stop running

---

### Step 9: Running the App Again Later

Every time you want to use the app again:

1. Open your terminal
2. Navigate to the project folder:
   ```
   cd Sentiment-Analysis-Claude
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Start the app:
   ```
   streamlit run app.py
   ```

---

## Troubleshooting Common Issues

### "python is not recognized"
- Make sure you installed Python with "Add to PATH" checked
- Try using `python3` instead of `python`
- Restart your terminal and try again

### "pip is not recognized"
- Try using `pip3` instead of `pip`
- On Windows, try `python -m pip install -r requirements.txt`

### "streamlit is not recognized"
- Make sure you activated your virtual environment (you should see `(venv)` at the start of your command line)
- Try `python -m streamlit run app.py`

### The app opens but shows an error
- Make sure you entered your Anthropic API key correctly
- Check that your API key hasn't expired
- Make sure you have internet access

### "Port 8501 is already in use"
- Another app might be using that port
- Try: `streamlit run app.py --server.port 8502`

---

## Project Structure

```
Sentiment-Analysis-Claude/
├── app.py                 # Main application file
├── utils/
│   ├── __init__.py
│   └── analysis.py        # Analysis helper functions
├── .streamlit/
│   └── config.toml        # Streamlit configuration
├── data/                  # Data storage (gitignored)
├── exports/               # Export files (gitignored)
├── requirements.txt       # Python dependencies
├── .env.example           # Example environment file
├── .gitignore
└── README.md
```

---

## Configuration Options

### API Key Setup (Optional Alternative)

Instead of entering your API key in the app every time, you can set it as an environment variable:

1. Copy the `.env.example` file to `.env`:
   ```
   cp .env.example .env
   ```
2. Open `.env` in a text editor and replace `your_api_key_here` with your actual API key
3. The app will automatically use this key

### Model Selection

The app supports multiple Claude models:
- **Claude Sonnet 4** (recommended) - Best balance of speed and quality
- **Claude 3.5 Haiku** - Fastest, most cost-effective
- **Claude 3.5 Sonnet** - High quality analysis

---

## API Usage & Costs

This app uses the Anthropic API, which has usage-based pricing. Each analysis costs a small amount based on:
- The length of text analyzed
- The model you select (Haiku is cheapest, Sonnet 4 is mid-range)

Check [Anthropic's pricing page](https://www.anthropic.com/pricing) for current rates.

---

## Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch
3. Submit a Pull Request

---

## License

This project is open source and available under the MIT License.

---

## Support

If you run into issues:
- Check the troubleshooting section above
- Open an issue on GitHub
- Review the [Anthropic Documentation](https://docs.anthropic.com/)
- Review the [Streamlit Documentation](https://docs.streamlit.io/)

---

Made with Claude AI
