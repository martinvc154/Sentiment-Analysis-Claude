import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import io
import time
from typing import List, Dict, Optional

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
def add_custom_css():
    st.markdown("""
    <style>
        /* Hide Streamlit default menu and footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Custom fonts */
        html, body, [class*="css"] {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        /* Card-style containers */
        .stPlotlyChart, .stDataFrame {
            background-color: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            margin: 1rem 0;
        }

        /* Metric cards */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 600;
            color: #1e3a5f;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.9rem;
            color: #666;
            font-weight: 500;
        }

        /* Better spacing */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* Accent color for buttons and links */
        .stButton>button {
            background-color: #1e3a5f;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 2rem;
            font-weight: 500;
            border: none;
            transition: all 0.3s;
        }

        .stButton>button:hover {
            background-color: #2c5282;
            box-shadow: 0 4px 12px rgba(30, 58, 95, 0.3);
        }

        /* File uploader styling */
        [data-testid="stFileUploader"] {
            background-color: #f8f9fa;
            border-radius: 12px;
            padding: 1.5rem;
            border: 2px dashed #1e3a5f;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
        }

        /* Info boxes */
        .stAlert {
            border-radius: 8px;
            border-left: 4px solid #1e3a5f;
        }

        /* Tooltips */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
            color: #1e3a5f;
            font-weight: 600;
            margin-left: 4px;
        }

        /* Custom headers */
        h1, h2, h3 {
            color: #1e3a5f;
            font-weight: 600;
        }

        /* Progress bar */
        .stProgress > div > div > div > div {
            background-color: #1e3a5f;
        }
    </style>
    """, unsafe_allow_html=True)

# Initialize models with caching
@st.cache_resource
def load_sentiment_model():
    """Load sentiment analysis model"""
    try:
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception as e:
        st.error(f"Error loading sentiment model: {str(e)}")
        return None

@st.cache_resource
def load_emotion_model():
    """Load emotion detection model"""
    try:
        return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    except Exception as e:
        st.error(f"Error loading emotion model: {str(e)}")
        return None

# Analysis functions
def analyze_sentiment(texts: List[str], sentiment_model) -> List[Dict]:
    """Analyze sentiment of texts with error handling"""
    results = []
    for text in texts:
        try:
            if not text or not isinstance(text, str):
                results.append({"label": "NEUTRAL", "score": 0.5})
                continue

            # Truncate long texts
            text = text[:512]
            result = sentiment_model(text)[0]
            results.append(result)
        except Exception as e:
            st.warning(f"Error analyzing text: {str(e)[:100]}")
            results.append({"label": "NEUTRAL", "score": 0.5})

    return results

def analyze_emotions(texts: List[str], emotion_model) -> List[Dict]:
    """Analyze emotions in texts with error handling"""
    results = []
    for text in texts:
        try:
            if not text or not isinstance(text, str):
                results.append({})
                continue

            # Truncate long texts
            text = text[:512]
            emotions = emotion_model(text)[0]
            emotion_dict = {e['label']: e['score'] for e in emotions}
            results.append(emotion_dict)
        except Exception as e:
            st.warning(f"Error analyzing emotions: {str(e)[:100]}")
            results.append({})

    return results

def calculate_divergence_score(sentiment_score: float, emotion_scores: Dict) -> float:
    """
    Calculate divergence score between sentiment and emotions.
    Higher score indicates more complex/mixed emotional state.
    """
    try:
        if not emotion_scores:
            return 0.0

        # Map sentiment to expected emotion distribution
        if sentiment_score > 0.6:  # Positive
            expected_emotions = {'joy': 0.7, 'surprise': 0.2, 'neutral': 0.1}
        elif sentiment_score < 0.4:  # Negative
            expected_emotions = {'sadness': 0.4, 'anger': 0.3, 'fear': 0.2, 'disgust': 0.1}
        else:  # Neutral
            expected_emotions = {'neutral': 0.8, 'surprise': 0.2}

        # Calculate KL divergence
        divergence = 0
        for emotion in emotion_scores:
            actual = emotion_scores.get(emotion, 0.0001)
            expected = expected_emotions.get(emotion, 0.0001)
            divergence += actual * np.log(actual / expected)

        return max(0, min(1, divergence))  # Normalize to 0-1
    except Exception:
        return 0.0

def perform_clustering(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """Perform K-means clustering on emotion features"""
    try:
        # Get emotion columns
        emotion_cols = [col for col in df.columns if col.startswith('emotion_')]

        if len(emotion_cols) == 0:
            return df

        # Prepare features
        X = df[emotion_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(X_scaled)

        return df
    except Exception as e:
        st.warning(f"Error performing clustering: {str(e)}")
        df['cluster'] = 0
        return df

# Visualization functions
def create_sentiment_distribution(df: pd.DataFrame):
    """Create sentiment distribution chart"""
    fig = px.histogram(
        df,
        x='sentiment_label',
        color='sentiment_label',
        title='Sentiment Distribution',
        labels={'sentiment_label': 'Sentiment', 'count': 'Count'},
        color_discrete_map={'POSITIVE': '#22c55e', 'NEGATIVE': '#ef4444', 'NEUTRAL': '#94a3b8'}
    )
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'system-ui', 'size': 12}
    )
    return fig

def create_emotion_heatmap(df: pd.DataFrame):
    """Create emotion heatmap"""
    emotion_cols = [col for col in df.columns if col.startswith('emotion_')]
    if len(emotion_cols) == 0:
        return None

    # Get top 20 texts for visualization
    emotion_data = df[emotion_cols].head(20)
    emotion_data.columns = [col.replace('emotion_', '').title() for col in emotion_cols]

    fig = px.imshow(
        emotion_data.T,
        labels=dict(x="Text Index", y="Emotion", color="Score"),
        title="Emotion Intensity Heatmap (First 20 Texts)",
        color_continuous_scale='Blues',
        aspect='auto'
    )
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'system-ui', 'size': 12}
    )
    return fig

def create_divergence_scatter(df: pd.DataFrame):
    """Create divergence score scatter plot"""
    fig = px.scatter(
        df,
        x='sentiment_score',
        y='divergence_score',
        color='sentiment_label',
        title='Sentiment vs Emotional Divergence',
        labels={'sentiment_score': 'Sentiment Score', 'divergence_score': 'Divergence Score'},
        color_discrete_map={'POSITIVE': '#22c55e', 'NEGATIVE': '#ef4444', 'NEUTRAL': '#94a3b8'},
        hover_data=['text'] if 'text' in df.columns else None
    )
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'system-ui', 'size': 12}
    )
    return fig

def create_cluster_visualization(df: pd.DataFrame):
    """Create cluster visualization using PCA"""
    try:
        from sklearn.decomposition import PCA

        emotion_cols = [col for col in df.columns if col.startswith('emotion_')]
        if len(emotion_cols) == 0 or 'cluster' not in df.columns:
            return None

        # PCA for visualization
        X = df[emotion_cols].fillna(0)
        pca = PCA(n_components=2)
        components = pca.fit_transform(X)

        plot_df = pd.DataFrame({
            'PC1': components[:, 0],
            'PC2': components[:, 1],
            'Cluster': df['cluster'].astype(str),
            'Sentiment': df['sentiment_label']
        })

        fig = px.scatter(
            plot_df,
            x='PC1',
            y='PC2',
            color='Cluster',
            symbol='Sentiment',
            title='Emotional Clusters (PCA Projection)',
            labels={'PC1': 'First Principal Component', 'PC2': 'Second Principal Component'}
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font={'family': 'system-ui', 'size': 12}
        )
        return fig
    except Exception as e:
        st.warning(f"Error creating cluster visualization: {str(e)}")
        return None

# Main app
def main():
    add_custom_css()

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configuration")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload your data (CSV, Excel, TXT)",
            type=['csv', 'xlsx', 'txt'],
            help="Upload a file containing text data for analysis"
        )

        # Settings
        st.subheader("Analysis Settings")

        enable_clustering = st.checkbox("Enable Clustering", value=True, help="Group similar emotional patterns")
        if enable_clustering:
            n_clusters = st.slider("Number of Clusters", 2, 10, 3)
        else:
            n_clusters = 3

        # About section
        with st.expander("📖 About This App"):
            st.markdown("""
            ### Methodology

            This app performs multi-dimensional sentiment analysis using:

            - **Sentiment Analysis**: DistilBERT fine-tuned on SST-2
            - **Emotion Detection**: DistilRoBERTa trained on emotion datasets
            - **Divergence Score**: KL divergence between expected and actual emotion distributions
            - **Clustering**: K-means clustering on emotion feature space

            ### Model Citations

            - Sentiment: `distilbert-base-uncased-finetuned-sst-2-english`
            - Emotions: `j-hartmann/emotion-english-distilroberta-base`

            ### Interpreting Results

            - **Divergence Score**: Measures emotional complexity. Higher scores indicate mixed or unexpected emotions.
            - **Emotions**: anger, disgust, fear, joy, neutral, sadness, surprise
            - **Clusters**: Groups of texts with similar emotional patterns

            ### Documentation

            For more information, visit: [GitHub Repository](https://github.com/anthropics/sentiment-analysis)
            """)

        st.markdown("---")
        st.caption("Built with Streamlit & Hugging Face Transformers")

    # Main content
    st.title("📊 Sentiment Analysis Dashboard")
    st.markdown("Analyze text sentiment and emotions with advanced ML models")

    if uploaded_file is None:
        st.info("👈 Upload a file to get started")

        # Show example
        with st.expander("ℹ️ How to use this app"):
            st.markdown("""
            1. **Upload your data** in CSV, Excel, or TXT format
            2. **Select the text column** to analyze
            3. **Configure settings** in the sidebar
            4. **View results** with interactive visualizations
            5. **Export** analyzed data with sentiment scores

            ### Expected File Format

            - **CSV/Excel**: Should contain a column with text data
            - **TXT**: One text entry per line

            ### Example CSV:
            ```
            text
            "I love this product!"
            "This is terrible."
            "It's okay, nothing special."
            ```
            """)
        return

    # Load data with error handling
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            texts = uploaded_file.read().decode('utf-8').split('\n')
            df = pd.DataFrame({'text': [t.strip() for t in texts if t.strip()]})
        else:
            st.error("Unsupported file format. Please upload CSV, Excel, or TXT file.")
            return
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.info("Please make sure your file is properly formatted and not corrupted.")
        return

    if df.empty:
        st.error("The uploaded file is empty. Please upload a file with data.")
        return

    # Select text column
    st.subheader("Select Text Column")
    text_column = st.selectbox(
        "Choose the column containing text to analyze",
        options=df.columns.tolist(),
        help="Select the column that contains the text you want to analyze"
    )

    if st.button("🚀 Start Analysis", type="primary"):
        # Validate data
        if text_column not in df.columns:
            st.error(f"Column '{text_column}' not found in the data.")
            return

        texts = df[text_column].dropna().astype(str).tolist()

        if len(texts) == 0:
            st.error("No valid text data found in the selected column.")
            return

        # Load models
        with st.spinner("Loading AI models..."):
            sentiment_model = load_sentiment_model()
            emotion_model = load_emotion_model()

        if sentiment_model is None or emotion_model is None:
            st.error("Failed to load models. Please try again.")
            return

        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Sentiment analysis
        status_text.text("Analyzing sentiment... 🔍")
        progress_bar.progress(20)

        with st.spinner("Processing sentiment..."):
            sentiment_results = analyze_sentiment(texts, sentiment_model)

        df['sentiment_label'] = [r['label'] for r in sentiment_results]
        df['sentiment_score'] = [r['score'] for r in sentiment_results]

        # Emotion analysis
        status_text.text("Detecting emotions... 😊😢😠")
        progress_bar.progress(50)

        with st.spinner("Processing emotions..."):
            emotion_results = analyze_emotions(texts, emotion_model)

        # Add emotion columns
        if emotion_results and emotion_results[0]:
            for emotion in emotion_results[0].keys():
                df[f'emotion_{emotion}'] = [r.get(emotion, 0) for r in emotion_results]

        # Calculate divergence
        status_text.text("Calculating divergence scores... 📊")
        progress_bar.progress(70)

        df['divergence_score'] = [
            calculate_divergence_score(
                sentiment_results[i]['score'],
                emotion_results[i]
            )
            for i in range(len(texts))
        ]

        # Clustering
        if enable_clustering:
            status_text.text("Clustering emotional patterns... 🎯")
            progress_bar.progress(85)
            df = perform_clustering(df, n_clusters)

        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()

        # Display results
        st.success(f"Successfully analyzed {len(texts)} texts!")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            pos_pct = (df['sentiment_label'] == 'POSITIVE').sum() / len(df) * 100
            st.metric(
                "Positive Sentiment",
                f"{pos_pct:.1f}%",
                help="Percentage of texts with positive sentiment"
            )

        with col2:
            avg_divergence = df['divergence_score'].mean()
            st.metric(
                "Avg Divergence",
                f"{avg_divergence:.2f}",
                help="Average emotional complexity across all texts"
            )

        with col3:
            dominant_emotion = None
            emotion_cols = [col for col in df.columns if col.startswith('emotion_')]
            if emotion_cols:
                emotion_means = df[emotion_cols].mean()
                dominant_emotion = emotion_means.idxmax().replace('emotion_', '').title()
            st.metric(
                "Dominant Emotion",
                dominant_emotion or "N/A",
                help="Most prevalent emotion across all texts"
            )

        with col4:
            if 'cluster' in df.columns:
                st.metric(
                    "Clusters Found",
                    n_clusters,
                    help="Number of distinct emotional pattern groups"
                )

        # Visualizations
        st.subheader("📈 Visualizations")

        tab1, tab2, tab3, tab4 = st.tabs(["Sentiment", "Emotions", "Divergence", "Clusters"])

        with tab1:
            fig = create_sentiment_distribution(df)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            with st.spinner("Rendering emotion heatmap..."):
                fig = create_emotion_heatmap(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No emotion data available")

            # Emotion breakdown
            emotion_cols = [col for col in df.columns if col.startswith('emotion_')]
            if emotion_cols:
                st.subheader("Emotion Breakdown")
                emotion_avg = df[emotion_cols].mean().sort_values(ascending=False)
                emotion_avg.index = [e.replace('emotion_', '').title() for e in emotion_avg.index]

                fig = px.bar(
                    x=emotion_avg.values,
                    y=emotion_avg.index,
                    orientation='h',
                    title='Average Emotion Scores',
                    labels={'x': 'Average Score', 'y': 'Emotion'}
                )
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font={'family': 'system-ui'}
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.markdown("""
            **What is Divergence Score?** 🤔

            The divergence score measures how much the detected emotions differ from what we'd expect
            based on the overall sentiment. A higher score indicates more complex or mixed emotions.

            - **Low (0-0.3)**: Emotions align with sentiment
            - **Medium (0.3-0.6)**: Some emotional complexity
            - **High (0.6-1.0)**: Strong mixed or unexpected emotions
            """)

            fig = create_divergence_scatter(df)
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            if enable_clustering and 'cluster' in df.columns:
                with st.spinner("Rendering cluster visualization..."):
                    fig = create_cluster_visualization(df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Unable to create cluster visualization")

                # Cluster summary
                st.subheader("Cluster Summary")
                for cluster_id in sorted(df['cluster'].unique()):
                    cluster_data = df[df['cluster'] == cluster_id]
                    dominant_sentiment = cluster_data['sentiment_label'].mode()[0] if len(cluster_data) > 0 else "N/A"
                    avg_div = cluster_data['divergence_score'].mean()

                    with st.expander(f"Cluster {cluster_id} ({len(cluster_data)} texts)"):
                        st.write(f"**Dominant Sentiment:** {dominant_sentiment}")
                        st.write(f"**Avg Divergence:** {avg_div:.2f}")

                        # Sample texts
                        st.write("**Sample texts:**")
                        samples = cluster_data[text_column].head(3).tolist()
                        for i, sample in enumerate(samples, 1):
                            st.write(f"{i}. {sample[:200]}...")
            else:
                st.info("Enable clustering in the sidebar to see cluster analysis")

        # Data table
        st.subheader("📋 Detailed Results")

        # Prepare display columns
        display_cols = [text_column, 'sentiment_label', 'sentiment_score', 'divergence_score']
        if 'cluster' in df.columns:
            display_cols.append('cluster')

        emotion_cols = [col for col in df.columns if col.startswith('emotion_')]
        display_cols.extend(emotion_cols[:5])  # Show first 5 emotions

        st.dataframe(
            df[display_cols].head(100),
            use_container_width=True,
            height=400
        )

        # Export functionality
        st.subheader("💾 Export Results")

        col1, col2 = st.columns(2)

        with col1:
            # Export CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="sentiment_analysis_results.csv",
                mime="text/csv"
            )

        with col2:
            # Export Excel
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Results')

            st.download_button(
                label="📥 Download Excel",
                data=buffer.getvalue(),
                file_name="sentiment_analysis_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()
