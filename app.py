"""
Sentiment Analysis Dashboard - Streamlit Application
"""
import streamlit as st
import pandas as pd
import numpy as np
from analyzer import SentimentAnalyzer
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO


# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_analyzer():
    """
    Load and cache the sentiment analyzer.
    Models are loaded once and reused across sessions.
    """
    return SentimentAnalyzer()


def apply_sentiment_color(sentiment):
    """Apply color coding to sentiment values."""
    if sentiment == 'positive':
        return 'background-color: #d4edda; color: #155724'
    elif sentiment == 'negative':
        return 'background-color: #f8d7da; color: #721c24'
    else:
        return 'background-color: #e2e3e5; color: #383d41'


def highlight_sentiment(row):
    """Highlight sentiment column based on value."""
    colors = [''] * len(row)
    if 'sentiment' in row.index:
        idx = row.index.get_loc('sentiment')
        if row['sentiment'] == 'positive':
            colors[idx] = 'background-color: #d4edda; color: #155724'
        elif row['sentiment'] == 'negative':
            colors[idx] = 'background-color: #f8d7da; color: #721c24'
        else:
            colors[idx] = 'background-color: #e2e3e5; color: #383d41'
    return colors


def run_analysis(df, text_column):
    """
    Run sentiment analysis on the dataframe.

    Args:
        df: Input DataFrame
        text_column: Name of the column containing text to analyze
    """
    analyzer = load_analyzer()

    # Extract texts
    texts = df[text_column].astype(str).tolist()

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    results = []

    def progress_callback(progress):
        progress_bar.progress(progress)
        status_text.text(f"Analyzing... {int(progress * 100)}% complete")

    # Run analysis
    analysis_results = analyzer.analyze_batch(texts, progress_callback)

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    # Create results dataframe
    results_df = df.copy()

    # Add analysis results
    for key in ['sentiment', 'sentiment_score', 'confidence', 'joy', 'sadness',
                'anger', 'fear', 'surprise', 'disgust', 'dominant_emotion', 'divergence']:
        results_df[key] = [result[key] for result in analysis_results]

    return results_df


def show_results_tab(df):
    """Display the results tab with filterable dataframe."""
    st.subheader("📋 Analysis Results")

    # Filter options
    col1, col2, col3 = st.columns(3)

    with col1:
        sentiment_filter = st.multiselect(
            "Filter by Sentiment",
            options=['positive', 'neutral', 'negative'],
            default=['positive', 'neutral', 'negative']
        )

    with col2:
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05
        )

    with col3:
        search_term = st.text_input("Search in text", "")

    # Apply filters
    filtered_df = df[df['sentiment'].isin(sentiment_filter)]
    filtered_df = filtered_df[filtered_df['confidence'] >= min_confidence]

    if search_term:
        # Search in all text columns
        text_cols = filtered_df.select_dtypes(include=['object']).columns
        mask = filtered_df[text_cols].apply(
            lambda x: x.str.contains(search_term, case=False, na=False)
        ).any(axis=1)
        filtered_df = filtered_df[mask]

    st.write(f"Showing {len(filtered_df)} of {len(df)} responses")

    # Display dataframe with styling
    st.dataframe(
        filtered_df.style.apply(highlight_sentiment, axis=1),
        use_container_width=True,
        height=500
    )

    # Download filtered results
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Results (CSV)",
        data=csv,
        file_name="filtered_results.csv",
        mime="text/csv"
    )


def show_dashboard_tab(df):
    """Display the dashboard tab with key metrics."""
    st.subheader("📊 Dashboard Overview")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Responses",
            len(df)
        )

    with col2:
        avg_sentiment = df['sentiment_score'].mean()
        st.metric(
            "Average Sentiment",
            f"{avg_sentiment:.3f}",
            delta=f"{avg_sentiment:.1%}" if avg_sentiment > 0 else None
        )

    with col3:
        positive_pct = (df['sentiment'] == 'positive').sum() / len(df) * 100
        st.metric(
            "Positive Responses",
            f"{positive_pct:.1f}%"
        )

    with col4:
        high_divergence = (df['divergence'] > 0.7).sum()
        st.metric(
            "High Divergence",
            high_divergence,
            delta=f"{high_divergence/len(df)*100:.1f}%"
        )

    st.divider()

    # Sentiment distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        sentiment_counts = df['sentiment'].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map={
                'positive': '#28a745',
                'neutral': '#6c757d',
                'negative': '#dc3545'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Dominant Emotions")
        emotion_counts = df['dominant_emotion'].value_counts()
        fig = px.bar(
            x=emotion_counts.index,
            y=emotion_counts.values,
            labels={'x': 'Emotion', 'y': 'Count'},
            color=emotion_counts.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Confidence distribution
    st.subheader("Confidence Distribution")
    fig = px.histogram(
        df,
        x='confidence',
        nbins=20,
        labels={'confidence': 'Confidence Score'},
        color_discrete_sequence=['#007bff']
    )
    st.plotly_chart(fig, use_container_width=True)


def show_visualizations_tab(df):
    """Display advanced visualizations."""
    st.subheader("📈 Advanced Visualizations")

    # Emotion heatmap
    st.subheader("Emotion Intensity Heatmap")
    emotion_cols = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
    emotion_data = df[emotion_cols].head(50)  # Show first 50 for readability

    fig = go.Figure(data=go.Heatmap(
        z=emotion_data.values.T,
        x=list(range(len(emotion_data))),
        y=emotion_cols,
        colorscale='RdYlGn',
        colorbar=dict(title="Intensity")
    ))
    fig.update_layout(
        xaxis_title="Response Index",
        yaxis_title="Emotion",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Scatter plot: Sentiment Score vs Divergence
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Score vs Divergence")
        fig = px.scatter(
            df,
            x='sentiment_score',
            y='divergence',
            color='sentiment',
            color_discrete_map={
                'positive': '#28a745',
                'neutral': '#6c757d',
                'negative': '#dc3545'
            },
            hover_data=['confidence', 'dominant_emotion']
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Confidence vs Divergence")
        fig = px.scatter(
            df,
            x='confidence',
            y='divergence',
            color='dominant_emotion',
            hover_data=['sentiment', 'sentiment_score']
        )
        st.plotly_chart(fig, use_container_width=True)

    # Average emotion scores
    st.subheader("Average Emotion Scores")
    avg_emotions = df[emotion_cols].mean().sort_values(ascending=False)
    fig = px.bar(
        x=avg_emotions.index,
        y=avg_emotions.values,
        labels={'x': 'Emotion', 'y': 'Average Score'},
        color=avg_emotions.values,
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig, use_container_width=True)


def show_export_tab(df):
    """Display export options."""
    st.subheader("📄 Export Data")

    st.write("Export your analysis results in various formats.")

    col1, col2, col3 = st.columns(3)

    with col1:
        # CSV Export
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="sentiment_analysis_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        # Excel Export
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Results')
        excel_data = buffer.getvalue()

        st.download_button(
            label="📥 Download Excel",
            data=excel_data,
            file_name="sentiment_analysis_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    with col3:
        # JSON Export
        json_data = df.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name="sentiment_analysis_results.json",
            mime="application/json",
            use_container_width=True
        )

    st.divider()

    # Summary statistics export
    st.subheader("Summary Statistics")

    summary_stats = {
        'Total Responses': len(df),
        'Positive': (df['sentiment'] == 'positive').sum(),
        'Neutral': (df['sentiment'] == 'neutral').sum(),
        'Negative': (df['sentiment'] == 'negative').sum(),
        'Average Sentiment Score': df['sentiment_score'].mean(),
        'Average Confidence': df['confidence'].mean(),
        'High Divergence Responses': (df['divergence'] > 0.7).sum()
    }

    summary_df = pd.DataFrame(summary_stats.items(), columns=['Metric', 'Value'])
    st.dataframe(summary_df, use_container_width=True)

    summary_csv = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Summary",
        data=summary_csv,
        file_name="analysis_summary.csv",
        mime="text/csv"
    )


def main():
    """Main application."""
    st.title("📊 Sentiment Analysis Dashboard")
    st.write("Upload your data and analyze sentiment and emotions in text responses.")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a file containing text data for analysis"
        )

        if uploaded_file is not None:
            # Load data
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                st.success(f"✅ Loaded {len(df)} rows")

                # Select text column
                text_column = st.selectbox(
                    "Select text column to analyze",
                    options=df.columns.tolist()
                )

                st.divider()

                # Run Analysis Button
                if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                    with st.spinner("Loading models and analyzing..."):
                        results_df = run_analysis(df, text_column)
                        st.session_state.results = results_df

                        # Show success message with summary
                        st.success("✅ Analysis Complete!")

                        # Summary metrics
                        avg_sentiment = results_df['sentiment_score'].mean()
                        high_divergence = (results_df['divergence'] > 0.7).sum()

                        st.info(
                            f"""
                            **Summary:**
                            - Analyzed {len(results_df)} responses
                            - Average sentiment: {avg_sentiment:.3f}
                            - High divergence responses: {high_divergence}
                            """
                        )

            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

    # Main content area
    if 'results' in st.session_state:
        results_df = st.session_state.results

        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Results",
            "📊 Dashboard",
            "📈 Visualizations",
            "📄 Export"
        ])

        with tab1:
            show_results_tab(results_df)

        with tab2:
            show_dashboard_tab(results_df)

        with tab3:
            show_visualizations_tab(results_df)

        with tab4:
            show_export_tab(results_df)

    else:
        # Welcome message
        st.info(
            """
            👋 **Welcome to the Sentiment Analysis Dashboard!**

            To get started:
            1. Upload a CSV or Excel file using the sidebar
            2. Select the column containing text to analyze
            3. Click "Run Analysis" to begin

            The analysis will provide:
            - Sentiment classification (positive/neutral/negative)
            - Sentiment scores and confidence levels
            - Emotion analysis (joy, sadness, anger, fear, surprise, disgust)
            - Divergence metrics for mixed emotions
            """
        )

        # Example data format
        with st.expander("📝 Example Data Format"):
            example_df = pd.DataFrame({
                'Response ID': [1, 2, 3],
                'Text': [
                    'I love this product! It works great.',
                    'Not sure how I feel about this.',
                    'Terrible experience, very disappointed.'
                ],
                'Date': ['2024-01-01', '2024-01-02', '2024-01-03']
            })
            st.dataframe(example_df, use_container_width=True)
            st.write("Your file should have at least one column containing text to analyze.")


if __name__ == "__main__":
    main()
