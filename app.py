import streamlit as st
import pandas as pd
from datetime import datetime
import os
from utils.analysis import (
    analyze_sentiment,
    analyze_batch,
    create_sentiment_chart,
    export_results
)

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis with Claude",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #ffc107;
        font-weight: bold;
    }
    .sentiment-mixed {
        color: #17a2b8;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Header
st.markdown('<div class="main-header">🎭 Sentiment Analysis with Claude</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by Anthropic\'s Claude AI</div>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # API Key input
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        help="Enter your Anthropic API key. Get one at https://console.anthropic.com/"
    )

    if api_key:
        os.environ['ANTHROPIC_API_KEY'] = api_key
        st.success("API Key set!")
    else:
        st.warning("Please enter your API key to continue")

    st.divider()

    # Model selection
    model = st.selectbox(
        "Claude Model",
        ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
        help="Select the Claude model to use for analysis"
    )

    st.divider()

    # Analysis options
    st.subheader("Analysis Options")
    include_confidence = st.checkbox("Include confidence scores", value=True)
    include_reasoning = st.checkbox("Include reasoning", value=True)

    st.divider()

    # History management
    if st.button("Clear History", use_container_width=True):
        st.session_state.analysis_history = []
        st.rerun()

# Main content
tab1, tab2, tab3 = st.tabs(["📝 Single Analysis", "📊 Batch Analysis", "📈 History"])

# Tab 1: Single Text Analysis
with tab1:
    st.header("Analyze Single Text")

    text_input = st.text_area(
        "Enter text to analyze",
        height=150,
        placeholder="Type or paste your text here...",
        help="Enter any text and Claude will analyze its sentiment"
    )

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        analyze_button = st.button("🔍 Analyze Sentiment", use_container_width=True, type="primary")

    if analyze_button and text_input and api_key:
        with st.spinner("Analyzing sentiment..."):
            result = analyze_sentiment(
                text_input,
                model=model,
                include_confidence=include_confidence,
                include_reasoning=include_reasoning
            )

            if result['status'] == 'success':
                # Add to history
                result['timestamp'] = datetime.now()
                result['text'] = text_input
                st.session_state.analysis_history.append(result)

                # Display results
                st.success("Analysis complete!")

                col1, col2 = st.columns(2)

                with col1:
                    sentiment = result['sentiment'].lower()
                    sentiment_class = f"sentiment-{sentiment}"
                    st.markdown(f"### Sentiment: <span class='{sentiment_class}'>{result['sentiment']}</span>", unsafe_allow_html=True)

                    if include_confidence and 'confidence' in result:
                        st.metric("Confidence", f"{result['confidence']:.1%}")

                with col2:
                    if 'emotions' in result and result['emotions']:
                        st.markdown("### Detected Emotions")
                        for emotion in result['emotions']:
                            st.write(f"• {emotion}")

                if include_reasoning and 'reasoning' in result:
                    st.markdown("### Analysis Reasoning")
                    st.info(result['reasoning'])

            else:
                st.error(f"Error: {result.get('error', 'Unknown error occurred')}")

    elif analyze_button and not api_key:
        st.warning("Please enter your Anthropic API key in the sidebar")
    elif analyze_button and not text_input:
        st.warning("Please enter some text to analyze")

# Tab 2: Batch Analysis
with tab2:
    st.header("Batch Analysis")

    upload_option = st.radio(
        "Input method",
        ["Upload CSV", "Paste Text (one per line)"],
        horizontal=True
    )

    texts_to_analyze = []

    if upload_option == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="CSV should have a column with text to analyze"
        )

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())

            text_column = st.selectbox(
                "Select column containing text",
                df.columns.tolist()
            )

            if text_column:
                texts_to_analyze = df[text_column].dropna().tolist()

    else:
        batch_text = st.text_area(
            "Enter texts (one per line)",
            height=200,
            placeholder="Enter multiple texts, one per line..."
        )

        if batch_text:
            texts_to_analyze = [line.strip() for line in batch_text.split('\n') if line.strip()]

    if texts_to_analyze:
        st.info(f"Ready to analyze {len(texts_to_analyze)} texts")

        if st.button("🚀 Start Batch Analysis", type="primary") and api_key:
            progress_bar = st.progress(0)
            status_text = st.empty()

            results = analyze_batch(
                texts_to_analyze,
                model=model,
                include_confidence=include_confidence,
                include_reasoning=include_reasoning,
                progress_callback=lambda current, total: (
                    progress_bar.progress(current / total),
                    status_text.text(f"Analyzing: {current}/{total}")
                )
            )

            progress_bar.empty()
            status_text.empty()

            # Create results dataframe
            results_df = pd.DataFrame(results)

            st.success(f"Analyzed {len(results)} texts!")

            # Display summary
            st.subheader("Summary")
            col1, col2, col3, col4 = st.columns(4)

            sentiment_counts = results_df['sentiment'].value_counts()

            with col1:
                st.metric("Positive", sentiment_counts.get('Positive', 0))
            with col2:
                st.metric("Negative", sentiment_counts.get('Negative', 0))
            with col3:
                st.metric("Neutral", sentiment_counts.get('Neutral', 0))
            with col4:
                st.metric("Mixed", sentiment_counts.get('Mixed', 0))

            # Display chart
            st.subheader("Sentiment Distribution")
            fig = create_sentiment_chart(results_df)
            st.plotly_chart(fig, use_container_width=True)

            # Display results table
            st.subheader("Results")
            st.dataframe(results_df, use_container_width=True)

            # Export options
            st.subheader("Export Results")
            col1, col2 = st.columns(2)

            with col1:
                csv_data = export_results(results_df, format='csv')
                st.download_button(
                    "📥 Download CSV",
                    csv_data,
                    f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )

            with col2:
                json_data = export_results(results_df, format='json')
                st.download_button(
                    "📥 Download JSON",
                    json_data,
                    f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json",
                    use_container_width=True
                )

# Tab 3: History
with tab3:
    st.header("Analysis History")

    if st.session_state.analysis_history:
        st.info(f"Total analyses: {len(st.session_state.analysis_history)}")

        for idx, item in enumerate(reversed(st.session_state.analysis_history)):
            with st.expander(
                f"{item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - {item['sentiment']}",
                expanded=(idx == 0)
            ):
                st.write(f"**Text:** {item['text']}")
                st.write(f"**Sentiment:** {item['sentiment']}")

                if 'confidence' in item:
                    st.write(f"**Confidence:** {item['confidence']:.1%}")

                if 'emotions' in item and item['emotions']:
                    st.write(f"**Emotions:** {', '.join(item['emotions'])}")

                if 'reasoning' in item:
                    st.write(f"**Reasoning:** {item['reasoning']}")

        # Export history
        if st.button("📥 Export History"):
            history_df = pd.DataFrame([
                {
                    'timestamp': item['timestamp'],
                    'text': item['text'],
                    'sentiment': item['sentiment'],
                    'confidence': item.get('confidence', ''),
                    'emotions': ', '.join(item.get('emotions', [])),
                    'reasoning': item.get('reasoning', '')
                }
                for item in st.session_state.analysis_history
            ])

            csv_data = export_results(history_df, format='csv')
            st.download_button(
                "📥 Download History as CSV",
                csv_data,
                f"analysis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
    else:
        st.info("No analysis history yet. Analyze some text to see it here!")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        Built with Streamlit and Claude AI |
        <a href='https://www.anthropic.com' target='_blank'>Anthropic</a>
    </div>
""", unsafe_allow_html=True)
