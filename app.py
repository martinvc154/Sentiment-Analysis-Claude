import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from anthropic import Anthropic
import json
import os
from typing import Dict, List, Optional

# Page config
st.set_page_config(page_title="Sentiment Analysis with Claude", layout="wide")

# Initialize session state
if 'analyzed_data' not in st.session_state:
    st.session_state.analyzed_data = None
if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {}
if 'filter_divergent' not in st.session_state:
    st.session_state.filter_divergent = False

def analyze_sentiment(text: str, api_key: str) -> Dict:
    """Analyze sentiment and emotions using Claude API."""
    client = Anthropic(api_key=api_key)

    prompt = f"""Analyze the sentiment and emotions in this text. Return ONLY a JSON object with this exact structure:
{{
    "sentiment": <float between -1 and 1>,
    "emotions": {{
        "joy": <float between 0 and 1>,
        "sadness": <float between 0 and 1>,
        "anger": <float between 0 and 1>,
        "fear": <float between 0 and 1>,
        "surprise": <float between 0 and 1>,
        "disgust": <float between 0 and 1>
    }}
}}

Text to analyze: "{text}"

Rules:
- sentiment: -1 (very negative) to 1 (very positive), 0 is neutral
- Each emotion: 0 (not present) to 1 (strongly present)
- Return ONLY the JSON, no other text"""

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    result = json.loads(message.content[0].text)

    # Calculate divergence (sentiment-emotion mismatch)
    emotions = result['emotions']
    positive_emotions = emotions['joy'] + emotions['surprise']
    negative_emotions = emotions['sadness'] + emotions['anger'] + emotions['fear'] + emotions['disgust']

    emotion_polarity = (positive_emotions - negative_emotions) / 2
    divergence = abs(result['sentiment'] - emotion_polarity)

    result['divergence'] = divergence

    return result

def categorize_sentiment(score: float) -> str:
    """Categorize sentiment score into positive/neutral/negative."""
    if score > 0.1:
        return 'positive'
    elif score < -0.1:
        return 'negative'
    else:
        return 'neutral'

def render_dashboard():
    """Render the Dashboard tab with all metrics and visualizations."""
    st.header("📊 Dashboard")

    if st.session_state.analyzed_data is None or len(st.session_state.analyzed_data) == 0:
        st.info("👈 Upload and analyze data to see the dashboard")
        return

    df = st.session_state.analyzed_data

    # ROW 1 - Metric Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Responses", len(df))

    with col2:
        org_col = st.session_state.column_mapping.get('organization')
        if org_col and org_col in df.columns:
            unique_orgs = df[org_col].nunique()
            st.metric("Organizations Analyzed", unique_orgs)
        else:
            st.metric("Organizations Analyzed", "N/A")

    with col3:
        avg_sentiment = df['sentiment'].mean()
        if avg_sentiment > 0.1:
            delta_color = "normal"
            sentiment_label = "Positive"
        elif avg_sentiment < -0.1:
            delta_color = "inverse"
            sentiment_label = "Negative"
        else:
            delta_color = "off"
            sentiment_label = "Neutral"

        st.metric(
            "Average Sentiment",
            f"{avg_sentiment:.3f}",
            delta=sentiment_label,
            delta_color=delta_color
        )

    with col4:
        high_divergence = len(df[df['divergence'] > 0.4])
        st.metric("High Divergence Count", high_divergence)

    st.divider()

    # ROW 2 - Two charts side by side
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Sentiment Distribution")

        # Count sentiment categories
        df['sentiment_category'] = df['sentiment'].apply(categorize_sentiment)
        sentiment_counts = df['sentiment_category'].value_counts()

        # Create donut chart
        colors = {
            'positive': '#55a868',
            'neutral': '#8c8c8c',
            'negative': '#c44e52'
        }

        fig_donut = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=0.4,
            marker=dict(colors=[colors.get(cat, '#8c8c8c') for cat in sentiment_counts.index]),
            textinfo='label+percent',
            textposition='auto',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])

        fig_donut.update_layout(
            showlegend=True,
            height=400,
            margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        st.subheader("Emotion Prevalence")

        # Calculate average emotion scores
        emotion_cols = ['emotion_joy', 'emotion_sadness', 'emotion_anger',
                       'emotion_fear', 'emotion_surprise', 'emotion_disgust']
        emotion_names = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']

        emotion_avgs = []
        for col, name in zip(emotion_cols, emotion_names):
            emotion_avgs.append({
                'emotion': name.capitalize(),
                'score': df[col].mean()
            })

        emotion_df = pd.DataFrame(emotion_avgs).sort_values('score', ascending=True)

        # Colors for emotions
        emotion_colors = {
            'Joy': '#f4b942',
            'Sadness': '#5c7fa3',
            'Anger': '#c44e52',
            'Fear': '#8b5cf6',
            'Surprise': '#06b6d4',
            'Disgust': '#84735a'
        }

        fig_bar = go.Figure(data=[go.Bar(
            y=emotion_df['emotion'],
            x=emotion_df['score'],
            orientation='h',
            marker=dict(color=[emotion_colors.get(e, '#8c8c8c') for e in emotion_df['emotion']]),
            text=emotion_df['score'].round(3),
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Average Score: %{x:.3f}<extra></extra>'
        )])

        fig_bar.update_layout(
            showlegend=False,
            height=400,
            margin=dict(t=20, b=20, l=20, r=20),
            xaxis_title="Average Score",
            yaxis_title="",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(range=[0, 1])
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ROW 3 - Alert Box
    if high_divergence > 0:
        st.warning(
            f"⚠️ {high_divergence} responses show sentiment-emotion divergence > 0.4, "
            "suggesting potential preference falsification"
        )

        if st.button("🔍 View Divergent Responses", type="primary"):
            st.session_state.filter_divergent = True
            st.rerun()

    st.divider()

    # ROW 4 - Organization Comparison (if org column mapped)
    org_col = st.session_state.column_mapping.get('organization')
    if org_col and org_col in df.columns:
        st.subheader("Sentiment by Organization")

        # Group by organization and sentiment category
        org_sentiment = df.groupby([org_col, 'sentiment_category']).size().unstack(fill_value=0)

        # Calculate percentages
        org_sentiment_pct = org_sentiment.div(org_sentiment.sum(axis=1), axis=0) * 100

        # Create grouped bar chart
        fig_org = go.Figure()

        for category in ['positive', 'neutral', 'negative']:
            if category in org_sentiment_pct.columns:
                fig_org.add_trace(go.Bar(
                    name=category.capitalize(),
                    x=org_sentiment_pct.index,
                    y=org_sentiment_pct[category],
                    marker_color=colors[category],
                    hovertemplate='<b>%{x}</b><br>' + category.capitalize() + ': %{y:.1f}%<extra></extra>'
                ))

        fig_org.update_layout(
            barmode='group',
            height=400,
            xaxis_title="Organization",
            yaxis_title="Percentage (%)",
            legend_title="Sentiment",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )

        st.plotly_chart(fig_org, use_container_width=True)

def render_upload():
    """Render the Upload tab."""
    st.header("📁 Upload & Configure")

    # API Key
    api_key = st.text_input("Claude API Key", type="password",
                           help="Enter your Anthropic API key")

    if not api_key:
        st.warning("Please enter your Claude API key to continue")
        return

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} rows")

        # Column mapping
        st.subheader("Column Mapping")

        col1, col2 = st.columns(2)

        with col1:
            text_col = st.selectbox("Text/Response Column", options=df.columns)

        with col2:
            org_col = st.selectbox("Organization Column (optional)",
                                  options=['None'] + list(df.columns))

        st.session_state.column_mapping = {
            'text': text_col,
            'organization': org_col if org_col != 'None' else None
        }

        # Analyze button
        if st.button("🚀 Analyze Sentiment", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            results = []

            for idx, row in df.iterrows():
                status_text.text(f"Analyzing response {idx + 1} of {len(df)}...")
                progress_bar.progress((idx + 1) / len(df))

                text = str(row[text_col])

                try:
                    analysis = analyze_sentiment(text, api_key)

                    result_row = row.to_dict()
                    result_row['sentiment'] = analysis['sentiment']
                    result_row['divergence'] = analysis['divergence']

                    for emotion, score in analysis['emotions'].items():
                        result_row[f'emotion_{emotion}'] = score

                    results.append(result_row)

                except Exception as e:
                    st.error(f"Error analyzing row {idx + 1}: {str(e)}")
                    continue

            st.session_state.analyzed_data = pd.DataFrame(results)
            status_text.text("Analysis complete!")
            st.success(f"✅ Analyzed {len(results)} responses")
            st.balloons()

def render_results():
    """Render the Results tab."""
    st.header("📋 Detailed Results")

    if st.session_state.analyzed_data is None:
        st.info("👈 Upload and analyze data to see results")
        return

    df = st.session_state.analyzed_data.copy()

    # Filter options
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        sentiment_filter = st.multiselect(
            "Filter by Sentiment",
            options=['positive', 'neutral', 'negative'],
            default=['positive', 'neutral', 'negative']
        )

    with col2:
        org_col = st.session_state.column_mapping.get('organization')
        if org_col and org_col in df.columns:
            orgs = df[org_col].unique().tolist()
            org_filter = st.multiselect("Filter by Organization", options=orgs, default=orgs)
        else:
            org_filter = None

    with col3:
        if st.button("Clear Filters"):
            st.session_state.filter_divergent = False
            st.rerun()

    # Apply filters
    df['sentiment_category'] = df['sentiment'].apply(categorize_sentiment)
    df = df[df['sentiment_category'].isin(sentiment_filter)]

    if org_filter and org_col:
        df = df[df[org_col].isin(org_filter)]

    if st.session_state.filter_divergent:
        df = df[df['divergence'] > 0.4]
        st.info(f"Showing {len(df)} responses with high divergence (> 0.4)")

    # Display results
    st.write(f"Showing {len(df)} of {len(st.session_state.analyzed_data)} responses")

    # Format display
    text_col = st.session_state.column_mapping.get('text')

    for idx, row in df.iterrows():
        with st.expander(f"Response {idx + 1} - {row['sentiment_category'].upper()} (sentiment: {row['sentiment']:.3f})"):
            if text_col:
                st.write("**Text:**", row[text_col])

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Sentiment Score", f"{row['sentiment']:.3f}")
                st.metric("Divergence", f"{row['divergence']:.3f}")

            with col2:
                if org_col and org_col in row:
                    st.write(f"**Organization:** {row[org_col]}")

            st.write("**Emotions:**")
            emotion_cols = {
                'Joy': 'emotion_joy',
                'Sadness': 'emotion_sadness',
                'Anger': 'emotion_anger',
                'Fear': 'emotion_fear',
                'Surprise': 'emotion_surprise',
                'Disgust': 'emotion_disgust'
            }

            cols = st.columns(3)
            for i, (emotion, col_name) in enumerate(emotion_cols.items()):
                with cols[i % 3]:
                    st.metric(emotion, f"{row[col_name]:.3f}")

# Main app
def main():
    st.title("🎭 Sentiment Analysis with Claude")
    st.caption("Analyze sentiment, emotions, and detect preference falsification")

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📁 Upload", "📊 Dashboard", "📋 Results"])

    with tab1:
        render_upload()

    with tab2:
        render_dashboard()

    with tab3:
        render_results()

if __name__ == "__main__":
    main()
