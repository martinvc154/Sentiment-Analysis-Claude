import anthropic
import os
import json
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Callable


def analyze_sentiment(
    text: str,
    model: str = "claude-3-5-sonnet-20241022",
    include_confidence: bool = True,
    include_reasoning: bool = True
) -> Dict[str, Any]:
    """
    Analyze sentiment of a single text using Claude.

    Args:
        text: The text to analyze
        model: Claude model to use
        include_confidence: Whether to include confidence score
        include_reasoning: Whether to include reasoning

    Returns:
        Dictionary containing sentiment analysis results
    """
    try:
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            return {
                'status': 'error',
                'error': 'ANTHROPIC_API_KEY not set'
            }

        client = anthropic.Anthropic(api_key=api_key)

        # Build prompt based on options
        prompt = f"""Analyze the sentiment of the following text and provide a structured response.

Text to analyze:
"{text}"

Please provide your analysis in the following JSON format:
{{
    "sentiment": "<Positive|Negative|Neutral|Mixed>",
    "confidence": <0.0 to 1.0>,
    "emotions": ["emotion1", "emotion2", ...],
    "reasoning": "Brief explanation of why you classified it this way"
}}

Guidelines:
- sentiment: Must be one of: Positive, Negative, Neutral, or Mixed
- confidence: A score from 0.0 to 1.0 indicating how confident you are
- emotions: List of specific emotions detected (e.g., joy, anger, sadness, fear, surprise, disgust)
- reasoning: Brief explanation of your classification

Respond only with the JSON object, no additional text."""

        # Call Claude API
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Parse response
        response_text = message.content[0].text.strip()

        # Handle potential markdown code blocks
        if response_text.startswith('```'):
            # Remove markdown code block syntax
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1] if len(lines) > 2 else lines)
            response_text = response_text.replace('```json', '').replace('```', '').strip()

        result = json.loads(response_text)

        # Filter result based on options
        filtered_result = {
            'status': 'success',
            'sentiment': result.get('sentiment', 'Unknown')
        }

        if include_confidence:
            filtered_result['confidence'] = result.get('confidence', 0.0)

        if 'emotions' in result:
            filtered_result['emotions'] = result.get('emotions', [])

        if include_reasoning:
            filtered_result['reasoning'] = result.get('reasoning', '')

        return filtered_result

    except json.JSONDecodeError as e:
        return {
            'status': 'error',
            'error': f'Failed to parse Claude response: {str(e)}',
            'raw_response': response_text if 'response_text' in locals() else None
        }
    except anthropic.APIError as e:
        return {
            'status': 'error',
            'error': f'Anthropic API error: {str(e)}'
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Unexpected error: {str(e)}'
        }


def analyze_batch(
    texts: List[str],
    model: str = "claude-3-5-sonnet-20241022",
    include_confidence: bool = True,
    include_reasoning: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Dict[str, Any]]:
    """
    Analyze sentiment of multiple texts.

    Args:
        texts: List of texts to analyze
        model: Claude model to use
        include_confidence: Whether to include confidence scores
        include_reasoning: Whether to include reasoning
        progress_callback: Optional callback function(current, total)

    Returns:
        List of dictionaries containing analysis results
    """
    results = []

    for idx, text in enumerate(texts):
        result = analyze_sentiment(
            text,
            model=model,
            include_confidence=include_confidence,
            include_reasoning=include_reasoning
        )

        # Add the original text to results
        result['text'] = text
        result['index'] = idx

        results.append(result)

        # Update progress
        if progress_callback:
            progress_callback(idx + 1, len(texts))

    return results


def create_sentiment_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a plotly chart showing sentiment distribution.

    Args:
        df: DataFrame containing sentiment analysis results

    Returns:
        Plotly figure object
    """
    # Count sentiments
    sentiment_counts = df['sentiment'].value_counts()

    # Define colors
    colors = {
        'Positive': '#28a745',
        'Negative': '#dc3545',
        'Neutral': '#ffc107',
        'Mixed': '#17a2b8'
    }

    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            marker_color=[colors.get(sent, '#6c757d') for sent in sentiment_counts.index],
            text=sentiment_counts.values,
            textposition='auto',
        )
    ])

    fig.update_layout(
        title="Sentiment Distribution",
        xaxis_title="Sentiment",
        yaxis_title="Count",
        showlegend=False,
        height=400,
        template="plotly_white"
    )

    return fig


def export_results(df: pd.DataFrame, format: str = 'csv') -> str:
    """
    Export analysis results to various formats.

    Args:
        df: DataFrame containing results
        format: Export format ('csv' or 'json')

    Returns:
        Exported data as string
    """
    if format == 'csv':
        return df.to_csv(index=False)
    elif format == 'json':
        return df.to_json(orient='records', indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")


def get_sentiment_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a summary of sentiment analysis results.

    Args:
        df: DataFrame containing sentiment analysis results

    Returns:
        Dictionary containing summary statistics
    """
    summary = {
        'total_analyzed': len(df),
        'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
    }

    if 'confidence' in df.columns:
        summary['average_confidence'] = df['confidence'].mean()
        summary['confidence_std'] = df['confidence'].std()

    if 'emotions' in df.columns:
        # Flatten emotions and count
        all_emotions = []
        for emotions in df['emotions'].dropna():
            if isinstance(emotions, list):
                all_emotions.extend(emotions)
        if all_emotions:
            emotion_series = pd.Series(all_emotions)
            summary['top_emotions'] = emotion_series.value_counts().head(5).to_dict()

    return summary
