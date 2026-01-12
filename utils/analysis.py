from transformers import pipeline
import numpy as np


class SentimentAnalyzer:
    """Sentiment and emotion analysis using transformer models."""

    # Emotion to valence mapping
    EMOTION_VALENCE = {
        'joy': 1.0,
        'surprise': 1.0,
        'anger': -1.0,
        'fear': -1.0,
        'sadness': -1.0,
        'disgust': -1.0
    }

    def __init__(self):
        """Initialize sentiment and emotion models."""
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        self.emotion_model = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )

    def _truncate_text(self, text, max_tokens=512):
        """Truncate text to maximum token length."""
        if not text:
            return ""

        # Simple approximation: ~4 characters per token
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            return text[:max_chars]
        return text

    def _convert_sentiment_to_score(self, sentiment_result):
        """
        Convert sentiment model output to -1 to 1 scale.

        Returns:
            tuple: (sentiment_label, sentiment_score, confidence)
        """
        label = sentiment_result['label'].lower()
        confidence = sentiment_result['score']

        # Map labels to scores
        if 'positive' in label:
            sentiment_score = confidence
            sentiment_label = 'positive'
        elif 'negative' in label:
            sentiment_score = -confidence
            sentiment_label = 'negative'
        else:  # neutral
            sentiment_score = 0.0
            sentiment_label = 'neutral'

        return sentiment_label, sentiment_score, confidence

    def _process_emotions(self, emotion_results):
        """
        Process emotion model output.

        Returns:
            tuple: (emotions_dict, dominant_emotion, emotion_valence)
        """
        # Convert to dict
        emotions = {result['label']: result['score'] for result in emotion_results}

        # Find dominant emotion
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]

        # Calculate weighted emotion valence
        emotion_valence = sum(
            score * self.EMOTION_VALENCE.get(emotion, 0)
            for emotion, score in emotions.items()
        )

        return emotions, dominant_emotion, emotion_valence

    def analyze(self, text):
        """
        Analyze sentiment and emotions for a single text.

        Args:
            text: String to analyze

        Returns:
            dict with keys:
                - sentiment: positive/neutral/negative
                - sentiment_score: float (-1 to 1)
                - sentiment_confidence: float (0 to 1)
                - emotions: dict of {emotion: score}
                - dominant_emotion: str
                - divergence: float
        """
        # Handle empty or invalid text
        if not text or not isinstance(text, str) or not text.strip():
            return {
                'sentiment': 'neutral',
                'sentiment_score': 0.0,
                'sentiment_confidence': 0.0,
                'emotions': {emotion: 0.0 for emotion in self.EMOTION_VALENCE.keys()},
                'dominant_emotion': 'neutral',
                'divergence': 0.0,
                'error': 'Empty or invalid text'
            }

        try:
            # Truncate long text
            text = self._truncate_text(text.strip())

            # Get sentiment
            sentiment_result = self.sentiment_model(text)[0]
            sentiment_label, sentiment_score, sentiment_confidence = \
                self._convert_sentiment_to_score(sentiment_result)

            # Get emotions
            emotion_results = self.emotion_model(text)[0]
            emotions, dominant_emotion, emotion_valence = \
                self._process_emotions(emotion_results)

            # Calculate divergence
            divergence = abs(sentiment_score - emotion_valence)

            return {
                'sentiment': sentiment_label,
                'sentiment_score': sentiment_score,
                'sentiment_confidence': sentiment_confidence,
                'emotions': emotions,
                'dominant_emotion': dominant_emotion,
                'divergence': divergence
            }

        except Exception as e:
            # Return neutral results on error
            return {
                'sentiment': 'neutral',
                'sentiment_score': 0.0,
                'sentiment_confidence': 0.0,
                'emotions': {emotion: 0.0 for emotion in self.EMOTION_VALENCE.keys()},
                'dominant_emotion': 'neutral',
                'divergence': 0.0,
                'error': str(e)
            }

    def analyze_batch(self, texts, progress_callback=None):
        """
        Analyze a batch of texts with optional progress tracking.

        Args:
            texts: List of strings to analyze
            progress_callback: Optional function(current, total) for progress updates

        Returns:
            list of result dicts (same format as analyze())
        """
        results = []
        total = len(texts)

        for i, text in enumerate(texts):
            # Analyze individual text
            result = self.analyze(text)
            results.append(result)

            # Update progress if callback provided
            if progress_callback:
                progress_callback(i + 1, total)

        return results
