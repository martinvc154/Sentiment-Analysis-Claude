"""
Sentiment Analysis Module using Transformer Models
"""
import numpy as np
from transformers import pipeline
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')


class SentimentAnalyzer:
    """
    A comprehensive sentiment and emotion analyzer using transformer models.
    """

    def __init__(self):
        """Initialize the sentiment and emotion analysis pipelines."""
        # Sentiment analysis pipeline
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # CPU
        )

        # Emotion analysis pipeline
        self.emotion_pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=-1,  # CPU
            top_k=None  # Return all emotion scores
        )

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze a single text for sentiment and emotions.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary containing sentiment and emotion analysis results
        """
        if not text or not isinstance(text, str):
            return self._get_empty_result()

        # Clean text
        text = str(text).strip()
        if not text:
            return self._get_empty_result()

        # Truncate if too long (max 512 tokens for most models)
        if len(text) > 2000:
            text = text[:2000]

        try:
            # Sentiment analysis
            sentiment_result = self.sentiment_pipeline(text)[0]
            sentiment_label = sentiment_result['label'].lower()
            sentiment_confidence = sentiment_result['score']

            # Map to positive/negative/neutral and calculate score
            if sentiment_label == 'positive':
                sentiment = 'positive'
                sentiment_score = sentiment_confidence
            elif sentiment_label == 'negative':
                sentiment = 'negative'
                sentiment_score = -sentiment_confidence
            else:
                sentiment = 'neutral'
                sentiment_score = 0.0

            # Emotion analysis
            emotion_results = self.emotion_pipeline(text)[0]

            # Create emotion score dictionary
            emotions = {
                'joy': 0.0,
                'sadness': 0.0,
                'anger': 0.0,
                'fear': 0.0,
                'surprise': 0.0,
                'disgust': 0.0
            }

            # Fill in emotion scores
            for emotion in emotion_results:
                emotion_name = emotion['label'].lower()
                if emotion_name in emotions:
                    emotions[emotion_name] = emotion['score']

            # Find dominant emotion
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]

            # Calculate divergence (uncertainty/entropy measure)
            # Higher divergence means more mixed emotions
            emotion_scores = list(emotions.values())
            divergence = self._calculate_divergence(emotion_scores)

            return {
                'sentiment': sentiment,
                'sentiment_score': round(sentiment_score, 3),
                'confidence': round(sentiment_confidence, 3),
                'joy': round(emotions['joy'], 3),
                'sadness': round(emotions['sadness'], 3),
                'anger': round(emotions['anger'], 3),
                'fear': round(emotions['fear'], 3),
                'surprise': round(emotions['surprise'], 3),
                'disgust': round(emotions['disgust'], 3),
                'dominant_emotion': dominant_emotion,
                'divergence': round(divergence, 3)
            }

        except Exception as e:
            print(f"Error analyzing text: {str(e)}")
            return self._get_empty_result()

    def analyze_batch(self, texts: List[str], progress_callback=None) -> List[Dict[str, Any]]:
        """
        Analyze a batch of texts.

        Args:
            texts: List of texts to analyze
            progress_callback: Optional callback function for progress updates

        Returns:
            List of analysis results
        """
        results = []
        total = len(texts)

        for idx, text in enumerate(texts):
            result = self.analyze_text(text)
            results.append(result)

            if progress_callback:
                progress_callback((idx + 1) / total)

        return results

    def _calculate_divergence(self, scores: List[float]) -> float:
        """
        Calculate divergence (entropy) of emotion scores.
        Higher value indicates more mixed/uncertain emotions.

        Args:
            scores: List of emotion probabilities

        Returns:
            Divergence score (0-1)
        """
        # Normalize scores to ensure they sum to 1
        scores_array = np.array(scores)
        if scores_array.sum() == 0:
            return 0.0

        probs = scores_array / scores_array.sum()

        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # Normalize to 0-1 range (max entropy for 6 emotions is log(6))
        max_entropy = np.log(6)
        normalized_divergence = entropy / max_entropy if max_entropy > 0 else 0.0

        return normalized_divergence

    def _get_empty_result(self) -> Dict[str, Any]:
        """Return empty result structure for invalid inputs."""
        return {
            'sentiment': 'neutral',
            'sentiment_score': 0.0,
            'confidence': 0.0,
            'joy': 0.0,
            'sadness': 0.0,
            'anger': 0.0,
            'fear': 0.0,
            'surprise': 0.0,
            'disgust': 0.0,
            'dominant_emotion': 'neutral',
            'divergence': 0.0
        }
