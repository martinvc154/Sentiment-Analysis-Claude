from transformers import pipeline
from typing import List, Dict, Optional, Callable


class SentimentAnalyzer:
    """
    Sentiment and emotion analyzer using Hugging Face transformers.
    Analyzes text for sentiment, emotions, and sentiment-emotion divergence.
    """

    def __init__(self):
        """Load sentiment and emotion models once during initialization."""
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        self.emotion_model = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )

        # Emotion valence mapping for divergence calculation
        self.emotion_valence_map = {
            'joy': 1.0,
            'surprise': 1.0,
            'anger': -1.0,
            'fear': -1.0,
            'sadness': -1.0,
            'disgust': -1.0
        }

        # Sentiment label to score mapping
        self.sentiment_score_map = {
            'positive': 1.0,
            'neutral': 0.0,
            'negative': -1.0
        }

    def _truncate_text(self, text: str, max_tokens: int = 512) -> str:
        """
        Truncate text to max_tokens tokens (approximated by splitting on whitespace).
        """
        tokens = text.split()
        if len(tokens) > max_tokens:
            return ' '.join(tokens[:max_tokens])
        return text

    def _calculate_emotion_valence(self, emotions: Dict[str, float]) -> float:
        """
        Calculate weighted average emotion valence.

        Args:
            emotions: Dict of emotion names to scores

        Returns:
            Emotion valence score between -1 and 1
        """
        total_score = 0.0
        total_weight = 0.0

        for emotion, score in emotions.items():
            if emotion in self.emotion_valence_map:
                valence = self.emotion_valence_map[emotion]
                total_score += valence * score
                total_weight += score

        if total_weight == 0:
            return 0.0

        return total_score / total_weight

    def analyze(self, text: str) -> Dict:
        """
        Analyze a single text for sentiment and emotions.

        Args:
            text: Input text to analyze

        Returns:
            Dict containing:
                - sentiment: str (positive/neutral/negative)
                - sentiment_score: float (-1 to 1)
                - sentiment_confidence: float (0 to 1)
                - emotions: dict of {emotion: score}
                - dominant_emotion: str
                - divergence: float (0 to 2)
        """
        try:
            # Handle empty text
            if not text or not text.strip():
                return {
                    'sentiment': 'neutral',
                    'sentiment_score': 0.0,
                    'sentiment_confidence': 0.0,
                    'emotions': {},
                    'dominant_emotion': 'neutral',
                    'divergence': 0.0,
                    'error': 'Empty text provided'
                }

            # Truncate long text
            processed_text = self._truncate_text(text.strip())

            # Get sentiment analysis
            sentiment_result = self.sentiment_model(processed_text)[0]
            sentiment_label = sentiment_result['label'].lower()
            sentiment_confidence = sentiment_result['score']

            # Map sentiment to score (-1 to 1)
            sentiment_score = self.sentiment_score_map.get(sentiment_label, 0.0)

            # Get emotion analysis
            emotion_results = self.emotion_model(processed_text)[0]

            # Convert emotion results to dict and find dominant emotion
            emotions = {}
            dominant_emotion = None
            max_emotion_score = -1

            for emotion_result in emotion_results:
                emotion = emotion_result['label'].lower()
                score = emotion_result['score']
                emotions[emotion] = score

                if score > max_emotion_score:
                    max_emotion_score = score
                    dominant_emotion = emotion

            # Calculate emotion valence
            emotion_valence = self._calculate_emotion_valence(emotions)

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
            # Return error result
            return {
                'sentiment': 'neutral',
                'sentiment_score': 0.0,
                'sentiment_confidence': 0.0,
                'emotions': {},
                'dominant_emotion': 'unknown',
                'divergence': 0.0,
                'error': str(e)
            }

    def analyze_batch(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict]:
        """
        Analyze a batch of texts.

        Args:
            texts: List of texts to analyze
            progress_callback: Optional callback function(current, total) for progress updates

        Returns:
            List of analysis result dicts
        """
        results = []
        total = len(texts)

        for i, text in enumerate(texts):
            result = self.analyze(text)
            results.append(result)

            # Call progress callback if provided
            if progress_callback is not None:
                progress_callback(i + 1, total)

        return results
