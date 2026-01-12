from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
from transformers import pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import chi2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sentiment Analysis API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
logger.info("Loading sentiment and emotion models...")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
logger.info("Models loaded successfully")


class TextInput(BaseModel):
    text: str
    organization: Optional[str] = None


class AnalysisRequest(BaseModel):
    texts: List[TextInput]


class EmotionScore(BaseModel):
    joy: float
    sadness: float
    anger: float
    fear: float
    surprise: float
    disgust: float


class AnalysisResult(BaseModel):
    text: str
    organization: Optional[str]
    sentiment_label: str
    sentiment_score: float
    emotions: EmotionScore


class ClusterInfo(BaseModel):
    cluster_id: int
    n: int
    avg_sentiment: float
    dominant_emotion: str
    suggested_label: str


class ClusterResult(BaseModel):
    optimal_k: int
    silhouette_score: float
    cluster_assignments: List[int]
    pca_coordinates: List[List[float]]
    centroids: List[List[float]]
    confidence_ellipses: List[Dict[str, Any]]
    cluster_info: List[ClusterInfo]


def analyze_emotions(text: str) -> Dict[str, float]:
    """Analyze emotions in text and return normalized scores."""
    results = emotion_analyzer(text)[0]

    # Map all emotions to their scores
    emotion_map = {item['label']: item['score'] for item in results}

    # Return in consistent order with all 6 emotions
    return {
        'joy': emotion_map.get('joy', 0.0),
        'sadness': emotion_map.get('sadness', 0.0),
        'anger': emotion_map.get('anger', 0.0),
        'fear': emotion_map.get('fear', 0.0),
        'surprise': emotion_map.get('surprise', 0.0),
        'disgust': emotion_map.get('disgust', 0.0),
    }


def analyze_sentiment(text: str) -> tuple:
    """Analyze sentiment and return label and score."""
    result = sentiment_analyzer(text)[0]
    label = result['label']
    score = result['score'] if label == 'POSITIVE' else -result['score']
    # Normalize to -1 to 1
    return label, score


def get_cluster_label(avg_sentiment: float, emotions: Dict[str, float], variance: float) -> str:
    """Determine cluster label based on sentiment and emotion profile."""
    dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]

    # High positive + joy → "Aligned Positive"
    if avg_sentiment > 0.3 and emotions['joy'] > 0.4:
        return "Aligned Positive"

    # Negative + low divergence → "Open Critics"
    elif avg_sentiment < -0.2 and variance < 0.1:
        return "Open Critics"

    # Neutral + high divergence + fear → "Strategic Adapters"
    elif abs(avg_sentiment) < 0.3 and variance > 0.15 and emotions['fear'] > 0.2:
        return "Strategic Adapters"

    # Positive sentiment + negative emotions → "Preference Falsifiers"
    elif avg_sentiment > 0.2 and (emotions['sadness'] > 0.2 or emotions['anger'] > 0.2 or emotions['fear'] > 0.2):
        return "Preference Falsifiers"

    # Default labels
    elif avg_sentiment > 0.3:
        return "Positive"
    elif avg_sentiment < -0.3:
        return "Negative"
    else:
        return "Neutral"


def calculate_confidence_ellipse(points: np.ndarray, n_std: float = 2.447) -> Dict[str, Any]:
    """
    Calculate 95% confidence ellipse parameters.
    n_std = 2.447 corresponds to 95% confidence for 2D data.
    """
    if len(points) < 2:
        return {"center": [0, 0], "width": 0, "height": 0, "angle": 0}

    mean = np.mean(points, axis=0)
    cov = np.cov(points.T)

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Calculate ellipse parameters
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])

    return {
        "center": mean.tolist(),
        "width": float(width),
        "height": float(height),
        "angle": float(angle)
    }


@app.post("/analyze", response_model=List[AnalysisResult])
async def analyze_texts(request: AnalysisRequest):
    """Analyze sentiment and emotions for multiple texts."""
    try:
        results = []

        for item in request.texts:
            # Analyze sentiment
            sentiment_label, sentiment_score = analyze_sentiment(item.text)

            # Analyze emotions
            emotions = analyze_emotions(item.text)

            results.append(AnalysisResult(
                text=item.text,
                organization=item.organization,
                sentiment_label=sentiment_label,
                sentiment_score=sentiment_score,
                emotions=EmotionScore(**emotions)
            ))

        return results

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cluster", response_model=ClusterResult)
async def cluster_analysis(analyses: List[AnalysisResult]):
    """Perform K-means clustering on sentiment and emotion data."""
    try:
        if len(analyses) < 5:
            raise HTTPException(status_code=400, detail="Need at least 5 samples for clustering")

        # Prepare feature matrix: sentiment_score + 6 emotions
        features = []
        for analysis in analyses:
            features.append([
                analysis.sentiment_score,
                analysis.emotions.joy,
                analysis.emotions.sadness,
                analysis.emotions.anger,
                analysis.emotions.fear,
                analysis.emotions.surprise,
                analysis.emotions.disgust
            ])

        X = np.array(features)

        # Test k=2,3,4,5 and pick best silhouette score
        best_k = 2
        best_score = -1
        best_kmeans = None

        max_k = min(5, len(analyses) - 1)

        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)

            if score > best_score:
                best_score = score
                best_k = k
                best_kmeans = kmeans

        # Get final cluster assignments
        cluster_assignments = best_kmeans.labels_.tolist()

        # PCA for 2D visualization
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(X)
        pca_coordinates = pca_coords.tolist()

        # Transform centroids to PCA space
        centroids_pca = pca.transform(best_kmeans.cluster_centers_)
        centroids = centroids_pca.tolist()

        # Calculate confidence ellipses for each cluster
        confidence_ellipses = []
        for i in range(best_k):
            cluster_points = pca_coords[best_kmeans.labels_ == i]
            ellipse = calculate_confidence_ellipse(cluster_points)
            ellipse['cluster_id'] = i
            confidence_ellipses.append(ellipse)

        # Calculate cluster characteristics
        cluster_info = []
        for i in range(best_k):
            cluster_mask = best_kmeans.labels_ == i
            cluster_data = X[cluster_mask]
            cluster_analyses = [analyses[j] for j in range(len(analyses)) if cluster_mask[j]]

            # Average sentiment
            avg_sentiment = float(np.mean(cluster_data[:, 0]))

            # Dominant emotion
            emotion_avgs = {
                'joy': float(np.mean(cluster_data[:, 1])),
                'sadness': float(np.mean(cluster_data[:, 2])),
                'anger': float(np.mean(cluster_data[:, 3])),
                'fear': float(np.mean(cluster_data[:, 4])),
                'surprise': float(np.mean(cluster_data[:, 5])),
                'disgust': float(np.mean(cluster_data[:, 6]))
            }
            dominant_emotion = max(emotion_avgs.items(), key=lambda x: x[1])[0]

            # Calculate variance for labeling
            variance = float(np.var(cluster_data[:, 0]))

            # Get suggested label
            suggested_label = get_cluster_label(avg_sentiment, emotion_avgs, variance)

            cluster_info.append(ClusterInfo(
                cluster_id=i,
                n=int(np.sum(cluster_mask)),
                avg_sentiment=avg_sentiment,
                dominant_emotion=dominant_emotion,
                suggested_label=suggested_label
            ))

        return ClusterResult(
            optimal_k=best_k,
            silhouette_score=float(best_score),
            cluster_assignments=cluster_assignments,
            pca_coordinates=pca_coordinates,
            centroids=centroids,
            confidence_ellipses=confidence_ellipses,
            cluster_info=cluster_info
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during clustering: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
