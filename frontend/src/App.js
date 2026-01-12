import React, { useState } from 'react';
import axios from 'axios';
import EmotionProfiles from './components/EmotionProfiles';
import ClusterVisualization from './components/ClusterVisualization';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [texts, setTexts] = useState([{ text: '', organization: '' }]);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [clusterResults, setClusterResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedView, setSelectedView] = useState(null); // 'emotions' or 'clusters'

  const addTextInput = () => {
    setTexts([...texts, { text: '', organization: '' }]);
  };

  const removeTextInput = (index) => {
    const newTexts = texts.filter((_, i) => i !== index);
    setTexts(newTexts.length > 0 ? newTexts : [{ text: '', organization: '' }]);
  };

  const updateText = (index, field, value) => {
    const newTexts = [...texts];
    newTexts[index][field] = value;
    setTexts(newTexts);
  };

  const analyzeTexts = async () => {
    setLoading(true);
    setError(null);
    setAnalysisResults(null);
    setClusterResults(null);
    setSelectedView(null);

    try {
      // Filter out empty texts
      const validTexts = texts.filter(t => t.text.trim() !== '');

      if (validTexts.length === 0) {
        setError('Please enter at least one text to analyze');
        setLoading(false);
        return;
      }

      // Analyze texts
      const response = await axios.post(`${API_URL}/analyze`, {
        texts: validTexts
      });

      setAnalysisResults(response.data);
      setLoading(false);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'An error occurred');
      setLoading(false);
    }
  };

  const performClustering = async () => {
    if (!analysisResults || analysisResults.length < 5) {
      setError('Need at least 5 samples for clustering');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API_URL}/cluster`, analysisResults);
      setClusterResults(response.data);
      setSelectedView('clusters');
      setLoading(false);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'An error occurred during clustering');
      setLoading(false);
    }
  };

  const loadSampleData = () => {
    const samples = [
      { text: "I absolutely love this product! It exceeded all my expectations and brought me so much joy.", organization: "Company A" },
      { text: "This is the worst experience I've ever had. I'm extremely disappointed and angry.", organization: "Company A" },
      { text: "The service was okay, nothing special but not terrible either.", organization: "Company B" },
      { text: "I'm really happy with the results! Everything worked perfectly.", organization: "Company B" },
      { text: "I'm worried about the quality. There are some concerns that make me anxious.", organization: "Company C" },
      { text: "What an amazing surprise! I didn't expect it to be this good.", organization: "Company C" },
      { text: "This makes me sad and frustrated. It didn't meet my needs at all.", organization: "Company A" },
      { text: "I'm cautiously optimistic, though there are still some fears about the outcome.", organization: "Company B" },
      { text: "Disgusted by the poor quality and lack of attention to detail.", organization: "Company C" },
      { text: "Everything is great! I'm delighted with the service and support.", organization: "Company A" },
      { text: "I feel neutral about this. It's acceptable but nothing remarkable.", organization: "Company B" },
      { text: "The experience filled me with joy and happiness beyond measure.", organization: "Company C" },
    ];
    setTexts(samples);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Sentiment & Emotion Analysis</h1>
        <p>Analyze sentiment and emotions with advanced clustering and visualization</p>
      </header>

      <div className="container">
        <div className="input-section">
          <div className="section-header">
            <h2>Input Texts</h2>
            <button onClick={loadSampleData} className="btn btn-secondary">
              Load Sample Data
            </button>
          </div>

          {texts.map((item, index) => (
            <div key={index} className="text-input-group">
              <div className="input-row">
                <textarea
                  placeholder="Enter text to analyze..."
                  value={item.text}
                  onChange={(e) => updateText(index, 'text', e.target.value)}
                  rows={3}
                />
                <input
                  type="text"
                  placeholder="Organization (optional)"
                  value={item.organization}
                  onChange={(e) => updateText(index, 'organization', e.target.value)}
                />
                {texts.length > 1 && (
                  <button
                    onClick={() => removeTextInput(index)}
                    className="btn btn-remove"
                    title="Remove"
                  >
                    ×
                  </button>
                )}
              </div>
            </div>
          ))}

          <button onClick={addTextInput} className="btn btn-secondary">
            + Add Another Text
          </button>

          <button
            onClick={analyzeTexts}
            className="btn btn-primary"
            disabled={loading}
          >
            {loading ? 'Analyzing...' : 'Analyze Texts'}
          </button>

          {error && <div className="error">{error}</div>}
        </div>

        {analysisResults && (
          <div className="results-section">
            <h2>Analysis Results ({analysisResults.length} items)</h2>

            <div className="view-selector">
              <button
                onClick={() => setSelectedView('emotions')}
                className={`btn ${selectedView === 'emotions' ? 'btn-primary' : 'btn-secondary'}`}
              >
                Emotion Profiles
              </button>
              <button
                onClick={performClustering}
                className={`btn ${selectedView === 'clusters' ? 'btn-primary' : 'btn-secondary'}`}
                disabled={loading || analysisResults.length < 5}
              >
                {loading ? 'Clustering...' : 'Cluster Analysis'}
              </button>
            </div>

            {selectedView === 'emotions' && (
              <EmotionProfiles analysisResults={analysisResults} />
            )}

            {selectedView === 'clusters' && clusterResults && (
              <ClusterVisualization
                analysisResults={analysisResults}
                clusterResults={clusterResults}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
