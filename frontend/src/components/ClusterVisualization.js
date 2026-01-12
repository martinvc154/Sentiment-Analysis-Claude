import React, { useRef } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';
import './ClusterVisualization.css';

const CLUSTER_COLORS = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
  '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B3D9'
];

function ClusterVisualization({ analysisResults, clusterResults }) {
  const vizRef = useRef(null);

  // Prepare scatter plot data
  const scatterData = analysisResults.map((result, idx) => ({
    x: clusterResults.pca_coordinates[idx][0],
    y: clusterResults.pca_coordinates[idx][1],
    cluster: clusterResults.cluster_assignments[idx],
    text: result.text.substring(0, 50) + (result.text.length > 50 ? '...' : ''),
    organization: result.organization || 'N/A',
    sentiment: result.sentiment_score.toFixed(2)
  }));

  // Prepare centroid data
  const centroidData = clusterResults.centroids.map((centroid, idx) => ({
    x: centroid[0],
    y: centroid[1],
    cluster: idx,
    isCentroid: true
  }));

  // Custom tooltip
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      if (data.isCentroid) {
        return (
          <div className="custom-tooltip">
            <p className="tooltip-title">Cluster {data.cluster} Centroid</p>
          </div>
        );
      }
      return (
        <div className="custom-tooltip">
          <p className="tooltip-title">Cluster {data.cluster}</p>
          <p className="tooltip-text">{data.text}</p>
          <p>Organization: {data.organization}</p>
          <p>Sentiment: {data.sentiment}</p>
        </div>
      );
    }
    return null;
  };

  // Render confidence ellipse as SVG
  const renderEllipse = (ellipse, color) => {
    const { center, width, height, angle } = ellipse;
    const key = `ellipse-${ellipse.cluster_id}`;

    // Note: This is a simplified representation
    // In a production app, you'd want to use proper ellipse rendering
    return (
      <ellipse
        key={key}
        cx={center[0]}
        cy={center[1]}
        rx={width / 2}
        ry={height / 2}
        fill={color}
        fillOpacity={0.1}
        stroke={color}
        strokeWidth={2}
        strokeDasharray="5,5"
        transform={`rotate(${angle} ${center[0]} ${center[1]})`}
      />
    );
  };

  // Export functionality
  const exportAsPNG = async () => {
    if (!vizRef.current) return;

    try {
      const canvas = await html2canvas(vizRef.current, {
        backgroundColor: '#ffffff',
        scale: 2
      });

      const link = document.createElement('a');
      link.download = 'cluster-analysis.png';
      link.href = canvas.toDataURL();
      link.click();
    } catch (error) {
      console.error('Error exporting as PNG:', error);
      alert('Failed to export as PNG');
    }
  };

  const exportAsPDF = async () => {
    if (!vizRef.current) return;

    try {
      const canvas = await html2canvas(vizRef.current, {
        backgroundColor: '#ffffff',
        scale: 2
      });

      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF({
        orientation: 'portrait',
        unit: 'px',
        format: 'a4'
      });

      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (canvas.height * pdfWidth) / canvas.width;

      pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
      pdf.save('cluster-analysis.pdf');
    } catch (error) {
      console.error('Error exporting as PDF:', error);
      alert('Failed to export as PDF');
    }
  };

  const exportAsJSON = () => {
    const data = {
      optimal_k: clusterResults.optimal_k,
      silhouette_score: clusterResults.silhouette_score,
      clusters: clusterResults.cluster_info,
      data_points: analysisResults.map((result, idx) => ({
        text: result.text,
        organization: result.organization,
        sentiment_score: result.sentiment_score,
        emotions: result.emotions,
        cluster: clusterResults.cluster_assignments[idx],
        pca_coordinates: clusterResults.pca_coordinates[idx]
      }))
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'cluster-analysis.json';
    link.click();
  };

  const exportTableAsCSV = () => {
    const headers = ['Cluster ID', 'N', 'Avg Sentiment', 'Dominant Emotion', 'Suggested Label'];
    const rows = clusterResults.cluster_info.map(info => [
      info.cluster_id,
      info.n,
      info.avg_sentiment.toFixed(3),
      info.dominant_emotion,
      info.suggested_label
    ]);

    const csv = [
      headers.join(','),
      ...rows.map(row => row.join(','))
    ].join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'cluster-characteristics.csv';
    link.click();
  };

  return (
    <div className="cluster-visualization" ref={vizRef}>
      <div className="section-header">
        <h3>Cluster Analysis</h3>
        <div className="export-buttons">
          <button onClick={exportAsPNG} className="btn btn-export">Export PNG</button>
          <button onClick={exportAsPDF} className="btn btn-export">Export PDF</button>
          <button onClick={exportAsJSON} className="btn btn-export">Export JSON</button>
        </div>
      </div>

      <div className="cluster-stats">
        <div className="stat-card">
          <span className="stat-label">Optimal K:</span>
          <span className="stat-value">{clusterResults.optimal_k}</span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Silhouette Score:</span>
          <span className="stat-value">{clusterResults.silhouette_score.toFixed(3)}</span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Total Samples:</span>
          <span className="stat-value">{analysisResults.length}</span>
        </div>
      </div>

      <div className="scatter-plot">
        <h4>PCA Visualization (2D Projection)</h4>
        <ResponsiveContainer width="100%" height={500}>
          <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              type="number"
              dataKey="x"
              name="PC1"
              label={{ value: 'Principal Component 1', position: 'insideBottom', offset: -10 }}
            />
            <YAxis
              type="number"
              dataKey="y"
              name="PC2"
              label={{ value: 'Principal Component 2', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />

            {/* Data points */}
            <Scatter
              name="Data Points"
              data={scatterData}
              fill="#8884d8"
            >
              {scatterData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={CLUSTER_COLORS[entry.cluster % CLUSTER_COLORS.length]} />
              ))}
            </Scatter>

            {/* Centroids */}
            <Scatter
              name="Centroids"
              data={centroidData}
              fill="#000000"
              shape="star"
            >
              {centroidData.map((entry, index) => (
                <Cell
                  key={`centroid-${index}`}
                  fill={CLUSTER_COLORS[entry.cluster % CLUSTER_COLORS.length]}
                  stroke="#000"
                  strokeWidth={2}
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>

        <div className="plot-legend">
          <p><strong>⭐</strong> = Cluster Centroid</p>
          <p>95% confidence ellipses shown around each cluster</p>
        </div>
      </div>

      <div className="cluster-table-section">
        <div className="section-header">
          <h4>Cluster Characteristics</h4>
          <button onClick={exportTableAsCSV} className="btn btn-export">Export CSV</button>
        </div>

        <table className="cluster-table">
          <thead>
            <tr>
              <th>Cluster ID</th>
              <th>N</th>
              <th>Avg Sentiment</th>
              <th>Dominant Emotion</th>
              <th>Suggested Label</th>
            </tr>
          </thead>
          <tbody>
            {clusterResults.cluster_info.map(info => (
              <tr key={info.cluster_id}>
                <td>
                  <span
                    className="cluster-badge"
                    style={{ backgroundColor: CLUSTER_COLORS[info.cluster_id % CLUSTER_COLORS.length] }}
                  >
                    {info.cluster_id}
                  </span>
                </td>
                <td>{info.n}</td>
                <td>
                  <span className={`sentiment-score ${info.avg_sentiment > 0 ? 'positive' : info.avg_sentiment < 0 ? 'negative' : 'neutral'}`}>
                    {info.avg_sentiment.toFixed(3)}
                  </span>
                </td>
                <td>
                  <span className="emotion-tag">{info.dominant_emotion}</span>
                </td>
                <td>
                  <span className="label-tag">{info.suggested_label}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        <div className="label-descriptions">
          <h5>Label Descriptions:</h5>
          <ul>
            <li><strong>Aligned Positive:</strong> High positive sentiment combined with joy</li>
            <li><strong>Open Critics:</strong> Negative sentiment with low divergence</li>
            <li><strong>Strategic Adapters:</strong> Neutral sentiment with high divergence and fear</li>
            <li><strong>Preference Falsifiers:</strong> Positive sentiment masking negative emotions</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default ClusterVisualization;
