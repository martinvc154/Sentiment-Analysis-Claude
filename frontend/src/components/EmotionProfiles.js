import React, { useRef } from 'react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';
import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';
import './EmotionProfiles.css';

const EMOTIONS = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust'];

const COLORS = {
  'joy': '#FFD700',
  'sadness': '#4169E1',
  'anger': '#DC143C',
  'fear': '#9370DB',
  'surprise': '#FF8C00',
  'disgust': '#228B22'
};

function EmotionProfiles({ analysisResults }) {
  const profilesRef = useRef(null);

  // Group results by organization
  const groupedByOrg = analysisResults.reduce((acc, result) => {
    const org = result.organization || 'All Data';
    if (!acc[org]) {
      acc[org] = [];
    }
    acc[org].push(result);
    return acc;
  }, {});

  const organizations = Object.keys(groupedByOrg);
  const hasOrganizations = organizations.length > 1 || (organizations.length === 1 && organizations[0] !== 'All Data');

  // Calculate average emotions for an organization
  const calculateAverageEmotions = (results) => {
    const avgEmotions = {};

    EMOTIONS.forEach(emotion => {
      const sum = results.reduce((acc, r) => acc + r.emotions[emotion], 0);
      avgEmotions[emotion] = sum / results.length;
    });

    return avgEmotions;
  };

  // Prepare data for radar chart
  const prepareRadarData = (emotions) => {
    return EMOTIONS.map(emotion => ({
      emotion: emotion.charAt(0).toUpperCase() + emotion.slice(1),
      value: emotions[emotion],
      fullMark: 1
    }));
  };

  // Calculate grid layout
  const getGridLayout = (count) => {
    if (count <= 4) return { cols: 2, rows: 2 };
    if (count <= 9) return { cols: 3, rows: 3 };
    return { cols: 4, rows: Math.ceil(count / 4) };
  };

  const gridLayout = hasOrganizations ? getGridLayout(organizations.length) : null;

  // Export functionality
  const exportAsPNG = async () => {
    if (!profilesRef.current) return;

    try {
      const canvas = await html2canvas(profilesRef.current, {
        backgroundColor: '#ffffff',
        scale: 2
      });

      const link = document.createElement('a');
      link.download = 'emotion-profiles.png';
      link.href = canvas.toDataURL();
      link.click();
    } catch (error) {
      console.error('Error exporting as PNG:', error);
      alert('Failed to export as PNG');
    }
  };

  const exportAsPDF = async () => {
    if (!profilesRef.current) return;

    try {
      const canvas = await html2canvas(profilesRef.current, {
        backgroundColor: '#ffffff',
        scale: 2
      });

      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF({
        orientation: canvas.width > canvas.height ? 'landscape' : 'portrait',
        unit: 'px',
        format: [canvas.width, canvas.height]
      });

      pdf.addImage(imgData, 'PNG', 0, 0, canvas.width, canvas.height);
      pdf.save('emotion-profiles.pdf');
    } catch (error) {
      console.error('Error exporting as PDF:', error);
      alert('Failed to export as PDF');
    }
  };

  const exportAsJSON = () => {
    const data = organizations.map(org => ({
      organization: org,
      emotions: calculateAverageEmotions(groupedByOrg[org]),
      sampleCount: groupedByOrg[org].length
    }));

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'emotion-profiles.json';
    link.click();
  };

  return (
    <div className="emotion-profiles">
      <div className="section-header">
        <h3>Emotion Profiles</h3>
        <div className="export-buttons">
          <button onClick={exportAsPNG} className="btn btn-export">Export PNG</button>
          <button onClick={exportAsPDF} className="btn btn-export">Export PDF</button>
          <button onClick={exportAsJSON} className="btn btn-export">Export JSON</button>
        </div>
      </div>

      <div
        ref={profilesRef}
        className={hasOrganizations ? 'profiles-grid' : 'profiles-single'}
        style={hasOrganizations ? {
          gridTemplateColumns: `repeat(${gridLayout.cols}, 1fr)`,
          gap: '20px'
        } : {}}
      >
        {organizations.map(org => {
          const orgResults = groupedByOrg[org];
          const avgEmotions = calculateAverageEmotions(orgResults);
          const radarData = prepareRadarData(avgEmotions);

          return (
            <div key={org} className="profile-card">
              <h4>{org}</h4>
              <p className="sample-count">n = {orgResults.length}</p>

              <ResponsiveContainer width="100%" height={hasOrganizations ? 250 : 400}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="#ccc" />
                  <PolarAngleAxis
                    dataKey="emotion"
                    tick={{ fill: '#333', fontSize: 12 }}
                  />
                  <PolarRadiusAxis
                    angle={90}
                    domain={[0, 1]}
                    tick={{ fill: '#666', fontSize: 10 }}
                  />
                  <Radar
                    name={org}
                    dataKey="value"
                    stroke="#8884d8"
                    fill="#8884d8"
                    fillOpacity={0.6}
                  />
                </RadarChart>
              </ResponsiveContainer>

              <div className="emotion-values">
                {EMOTIONS.map(emotion => (
                  <div key={emotion} className="emotion-value">
                    <span
                      className="emotion-dot"
                      style={{ backgroundColor: COLORS[emotion] }}
                    ></span>
                    <span className="emotion-name">
                      {emotion.charAt(0).toUpperCase() + emotion.slice(1)}:
                    </span>
                    <span className="emotion-score">
                      {avgEmotions[emotion].toFixed(3)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      <div className="legend">
        <h4>Emotion Scale</h4>
        <p>Each emotion is scored from 0 (not present) to 1 (strongly present)</p>
      </div>
    </div>
  );
}

export default EmotionProfiles;
