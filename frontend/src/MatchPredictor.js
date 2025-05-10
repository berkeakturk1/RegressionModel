import React, { useState, useEffect } from 'react';
import _ from 'lodash';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';
import './App.css';
const API_URL = 'http://127.0.0.1:5000/api';

const MatchPredictor = () => {
  // State for application data
  const [teams, setTeams] = useState([]);
  const [homeTeam, setHomeTeam] = useState('');
  const [awayTeam, setAwayTeam] = useState('');
  const [homeRedCards, setHomeRedCards] = useState(0);
  const [awayRedCards, setAwayRedCards] = useState(0);
  const [predictions, setPredictions] = useState([]);
  const [currentPrediction, setCurrentPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [modelStatus, setModelStatus] = useState('');
  const [error, setError] = useState('');
  
  // State for statistics and visualization
  const [showDetails, setShowDetails] = useState(false);
  const [showModelStats, setShowModelStats] = useState(false);
  const [statisticalData, setStatisticalData] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [teamStats, setTeamStats] = useState(null);
  const [loadingStats, setLoadingStats] = useState(false);
  
  // State for model architecture visualization
  const [featureImportance, setFeatureImportance] = useState(null);
  const [predictionDistribution, setPredictionDistribution] = useState(null);
  
  // Initialize by loading models
  useEffect(() => {
    loadModels();
  }, []);
  
  // Fetch feature importance when model stats are shown
  useEffect(() => {
    if (showModelStats) {
      fetchFeatureImportance();
    }
  }, [showModelStats]);
  
  // Fetch prediction from the Flask API
  const predictMatch = async (home, away, homeRed, awayRed) => {
    setIsLoading(true);
    setError('');
  
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          homeTeam: home,
          awayTeam: away,
          homeRedCards: homeRed,
          awayRedCards: awayRed
        }),
      });
  
      const data = await response.json();
      console.log("[DEBUG] API /predict response:", data);
  
      if (data.success && data.prediction) {
        const prediction = {
          ...data.prediction,
          homeTeam: home,
          awayTeam: away,
          homeRedCards: homeRed,
          awayRedCards: awayRed,
          timestamp: new Date().toLocaleString()
        };
  
        setCurrentPrediction(prediction);
        setPredictions(prev => [prediction, ...prev]);
        fetchStatisticalData(home, away);
      } else {
        throw new Error(data.message || 'Prediction failed.');
      }
    } catch (err) {
      console.error("[ERROR] predictMatch:", err);
      setError(err.message || 'Prediction error');
    } finally {
      setIsLoading(false);
    }
  };
  
  // Fetch statistical data
  const fetchStatisticalData = async (homeTeam, awayTeam) => {
    setLoadingStats(true);
    
    try {
      const response = await fetch(`${API_URL}/stats`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          homeTeam,
          awayTeam
        }),
      });
      
      const data = await response.json();
      
      if (data.success) {
        setStatisticalData(data.modelStats);
        setTeamStats(data.teamStats);
      } else {
        console.error('Failed to fetch statistical data');
      }
    } catch (err) {
      console.error('Error fetching statistical data:', err);
    } finally {
      setLoadingStats(false);
    }
  };

  // Fetch model info
  const fetchModelInfo = async () => {
    try {
      const response = await fetch(`${API_URL}/model_info`);
      const data = await response.json();
      
      if (data.success) {
        setModelInfo(data);
      }
    } catch (err) {
      console.error('Error fetching model info:', err);
    }
  };
  
  // Fetch feature importance
  const fetchFeatureImportance = async () => {
    try {
      const response = await fetch(`${API_URL}/feature_importance`);
      const data = await response.json();
      if (data.success) {
        setFeatureImportance(data.featureImportance);
      }
    } catch (err) {
      console.error('Error fetching feature importance:', err);
    }
  };
  
  // Fetch prediction distribution
  const fetchPredictionDistribution = async () => {
    if (!homeTeam || !awayTeam) return;
    
    try {
      const response = await fetch(`${API_URL}/prediction_distribution`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          homeTeam,
          awayTeam,
          homeRedCards,
          awayRedCards
        }),
      });
      
      const data = await response.json();
      if (data.success) {
        setPredictionDistribution(data);
      }
    } catch (err) {
      console.error('Error fetching prediction distribution:', err);
    }
  };
  
  // Update model
  const updateModel = async () => {
    setModelStatus('Updating model...');
    setError('');
    
    try {
        const response = await fetch(`${API_URL}/train`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                num_seasons: 3  // Default to 3 seasons
            }),
        });
        
        const data = await response.json();
        
        if (data.success) {
            setModelStatus('Model updated successfully!');
            setTeams(data.teams);
            
            if (data.teams.length >= 2) {
                setHomeTeam(data.teams[0]);
                setAwayTeam(data.teams[1]);
            }
            
            // Refresh model info
            fetchModelInfo();
        } else {
            setError(data.message || 'Failed to update model');
            setModelStatus('Update failed');
        }
    } catch (err) {
        setError('Network error: ' + err.message);
        setModelStatus('Update failed');
    }
};
  
  // Load existing models and team data
  const loadModels = async () => {
    setModelStatus('Loading models...');
    setError('');
    
    try {
      const response = await fetch(`${API_URL}/load_models`);
      const data = await response.json();
      
      if (data.success) {
        setModelStatus('Models loaded successfully!');
        setTeams(data.teams);
        
        // Set default selections
        if (data.teams.length >= 2) {
          setHomeTeam(data.teams[0]);
          setAwayTeam(data.teams[1]);
        }
        
        // Fetch model info
        fetchModelInfo();
      } else {
        fetchTeams(); // Try to at least get teams list
        setError(data.message || 'Failed to load models');
        setModelStatus('No models loaded');
      }
    } catch (err) {
      fetchTeams(); // Try to at least get teams list
      setError('Network error: ' + err.message);
      setModelStatus('No models loaded');
    }
  };
  
  // Fetch teams if models are already trained
  const fetchTeams = async () => {
    try {
      const response = await fetch(`${API_URL}/teams`);
      const data = await response.json();
      
      if (data.success) {
        setTeams(data.teams);
        
        // Set default selections
        if (data.teams.length >= 2) {
          setHomeTeam(data.teams[0]);
          setAwayTeam(data.teams[1]);
        }
        
        setModelStatus('Models ready for prediction');
      }
    } catch (err) {
      console.error('Error fetching teams:', err);
    }
  };
  
  // Handle prediction button click
  const handlePredict = () => {
    if (homeTeam && awayTeam) {
      predictMatch(homeTeam, awayTeam, homeRedCards, awayRedCards);
    }
  };
  
  // Handle clearing predictions
  const handleClearPredictions = () => {
    setPredictions([]);
    setCurrentPrediction(null);
    setStatisticalData(null);
    setTeamStats(null);
    setShowDetails(false);
  };
  
  // Helper function to determine result color
  const getResultColor = (home, away) => {
    if (home > away) return 'text-green-600';
    if (away > home) return 'text-red-600';
    return 'text-yellow-500';
  };
  
  // Helper function to determine match result text
  const getResultText = (home, away) => {
    if (home > away) return 'Home Win';
    if (away > home) return 'Away Win';
    return 'Draw';
  };
  
  // RENDER FUNCTIONS - All rendering functions defined here
  
  // Generate performance metrics component
  const renderPerformanceMetrics = () => {
    if (!statisticalData?.metrics) return null;
    
    const metrics = statisticalData.metrics;
    
    return (
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-blue-50 p-4 rounded-lg shadow-sm">
          <h3 className="font-semibold text-blue-800 mb-1">MSE (Mean Squared Error)</h3>
          <div className="flex justify-between">
            <div>Home: <span className="font-bold">{metrics.homeGoalsMSE.toFixed(2)}</span></div>
            <div>Away: <span className="font-bold">{metrics.awayGoalsMSE.toFixed(2)}</span></div>
          </div>
          <p className="text-xs text-gray-600 mt-1">Lower is better - measures average squared prediction error</p>
        </div>
        
        <div className="bg-green-50 p-4 rounded-lg shadow-sm">
          <h3 className="font-semibold text-green-800 mb-1">MAE (Mean Absolute Error)</h3>
          <div className="flex justify-between">
            <div>Home: <span className="font-bold">{metrics.homeGoalsMAE.toFixed(2)}</span></div>
            <div>Away: <span className="font-bold">{metrics.awayGoalsMAE.toFixed(2)}</span></div>
          </div>
          <p className="text-xs text-gray-600 mt-1">Average goal prediction error</p>
        </div>
        
        <div className="bg-purple-50 p-4 rounded-lg shadow-sm">
          <h3 className="font-semibold text-purple-800 mb-1">R² Score</h3>
          <div className="flex justify-between">
            <div>Home: <span className="font-bold">{metrics.homeGoalsR2.toFixed(2)}</span></div>
            <div>Away: <span className="font-bold">{metrics.awayGoalsR2.toFixed(2)}</span></div>
          </div>
          <p className="text-xs text-gray-600 mt-1">1.0 is perfect - measures prediction accuracy</p>
        </div>
      </div>
    );
  };
  
  // Render regression graph
  const renderRegressionGraph = () => {
    if (!statisticalData?.regressionData) return null;
    
    const regressionData = statisticalData.regressionData;
    
    // Find the min and max values
    const allPredicted = regressionData.map(p => p.predicted);
    const allActual = regressionData.map(p => p.actual);

    const minValue = Math.floor(Math.min(...allPredicted, ...allActual));
    const maxValue = Math.ceil(Math.max(...allPredicted, ...allActual));

    // Add some padding
    const axisMin = Math.max(0, minValue - 1);
    const axisMax = maxValue + 1;

    return (
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-3">Prediction Accuracy</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid />
              <XAxis
                type="number"
                dataKey="predicted"
                name="Predicted"
                unit=" goals"
                domain={[axisMin, axisMax]}
              />
              <YAxis
                type="number"
                dataKey="actual"
                name="Actual"
                unit=" goals"
                domain={[axisMin, axisMax]}
              />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Legend />
              
              {/* Predicted vs Actual scatter points */}
              <Scatter name="Goals" data={regressionData} fill="#8884d8" />
              
              {/* Perfect line: Predicted == Actual */}
              <Scatter
                name="Perfect Prediction"
                data={[
                  { predicted: 0, actual: 0 },
                  { predicted: 5, actual: 5 }
                ]}
                line={{ stroke: '#ff7300', strokeWidth: 2 }}
                shape="none"
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div className="text-sm text-gray-600 text-center">
          Perfect predictions would fall on the orange line. Scatter points show actual vs predicted goals.
        </div>
      </div>
    );
  };
  
  // Render team comparison
  const renderTeamComparison = () => {
    if (!teamStats) return null;
    
    const homeTeamData = teamStats.homeTeam;
    const awayTeamData = teamStats.awayTeam;
    
    const comparisonData = [
      {
        name: 'Goals For',
        home: homeTeamData.avgHomeGoals,
        away: awayTeamData.avgAwayGoals,
      },
      {
        name: 'Goals Against',
        home: homeTeamData.avgHomeGoalsAgainst,
        away: awayTeamData.avgAwayGoalsAgainst,
      },
      {
        name: 'Win Rate',
        home: homeTeamData.homeWinRate * 100,
        away: awayTeamData.awayWinRate * 100,
      },
      {
        name: 'Draw Rate',
        home: homeTeamData.homeDrawRate * 100,
        away: awayTeamData.awayDrawRate * 100,
      },
      {
        name: 'Loss Rate',
        home: homeTeamData.homeLossRate * 100,
        away: awayTeamData.awayLossRate * 100,
      }
    ];
    
    return (
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-3">Team Comparison</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={comparisonData}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar name={`${homeTeamData.name} (Home)`} dataKey="home" fill="#3182ce" />
              <Bar name={`${awayTeamData.name} (Away)`} dataKey="away" fill="#e53e3e" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    );
  };
  
  // Feature importance visualization
  const renderFeatureImportance = () => {
    if (!featureImportance) {
      return (
        <div className="bg-white p-4 rounded-lg shadow mb-6">
          <h4 className="font-medium mb-2">Feature Importance</h4>
          <div className="p-4 text-center text-gray-500">
            No feature importance data available. Feature importance shows which factors most strongly influence the model's predictions.
          </div>
          <button
            className="mt-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            onClick={fetchFeatureImportance}
          >
            Load Feature Importance
          </button>
        </div>
      );
    }
    
    return (
      <div className="bg-white p-4 rounded-lg shadow mb-6">
        <h4 className="font-medium mb-2">Feature Importance</h4>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={featureImportance}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 150, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis type="category" dataKey="feature" width={150} />
              <Tooltip formatter={(value) => value.toFixed(4)} />
              <Bar dataKey="importance" fill="#8884d8" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="text-sm text-gray-600 mt-2">
          This shows which features have the strongest influence on the model's predictions.
          Higher values indicate greater importance in determining goal predictions.
        </p>
      </div>
    );
  };
  
  // Prediction distribution visualization
  const renderPredictionDistribution = () => {
    if (!predictionDistribution) return null;
    
    const homeData = predictionDistribution.homeDistribution;
    const awayData = predictionDistribution.awayDistribution;
    
    return (
      <div className="bg-white p-4 rounded-lg shadow mb-6">
        <h4 className="font-medium mb-2">Prediction Distribution Across Trees</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h5 className="text-sm font-medium text-center mb-2">Home Goals Distribution</h5>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={homeData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 25 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="goals"
                    label={{ value: 'Predicted Goals', position: 'bottom' }}
                  />
                  <YAxis label={{ value: 'Number of Trees', angle: -90, position: 'left' }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#3182ce" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="text-xs text-center text-gray-600">
              Average: {predictionDistribution.homeAverage?.toFixed(2) || 'N/A'} goals
            </p>
          </div>
          
          <div>
            <h5 className="text-sm font-medium text-center mb-2">Away Goals Distribution</h5>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={awayData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 25 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="goals"
                    label={{ value: 'Predicted Goals', position: 'bottom' }}
                  />
                  <YAxis label={{ value: 'Number of Trees', angle: -90, position: 'left' }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#e53e3e" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="text-xs text-center text-gray-600">
              Average: {predictionDistribution.awayAverage?.toFixed(2) || 'N/A'} goals
            </p>
          </div>
        </div>
        <p className="text-sm text-gray-600 mt-2">
          These charts show how the individual decision trees in the Random Forest voted.
          The final prediction is the average of all these votes (rounded to the nearest integer).
        </p>
      </div>
    );
  };
  
  // Render model architecture section
  const renderModelArchitecture = () => {
    return (
      <div className="mb-6">
        <h3 className="text-xl font-semibold mb-4">Random Forest Model Architecture</h3>
        
        {renderFeatureImportance()}
        
        <div className="flex justify-center mb-6">
          <button
            className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-6 rounded-md"
            onClick={fetchPredictionDistribution}
            disabled={!homeTeam || !awayTeam}
          >
            Show Prediction Distribution for Current Match
          </button>
        </div>
        
        {renderPredictionDistribution()}
        
        <div className="p-4 bg-blue-50 text-blue-800 rounded-md">
          <p className="text-sm">
            <strong>How Random Forest Works:</strong> Unlike linear regression which fits a single line through data,
            Random Forest builds 100 different decision trees, each trained on a random subset of the data.
            Each tree makes its own prediction, and the final prediction is the average of all trees.
            This ensemble approach captures complex patterns and non-linear relationships that a single model cannot.
          </p>
        </div>
      </div>
    );
  };
  
  // Main component render
  return (
    <div className="mx-auto max-w-6xl p-4">
      <h1 className="text-3xl font-bold text-center mb-6">Turkish Super League Match Predictor</h1>
      
      {/* Model Status Card */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold">Model Status</h2>
          <button
            className={`flex items-center px-4 py-2 rounded-md text-sm font-medium ${
              showModelStats ? 'bg-gray-200 text-gray-700' : 'bg-blue-100 text-blue-700'
            }`}
            onClick={() => setShowModelStats(!showModelStats)}
          >
            {showModelStats ? 'Hide Model Statistics' : 'Show Model Statistics'}
            <svg
              className={`ml-1 h-4 w-4 transition-transform ${showModelStats ? 'transform rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-semibold text-gray-700 mb-1">Status</h3>
            <p className="text-gray-800">{modelStatus || 'Unknown'}</p>
          </div>
          
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-semibold text-gray-700 mb-1">Teams</h3>
            <p className="text-gray-800">{teams.length} teams available</p>
          </div>
          
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-semibold text-gray-700 mb-1">Dataset Size</h3>
            <p className="text-gray-800">{modelInfo?.datasetRows || 'Unknown'} matches</p>
          </div>
        </div>
        
        <div className="flex justify-center">
          <button
            className="bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-2 px-6 rounded-md"
            onClick={updateModel}
            disabled={isLoading}
          >
            Update Model
          </button>
        </div>
        
        {/* Model Statistics Section */}
        {showModelStats && (
          <div className="mt-6 border-t pt-6">
            <h3 className="text-xl font-semibold mb-4">Model Performance Statistics</h3>
            
            {loadingStats ? (
              <div className="text-center py-8">
                <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-blue-600 border-t-transparent"></div>
                <div className="mt-2 text-gray-600">Loading statistical data...</div>
              </div>
            ) : (
              <>
                {renderPerformanceMetrics()}
                {renderRegressionGraph()}
                {renderModelArchitecture()}
              </>
            )}
          </div>
        )}
      </div>
      
      {/* Match Prediction Form */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">Predict Match Result</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Team Selection */}
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">Home Team</label>
              <select
                className="w-full p-2 border rounded-md"
                value={homeTeam}
                onChange={(e) => setHomeTeam(e.target.value)}
              >
                <option value="">Select a team</option>
                {teams.map(team => (
                  <option key={`home-${team}`} value={team}>{team}</option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">Away Team</label>
              <select
                className="w-full p-2 border rounded-md"
                value={awayTeam}
                onChange={(e) => setAwayTeam(e.target.value)}
              >
                <option value="">Select a team</option>
                {teams.map(team => (
                  <option key={`away-${team}`} value={team}>{team}</option>
                ))}
              </select>
            </div>
          </div>
          
          {/* Red Card Selection */}
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">Home Team Red Cards</label>
              <select
                className="w-full p-2 border rounded-md"
                value={homeRedCards}
                onChange={(e) => setHomeRedCards(parseInt(e.target.value))}
              >
                {[0, 1, 2, 3].map(num => (
                  <option key={`home-red-${num}`} value={num}>{num}</option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">Away Team Red Cards</label>
              <select
                className="w-full p-2 border rounded-md"
                value={awayRedCards}
                onChange={(e) => setAwayRedCards(parseInt(e.target.value))}
              >
                {[0, 1, 2, 3].map(num => (
                  <option key={`away-red-${num}`} value={num}>{num}</option>
                ))}
              </select>
            </div>
          </div>
        </div>
        
        <div className="mt-6 flex flex-col items-center">
          {error && (
            <div className="mb-4 p-3 bg-red-100 text-red-700 rounded-md w-full max-w-md text-center">
              {error}
            </div>
          )}
          
          <button
            className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-6 rounded-md"
            onClick={handlePredict}
            disabled={isLoading || !homeTeam || !awayTeam}
          >
            {isLoading ? 'Predicting...' : 'Predict Result'}
          </button>
        </div>
      </div>
      
      {/* Current Prediction */}
      {currentPrediction && (
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Latest Prediction</h2>
          
          <div className="flex items-center justify-center mb-6">
            <div className="text-center w-2/5">
              <div className="font-bold text-lg">{currentPrediction.homeTeam}</div>
              <div className="text-xs text-gray-500">Home Team</div>
              {currentPrediction.homeRedCards > 0 && (
                <div className="mt-1">
                  <span className="inline-block bg-red-600 text-white text-xs px-2 py-1 rounded">
                    {currentPrediction.homeRedCards} Red Card{currentPrediction.homeRedCards > 1 ? 's' : ''}
                  </span>
                </div>
              )}
            </div>
            
            <div className="w-1/5 text-center">
              <div className="text-3xl font-bold">
                <span className={getResultColor(currentPrediction.predictedHomeGoals, currentPrediction.predictedAwayGoals)}>
                  {currentPrediction.predictedHomeGoals} - {currentPrediction.predictedAwayGoals}
               </span>
             </div>
             <div className="text-xs text-gray-500 mt-1">
               {getResultText(currentPrediction.predictedHomeGoals, currentPrediction.predictedAwayGoals)}
             </div>
           </div>
           
           <div className="text-center w-2/5">
             <div className="font-bold text-lg">{currentPrediction.awayTeam}</div>
             <div className="text-xs text-gray-500">Away Team</div>
             {currentPrediction.awayRedCards > 0 && (
               <div className="mt-1">
                 <span className="inline-block bg-red-600 text-white text-xs px-2 py-1 rounded">
                   {currentPrediction.awayRedCards} Red Card{currentPrediction.awayRedCards > 1 ? 's' : ''}
                 </span>
               </div>
             )}
           </div>
         </div>
         
         {/* Show/Hide Match Details Button */}
         <div className="text-center">
           <button
             className={`flex items-center mx-auto px-4 py-2 rounded-md text-sm font-medium ${
               showDetails ? 'bg-gray-200 text-gray-700' : 'bg-blue-100 text-blue-700'
             }`}
             onClick={() => setShowDetails(!showDetails)}
           >
             {showDetails ? 'Hide Match Details' : 'Show Match Details'}
             <svg
               className={`ml-1 h-4 w-4 transition-transform ${showDetails ? 'transform rotate-180' : ''}`}
               fill="none"
               stroke="currentColor"
               viewBox="0 0 24 24"
             >
               <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
             </svg>
           </button>
         </div>
         
         {/* Match-Specific Details */}
         {showDetails && (
           <div className="mt-6 border-t pt-6">
             <h3 className="text-xl font-semibold mb-4">Match Analysis</h3>
             
             {loadingStats ? (
               <div className="text-center py-8">
                 <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-blue-600 border-t-transparent"></div>
                 <div className="mt-2 text-gray-600">Loading match data...</div>
               </div>
             ) : (
               <>
                 {/* Only the team comparison stays in match details */}
                 {renderTeamComparison() || (
                   <div className="text-center text-gray-500 py-8">
                     No historical data available for this team matchup
                   </div>
                 )}
               </>
             )}
           </div>
         )}
       </div>
     )}
     
     {/* Prediction History */}
     {predictions.length > 0 && (
       <div className="bg-white rounded-lg shadow-md p-6">
         <div className="flex justify-between items-center mb-4">
           <h2 className="text-xl font-semibold">Prediction History</h2>
           <button 
             className="text-sm text-red-600 hover:text-red-800"
             onClick={handleClearPredictions}
           >
             Clear All
           </button>
         </div>
         
         <div className="overflow-x-auto">
           <table className="w-full border-collapse">
             <thead>
               <tr className="bg-gray-100">
                 <th className="py-2 px-3 text-left">Home Team</th>
                 <th className="py-2 px-3 text-center">Red Cards</th>
                 <th className="py-2 px-3 text-center">Score</th>
                 <th className="py-2 px-3 text-center">Red Cards</th>
                 <th className="py-2 px-3 text-right">Away Team</th>
                 <th className="py-2 px-3 text-center">Result</th>
                 <th className="py-2 px-3 text-right">Time</th>
               </tr>
             </thead>
             <tbody>
               {predictions.map((pred, index) => (
                 <tr key={index} className="border-t hover:bg-gray-50">
                   <td className="py-2 px-3 text-left font-medium">{pred.homeTeam}</td>
                   <td className="py-2 px-3 text-center">
                     {pred.homeRedCards > 0 ? (
                       <span className="inline-block bg-red-600 text-white text-xs px-2 py-1 rounded">
                         {pred.homeRedCards}
                       </span>
                     ) : '0'}
                   </td>
                   <td className="py-2 px-3 text-center font-bold">
                     <span className={getResultColor(pred.predictedHomeGoals, pred.predictedAwayGoals)}>
                       {pred.predictedHomeGoals} - {pred.predictedAwayGoals}
                     </span>
                   </td>
                   <td className="py-2 px-3 text-center">
                     {pred.awayRedCards > 0 ? (
                       <span className="inline-block bg-red-600 text-white text-xs px-2 py-1 rounded">
                         {pred.awayRedCards}
                       </span>
                     ) : '0'}
                   </td>
                   <td className="py-2 px-3 text-right font-medium">{pred.awayTeam}</td>
                   <td className="py-2 px-3 text-center">
                     <span className={`inline-block text-xs px-2 py-1 rounded ${
                       getResultColor(pred.predictedHomeGoals, pred.predictedAwayGoals) === 'text-green-600'
                       ? 'bg-green-100 text-green-800'
                       : getResultColor(pred.predictedHomeGoals, pred.predictedAwayGoals) === 'text-red-600'
                       ? 'bg-red-100 text-red-800'
                       : 'bg-yellow-100 text-yellow-800'
                     }`}>
                       {getResultText(pred.predictedHomeGoals, pred.predictedAwayGoals)}
                     </span>
                   </td>
                   <td className="py-2 px-3 text-right text-xs text-gray-500">{pred.timestamp}</td>
                 </tr>
               ))}
             </tbody>
           </table>
         </div>
       </div>
     )}
     
     <div className="mt-6 text-center text-sm text-gray-500">
       Turkish Super League Match Predictor - Random Forest Regression Model
     </div>
   </div>
 );
};

export default MatchPredictor;