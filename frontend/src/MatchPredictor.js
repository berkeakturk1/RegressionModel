import React, { useState, useEffect } from 'react';
import _ from 'lodash';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';
import './App.css';
const API_URL = 'https://regressionmodel.duckdns.org/api';

const MatchPredictor = () => {
  // Original state
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
  
  // New state for statistics and visualization
  const [showDetails, setShowDetails] = useState(false);
  const [statisticalData, setStatisticalData] = useState(null);
  const [teamStats, setTeamStats] = useState(null);
  const [loadingStats, setLoadingStats] = useState(false);
  
  // Fetch prediction from the Flask API
  const predictMatch = async (home, away, homeRed, awayRed) => {
    setIsLoading(true);
    setError('');
    
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          homeTeam: home,
          awayTeam: away,
          homeRedCards: homeRed,
          awayRedCards: awayRed
        }),
      });
      console.log("[FETCH] Predicting match", home, "vs", away, "with red cards", homeRed, awayRed);
      const data = await response.json();
      
      if (data.success) {
        const prediction = {
          ...data.prediction,
          timestamp: new Date().toLocaleString()
        };
        
        setCurrentPrediction(prediction);
        setPredictions(prev => [prediction, ...prev]);
        
        // When we have a prediction, fetch the statistical data
        if (data.prediction) {
          fetchStatisticalData(home, away);
        }
      } else {
        setError(data.error || 'Failed to get prediction');
      }
    } catch (err) {
      setError('Network error: ' + err.message);
      console.error("[ERROR] Prediction fetch failed:", err);
    } finally {
      setIsLoading(false);
    }
  };
  
  // New function to fetch statistical data
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
  
  // Train the model
  const trainModel = async () => {
    setModelStatus('Training model...');
    setError('');
    
    try {
      const response = await fetch(`${API_URL}/train`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        // ❗ No body!
      });
      
      const data = await response.json();
      
      if (data.success) {
        setModelStatus('Model trained successfully!');
        setTeams(data.teams);
        
        if (data.teams.length >= 2) {
          setHomeTeam(data.teams[0]);
          setAwayTeam(data.teams[1]);
        }
      } else {
        setError(data.error || 'Failed to train model');
        setModelStatus('Training failed');
      }
    } catch (err) {
      setError('Network error: ' + err.message);
      setModelStatus('Training failed');
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
      } else {
        fetchTeams(); // Try to at least get teams list
        setError(data.error || 'Failed to load models');
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
  
  // Initialize by loading models
  useEffect(() => {
    loadModels();
  }, []);
  
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
  
  // Mock data for development - in a real app this would come from the API
  const sampleRegressionData = [
    { predicted: 1, actual: 1.2, index: 1 },
    { predicted: 2, actual: 1.8, index: 2 },
    { predicted: 0, actual: 0.3, index: 3 },
    { predicted: 3, actual: 2.7, index: 4 },
    { predicted: 1, actual: 1.4, index: 5 },
    { predicted: 2, actual: 2.1, index: 6 },
    { predicted: 4, actual: 3.6, index: 7 },
    { predicted: 0, actual: 0.2, index: 8 },
    { predicted: 3, actual: 3.3, index: 9 },
    { predicted: 2, actual: 1.7, index: 10 },
  ];
  
  const sampleTeamGoalAvg = [
    { team: 'Galatasaray', homeGoals: 2.3, awayGoals: 1.5 },
    { team: 'Fenerbahçe', homeGoals: 2.1, awayGoals: 1.7 },
    { team: 'Beşiktaş', homeGoals: 1.9, awayGoals: 1.2 },
    { team: 'Trabzonspor', homeGoals: 1.8, awayGoals: 1.0 },
    { team: 'İstanbul Başakşehir', homeGoals: 1.5, awayGoals: 1.1 },
  ];
  
  // Generate performance metrics component
  const renderPerformanceMetrics = () => {
    // In a real app, this would use actual data from API
    // Using placeholder data for now
    const metrics = statisticalData?.metrics || {
      homeGoalsMSE: 0.82,
      awayGoalsMSE: 0.75,
      homeGoalsMAE: 0.64,
      awayGoalsMAE: 0.58,
      homeGoalsR2: 0.56,
      awayGoalsR2: 0.61
    };
    
    return (
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-blue-50 p-4 rounded-lg shadow-sm">
          <h3 className="font-semibold text-blue-800 mb-1">MSE (Mean Squared Error)</h3>
          <div className="flex justify-between">
            <div>Home: <span className="font-bold">{metrics.homeGoalsMSE.toFixed(2)}</span></div>
            <div>Away: <span className="font-bold">{metrics.awayGoalsMSE.toFixed(2)}</span></div>
          </div>
        </div>
        
        <div className="bg-green-50 p-4 rounded-lg shadow-sm">
          <h3 className="font-semibold text-green-800 mb-1">MAE (Mean Absolute Error)</h3>
          <div className="flex justify-between">
            <div>Home: <span className="font-bold">{metrics.homeGoalsMAE.toFixed(2)}</span></div>
            <div>Away: <span className="font-bold">{metrics.awayGoalsMAE.toFixed(2)}</span></div>
          </div>
        </div>
        
        <div className="bg-purple-50 p-4 rounded-lg shadow-sm">
          <h3 className="font-semibold text-purple-800 mb-1">R² Score</h3>
          <div className="flex justify-between">
            <div>Home: <span className="font-bold">{metrics.homeGoalsR2.toFixed(2)}</span></div>
            <div>Away: <span className="font-bold">{metrics.awayGoalsR2.toFixed(2)}</span></div>
          </div>
        </div>
      </div>
    );
  };
  
  // Render team comparison
  const renderTeamComparison = () => {
    const data = teamStats || {
      homeTeam: {
        name: homeTeam,
        avgHomeGoals: 1.8,
        avgHomeGoalsAgainst: 0.9,
        homeWinRate: 0.65,
        homeDrawRate: 0.15,
        homeLossRate: 0.2
      },
      awayTeam: {
        name: awayTeam,
        avgAwayGoals: 1.2,
        avgAwayGoalsAgainst: 1.5,
        awayWinRate: 0.35,
        awayDrawRate: 0.25,
        awayLossRate: 0.4
      }
    };
    
    const homeTeamData = data.homeTeam;
    const awayTeamData = data.awayTeam;
    
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
  
  // Render regression graph
  const renderRegressionGraph = () => {
    const regressionData = statisticalData?.regressionData || sampleRegressionData;
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
          <ScatterChart
  margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
>
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
  
  // Render team goal statistics
  const renderTeamGoalStats = () => {
    const data = statisticalData?.teamStats || sampleTeamGoalAvg;
    
    return (
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-3">Team Goal Statistics</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={data}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="team" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar name="Home Goals Avg" dataKey="homeGoals" fill="#3182ce" />
              <Bar name="Away Goals Avg" dataKey="awayGoals" fill="#ed64a6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    );
  };
  
  return (
    <div className="mx-auto max-w-6xl p-4">
      <h1 className="text-3xl font-bold text-center mb-6">Turkish Super League Match Predictor</h1>
      
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
          
          {modelStatus && (
            <div className="mb-4 text-sm font-medium text-gray-600">
              Model Status: {modelStatus}
            </div>
          )}
          
          <div className="flex space-x-4">
            <button
              className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-6 rounded-md"
              onClick={handlePredict}
              disabled={isLoading}
            >
              {isLoading ? 'Predicting...' : 'Predict Result'}
            </button>
            
            <button
              className="bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-6 rounded-md"
              onClick={trainModel}
              disabled={isLoading}
            >
              Train Model
            </button>
          </div>
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
          
          {/* Show/Hide Details Button */}
          <div className="text-center">
            <button
              className={`flex items-center mx-auto px-4 py-2 rounded-md text-sm font-medium ${
                showDetails ? 'bg-gray-200 text-gray-700' : 'bg-blue-100 text-blue-700'
              }`}
              onClick={() => setShowDetails(!showDetails)}
            >
              {showDetails ? 'Hide Details' : 'Show Details'}
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
          
          {/* Statistical Details */}
          {showDetails && (
            <div className="mt-6 border-t pt-6">
              <h3 className="text-xl font-semibold mb-4">Statistical Analysis</h3>
              
              {loadingStats ? (
                <div className="text-center py-8">
                  <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-blue-600 border-t-transparent"></div>
                  <div className="mt-2 text-gray-600">Loading statistical data...</div>
                </div>
              ) : (
                <>
                  {renderPerformanceMetrics()}
                  {renderTeamComparison()}
                  {renderRegressionGraph()}
                  {renderTeamGoalStats()}
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
        
      </div>
    </div>
  );
};

export default MatchPredictor;