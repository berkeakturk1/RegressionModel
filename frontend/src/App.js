import React from 'react';
import './App.css';
import MatchPredictor from './MatchPredictor';
import TurkishCupSimulator from './TurkishCupSimulator';

function App() {
  const [activeTab, setActiveTab] = React.useState('predictor');

  return (
    <div className="App min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-blue-700 text-white shadow-md">
        <div className="container mx-auto px-4 py-6">
          <h1 className="text-4xl font-bold text-center">Turkish Soccer Analysis</h1>
          <p className="text-center mt-2 text-blue-100">
            Match predictions and tournament simulations for Turkish soccer
          </p>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="bg-white shadow-md">
        <div className="container mx-auto px-4">
          <div className="flex space-x-8">
            <button
              className={`px-4 py-4 font-medium border-b-2 transition-colors ${
                activeTab === 'predictor' 
                  ? 'border-blue-600 text-blue-600' 
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
              onClick={() => setActiveTab('predictor')}
            >
              Match Predictor
            </button>
            <button
              className={`px-4 py-4 font-medium border-b-2 transition-colors ${
                activeTab === 'tournament' 
                  ? 'border-blue-600 text-blue-600' 
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
              onClick={() => setActiveTab('tournament')}
            >
              Turkish Cup Simulator
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="container mx-auto px-4 py-8">
        {activeTab === 'predictor' && <MatchPredictor />}
        {activeTab === 'tournament' && <TurkishCupSimulator />}
      </div>

      {/* Footer */}
      <footer className="bg-gray-800 text-white py-6">
        <div className="container mx-auto px-4">
          <div className="text-center">
            <p className="text-gray-300">
              Turkish Soccer Analysis Tool &copy; {new Date().getFullYear()}
            </p>
            <p className="text-gray-400 text-sm mt-2">
              All match predictions and tournament simulations are for entertainment purposes only.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;