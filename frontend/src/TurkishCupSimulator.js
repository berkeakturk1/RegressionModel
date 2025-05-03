import React, { useState, useEffect } from 'react';
import _ from 'lodash';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const API_URL = 'https://regressionmodel.duckdns.org/api';//'http://ec2-16-171-14-181.eu-north-1.compute.amazonaws.com:5000/api';

// Turkish team names
const ALL_TEAMS = [
  "Galatasaray", "Fenerbahçe", "Beşiktaş", "Trabzonspor",
  "Başakşehir", "Adana Demirspor", "Konyaspor", "Alanyaspor",
  "Antalyaspor", "Kayserispor", "Sivasspor", "Gaziantep FK",
  "Kasımpaşa", "Hatayspor", "Giresunspor", "Göztepe"
];

// Colors for teams
const TEAM_COLORS = {
  "Galatasaray": "#FDB912",
  "Fenerbahçe": "#FFED00",
  "Beşiktaş": "#000000",
  "Trabzonspor": "#841E31",
  "Default": "#3182CE"
};

const TurkishCupSimulator = () => {
  const [teams, setTeams] = useState([]);
  const [groupStage, setGroupStage] = useState(null);
  const [knockoutStage, setKnockoutStage] = useState(null);
  const [champions, setChampions] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('groups');
  const [selectedMatchDetails, setSelectedMatchDetails] = useState(null);
  const [simulationComplete, setSimulationComplete] = useState(false);
  const [topScorers, setTopScorers] = useState([]);

  // Tournament structure
  const tournamentStructure = {
    groupStage: {
      groups: 4,  // Groups A, B, C, D
      teamsPerGroup: 4, // 4 teams per group
      advancePerGroup: 2 // Top 2 teams advance to knockout
    },
    knockoutStage: {
      rounds: ["Quarter-Finals", "Semi-Finals", "Final"]
    }
  };

  // Fetch teams from API or use defaults
  useEffect(() => {
    const fetchTeams = async () => {
      try {
        const response = await fetch(`${API_URL}/teams`);
        const data = await response.json();
        
        if (data.success && data.teams.length > 0) {
          // Make sure big teams are included
          const mustHaveTeams = ["Galatasaray", "Fenerbahçe", "Beşiktaş", "Trabzonspor"];
          
          // Shuffle remaining teams
          const otherTeams = _.shuffle(data.teams.filter(team => !mustHaveTeams.includes(team)));
          
          // Select enough other teams to reach 16 teams total
          const selectedTeams = [...mustHaveTeams, ...otherTeams.slice(0, 12)];
          
          setTeams(selectedTeams);
        } else {
          setTeams(ALL_TEAMS);
        }
      } catch (err) {
        console.error('Error fetching teams:', err);
        setTeams(ALL_TEAMS);
      }
    };
    
    fetchTeams();
  }, []);

  // Helper function to predict match with small randomness
  const predictMatch = async (homeTeam, awayTeam) => {
    // Try to use the API if available
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          homeTeam: homeTeam,
          awayTeam: awayTeam,
          homeRedCards: 0,
          awayRedCards: 0
        }),
      });
      
      const data = await response.json();
      
      if (data.success) {
        // Add some randomness to the API prediction
        const homeGoalsBase = data.prediction.predictedHomeGoals;
        const awayGoalsBase = data.prediction.predictedAwayGoals;
        
        const randomFactor = 0.8; // Randomness factor
        const homeGoals = Math.max(0, Math.round(homeGoalsBase + (Math.random() - 0.5) * randomFactor * 2));
        const awayGoals = Math.max(0, Math.round(awayGoalsBase + (Math.random() - 0.5) * randomFactor * 2));
        
        return {
          homeTeam,
          awayTeam,
          homeGoals,
          awayGoals,
          timestamp: new Date().toLocaleString()
        };
      }
    } catch (err) {
      console.log('API not available, using simulation');
    }
    
    // Fallback to simulation if API fails
    return simulateMatch(homeTeam, awayTeam);
  };
  
  // Simulate a match with randomness
  const simulateMatch = (homeTeam, awayTeam, isKnockout = false) => {
    // Team strengths (biased for popular teams)
    const teamStrength = {
      "Galatasaray": 90,
      "Fenerbahçe": 87,
      "Beşiktaş": 85,
      "Trabzonspor": 82,
      "Başakşehir": 78,
      "Default": 70
    };
    
    // Get team strengths with fallback to default
    const homeStrength = teamStrength[homeTeam] || teamStrength["Default"];
    const awayStrength = teamStrength[awayTeam] || teamStrength["Default"];
    
    // Home advantage factor (20% boost)
    const homeAdvantage = 1.2;
    
    // Calculate base expected goals with home advantage
    let homeGoalsExp = (homeStrength / 100) * 2.5 * homeAdvantage;
    let awayGoalsExp = (awayStrength / 100) * 2;
    
    // Add randomness (±1.2 goals for more variance)
    const randomFactor = 1.2;
    const homeGoals = Math.max(0, Math.round(homeGoalsExp + (Math.random() - 0.5) * randomFactor * 2));
    const awayGoals = Math.max(0, Math.round(awayGoalsExp + (Math.random() - 0.5) * randomFactor * 2));
    
    // For knockout matches that end in a draw, simulate penalty shootout
    let penalties = null;
    if (isKnockout && homeGoals === awayGoals) {
      // Simulate penalties (4-5 goals each side on average)
      const homePenalties = Math.floor(Math.random() * 3) + 3; // 3-5 goals
      const awayPenalties = Math.floor(Math.random() * 3) + 3; // 3-5 goals
      
      // Ensure we have a winner
      penalties = homePenalties !== awayPenalties ? 
        { home: homePenalties, away: awayPenalties } : 
        { home: homePenalties, away: awayPenalties - 1 };
    }
    
    return {
      homeTeam,
      awayTeam,
      homeGoals,
      awayGoals,
      penalties,
      timestamp: new Date().toLocaleString()
    };
  };
  
  // Start tournament simulation
  const startSimulation = async () => {
    if (teams.length < 16) {
      setError("Need at least 16 teams to start the tournament");
      return;
    }
    
    setIsLoading(true);
    setError('');
    setStatus('Organizing tournament groups...');
    
    try {
      // Create groups
      await simulateGroupStage();
    } catch (err) {
      setError(`Simulation error: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Simulate group stage
  const simulateGroupStage = async () => {
    // Create groups (A, B, C, D)
    const shuffledTeams = _.shuffle(teams);
    const groups = [];
    
    for (let i = 0; i < tournamentStructure.groupStage.groups; i++) {
      const startIdx = i * tournamentStructure.groupStage.teamsPerGroup;
      const groupTeams = shuffledTeams.slice(startIdx, startIdx + tournamentStructure.groupStage.teamsPerGroup);
      
      const groupName = String.fromCharCode(65 + i); // A, B, C, D
      const groupMatches = [];
      const statistics = {};
      
      // Initialize statistics for each team
      groupTeams.forEach(team => {
        statistics[team] = {
          team,
          played: 0,
          won: 0,
          drawn: 0,
          lost: 0,
          goalsFor: 0,
          goalsAgainst: 0,
          goalDifference: 0,
          points: 0,
          scorers: {}
        };
      });
      
      // Create all matches in group (each team plays against every other team)
      for (let j = 0; j < groupTeams.length; j++) {
        for (let k = j + 1; k < groupTeams.length; k++) {
          groupMatches.push({
            homeTeam: groupTeams[j],
            awayTeam: groupTeams[k],
            played: false,
            result: null
          });
        }
      }
      
      groups.push({
        name: `Group ${groupName}`,
        teams: groupTeams,
        matches: groupMatches,
        statistics
      });
    }
    
    setGroupStage(groups);
    setStatus('Groups created. Simulating matches...');
    
    // Play group matches with a small delay to show progress
    setTimeout(() => simulateGroupMatches(groups), 500);
  };
  
  // Simulate all group matches
  const simulateGroupMatches = async (groups) => {
    const updatedGroups = [...groups];
    const allScorers = {};
    
    // Simulate each group's matches
    for (let i = 0; i < updatedGroups.length; i++) {
      const group = updatedGroups[i];
      
      // Play all matches in the group
      for (let j = 0; j < group.matches.length; j++) {
        const match = group.matches[j];
        
        // Simulate the match
        const result = await predictMatch(match.homeTeam, match.awayTeam);
        
        // Mark match as played
        match.played = true;
        match.result = result;
        
        // Update statistics
        updateStats(group.statistics, match.homeTeam, match.awayTeam, result.homeGoals, result.awayGoals);
        
        // Generate random goalscorers
        const goalscorers = generateGoalscorers(match.homeTeam, match.awayTeam, result.homeGoals, result.awayGoals);
        match.goalscorers = goalscorers;
        
        // Update top scorers
        goalscorers.forEach(scorer => {
          if (!allScorers[scorer.name]) {
            allScorers[scorer.name] = { name: scorer.name, team: scorer.team, goals: 0 };
          }
          allScorers[scorer.name].goals += 1;
        });
      }
      
      // Sort statistics by points, then goal difference
      const sortedStats = Object.values(group.statistics).sort((a, b) => {
        if (b.points !== a.points) return b.points - a.points;
        if (b.goalDifference !== a.goalDifference) return b.goalDifference - a.goalDifference;
        return b.goalsFor - a.goalsFor;
      });
      
      group.standings = sortedStats;
      group.qualifiers = sortedStats.slice(0, tournamentStructure.groupStage.advancePerGroup);
    }
    
    setGroupStage(updatedGroups);
    setStatus('Group stage completed. Setting up knockout phase...');
    
    // Set top scorers
    const topScorersList = Object.values(allScorers)
      .sort((a, b) => b.goals - a.goals)
      .slice(0, 10);
    setTopScorers(topScorersList);
    
    // Proceed to knockout stage
    setTimeout(() => setupKnockoutStage(updatedGroups), 500);
  };
  
  // Update team statistics based on match result
  const updateStats = (statistics, homeTeam, awayTeam, homeGoals, awayGoals) => {
    // Update home team stats
    statistics[homeTeam].played += 1;
    statistics[homeTeam].goalsFor += homeGoals;
    statistics[homeTeam].goalsAgainst += awayGoals;
    
    // Update away team stats
    statistics[awayTeam].played += 1;
    statistics[awayTeam].goalsFor += awayGoals;
    statistics[awayTeam].goalsAgainst += homeGoals;
    
    // Update results based on score
    if (homeGoals > awayGoals) {
      // Home win
      statistics[homeTeam].won += 1;
      statistics[homeTeam].points += 3;
      statistics[awayTeam].lost += 1;
    } else if (homeGoals < awayGoals) {
      // Away win
      statistics[awayTeam].won += 1;
      statistics[awayTeam].points += 3;
      statistics[homeTeam].lost += 1;
    } else {
      // Draw
      statistics[homeTeam].drawn += 1;
      statistics[homeTeam].points += 1;
      statistics[awayTeam].drawn += 1;
      statistics[awayTeam].points += 1;
    }
    
    // Update goal differences
    statistics[homeTeam].goalDifference = statistics[homeTeam].goalsFor - statistics[homeTeam].goalsAgainst;
    statistics[awayTeam].goalDifference = statistics[awayTeam].goalsFor - statistics[awayTeam].goalsAgainst;
  };
  
  // Generate random goalscorers for a match
  const generateGoalscorers = (homeTeam, awayTeam, homeGoals, awayGoals) => {
    const scorers = [];
    
    // Create player names for the team (could be expanded to use real player names)
    const createPlayers = (team, count) => {
      const positions = ["FW", "MF", "MF", "DF"];
      const players = [];
      
      for (let i = 0; i < count; i++) {
        // Attackers are more likely to score
        const position = positions[Math.floor(Math.random() * positions.length)];
        const number = Math.floor(Math.random() * 20) + 1; // Jersey numbers 1-20
        const name = `${position} ${team.substring(0, 3).toUpperCase()}${number}`;
        
        players.push({
          name,
          team,
          minute: Math.floor(Math.random() * 90) + 1 // Random minute 1-90
        });
      }
      
      return players;
    };
    
    // Add home team scorers
    scorers.push(...createPlayers(homeTeam, homeGoals));
    
    // Add away team scorers
    scorers.push(...createPlayers(awayTeam, awayGoals));
    
    // Sort by minute
    return scorers.sort((a, b) => a.minute - b.minute);
  };
  
  // Set up knockout stage brackets
  const setupKnockoutStage = (groups) => {
    // Get qualified teams from each group
    const qualifiedTeams = [];
    
    groups.forEach(group => {
      qualifiedTeams.push(...group.qualifiers.map(stats => stats.team));
    });
    
    // Create quarter-final matches
    // Group A 1st vs Group B 2nd
    // Group C 1st vs Group D 2nd
    // Group B 1st vs Group A 2nd
    // Group D 1st vs Group C 2nd
    const quarterFinals = [
      {
        id: 'QF1',
        homeTeam: groups[0].qualifiers[0].team, // A1
        awayTeam: groups[1].qualifiers[1].team, // B2
        played: false,
        result: null
      },
      {
        id: 'QF2',
        homeTeam: groups[2].qualifiers[0].team, // C1
        awayTeam: groups[3].qualifiers[1].team, // D2
        played: false,
        result: null
      },
      {
        id: 'QF3',
        homeTeam: groups[1].qualifiers[0].team, // B1
        awayTeam: groups[0].qualifiers[1].team, // A2
        played: false,
        result: null
      },
      {
        id: 'QF4',
        homeTeam: groups[3].qualifiers[0].team, // D1
        awayTeam: groups[2].qualifiers[1].team, // C2
        played: false,
        result: null
      }
    ];
    
    // Create semi-final placeholders
    const semiFinals = [
      {
        id: 'SF1',
        homeTeam: null, // Will be winner of QF1
        awayTeam: null, // Will be winner of QF2
        played: false,
        result: null,
        previousMatches: ['QF1', 'QF2']
      },
      {
        id: 'SF2',
        homeTeam: null, // Will be winner of QF3
        awayTeam: null, // Will be winner of QF4
        played: false,
        result: null,
        previousMatches: ['QF3', 'QF4']
      }
    ];
    
    // Create final placeholder
    const final = [
      {
        id: 'F1',
        homeTeam: null, // Will be winner of SF1
        awayTeam: null, // Will be winner of SF2
        played: false,
        result: null,
        previousMatches: ['SF1', 'SF2']
      }
    ];
    
    const knockoutData = {
      rounds: [
        {
          name: 'Quarter-Finals',
          matches: quarterFinals
        },
        {
          name: 'Semi-Finals',
          matches: semiFinals
        },
        {
          name: 'Final',
          matches: final
        }
      ]
    };
    
    setKnockoutStage(knockoutData);
    setStatus('Knockout stage setup complete. Simulating quarter-finals...');
    
    // Start simulating knockout matches
    setTimeout(() => simulateKnockoutRound(knockoutData, 0), 500);
  };
  
  // Simulate matches for a knockout round
  const simulateKnockoutRound = async (knockoutData, roundIndex) => {
    const updatedKnockout = _.cloneDeep(knockoutData);
    const currentRound = updatedKnockout.rounds[roundIndex];
    
    // Simulate all matches in the current round
    for (let i = 0; i < currentRound.matches.length; i++) {
      const match = currentRound.matches[i];
      
      if (!match.homeTeam || !match.awayTeam) {
        // Fill in teams from previous rounds if needed
        if (match.previousMatches) {
          const prevRound = updatedKnockout.rounds[roundIndex - 1];
          
          for (let j = 0; j < match.previousMatches.length; j++) {
            const prevMatchId = match.previousMatches[j];
            const prevMatch = prevRound.matches.find(m => m.id === prevMatchId);
            
            if (prevMatch && prevMatch.result) {
              const winner = getMatchWinner(prevMatch);
              
              if (j === 0) {
                match.homeTeam = winner;
              } else {
                match.awayTeam = winner;
              }
            }
          }
        }
      }
      
      // Only simulate if we have both teams
      if (match.homeTeam && match.awayTeam) {
        // Simulate match as knockout (allows penalties)
        const result = simulateMatch(match.homeTeam, match.awayTeam, true);
        match.played = true;
        match.result = result;
        
        // Generate goalscorers
        match.goalscorers = generateGoalscorers(match.homeTeam, match.awayTeam, result.homeGoals, result.awayGoals);
      }
    }
    
    setKnockoutStage(updatedKnockout);
    
    // Move to next round or finish
    if (roundIndex < updatedKnockout.rounds.length - 1) {
      const nextRound = updatedKnockout.rounds[roundIndex + 1].name;
      setStatus(`Simulating ${nextRound}...`);
      setTimeout(() => simulateKnockoutRound(updatedKnockout, roundIndex + 1), 500);
    } else {
      // Tournament complete
      const finalMatch = updatedKnockout.rounds[roundIndex].matches[0];
      const champion = getMatchWinner(finalMatch);
      
      setChampions(champion);
      setStatus('Tournament completed!');
      setSimulationComplete(true);
    }
  };
  
  // Get winner of a match (considering penalties if necessary)
  const getMatchWinner = (match) => {
    const result = match.result;
    
    if (result.homeGoals > result.awayGoals) {
      return result.homeTeam;
    } else if (result.homeGoals < result.awayGoals) {
      return result.awayTeam;
    } else if (result.penalties) {
      // Decide by penalties
      return result.penalties.home > result.penalties.away ? 
        result.homeTeam : result.awayTeam;
    }
    
    // Fallback (should not happen)
    return result.homeTeam;
  };
  
  // Get team color
  const getTeamColor = (team) => {
    return TEAM_COLORS[team] || TEAM_COLORS.Default;
  };
  
  // Reset the simulation
  const resetSimulation = () => {
    setGroupStage(null);
    setKnockoutStage(null);
    setChampions(null);
    setStatus('');
    setError('');
    setSelectedMatchDetails(null);
    setSimulationComplete(false);
    setTopScorers([]);
  };
  
  // Render group tables
  const renderGroupTables = () => {
    if (!groupStage) return null;
    
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        {groupStage.map((group, index) => (
          <div key={index} className="bg-white rounded-lg shadow-md p-4">
            <h3 className="text-lg font-semibold mb-3">{group.name}</h3>
            
            <div className="overflow-x-auto">
              <table className="w-full border-collapse">
                <thead>
                  <tr className="bg-gray-100">
                    <th className="py-2 px-2 text-left">Team</th>
                    <th className="py-2 px-2 text-center">P</th>
                    <th className="py-2 px-2 text-center">W</th>
                    <th className="py-2 px-2 text-center">D</th>
                    <th className="py-2 px-2 text-center">L</th>
                    <th className="py-2 px-2 text-center">GF</th>
                    <th className="py-2 px-2 text-center">GA</th>
                    <th className="py-2 px-2 text-center">GD</th>
                    <th className="py-2 px-2 text-center">Pts</th>
                  </tr>
                </thead>
                <tbody>
                  {group.standings?.map((team, idx) => (
                    <tr 
                      key={idx} 
                      className={`border-t hover:bg-gray-50 ${
                        idx < tournamentStructure.groupStage.advancePerGroup ? 'bg-green-50' : ''
                      }`}
                    >
                      <td className="py-2 px-2 font-medium flex items-center">
                        <div 
                          className="w-3 h-3 rounded-full mr-2" 
                          style={{ backgroundColor: getTeamColor(team.team) }}
                        ></div>
                        {team.team}
                      </td>
                      <td className="py-2 px-2 text-center">{team.played}</td>
                      <td className="py-2 px-2 text-center">{team.won}</td>
                      <td className="py-2 px-2 text-center">{team.drawn}</td>
                      <td className="py-2 px-2 text-center">{team.lost}</td>
                      <td className="py-2 px-2 text-center">{team.goalsFor}</td>
                      <td className="py-2 px-2 text-center">{team.goalsAgainst}</td>
                      <td className="py-2 px-2 text-center">{team.goalDifference}</td>
                      <td className="py-2 px-2 text-center font-bold">{team.points}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            <div className="mt-4">
              <h4 className="text-sm font-semibold mb-2">Group Matches</h4>
              <div className="space-y-2">
                {group.matches.map((match, idx) => (
                  <div 
                    key={idx} 
                    className={`p-2 rounded-md ${match.played ? 'bg-gray-50' : 'bg-gray-100'} cursor-pointer hover:bg-gray-200`}
                    onClick={() => setSelectedMatchDetails(match)}
                  >
                    <div className="flex justify-between items-center">
                      <div className="text-sm">{match.homeTeam}</div>
                      <div className="font-semibold">
                        {match.played ? 
                          `${match.result.homeGoals} - ${match.result.awayGoals}` : 
                          'vs'
                        }
                      </div>
                      <div className="text-sm">{match.awayTeam}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  };
  
  // Render knockout bracket
  const renderKnockoutBracket = () => {
    if (!knockoutStage) return null;
    
    return (
      <div className="mb-8">
        <div className="overflow-x-auto">
          <div className="min-w-max">
            <div className="flex justify-center space-x-12">
              {knockoutStage.rounds.map((round, roundIdx) => (
                <div key={roundIdx} className="flex-1 px-2">
                  <h3 className="text-lg font-semibold mb-3 text-center">{round.name}</h3>
                  <div className="space-y-4">
                    {round.matches.map((match, matchIdx) => (
                      <div 
                        key={matchIdx} 
                        className={`
                          p-3 rounded-md border border-gray-200 bg-white shadow-sm
                          ${match.played ? 'hover:shadow-md' : 'opacity-80'}
                          cursor-pointer transition-all
                        `}
                        onClick={() => match.played && setSelectedMatchDetails(match)}
                      >
                        <div className="flex flex-col space-y-2">
                          <div className="flex justify-between items-center">
                            <div className="flex items-center">
                              <div 
                                className="w-3 h-3 rounded-full mr-2" 
                                style={{ backgroundColor: match.homeTeam ? getTeamColor(match.homeTeam) : '#ccc' }}
                              ></div>
                              <span className={match.homeTeam ? 'font-medium' : 'text-gray-400'}>
                                {match.homeTeam || 'TBD'}
                              </span>
                            </div>
                            <div className="font-semibold text-sm">
                              {match.played ? match.result.homeGoals : '-'}
                            </div>
                          </div>
                          
                          <div className="flex justify-between items-center">
                            <div className="flex items-center">
                              <div 
                                className="w-3 h-3 rounded-full mr-2" 
                                style={{ backgroundColor: match.awayTeam ? getTeamColor(match.awayTeam) : '#ccc' }}
                              ></div>
                              <span className={match.awayTeam ? 'font-medium' : 'text-gray-400'}>
                                {match.awayTeam || 'TBD'}
                              </span>
                            </div>
                            <div className="font-semibold text-sm">
                              {match.played ? match.result.awayGoals : '-'}
                            </div>
                          </div>
                          
                          {match.played && match.result.penalties && (
                            <div className="text-xs text-center italic mt-1 text-gray-600">
                              Penalties: {match.result.penalties.home} - {match.result.penalties.away}
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  };
  
  // Render selected match details
  // Render selected match details
  const renderMatchDetails = () => {
    if (!selectedMatchDetails || !selectedMatchDetails.played) return null;
    
    const match = selectedMatchDetails;
    const result = match.result;
    
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-10">
        <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4 p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-xl font-bold">Match Details</h3>
            <button 
              onClick={() => setSelectedMatchDetails(null)}
              className="text-gray-500 hover:text-gray-700"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          
          <div className="flex justify-center items-center space-x-8 mb-6">
            <div className="text-center">
              <div className="font-bold text-lg">{result.homeTeam}</div>
              <div className="text-4xl font-bold mt-2">{result.homeGoals}</div>
            </div>
            
            <div className="text-lg font-medium text-gray-500">vs</div>
            
            <div className="text-center">
              <div className="font-bold text-lg">{result.awayTeam}</div>
              <div className="text-4xl font-bold mt-2">{result.awayGoals}</div>
            </div>
          </div>
          
          {/* Display penalties if applicable */}
          {result.penalties && (
            <div className="mb-4 p-2 bg-gray-100 rounded-md text-center">
              <div className="text-sm text-gray-600">Penalties</div>
              <div className="font-bold">
                {result.penalties.home} - {result.penalties.away}
              </div>
            </div>
          )}
          
          {/* Goal timeline */}
          {match.goalscorers && match.goalscorers.length > 0 && (
            <div className="mb-6">
              <h4 className="font-semibold mb-2">Goals</h4>
              <div className="h-20 relative border-b border-gray-300">
                {/* Time markers */}
                {[0, 15, 30, 45, 60, 75, 90].map(minute => (
                  <div 
                    key={minute} 
                    className="absolute bottom-0 transform -translate-x-1/2"
                    style={{ left: `${(minute / 90) * 100}%` }}
                  >
                    <div className="h-2 border-l border-gray-300"></div>
                    <div className="text-xs text-gray-500">{minute}'</div>
                  </div>
                ))}
                
                {/* Goal events */}
                {match.goalscorers.map((scorer, idx) => (
                  <div 
                    key={idx}
                    className="absolute transform -translate-x-1/2"
                    style={{ 
                      left: `${(scorer.minute / 90) * 100}%`,
                      bottom: scorer.team === result.homeTeam ? '15px' : '30px'
                    }}
                  >
                    <div className="flex flex-col items-center">
                      <svg 
                        className="w-4 h-4" 
                        fill={scorer.team === result.homeTeam ? getTeamColor(result.homeTeam) : getTeamColor(result.awayTeam)}
                        viewBox="0 0 20 20"
                      >
                        <path d="M10 18a8 8 0 100-16 8 8 0 000 16z" />
                      </svg>
                      <div className="text-xs mt-1">{scorer.name}</div>
                      <div className="text-xs text-gray-500">{scorer.minute}'</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Match statistics (random data) */}
          <div>
            <h4 className="font-semibold mb-2">Match Statistics</h4>
            
            <div className="space-y-3">
              {/* Possession */}
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>{Math.floor(Math.random() * 20) + 40}%</span>
                  <span>Possession</span>
                  <span>{Math.floor(Math.random() * 20) + 40}%</span>
                </div>
                <div className="flex h-2 rounded-full overflow-hidden bg-gray-200">
                  <div 
                    className="bg-blue-600" 
                    style={{ width: `${Math.floor(Math.random() * 20) + 40}%` }}
                  ></div>
                  <div 
                    className="bg-red-600" 
                    style={{ width: `${Math.floor(Math.random() * 20) + 40}%` }}
                  ></div>
                </div>
              </div>
              
              {/* Shots */}
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>{Math.floor(Math.random() * 10) + 5}</span>
                  <span>Shots</span>
                  <span>{Math.floor(Math.random() * 10) + 5}</span>
                </div>
                <div className="flex h-2 rounded-full overflow-hidden bg-gray-200">
                  <div 
                    className="bg-blue-600" 
                    style={{ width: `${Math.floor(Math.random() * 30) + 35}%` }}
                  ></div>
                  <div 
                    className="bg-red-600" 
                    style={{ width: `${Math.floor(Math.random() * 30) + 35}%` }}
                  ></div>
                </div>
              </div>
              
              {/* Corners */}
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>{Math.floor(Math.random() * 6) + 2}</span>
                  <span>Corners</span>
                  <span>{Math.floor(Math.random() * 6) + 2}</span>
                </div>
                <div className="flex h-2 rounded-full overflow-hidden bg-gray-200">
                  <div 
                    className="bg-blue-600" 
                    style={{ width: `${Math.floor(Math.random() * 30) + 35}%` }}
                  ></div>
                  <div 
                    className="bg-red-600" 
                    style={{ width: `${Math.floor(Math.random() * 30) + 35}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };
  
  // Render top scorers
  const renderTopScorers = () => {
    if (!topScorers || topScorers.length === 0) return null;
    
    return (
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h3 className="text-xl font-semibold mb-4">Top Goalscorers</h3>
        
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr className="bg-gray-100">
                <th className="py-2 px-4 text-left">Rank</th>
                <th className="py-2 px-4 text-left">Player</th>
                <th className="py-2 px-4 text-left">Team</th>
                <th className="py-2 px-4 text-center">Goals</th>
              </tr>
            </thead>
            <tbody>
              {topScorers.map((scorer, index) => (
                <tr key={index} className="border-t hover:bg-gray-50">
                  <td className="py-2 px-4">{index + 1}</td>
                  <td className="py-2 px-4 font-medium">{scorer.name}</td>
                  <td className="py-2 px-4">{scorer.team}</td>
                  <td className="py-2 px-4 text-center font-bold">{scorer.goals}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };
  
  // Render champions section
  const renderChampions = () => {
    if (!champions) return null;
    
    // Get confetti colors based on team
    const confettiColors = [getTeamColor(champions), "#FFD700", "#FFFFFF"];
    
    return (
      <div className="bg-gradient-to-br from-yellow-400 to-yellow-600 rounded-lg shadow-xl p-8 mb-6 text-center relative overflow-hidden">
        {/* Confetti-like elements */}
        {Array.from({ length: 30 }).map((_, i) => (
          <div 
            key={i}
            className="absolute w-4 h-4 rounded-full opacity-70 animate-pulse"
            style={{
              backgroundColor: confettiColors[i % confettiColors.length],
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDuration: `${Math.random() * 3 + 2}s`,
              animationDelay: `${Math.random() * 2}s`
            }}
          ></div>
        ))}
        
        <h2 className="text-3xl font-bold text-white mb-2">Tournament Champions</h2>
        <div className="text-5xl font-bold text-white mb-6">{champions}</div>
        
        <div className="bg-white bg-opacity-20 rounded-lg p-4 inline-block">
          <div className="text-white">Congratulations to {champions} for winning the Turkish Cup!</div>
        </div>
      </div>
    );
  };
  
  // Render team performance chart
  const renderTeamPerformanceChart = () => {
    if (!groupStage) return null;
    
    // Collect all teams' goal data
    const teamsData = [];
    groupStage.forEach(group => {
      const groupTeams = Object.values(group.statistics);
      teamsData.push(...groupTeams);
    });
    
    // Sort by goals scored
    const sortedTeams = teamsData.sort((a, b) => b.goalsFor - a.goalsFor).slice(0, 8);
    
    return (
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h3 className="text-xl font-semibold mb-4">Team Performance</h3>
        
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={sortedTeams}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="team" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar name="Goals Scored" dataKey="goalsFor" fill="#3182CE" />
              <Bar name="Goals Conceded" dataKey="goalsAgainst" fill="#E53E3E" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    );
  };

  // HERE IS THE MISSING RETURN STATEMENT
  return (
    <div className="mx-auto max-w-6xl p-4">
      <h1 className="text-3xl font-bold text-center mb-6">Turkish Cup Tournament Simulator</h1>
      
      {/* Tournament controls */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="mb-4 md:mb-0">
            <h2 className="text-xl font-semibold">Tournament Control Panel</h2>
            <p className="text-sm text-gray-600 mt-1">
              Simulate a complete Turkish Cup tournament with group stages and knockout rounds
            </p>
          </div>
          
          <div className="flex space-x-4">
            <button
              className={`
                px-4 py-2 rounded-md text-white font-medium
                ${isLoading ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'}
              `}
              onClick={startSimulation}
              disabled={isLoading}
            >
              {isLoading ? 'Simulating...' : 'Start New Tournament'}
            </button>
            
            {simulationComplete && (
              <button
                className="px-4 py-2 rounded-md bg-gray-200 hover:bg-gray-300 font-medium"
                onClick={resetSimulation}
              >
                Reset
              </button>
            )}
          </div>
        </div>
        
        {/* Status message */}
        {status && (
          <div className="mt-4 p-2 bg-blue-50 text-blue-700 rounded-md text-center">
            {status}
          </div>
        )}
        
        {/* Error message */}
        {error && (
          <div className="mt-4 p-2 bg-red-50 text-red-700 rounded-md text-center">
            {error}
          </div>
        )}
      </div>
      
      {/* Champions section (if tournament complete) */}
      {champions && renderChampions()}
      
      {/* Tab navigation for tournament data */}
      {(groupStage || knockoutStage) && (
        <div className="mb-6">
          <div className="border-b border-gray-200">
            <nav className="flex space-x-8">
              <button
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'groups' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
                onClick={() => setActiveTab('groups')}
              >
                Group Stage
              </button>
              <button
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'knockout' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
                onClick={() => setActiveTab('knockout')}
              >
                Knockout Stage
              </button>
              <button
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'stats' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
                onClick={() => setActiveTab('stats')}
              >
                Statistics
              </button>
            </nav>
          </div>
        </div>
      )}
      
      {/* Tab content */}
      {activeTab === 'groups' && groupStage && renderGroupTables()}
      {activeTab === 'knockout' && knockoutStage && renderKnockoutBracket()}
      {activeTab === 'stats' && (
        <>
          {renderTopScorers()}
          {renderTeamPerformanceChart()}
        </>
      )}
      
      {/* Match details modal */}
      {selectedMatchDetails && renderMatchDetails()}
    </div>
  );
};

export default TurkishCupSimulator;