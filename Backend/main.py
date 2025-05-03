from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# Global variables to store models and data
home_goals_model = None
away_goals_model = None
teams = []
dataset = None

# Load dataset and train models
def train_models(dataset_path):
    global home_goals_model, away_goals_model, teams, dataset
    
    try:
        # Load the dataset
        dataset = pd.read_csv(dataset_path)
        
        # Clean and prepare data
        dataset = preprocess_data(dataset)
        
        # Extract list of teams
        teams = sorted(list(set(dataset['Home'].unique()) | set(dataset['Away'].unique())))
        
        # Prepare feature matrix
        X = prepare_features(dataset)
        
        # Train home goals model
        home_goals_model = RandomForestRegressor(n_estimators=100, random_state=42)
        home_goals_model.fit(X, dataset['HomeGoals'])
        
        # Train away goals model
        away_goals_model = RandomForestRegressor(n_estimators=100, random_state=42)
        away_goals_model.fit(X, dataset['AwayGoals'])
        
        # Save models
        os.makedirs('models', exist_ok=True)
        joblib.dump(home_goals_model, 'models/home_goals_model.pkl')
        joblib.dump(away_goals_model, 'models/away_goals_model.pkl')
        
        return True, "Models trained successfully"
    
    except Exception as e:
        return False, str(e)

# Preprocess the dataset
def preprocess_data(df):
    # Basic preprocessing
    df = df.copy()
    df = df.dropna(subset=['Home', 'Away', 'HomeGoals', 'AwayGoals'])
    
    # Ensure required columns exist
    if 'HomeRedCards' not in df.columns:
        df['HomeRedCards'] = 0
    if 'AwayRedCards' not in df.columns:
        df['AwayRedCards'] = 0
    
    return df

# Prepare features for prediction
def prepare_features(df):
    # One-hot encode team names
    home_dummies = pd.get_dummies(df['Home'], prefix='home')
    away_dummies = pd.get_dummies(df['Away'], prefix='away')
    
    # Combine features
    features = pd.concat([home_dummies, away_dummies, df[['HomeRedCards', 'AwayRedCards']]], axis=1)
    
    return features

# Load existing models
def load_existing_models():
    global home_goals_model, away_goals_model, teams, dataset

    try:
        # Correct file names
        home_model_path = 'models/home_goals_model.pkl'
        away_model_path = 'models/away_goals_model.pkl'

        if not os.path.exists(home_model_path) or not os.path.exists(away_model_path):
            return False, "Model files not found"

        # Load models
        home_goals_model = joblib.load(home_model_path)
        away_goals_model = joblib.load(away_model_path)
        print("✅ Models loaded successfully")

        # Load teams list
        if not teams:
            if os.path.exists('models/teams.txt'):
                with open('models/teams.txt', 'r') as f:
                    teams = [line.strip() for line in f.readlines()]
                    print("✅ Teams loaded from teams.txt")
            elif os.path.exists('models/dataset.csv'):
                temp_dataset = pd.read_csv('models/dataset.csv')
                if 'Home' in temp_dataset.columns and 'Away' in temp_dataset.columns:
                    teams = sorted(list(set(temp_dataset['Home'].unique()) | set(temp_dataset['Away'].unique())))
                    print("✅ Teams extracted from dataset.csv")

        # Load dataset
        if dataset is None and os.path.exists('models/dataset.csv'):
            dataset = pd.read_csv('models/dataset.csv')
            dataset = preprocess_data(dataset)
            print("✅ Dataset loaded and preprocessed")

        # Fallback if teams still not loaded
        if not teams:
            teams = ["FallbackTeam"]
            print("⚠️ Using fallback team list")

        return True, "Models loaded successfully"

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False, str(e)
    
# Predict match result
def predict_match(home_team, away_team, home_red_cards, away_red_cards):
    if home_goals_model is None or away_goals_model is None:
        return False, "Models not loaded", None
    
    try:
        # Create feature vector for this match
        feature_vector = pd.DataFrame(columns=[f'home_{team}' for team in teams] + 
                                     [f'away_{team}' for team in teams] + 
                                     ['HomeRedCards', 'AwayRedCards'])
        
        # Initialize with zeros
        feature_vector.loc[0] = [0] * len(feature_vector.columns)
        
        # Set team indicators
        if f'home_{home_team}' in feature_vector.columns:
            feature_vector[f'home_{home_team}'] = 1
        if f'away_{away_team}' in feature_vector.columns:
            feature_vector[f'away_{away_team}'] = 1
        
        # Set red cards
        feature_vector['HomeRedCards'] = home_red_cards
        feature_vector['AwayRedCards'] = away_red_cards
        
        # Make predictions
        predicted_home_goals = max(0, round(home_goals_model.predict(feature_vector)[0]))
        predicted_away_goals = max(0, round(away_goals_model.predict(feature_vector)[0]))
        
        return True, "Prediction successful", {
            'homeTeam': home_team,
            'awayTeam': away_team,
            'homeRedCards': home_red_cards,
            'awayRedCards': away_red_cards,
            'predictedHomeGoals': int(predicted_home_goals),
            'predictedAwayGoals': int(predicted_away_goals)
        }
    
    except Exception as e:
        return False, str(e), None

# Get statistical analysis data
def get_statistical_data(home_team, away_team):
    if dataset is None or home_goals_model is None or away_goals_model is None:
        return False, "No dataset or models available"
    
    try:
        # Prepare metrics
        X = prepare_features(dataset)
        home_preds = home_goals_model.predict(X)
        away_preds = away_goals_model.predict(X)
        
        home_mse = mean_squared_error(dataset['HomeGoals'], home_preds)
        away_mse = mean_squared_error(dataset['AwayGoals'], away_preds)
        home_mae = mean_absolute_error(dataset['HomeGoals'], home_preds)
        away_mae = mean_absolute_error(dataset['AwayGoals'], away_preds)
        home_r2 = r2_score(dataset['HomeGoals'], home_preds)
        away_r2 = r2_score(dataset['AwayGoals'], away_preds)
        
        # Generate regression data
        regression_data = []
        for i in range(min(100, len(dataset))):  # Limit to 100 samples
            regression_data.append({
                'predicted': float(home_preds[i]),
                'actual': float(dataset['HomeGoals'].iloc[i]),
                'index': i
            })
        
        # Generate team stats
        home_matches = dataset[dataset['Home'] == home_team]
        away_matches = dataset[dataset['Away'] == away_team]
        
        home_team_stats = {
            'name': home_team,
            'avgHomeGoals': float(home_matches['HomeGoals'].mean()) if len(home_matches) > 0 else 0,
            'avgHomeGoalsAgainst': float(home_matches['AwayGoals'].mean()) if len(home_matches) > 0 else 0,
            'homeWinRate': float((home_matches['HomeGoals'] > home_matches['AwayGoals']).mean()) if len(home_matches) > 0 else 0,
            'homeDrawRate': float((home_matches['HomeGoals'] == home_matches['AwayGoals']).mean()) if len(home_matches) > 0 else 0,
            'homeLossRate': float((home_matches['HomeGoals'] < home_matches['AwayGoals']).mean()) if len(home_matches) > 0 else 0
        }
        
        away_team_stats = {
            'name': away_team,
            'avgAwayGoals': float(away_matches['AwayGoals'].mean()) if len(away_matches) > 0 else 0,
            'avgAwayGoalsAgainst': float(away_matches['HomeGoals'].mean()) if len(away_matches) > 0 else 0,
            'awayWinRate': float((away_matches['AwayGoals'] > away_matches['HomeGoals']).mean()) if len(away_matches) > 0 else 0,
            'awayDrawRate': float((away_matches['AwayGoals'] == away_matches['HomeGoals']).mean()) if len(away_matches) > 0 else 0,
            'awayLossRate': float((away_matches['AwayGoals'] < away_matches['HomeGoals']).mean()) if len(away_matches) > 0 else 0
        }
        
        # Get top team stats for comparison
        top_teams = dataset['Home'].value_counts().head(5).index.tolist()
        team_goal_stats = []
        
        for team in top_teams:
            home_data = dataset[dataset['Home'] == team]
            away_data = dataset[dataset['Away'] == team]
            
            team_goal_stats.append({
                'team': team,
                'homeGoals': float(home_data['HomeGoals'].mean()) if len(home_data) > 0 else 0,
                'awayGoals': float(away_data['AwayGoals'].mean()) if len(away_data) > 0 else 0
            })
        
        return True, {
            'metrics': {
                'homeGoalsMSE': home_mse,
                'awayGoalsMSE': away_mse,
                'homeGoalsMAE': home_mae,
                'awayGoalsMAE': away_mae,
                'homeGoalsR2': home_r2,
                'awayGoalsR2': away_r2
            },
            'regressionData': regression_data,
            'teamStats': team_goal_stats,
            'homeTeam': home_team_stats,
            'awayTeam': away_team_stats
        }
    
    except Exception as e:
        return False, str(e)

# API Routes
@app.route('/api/train', methods=['POST'])
def api_train():
    # Define the path more carefully
    dataset_path = os.path.abspath('./tsl_dataset.csv')
    
    # Check if file exists
    if not os.path.exists(dataset_path):
        return jsonify({
            'success': False,
            'message': f'Dataset file not found at {dataset_path}',
            'teams': []
        }), 404
    
    try:
        success, message = train_models(dataset_path)
        
        return jsonify({
            'success': success,
            'message': message,
            'teams': teams if success else []
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}',
            'teams': []
        }), 500
    

@app.route('/api/load_models', methods=['GET'])
def api_load_models():
    success, message = load_existing_models()
    
    return jsonify({
        'success': success,
        'message': message,
        'teams': teams if success else []
    })

@app.route('/api/teams', methods=['GET'])
def api_get_teams():
    return jsonify({
        'success': True,
        'teams': teams
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    home_team = data.get('homeTeam')
    away_team = data.get('awayTeam')
    home_red_cards = data.get('homeRedCards', 0)
    away_red_cards = data.get('awayRedCards', 0)
    
    if not home_team or not away_team:
        return jsonify({'success': False, 'error': 'Home team and away team are required'})
    
    success, message, prediction = predict_match(home_team, away_team, home_red_cards, away_red_cards)
    
    return jsonify({
        'success': success,
        'message': message,
        'prediction': prediction
    })

@app.route('/api/stats', methods=['POST'])
def api_get_stats():
    data = request.json
    home_team = data.get('homeTeam')
    away_team = data.get('awayTeam')
    
    if not home_team or not away_team:
        return jsonify({'success': False, 'error': 'Home team and away team are required'})
    
    success, result = get_statistical_data(home_team, away_team)
    
    if success:
        return jsonify({
            'success': True,
            'modelStats': result
        })
    else:
        return jsonify({
            'success': False,
            'error': result
        })

if __name__ == '__main__':
    # Try to load existing models on startup
    load_existing_models()
    app.run(debug=True)