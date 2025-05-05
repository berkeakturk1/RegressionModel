from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("api.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Global variables to store models and data
home_goals_model = None
away_goals_model = None
teams = []
dataset = None

# Load dataset and train models
def train_models(dataset_path):
    global home_goals_model, away_goals_model, teams, dataset
    
    try:
        # Log the dataset path
        logger.info(f"Training models using dataset: {dataset_path}")
        
        # Check if file exists
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset file not found at {dataset_path}")
            return False, f"Dataset file not found at {dataset_path}"
        
        # Load the dataset
        dataset = pd.read_csv(dataset_path)
        logger.info(f"Dataset loaded with shape: {dataset.shape}")
        
        # Clean and prepare data
        dataset = preprocess_data(dataset)
        logger.info(f"After preprocessing, dataset shape: {dataset.shape}")
        
        # Extract list of teams
        teams = sorted(list(set(dataset['Home'].unique()) | set(dataset['Away'].unique())))
        logger.info(f"Found {len(teams)} teams: {teams}")
        
        # Save teams to file for later reference
        os.makedirs('models', exist_ok=True)
        with open('models/teams.txt', 'w') as f:
            for team in teams:
                f.write(f"{team}\n")
        
        # Prepare feature matrix
        X = prepare_features(dataset)
        logger.info(f"Feature matrix shape: {X.shape}")
        
        # Train home goals model
        home_goals_model = RandomForestRegressor(n_estimators=100, random_state=42)
        home_goals_model.fit(X, dataset['HomeGoals'])
        logger.info("Home goals model trained successfully")
        
        # Train away goals model
        away_goals_model = RandomForestRegressor(n_estimators=100, random_state=42)
        away_goals_model.fit(X, dataset['AwayGoals'])
        logger.info("Away goals model trained successfully")
        
        # Save models
        joblib.dump(home_goals_model, 'models/model_home.pkl')
        joblib.dump(away_goals_model, 'models/model_away.pkl')
        
        # Save dataset for later reference
        dataset.to_csv('models/dataset.csv', index=False)
        
        logger.info("Models and dataset saved successfully")
        return True, "Models trained successfully"
    
    except Exception as e:
        logger.error(f"Error training models: {str(e)}", exc_info=True)
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

# Create prediction feature vector with more robust error handling
def create_feature_vector(home_team, away_team, home_red_cards, away_red_cards):
    # Create a list of all possible feature columns
    home_team_cols = [f'home_{team}' for team in teams]
    away_team_cols = [f'away_{team}' for team in teams]
    all_cols = home_team_cols + away_team_cols + ['HomeRedCards', 'AwayRedCards']
    
    # Create empty DataFrame with all possible columns
    feature_vector = pd.DataFrame(columns=all_cols)
    
    # Initialize with zeros
    feature_vector.loc[0] = [0] * len(all_cols)
    
    # Set team indicators
    home_col = f'home_{home_team}'
    away_col = f'away_{away_team}'
    
    if home_col in feature_vector.columns:
        feature_vector[home_col] = 1
    else:
        # If team not found, log error and return None
        logger.error(f"Home team '{home_team}' not found in training data. Available teams: {teams}")
        return None, f"Home team '{home_team}' not found in training data."
    
    if away_col in feature_vector.columns:
        feature_vector[away_col] = 1
    else:
        # If team not found, log error and return None
        logger.error(f"Away team '{away_team}' not found in training data. Available teams: {teams}")
        return None, f"Away team '{away_team}' not found in training data."
    
    # Set red cards
    feature_vector['HomeRedCards'] = home_red_cards
    feature_vector['AwayRedCards'] = away_red_cards
    
    return feature_vector, None

# Load existing models
def load_existing_models():
    global home_goals_model, away_goals_model, teams, dataset
    
    try:
        logger.info("Attempting to load existing models...")
        
        # Check if models exist
        if not os.path.exists('models/model_home.pkl') or not os.path.exists('models/model_away.pkl'):
            logger.warning("Models not found in expected location")
            return False, "Models not found"
        
        # Load models
        home_goals_model = joblib.load('models/model_home.pkl')
        away_goals_model = joblib.load('models/model_away.pkl')
        logger.info("Models loaded successfully")
        
        # If teams list is not yet populated, try to get it from the teams file
        if not teams:
            if os.path.exists('models/teams.txt'):
                with open('models/teams.txt', 'r') as f:
                    teams = [line.strip() for line in f.readlines()]
                logger.info(f"Loaded {len(teams)} teams from teams.txt")
            
            # If still no teams, try to load from dataset
            if not teams and os.path.exists('models/dataset.csv'):
                temp_dataset = pd.read_csv('models/dataset.csv')
                if 'Home' in temp_dataset.columns and 'Away' in temp_dataset.columns:
                    teams = sorted(list(set(temp_dataset['Home'].unique()) | set(temp_dataset['Away'].unique())))
                    logger.info(f"Loaded {len(teams)} teams from dataset.csv")
            
            # If still no teams, try to load from the original dataset path
            if not teams:
                # Try multiple possible locations
                dataset_paths = [
                    'tsl_dataset.csv',
                    './tsl_dataset.csv',
                    '../tsl_dataset.csv',
                    '/app/tsl_dataset.csv'
                ]
                
                for path in dataset_paths:
                    if os.path.exists(path):
                        try:
                            temp_dataset = pd.read_csv(path)
                            temp_dataset = temp_dataset.dropna(subset=['Home', 'Away'])
                            teams = sorted(list(set(temp_dataset['Home'].unique()) | set(temp_dataset['Away'].unique())))
                            logger.info(f"Loaded {len(teams)} teams from {path}")
                            break
                        except Exception as dataset_error:
                            logger.error(f"Error loading from {path}: {dataset_error}")
            
            # If still no teams, use dummy data as fallback
            if not teams:
                teams = [
                    "Galatasaray", "Fenerbahçe", "Beşiktaş", "Trabzonspor", 
                    "İstanbul Başakşehir", "Adana Demirspor", "Antalyaspor", 
                    "Konyaspor", "Kayserispor", "Hatayspor"
                ]
                logger.warning(f"Using fallback team list: {teams}")
        
        # Try to load dataset if available
        if dataset is None and os.path.exists('models/dataset.csv'):
            dataset = pd.read_csv('models/dataset.csv')
            dataset = preprocess_data(dataset)
            logger.info(f"Loaded dataset with shape: {dataset.shape}")
        
        return True, "Models loaded successfully"
    
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}", exc_info=True)
        
        # If there was an error loading models, still try to populate teams
        if not teams:
            teams = [
                "Galatasaray", "Fenerbahçe", "Beşiktaş", "Trabzonspor", 
                "İstanbul Başakşehir", "Adana Demirspor", "Antalyaspor", 
                "Konyaspor", "Kayserispor", "Hatayspor"
            ]
            logger.warning(f"Using fallback team list due to error: {teams}")
        
        return False, str(e)
    
# Predict match result
def predict_match(home_team, away_team, home_red_cards, away_red_cards):
    if home_goals_model is None or away_goals_model is None:
        logger.error("Models not loaded")
        return False, "Models not loaded", None
    
    try:
        logger.info(f"Predicting match: {home_team} vs {away_team} (Red cards: {home_red_cards}-{away_red_cards})")
        
        # Create feature vector for this match
        feature_vector, error = create_feature_vector(home_team, away_team, home_red_cards, away_red_cards)
        
        if feature_vector is None:
            return False, error, None
        
        # Make predictions
        logger.info("Making predictions...")
        predicted_home_goals = max(0, round(home_goals_model.predict(feature_vector)[0]))
        predicted_away_goals = max(0, round(away_goals_model.predict(feature_vector)[0]))
        
        prediction = {
            'homeTeam': home_team,
            'awayTeam': away_team,
            'homeRedCards': home_red_cards,
            'awayRedCards': away_red_cards,
            'predictedHomeGoals': int(predicted_home_goals),
            'predictedAwayGoals': int(predicted_away_goals)
        }
        
        logger.info(f"Prediction result: {home_team} {predicted_home_goals} - {predicted_away_goals} {away_team}")
        return True, "Prediction successful", prediction
    
    except Exception as e:
        logger.error(f"Error predicting match: {str(e)}", exc_info=True)
        return False, str(e), None

# Get statistical analysis data
def get_statistical_data(home_team, away_team):
    if dataset is None or home_goals_model is None or away_goals_model is None:
        logger.error("No dataset or models available for statistical analysis")
        return False, "No dataset or models available"
    
    try:
        logger.info(f"Getting statistical data for {home_team} vs {away_team}")
        
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
        
        stats_result = {
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
        
        logger.info("Statistical data generated successfully")
        return True, stats_result
    
    except Exception as e:
        logger.error(f"Error getting statistical data: {str(e)}", exc_info=True)
        return False, str(e)

# API Routes
@app.route('/api/train', methods=['POST'])
def api_train():
    logger.info("Train endpoint called")
    
    # Define multiple possible dataset paths
    dataset_paths = [
        'tsl_dataset.csv',
        './tsl_dataset.csv',
        '../tsl_dataset.csv',
        '/app/tsl_dataset.csv'
    ]
    
    # Try each path until we find one that exists
    dataset_path = None
    for path in dataset_paths:
        if os.path.exists(path):
            dataset_path = os.path.abspath(path)
            logger.info(f"Found dataset at {dataset_path}")
            break
    
    # If no dataset found, return error
    if dataset_path is None:
        error_msg = f"Dataset file not found. Tried: {dataset_paths}"
        logger.error(error_msg)
        return jsonify({
            'success': False,
            'message': error_msg,
            'teams': []
        }), 404
    
    try:
        success, message = train_models(dataset_path)
        
        response = {
            'success': success,
            'message': message,
            'teams': teams if success else []
        }
        
        logger.info(f"Train response: {response}")
        return jsonify(response)
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({
            'success': False,
            'message': error_msg,
            'teams': []
        }), 500
    

@app.route('/api/load_models', methods=['GET'])
def api_load_models():
    logger.info("Load models endpoint called")
    success, message = load_existing_models()
    
    response = {
        'success': success,
        'message': message,
        'teams': teams if success else []
    }
    
    logger.info(f"Load models response: {response}")
    return jsonify(response)

@app.route('/api/teams', methods=['GET'])
def api_get_teams():
    logger.info("Teams endpoint called")
    return jsonify({
        'success': True,
        'teams': teams
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    # Log request data
    request_data = request.json
    logger.info(f"Predict endpoint called with data: {request_data}")
    
    # Extract data from request
    home_team = request_data.get('homeTeam')
    away_team = request_data.get('awayTeam')
    home_red_cards = request_data.get('homeRedCards', 0)
    away_red_cards = request_data.get('awayRedCards', 0)
    
    # Validate input
    if not home_team or not away_team:
        error_msg = 'Home team and away team are required'
        logger.error(error_msg)
        return jsonify({'success': False, 'error': error_msg})
    
    # Convert red cards to integers if they're strings
    try:
        home_red_cards = int(home_red_cards)
        away_red_cards = int(away_red_cards)
    except (ValueError, TypeError):
        error_msg = 'Red cards must be integers'
        logger.error(error_msg)
        return jsonify({'success': False, 'error': error_msg})
    
    # Make prediction
    success, message, prediction = predict_match(home_team, away_team, home_red_cards, away_red_cards)
    
    # Prepare response
    response = {
        'success': success,
        'message': message,
        'prediction': prediction
    }
    
    # Log response
    logger.info(f"Predict response: {response}")
    
    return jsonify(response)

@app.route('/api/stats', methods=['POST'])
def api_get_stats():
    # Log request data
    request_data = request.json
    logger.info(f"Stats endpoint called with data: {request_data}")
    
    # Extract data from request
    home_team = request_data.get('homeTeam')
    away_team = request_data.get('awayTeam')
    
    # Validate input
    if not home_team or not away_team:
        error_msg = 'Home team and away team are required'
        logger.error(error_msg)
        return jsonify({'success': False, 'error': error_msg})
    
    # Get statistical data
    success, result = get_statistical_data(home_team, away_team)
    
    # Prepare response
    if success:
        response = {
            'success': True,
            'modelStats': result
        }
    else:
        response = {
            'success': False,
            'error': result
        }
    
    # Log response summary (without the full data)
    logger.info(f"Stats response success: {success}")
    
    return jsonify(response)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint to verify the API is running"""
    logger.info("Health check endpoint called")
    models_loaded = home_goals_model is not None and away_goals_model is not None
    teams_loaded = len(teams) > 0
    
    return jsonify({
        'status': 'ok',
        'models_loaded': models_loaded,
        'teams_loaded': teams_loaded,
        'team_count': len(teams)
    })

if __name__ == '__main__':
    # Try to load existing models on startup
    load_existing_models()
    app.run(debug=True, host='0.0.0.0')