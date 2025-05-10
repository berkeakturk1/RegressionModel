from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# Model paths
MODEL_DIR = 'models'
HOME_MODEL_PATH = os.path.join(MODEL_DIR, 'home_goals_model.pkl')
AWAY_MODEL_PATH = os.path.join(MODEL_DIR, 'away_goals_model.pkl')
TEAMS_PATH = os.path.join(MODEL_DIR, 'teams.txt')
PROCESSED_DATASET_PATH = os.path.join(MODEL_DIR, 'processed_dataset.csv')
TEST_METRICS_PATH = os.path.join(MODEL_DIR, 'test_metrics.json')
TEST_DATA_PATH = os.path.join(MODEL_DIR, 'test_data.pkl')

# Dataset path
DATASET_PATH = 'tsl_dataset.csv'

# Global variables to store models and data
home_goals_model = None
away_goals_model = None
teams = []
dataset = None
test_metrics = None
test_data = None

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset and train models
def train_models(dataset_path=DATASET_PATH, num_seasons=3):
    """Train models and save them to disk"""
    global home_goals_model, away_goals_model, teams, dataset, test_metrics, test_data
    
    try:
        logger.info(f"Training models using dataset: {dataset_path}")
        
        # Load the dataset
        dataset = pd.read_csv(dataset_path)
        
        # Clean and prepare data
        dataset = preprocess_data(dataset, num_seasons=num_seasons)
        
        # Save processed dataset for future reference
        dataset.to_csv(PROCESSED_DATASET_PATH, index=False)
        logger.info(f"Saved processed dataset to {PROCESSED_DATASET_PATH}")
        
        # Extract list of teams
        teams = sorted(list(set(dataset['Home'].unique()) | set(dataset['Away'].unique())))
        
        # Save teams list
        with open(TEAMS_PATH, 'w') as f:
            for team in teams:
                f.write(f"{team}\n")
        logger.info(f"Saved {len(teams)} teams to {TEAMS_PATH}")
        
        # Prepare feature matrix
        X = prepare_features(dataset)
        
        # Split data into training and test sets (80/20 split)
        X_train, X_test, y_home_train, y_home_test = train_test_split(
            X, dataset['HomeGoals'], test_size=0.2, random_state=42
        )
        
        # We use the same X split but different y values for away goals
        _, _, y_away_train, y_away_test = train_test_split(
            X, dataset['AwayGoals'], test_size=0.2, random_state=42
        )
        
        # Save the test data for later evaluation
        test_data = {
            'X_test': X_test,
            'y_home_test': y_home_test,
            'y_away_test': y_away_test,
            'indices': y_home_test.index  # Save indices to find original samples in dataset
        }
        joblib.dump(test_data, TEST_DATA_PATH)
        
        # Train home goals model
        home_goals_model = RandomForestRegressor(n_estimators=100, random_state=42)
        home_goals_model.fit(X_train, y_home_train)
        
        # Evaluate on test set
        home_test_predictions = home_goals_model.predict(X_test)
        home_test_mse = mean_squared_error(y_home_test, home_test_predictions)
        home_test_mae = mean_absolute_error(y_home_test, home_test_predictions)
        home_test_r2 = r2_score(y_home_test, home_test_predictions)
        
        # Train away goals model
        away_goals_model = RandomForestRegressor(n_estimators=100, random_state=42)
        away_goals_model.fit(X_train, y_away_train)
        
        # Evaluate on test set
        away_test_predictions = away_goals_model.predict(X_test)
        away_test_mse = mean_squared_error(y_away_test, away_test_predictions)
        away_test_mae = mean_absolute_error(y_away_test, away_test_predictions)
        away_test_r2 = r2_score(y_away_test, away_test_predictions)
        
        # Save test metrics
        test_metrics = {
            'home_test_mse': float(home_test_mse),
            'home_test_mae': float(home_test_mae),
            'home_test_r2': float(home_test_r2),
            'away_test_mse': float(away_test_mse),
            'away_test_mae': float(away_test_mae),
            'away_test_r2': float(away_test_r2),
            'test_size': len(y_home_test),
            'train_size': len(y_home_train)
        }
        
        with open(TEST_METRICS_PATH, 'w') as f:
            json.dump(test_metrics, f)
        
        # Save models
        joblib.dump(home_goals_model, HOME_MODEL_PATH)
        joblib.dump(away_goals_model, AWAY_MODEL_PATH)
        
        logger.info(f"Models trained and evaluated successfully")
        logger.info(f"Home goals model - Test MSE: {home_test_mse:.4f}, MAE: {home_test_mae:.4f}, R²: {home_test_r2:.4f}")
        logger.info(f"Away goals model - Test MSE: {away_test_mse:.4f}, MAE: {away_test_mae:.4f}, R²: {away_test_r2:.4f}")
        
        return True, "Models trained and saved successfully"
    
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        return False, str(e)

# Preprocess the dataset
# Add this to your preprocess_data function
def preprocess_data(df, num_seasons=3):
    """Clean and prepare data for training
    
    Args:
        df: The dataframe to preprocess
        num_seasons: Number of most recent seasons to include
    """
    # Basic preprocessing
    df = df.copy()
    logger.info(f"Preprocessing dataset with {len(df)} rows")
    
    # Filter for last N seasons only
    if 'Season' in df.columns and num_seasons > 0:
        # Get the most recent season in the dataset
        most_recent_season = df['Season'].max()
        # Create a list of seasons to include
        seasons_to_include = list(range(most_recent_season - num_seasons + 1, most_recent_season + 1))
        df = df[df['Season'].isin(seasons_to_include)]
        logger.info(f"Filtered for last {num_seasons} seasons: {seasons_to_include}, remaining rows: {len(df)}")
    
    # Rest of preprocessing remains the same
    df = df.dropna(subset=['Home', 'Away', 'HomeGoals', 'AwayGoals'])
    
    # Ensure required columns exist
    if 'HomeRedCards' not in df.columns:
        df['HomeRedCards'] = 0
    if 'AwayRedCards' not in df.columns:
        df['AwayRedCards'] = 0
    
    logger.info(f"After preprocessing: {len(df)} rows")
    return df

# Prepare features for prediction
def prepare_features(df):
    """Convert data to feature matrix suitable for model training/prediction"""
    # One-hot encode team names
    home_dummies = pd.get_dummies(df['Home'], prefix='home')
    away_dummies = pd.get_dummies(df['Away'], prefix='away')
    
    # Combine features
    features = pd.concat([home_dummies, away_dummies, df[['HomeRedCards', 'AwayRedCards']]], axis=1)
    
    return features

# Load existing models
def load_existing_models():
    """Load pre-trained models from disk"""
    global home_goals_model, away_goals_model, teams, dataset, test_metrics, test_data
    
    try:
        # Check if all required files exist
        required_files = [HOME_MODEL_PATH, AWAY_MODEL_PATH, TEAMS_PATH]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            return False, f"Missing files: {', '.join(missing_files)}"
        
        # Load models
        logger.info("Loading models from disk")
        home_goals_model = joblib.load(HOME_MODEL_PATH)
        away_goals_model = joblib.load(AWAY_MODEL_PATH)
        
        # Load teams list
        with open(TEAMS_PATH, 'r') as f:
            teams = [line.strip() for line in f.readlines()]
        logger.info(f"Loaded {len(teams)} teams")
        
        # Load test metrics if available
        if os.path.exists(TEST_METRICS_PATH):
            with open(TEST_METRICS_PATH, 'r') as f:
                test_metrics = json.load(f)
            logger.info("Loaded test metrics")
        
        # Load test data if available
        if os.path.exists(TEST_DATA_PATH):
            test_data = joblib.load(TEST_DATA_PATH)
            logger.info("Loaded test data")
        
        # Load dataset if available
        if os.path.exists(PROCESSED_DATASET_PATH):
            logger.info(f"Loading processed dataset from {PROCESSED_DATASET_PATH}")
            dataset = pd.read_csv(PROCESSED_DATASET_PATH)
        elif os.path.exists(DATASET_PATH):
            logger.info(f"Loading original dataset from {DATASET_PATH}")
            dataset = pd.read_csv(DATASET_PATH)
            dataset = preprocess_data(dataset)
        
        return True, "Models loaded successfully"
    
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False, str(e)
    
# Predict match result
def predict_match(home_team, away_team, home_red_cards=0, away_red_cards=0):
    """Make prediction for a match"""
    global home_goals_model, away_goals_model, teams
    
    if home_goals_model is None or away_goals_model is None:
        return False, "Models not loaded", None
    
    try:
        logger.info(f"Predicting match: {home_team} vs {away_team} (Red cards: {home_red_cards}-{away_red_cards})")
        
        # Validate teams
        if home_team not in teams:
            return False, f"Home team '{home_team}' not found in training data", None
        if away_team not in teams:
            return False, f"Away team '{away_team}' not found in training data", None
        
        # Create feature vector for this match
        feature_columns = [f'home_{team}' for team in teams] + \
                          [f'away_{team}' for team in teams] + \
                          ['HomeRedCards', 'AwayRedCards']
        
        feature_vector = pd.DataFrame(columns=feature_columns)
        
        # Initialize with zeros
        feature_vector.loc[0] = [0] * len(feature_columns)
        
        # Set team indicators
        feature_vector[f'home_{home_team}'] = 1
        feature_vector[f'away_{away_team}'] = 1
        
        # Set red cards
        feature_vector['HomeRedCards'] = home_red_cards
        feature_vector['AwayRedCards'] = away_red_cards
        
        # Make predictions
        predicted_home_goals = max(0, round(home_goals_model.predict(feature_vector)[0]))
        predicted_away_goals = max(0, round(away_goals_model.predict(feature_vector)[0]))
        
        logger.info(f"Prediction result: {home_team} {predicted_home_goals} - {predicted_away_goals} {away_team}")
        
        return True, "Prediction successful", {
            'homeTeam': home_team,
            'awayTeam': away_team,
            'homeRedCards': home_red_cards,
            'awayRedCards': away_red_cards,
            'predictedHomeGoals': int(predicted_home_goals),
            'predictedAwayGoals': int(predicted_away_goals)
        }
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return False, str(e), None

# Get statistical analysis data
def get_statistical_data(home_team, away_team):
    """Get model performance statistics and team comparisons"""
    global dataset, home_goals_model, away_goals_model, teams, test_metrics, test_data
    
    if dataset is None or home_goals_model is None or away_goals_model is None:
        return False, "No dataset or models available"
    
    try:
        logger.info(f"Generating statistical data for {home_team} vs {away_team}")
        
        # Validate teams
        if home_team not in teams:
            return False, f"Home team '{home_team}' not found in training data"
        if away_team not in teams:
            return False, f"Away team '{away_team}' not found in training data"
        
        # Use test metrics instead of recalculating on the entire dataset
        metrics = {}
        if test_metrics:
            metrics = {
                'homeGoalsMSE': test_metrics['home_test_mse'],
                'awayGoalsMSE': test_metrics['away_test_mse'],
                'homeGoalsMAE': test_metrics['home_test_mae'],
                'awayGoalsMAE': test_metrics['away_test_mae'],
                'homeGoalsR2': test_metrics['home_test_r2'],
                'awayGoalsR2': test_metrics['away_test_r2'],
                'testSize': test_metrics['test_size'],
                'trainSize': test_metrics['train_size']
            }
        else:
            # Fallback if test metrics are not available (should not happen)
            logger.warning("Test metrics not available, using training data for metrics (less reliable)")
            X = prepare_features(dataset)
            home_preds = home_goals_model.predict(X)
            away_preds = away_goals_model.predict(X)
            
            metrics = {
                'homeGoalsMSE': float(mean_squared_error(dataset['HomeGoals'], home_preds)),
                'awayGoalsMSE': float(mean_squared_error(dataset['AwayGoals'], away_preds)),
                'homeGoalsMAE': float(mean_absolute_error(dataset['HomeGoals'], home_preds)),
                'awayGoalsMAE': float(mean_absolute_error(dataset['AwayGoals'], away_preds)),
                'homeGoalsR2': float(r2_score(dataset['HomeGoals'], home_preds)),
                'awayGoalsR2': float(r2_score(dataset['AwayGoals'], away_preds)),
                'testSize': 0,
                'trainSize': len(dataset)
            }
        
        # Generate regression data from test set if available
        regression_data = []
        if test_data is not None:
            # Get predictions on test data
            home_preds = home_goals_model.predict(test_data['X_test'])
            
            # Create regression data from test set
            sample_indices = np.random.choice(len(home_preds), min(100, len(home_preds)), replace=False)
            
            for idx in sample_indices:
                regression_data.append({
                    'predicted': float(home_preds[idx]),
                    'actual': float(test_data['y_home_test'].iloc[idx]),
                    'index': int(idx)
                })
        else:
            # Fallback - create from training data (less reliable)
            X = prepare_features(dataset)
            home_preds = home_goals_model.predict(X)
            
            sample_indices = np.random.choice(len(home_preds), min(100, len(home_preds)), replace=False)
            
            for idx in sample_indices:
                regression_data.append({
                    'predicted': float(home_preds[idx]),
                    'actual': float(dataset['HomeGoals'].iloc[idx]),
                    'index': int(idx)
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
        
        logger.info(f"Generated statistical data successfully")
        return True, {
            'metrics': metrics,
            'regressionData': regression_data,
            'teamStats': team_goal_stats,
            'homeTeam': home_team_stats,
            'awayTeam': away_team_stats
        }
    
    except Exception as e:
        logger.error(f"Error generating statistics: {str(e)}")
        return False, str(e)

# API Routes - keeping all your existing routes intact

@app.route('/api/train', methods=['POST'])
def api_train():
    """API endpoint to train models - should be restricted to admin users"""
    try:
        logger.info("API request to train models")
        
        # Safely get JSON data or use empty dict
        data = {}
        if request.is_json and request.data:
            data = request.json
        
        num_seasons = data.get('num_seasons', 3)  # Default to 3 seasons
        
        # Pass the parameter to train_models
        success, message = train_models(num_seasons=num_seasons)
        
        return jsonify({
            'success': success,
            'message': message,
            'teams': teams if success else [],
            'seasons_used': num_seasons
        })
    except Exception as e:
        logger.error(f"Error in /api/train: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}',
            'teams': []
        }), 500

@app.route('/api/load_models', methods=['GET'])
def api_load_models():
    """API endpoint to load models (useful for checking status)"""
    success, message = load_existing_models()
    
    return jsonify({
        'success': success,
        'message': message,
        'teams': teams if success else []
    })

@app.route('/api/teams', methods=['GET'])
def api_get_teams():
    """Get list of all teams in the dataset"""
    return jsonify({
        'success': True,
        'teams': teams
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Make a match prediction"""
    try:
        data = request.json
        home_team = data.get('homeTeam')
        away_team = data.get('awayTeam')
        home_red_cards = int(data.get('homeRedCards', 0))
        away_red_cards = int(data.get('awayRedCards', 0))
        
        if not home_team or not away_team:
            return jsonify({'success': False, 'message': 'Home team and away team are required'})
        
        success, message, prediction = predict_match(home_team, away_team, home_red_cards, away_red_cards)
        
        return jsonify({
            'success': success,
            'message': message,
            'prediction': prediction
        })
    except Exception as e:
        logger.error(f"Error in /api/predict: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}',
            'prediction': None
        }), 500

@app.route('/api/stats', methods=['POST'])
def api_get_stats():
    """Get statistical data for teams"""
    try:
        data = request.json
        home_team = data.get('homeTeam')
        away_team = data.get('awayTeam')
        
        if not home_team or not away_team:
            return jsonify({'success': False, 'message': 'Home team and away team are required'})
        
        success, result = get_statistical_data(home_team, away_team)
        
        if success:
            return jsonify({
                'success': True,
                'modelStats': result
            })
        else:
            return jsonify({
                'success': False,
                'message': result
            })
    except Exception as e:
        logger.error(f"Error in /api/stats: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}',
            'modelStats': None
        }), 500

# Keep all your other existing routes (feature_importance, prediction_distribution, etc.)

@app.route('/api/feature_importance', methods=['GET'])
def api_feature_importance():
    """Get feature importance from the model"""
    global home_goals_model, away_goals_model, teams
    
    if home_goals_model is None:
        return jsonify({'success': False, 'message': 'Model not loaded'})
    
    try:
        # Get feature importance
        home_importances = home_goals_model.feature_importances_
        away_importances = away_goals_model.feature_importances_
        
        # Get feature names (including team names)
        feature_names = []
        for team in teams:
            feature_names.append(f'home_{team}')
            feature_names.append(f'away_{team}')
        feature_names.extend(['HomeRedCards', 'AwayRedCards'])
        
        # Create feature importance data
        feature_importance = []
        
        # Add team-level aggregated importance
        team_home_importance = {}
        team_away_importance = {}
        
        for i, name in enumerate(feature_names):
            if i < len(home_importances):
                if name.startswith('home_'):
                    team = name[5:]  # Extract team name
                    if team not in team_home_importance:
                        team_home_importance[team] = 0
                    team_home_importance[team] += home_importances[i]
                elif name.startswith('away_'):
                    team = name[5:]  # Extract team name
                    if team not in team_away_importance:
                        team_away_importance[team] = 0
                    team_away_importance[team] += home_importances[i]
        
        # Get top 10 teams by importance
        top_home_teams = sorted(team_home_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        top_away_teams = sorted(team_away_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Add top teams to feature importance
        for team, importance in top_home_teams:
            feature_importance.append({
                'feature': f'Home: {team}',
                'importance': float(importance)
            })
        
        for team, importance in top_away_teams:
            feature_importance.append({
                'feature': f'Away: {team}',
                'importance': float(importance)
            })
        
        # Add red cards importance
        red_card_index = len(feature_names) - 2  # HomeRedCards index
        if red_card_index < len(home_importances):
            feature_importance.append({
                'feature': 'Home Red Cards',
                'importance': float(home_importances[red_card_index])
            })
        
        red_card_index = len(feature_names) - 1  # AwayRedCards index
        if red_card_index < len(home_importances):
            feature_importance.append({
                'feature': 'Away Red Cards',
                'importance': float(home_importances[red_card_index])
            })
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        return jsonify({
            'success': True,
            'featureImportance': feature_importance
        })
    
    except Exception as e:
        logger.error(f"Error generating feature importance: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })

@app.route('/api/prediction_distribution', methods=['POST'])
def api_prediction_distribution():
    """Get distribution of predictions from all trees in the forest for a specific match"""
    global home_goals_model, away_goals_model, teams
    
    try:
        data = request.json
        home_team = data.get('homeTeam')
        away_team = data.get('awayTeam')
        home_red_cards = int(data.get('homeRedCards', 0))
        away_red_cards = int(data.get('awayRedCards', 0))
        
        if not home_team or not away_team:
            return jsonify({'success': False, 'message': 'Home team and away team are required'})
        
        # Validate teams
        if home_team not in teams or away_team not in teams:
            return jsonify({'success': False, 'message': 'Team not found'})
        
        # Create feature vector
        feature_columns = [f'home_{team}' for team in teams] + \
                         [f'away_{team}' for team in teams] + \
                         ['HomeRedCards', 'AwayRedCards']
        
        feature_vector = pd.DataFrame(columns=feature_columns)
        feature_vector.loc[0] = [0] * len(feature_columns)
        feature_vector[f'home_{home_team}'] = 1
        feature_vector[f'away_{away_team}'] = 1
        feature_vector['HomeRedCards'] = home_red_cards
        feature_vector['AwayRedCards'] = away_red_cards
        
        # Get predictions from all trees in the forest
        home_predictions = [tree.predict(feature_vector)[0] for tree in home_goals_model.estimators_]
        away_predictions = [tree.predict(feature_vector)[0] for tree in away_goals_model.estimators_]
        
        # Create binned distribution data
        home_distribution = {}
        away_distribution = {}
        
        for pred in home_predictions:
            rounded = round(pred)
            if rounded not in home_distribution:
                home_distribution[rounded] = 0
            home_distribution[rounded] += 1
        
        for pred in away_predictions:
            rounded = round(pred)
            if rounded not in away_distribution:
                away_distribution[rounded] = 0
            away_distribution[rounded] += 1
        
        # Convert to list format for frontend
        home_dist_data = [{'goals': k, 'count': v} for k, v in sorted(home_distribution.items())]
        away_dist_data = [{'goals': k, 'count': v} for k, v in sorted(away_distribution.items())]
        
        return jsonify({
            'success': True,
            'homeDistribution': home_dist_data,
            'awayDistribution': away_dist_data,
            'homeAverage': sum(home_predictions) / len(home_predictions),
            'awayAverage': sum(away_predictions) / len(away_predictions)
        })
        
    except Exception as e:
        logger.error(f"Error generating prediction distribution: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/api/model_info', methods=['GET'])
def api_model_info():
    """Get information about the current model state"""
    model_status = "loaded" if home_goals_model is not None and away_goals_model is not None else "not loaded"
    
    # Add test info if available
    test_info = {}
    if test_metrics:
        test_info = {
            'testSize': test_metrics.get('test_size', 0),
            'trainSize': test_metrics.get('train_size', 0),
            'homeR2': test_metrics.get('home_test_r2', 0),
            'awayR2': test_metrics.get('away_test_r2', 0)
        }
    
    return jsonify({
        'success': True,
        'modelStatus': model_status,
        'teamCount': len(teams),
        'datasetRows': len(dataset) if dataset is not None else 0,
        'testInfo': test_info
    })

# Initialize the server
def initialize_server():
    """Initial setup when the server starts"""
    global home_goals_model, away_goals_model, teams, dataset, test_metrics
    
    logger.info("Initializing server...")
    
    # Try to load existing models
    success, message = load_existing_models()
    
    if success:
        logger.info("Successfully loaded existing models")
    else:
        logger.warning(f"Could not load models: {message}")
        
        # Check if dataset exists
        if os.path.exists(DATASET_PATH):
            logger.info(f"Training new models using dataset: {DATASET_PATH}")
            success, train_message = train_models()
            if success:
                logger.info("Successfully trained new models")
            else:
                logger.error(f"Failed to train models: {train_message}")
        else:
            logger.error(f"Dataset not found at {DATASET_PATH}")
    
    model_status = "ready" if home_goals_model is not None and away_goals_model is not None else "not available"
    logger.info(f"Server initialized. Model status: {model_status}")

if __name__ == '__main__':
   # Initialize the server
   initialize_server()
   
   # Start the Flask application
   app.run(host='127.0.0.1', port=5000, debug=True)