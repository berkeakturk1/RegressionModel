import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Base directory for resolving paths
BASE_DIR = os.path.dirname(__file__)

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
        # Load and preprocess the dataset
        df = pd.read_csv(dataset_path)
        df = preprocess_data(df)

        # Extract list of teams
        teams = sorted(set(df['Home']) | set(df['Away']))

        # Prepare feature matrix
        X = prepare_features(df)
        y_home = df['HomeGoals']
        y_away = df['AwayGoals']

        # Train models
        home_goals_model = RandomForestRegressor(n_estimators=100, random_state=42)
        home_goals_model.fit(X, y_home)

        away_goals_model = RandomForestRegressor(n_estimators=100, random_state=42)
        away_goals_model.fit(X, y_away)

        # Save models and additional artifacts
        models_dir = os.path.join(BASE_DIR, 'models')
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(home_goals_model, os.path.join(models_dir, 'home_goals_model.pkl'))
        joblib.dump(away_goals_model, os.path.join(models_dir, 'away_goals_model.pkl'))
        with open(os.path.join(models_dir, 'teams.txt'), 'w') as f:
            for t in teams:
                f.write(f"{t}\n")
        df.to_csv(os.path.join(models_dir, 'dataset.csv'), index=False)

        dataset = df
        return True, "Models trained successfully"
    except Exception as e:
        return False, str(e)

# Preprocess the dataset
def preprocess_data(df):
    # Rename columns to match pipeline
    df = df.rename(columns={
        'home': 'Home',
        'visitor': 'Away',
        'hgoal': 'HomeGoals',
        'vgoal': 'AwayGoals',
        'home_red_card': 'HomeRedCards',
        'visitor_red_card': 'AwayRedCards'
    })
    df = df.dropna(subset=['Home', 'Away', 'HomeGoals', 'AwayGoals'])

    # Ensure red-card columns exist
    if 'HomeRedCards' not in df.columns:
        df['HomeRedCards'] = 0
    if 'AwayRedCards' not in df.columns:
        df['AwayRedCards'] = 0
    return df

# Prepare features for prediction
def prepare_features(df):
    home_dummies = pd.get_dummies(df['Home'], prefix='home')
    away_dummies = pd.get_dummies(df['Away'], prefix='away')
    features = pd.concat([home_dummies, away_dummies, df[['HomeRedCards', 'AwayRedCards']]], axis=1)
    return features

# Load existing models
def load_existing_models():
    global home_goals_model, away_goals_model, teams, dataset
    try:
        models_dir = os.path.join(BASE_DIR, 'models')
        home_path = os.path.join(models_dir, 'home_goals_model.pkl')
        away_path = os.path.join(models_dir, 'away_goals_model.pkl')
        teams_path = os.path.join(models_dir, 'teams.txt')
        data_path = os.path.join(models_dir, 'dataset.csv')

        if not os.path.exists(home_path) or not os.path.exists(away_path):
            return False, "Models not found"

        home_goals_model = joblib.load(home_path)
        away_goals_model = joblib.load(away_path)

        # Load teams
        if os.path.exists(teams_path):
            with open(teams_path) as f:
                teams = [line.strip() for line in f]

        # Load dataset for stats endpoint
        if os.path.exists(data_path):
            dataset = pd.read_csv(data_path)
            dataset = preprocess_data(dataset)

        return True, "Models loaded successfully"
    except Exception as e:
        # Fallback team list on error
        if not teams:
            teams.extend([
                "Galatasaray", "Fenerbahçe", "Beşiktaş", "Trabzonspor",
                "İstanbul Başakşehir", "Adana Demirspor", "Antalyaspor",
                "Konyaspor", "Kayserispor", "Hatayspor"
            ])
        return False, str(e)

# Prediction and statistical functions unchanged (predict_match, get_statistical_data)

# API Routes
@app.route('/api/train', methods=['POST'])
def api_train():
    dataset_path = os.path.join(BASE_DIR, 'Backend', 'tsl_dataset.csv')
    if not os.path.exists(dataset_path):
        return jsonify({
            'success': False,
            'message': f'Dataset file not found at {dataset_path}',
            'teams': []
        }), 404
    success, message = train_models(dataset_path)
    return jsonify({
        'success': success,
        'message': message,
        'teams': teams if success else []
    })

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
    return jsonify({'success': True, 'teams': teams})

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
    return jsonify({'success': success, 'message': message, 'prediction': prediction})

@app.route('/api/stats', methods=['POST'])
def api_get_stats():
    data = request.json
    home_team = data.get('homeTeam')
    away_team = data.get('awayTeam')
    if not home_team or not away_team:
        return jsonify({'success': False, 'error': 'Home team and away team are required'})
    success, result = get_statistical_data(home_team, away_team)
    if success:
        return jsonify({'success': True, 'modelStats': result})
    else:
        return jsonify({'success': False, 'error': result})

if __name__ == '__main__':
    load_existing_models()
    app.run(debug=True)
