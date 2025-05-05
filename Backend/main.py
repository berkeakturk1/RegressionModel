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

# Global state
home_goals_model = None
away_goals_model = None
teams = []
dataset = None

# Preprocess the dataset
def preprocess_data(df):
    df = df.copy()
    df = df.dropna(subset=['Home', 'Away', 'HomeGoals', 'AwayGoals'])

    df.rename(columns={
        'Home': 'homeTeam',
        'Away': 'awayTeam',
        'HomeGoals': 'homeGoals',
        'AwayGoals': 'awayGoals',
        'HomeRedCards': 'homeRedCards',
        'AwayRedCards': 'awayRedCards'
    }, inplace=True)

    df['homeRedCards'] = df.get('homeRedCards', 0)
    df['awayRedCards'] = df.get('awayRedCards', 0)

    return df

# Prepare one-hot encoded features
def prepare_features(df):
    home_dummies = pd.get_dummies(df['homeTeam'], prefix='home')
    away_dummies = pd.get_dummies(df['awayTeam'], prefix='away')
    return pd.concat([home_dummies, away_dummies, df[['homeRedCards', 'awayRedCards']]], axis=1)

# Train and save models
def train_models(dataset_path):
    global home_goals_model, away_goals_model, teams, dataset

    try:
        dataset = pd.read_csv(dataset_path)
        dataset = preprocess_data(dataset)

        teams = sorted(list(set(dataset['homeTeam'].unique()) | set(dataset['awayTeam'].unique())))
        X = prepare_features(dataset)

        home_goals_model = RandomForestRegressor(n_estimators=100, random_state=42)
        home_goals_model.fit(X, dataset['homeGoals'])

        away_goals_model = RandomForestRegressor(n_estimators=100, random_state=42)
        away_goals_model.fit(X, dataset['awayGoals'])

        os.makedirs('models', exist_ok=True)
        joblib.dump(home_goals_model, 'models/home_goals_model.pkl')
        joblib.dump(away_goals_model, 'models/away_goals_model.pkl')
        dataset.to_csv('models/dataset.csv', index=False)

        with open('models/teams.txt', 'w') as f:
            f.writelines([t + '\n' for t in teams])

        return True, "Models trained successfully"
    except Exception as e:
        return False, str(e)

# Load saved models
def load_existing_models():
    global home_goals_model, away_goals_model, teams, dataset

    try:
        home_goals_model = joblib.load('models/home_goals_model.pkl')
        away_goals_model = joblib.load('models/away_goals_model.pkl')

        if os.path.exists('models/teams.txt'):
            with open('models/teams.txt', 'r') as f:
                teams = [line.strip() for line in f.readlines()]

        if os.path.exists('models/dataset.csv'):
            dataset = pd.read_csv('models/dataset.csv')
            dataset = preprocess_data(dataset)

        return True, "Models loaded successfully"
    except Exception as e:
        return False, str(e)

# Predict a single match
def predict_match(home_team, away_team, home_red_cards, away_red_cards):
    if home_goals_model is None or away_goals_model is None:
        return False, "Models not loaded", None

    try:
        feature_vector = pd.DataFrame(columns=[f'home_{t}' for t in teams] + [f'away_{t}' for t in teams] + ['homeRedCards', 'awayRedCards'])
        feature_vector.loc[0] = 0

        if f'home_{home_team}' in feature_vector.columns:
            feature_vector[f'home_{home_team}'] = 1
        if f'away_{away_team}' in feature_vector.columns:
            feature_vector[f'away_{away_team}'] = 1

        feature_vector['homeRedCards'] = home_red_cards
        feature_vector['awayRedCards'] = away_red_cards

        predicted_home_goals = round(home_goals_model.predict(feature_vector)[0])
        predicted_away_goals = round(away_goals_model.predict(feature_vector)[0])

        return True, "Prediction successful", {
            'homeTeam': home_team,
            'awayTeam': away_team,
            'homeRedCards': home_red_cards,
            'awayRedCards': away_red_cards,
            'predictedHomeGoals': predicted_home_goals,
            'predictedAwayGoals': predicted_away_goals
        }
    except Exception as e:
        return False, str(e), None

# Get evaluation metrics + team stats
def get_statistical_data(home_team, away_team):
    if dataset is None or home_goals_model is None or away_goals_model is None:
        return False, "No dataset or models loaded"

    try:
        X = prepare_features(dataset)
        home_preds = home_goals_model.predict(X)
        away_preds = away_goals_model.predict(X)

        regression_data = [
            {
                'predicted': float(home_preds[i]),
                'actual': float(dataset['homeGoals'].iloc[i]),
                'index': i
            } for i in range(min(100, len(dataset)))
        ]

        metrics = {
            'homeGoalsMSE': mean_squared_error(dataset['homeGoals'], home_preds),
            'awayGoalsMSE': mean_squared_error(dataset['awayGoals'], away_preds),
            'homeGoalsMAE': mean_absolute_error(dataset['homeGoals'], home_preds),
            'awayGoalsMAE': mean_absolute_error(dataset['awayGoals'], away_preds),
            'homeGoalsR2': r2_score(dataset['homeGoals'], home_preds),
            'awayGoalsR2': r2_score(dataset['awayGoals'], away_preds),
        }

        home_matches = dataset[dataset['homeTeam'] == home_team]
        away_matches = dataset[dataset['awayTeam'] == away_team]

        home_team_stats = {
            'name': home_team,
            'avgHomeGoals': home_matches['homeGoals'].mean(),
            'avgHomeGoalsAgainst': home_matches['awayGoals'].mean(),
            'homeWinRate': (home_matches['homeGoals'] > home_matches['awayGoals']).mean(),
            'homeDrawRate': (home_matches['homeGoals'] == home_matches['awayGoals']).mean(),
            'homeLossRate': (home_matches['homeGoals'] < home_matches['awayGoals']).mean()
        }

        away_team_stats = {
            'name': away_team,
            'avgAwayGoals': away_matches['awayGoals'].mean(),
            'avgAwayGoalsAgainst': away_matches['homeGoals'].mean(),
            'awayWinRate': (away_matches['awayGoals'] > away_matches['homeGoals']).mean(),
            'awayDrawRate': (away_matches['awayGoals'] == away_matches['homeGoals']).mean(),
            'awayLossRate': (away_matches['awayGoals'] < away_matches['homeGoals']).mean()
        }

        top_teams = dataset['homeTeam'].value_counts().head(5).index.tolist()
        team_goal_stats = []

        for team in top_teams:
            team_goal_stats.append({
                'team': team,
                'homeGoals': dataset[dataset['homeTeam'] == team]['homeGoals'].mean(),
                'awayGoals': dataset[dataset['awayTeam'] == team]['awayGoals'].mean()
            })

        return True, {
            'metrics': metrics,
            'regressionData': regression_data,
            'teamStats': team_goal_stats,
            'homeTeam': home_team_stats,
            'awayTeam': away_team_stats
        }
    except Exception as e:
        return False, str(e)

# ---------- ROUTES ----------

@app.route('/api/train', methods=['POST'])
def api_train():
    dataset_path = os.path.abspath('./tsl_dataset.csv')
    if not os.path.exists(dataset_path):
        return jsonify({'success': False, 'message': 'Dataset file not found.', 'teams': []}), 404

    success, message = train_models(dataset_path)
    return jsonify({'success': success, 'message': message, 'teams': teams if success else []})

@app.route('/api/load_models', methods=['GET'])
def api_load_models():
    success, message = load_existing_models()
    return jsonify({'success': success, 'message': message, 'teams': teams if success else []})

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
        return jsonify({'success': False, 'error': 'Both teams are required'})

    success, message, prediction = predict_match(home_team, away_team, home_red_cards, away_red_cards)
    return jsonify({'success': success, 'message': message, 'prediction': prediction})

@app.route('/api/stats', methods=['POST'])
def api_get_stats():
    data = request.json
    home_team = data.get('homeTeam')
    away_team = data.get('awayTeam')

    if not home_team or not away_team:
        return jsonify({'success': False, 'error': 'Both teams are required'})

    success, result = get_statistical_data(home_team, away_team)
    if success:
        return jsonify({'success': True, 'modelStats': result})
    else:
        return jsonify({'success': False, 'error': result})

if __name__ == '__main__':
    load_existing_models()
    app.run(debug=True)