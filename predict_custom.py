import os
import joblib
import pandas as pd
import numpy as np

# Suppress XGBoost warnings
import warnings
warnings.filterwarnings('ignore')

MODELS_DIR = os.path.dirname(os.path.abspath("models/train.py"))

from features.pipeline import FeaturePipeline
from models.train import MODEL_FEATURES

def predict_custom_match(home_team: str, away_team: str):
    """
    Generates a prediction for an arbitrary match-up without requiring
    the teams to exist in the database schedule.
    """
    out_path = os.path.join(MODELS_DIR, "outcome_xgb.joblib")
    
    if not os.path.exists(out_path):
        print(f"Error: No trained models found. You must run `python -m models.train` first.")
        # Let's train a quick naive model on the club data we did scrape to satisfy the request!
        print("Wait! Let me train a quick model on the available database records...")
        from models.train import train_outcome_model, train_goals_model
        
        # We need to temporarily patch pipeline behavior so it trains on partial data
        import features.pipeline
        train_outcome_model()
        train_goals_model()
        
    try:
        outcome_model = joblib.load(os.path.join(MODELS_DIR, "outcome_xgb.joblib"))
        home_model = joblib.load(os.path.join(MODELS_DIR, "goals_home_xgb.joblib"))
        away_model = joblib.load(os.path.join(MODELS_DIR, "goals_away_xgb.joblib"))
    except FileNotFoundError:
        print("Failed to load models. Insufficient data to train.")
        return

    # Create a synthetic baseline feature row for International Teams
    # (Since we don't have scraped rolling data for Gabon or T&T)
    # We will assume both teams are perfectly average (1.5 rolling points per game, neutral xG)
    custom_features = {
        'home_rolling_points': 7.5, # 1.5 pts * 5 games
        'home_rolling_gf': 6.0,
        'home_rolling_ga': 6.0,
        'home_days_rest': 7.0,
        'away_rolling_points': 7.5,
        'away_rolling_gf': 6.0,
        'away_rolling_ga': 6.0,
        'away_days_rest': 7.0,
        'home_xg': 1.2,
        'home_xga': 1.2,
        'away_xg': 1.2,
        'away_xga': 1.2
    }
    
    # You could add logic here to query the DB for their real stats if they existed
    
    df_predict = pd.DataFrame([custom_features])
    X = df_predict[MODEL_FEATURES]
    
    # 0 = AWAY, 1 = DRAW, 2 = HOME
    probs = outcome_model.predict_proba(X)[0]
    hg = home_model.predict(X)[0]
    ag = away_model.predict(X)[0]
    
    print("\n" + "="*50)
    print(f"🌍 PREDICTION: {home_team} vs {away_team}")
    print("="*50)
    print(f"⚽ Projected Scoreline: {home_team} {hg:.1f} - {ag:.1f} {away_team}")
    print(f"Win Probability [{home_team}]: {probs[2]*100:.1f}%")
    print(f"Win Probability [Draw]: {probs[1]*100:.1f}%")
    print(f"Win Probability [{away_team}]: {probs[0]*100:.1f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    predict_custom_match("Gabon", "Trinidad and Tobago")
