import os
import joblib
import pandas as pd
import numpy as np
import warnings
import logging
from datetime import datetime

# Suppress XGBoost warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.ERROR)

from db.database import Database
from features.pipeline import FeaturePipeline
from models.train import MODEL_FEATURES

MODELS_DIR = os.path.dirname(os.path.abspath("models/train.py"))

# Mapping user names to DB names
TEAM_MAPPING = {
    "Inter": "FC Internazionale Milano",
    "Cagliari": "Cagliari Calcio",
    "RC Lens": "Racing Club de Lens",
    "Toulouse": "Toulouse FC",
    "Sassuolo": "US Sassuolo Calcio",
    "Como": "Como 1907"
}

def get_team_features(pipeline, team_name):
    """Fetches rolling stats and xG for a team from the database if available."""
    db_name = TEAM_MAPPING.get(team_name, team_name)
    
    # Try to find recent matches for rolling stats
    recent = pipeline.db.get_recent_matches(db_name, n=5)
    
    if len(recent) < 5:
        # Not enough data, use baseline
        return {
            'rolling_points': 7.5,
            'rolling_gf': 6.0,
            'rolling_ga': 6.0,
            'days_rest': 7.0,
            'xg': 1.2,
            'xga': 1.2
        }
    
    # Calculate rolling stats from recent matches
    points = 0
    gf = 0
    ga = 0
    for m in recent:
        if m['home_team_name'] == db_name:
            gf += m['home_goals_ft'] or 0
            ga += m['away_goals_ft'] or 0
            if m['winner'] == 'HOME_TEAM': points += 3
            elif m['winner'] == 'DRAW': points += 1
        else:
            gf += m['away_goals_ft'] or 0
            ga += m['home_goals_ft'] or 0
            if m['winner'] == 'AWAY_TEAM': points += 3
            elif m['winner'] == 'DRAW': points += 1
            
    # xG (simplified: try to find any xG record)
    with pipeline.db.engine.connect() as conn:
        from sqlalchemy import text
        xg_rec = conn.execute(text("SELECT xg, xga FROM xg_stats WHERE team_name = :t ORDER BY scraped_at DESC LIMIT 1"), {"t": db_name}).fetchone()
    
    xg = xg_rec[0] if xg_rec else 1.2
    xga = xg_rec[1] if xg_rec else 1.2
    
    # Fatigue
    last_date = pd.to_datetime(recent[0]['utc_date'])
    days_rest = (pd.Timestamp.now(tz='UTC').replace(tzinfo=None) - last_date.replace(tzinfo=None)).days
    
    return {
        'rolling_points': float(points),
        'rolling_gf': float(gf),
        'rolling_ga': float(ga),
        'days_rest': float(min(days_rest, 14)), # cap at 14
        'xg': float(xg),
        'xga': float(xga)
    }

def predict_custom_batch(matches_list):
    db = Database()
    pipeline = FeaturePipeline(db)
    
    outcome_model = joblib.load(os.path.join(MODELS_DIR, "outcome_xgb.joblib"))
    home_model = joblib.load(os.path.join(MODELS_DIR, "goals_home_xgb.joblib"))
    away_model = joblib.load(os.path.join(MODELS_DIR, "goals_away_xgb.joblib"))

    results = []
    
    for home_team, away_team in matches_list:
        h_feats = get_team_features(pipeline, home_team)
        a_feats = get_team_features(pipeline, away_team)
        
        custom_features = {
            'home_rolling_points': h_feats['rolling_points'],
            'home_rolling_gf': h_feats['rolling_gf'],
            'home_rolling_ga': h_feats['rolling_ga'],
            'home_days_rest': h_feats['days_rest'],
            'away_rolling_points': a_feats['rolling_points'],
            'away_rolling_gf': a_feats['rolling_gf'],
            'away_rolling_ga': a_feats['rolling_ga'],
            'away_days_rest': a_feats['days_rest'],
            'home_xg': h_feats['xg'],
            'home_xga': h_feats['xga'],
            'away_xg': a_feats['xg'],
            'away_xga': a_feats['xga']
        }
        
        df_predict = pd.DataFrame([custom_features])
        X = df_predict[MODEL_FEATURES]
        
        probs = outcome_model.predict_proba(X)[0]
        hg = home_model.predict(X)[0]
        ag = away_model.predict(X)[0]
        
        results.append({
            'home': home_team,
            'away': away_team,
            'hg': hg,
            'ag': ag,
            'prob_home': probs[2],
            'prob_draw': probs[1],
            'prob_away': probs[0]
        })

    print("\n🚀 BATCH PREDICTIONS (Updated with DB stats where available)\n")
    for r in results:
        print("="*60)
        print(f"🌍 {r['home']} vs {r['away']}")
        print(f"⚽ Projected: {r['home']} {r['hg']:.1f} - {r['ag']:.1f} {r['away']}")
        print(f"Win%: {r['home']} {r['prob_home']*100:.1f}% | Draw {r['prob_draw']*100:.1f}% | {r['away']} {r['prob_away']*100:.1f}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    matches = [
        ("Inter", "Cagliari"),
        ("FC Dynamo Kyiv", "FC Zorya Luhansk"),
        ("Vitesse Arnhem", "MVV Maastricht"),
        ("Al Ahli Saudi FC", "Johor Darul Tazim FC"),
        ("Fenerbahce Istanbul", "Caykur Rizespor"),
        ("FK Rostov", "FK Sochi"),
        ("SV 07 Elversberg", "Karlsruher SC"),
        ("Roda JC Kerkrade", "FC Emmen"),
        ("RC Lens", "Toulouse"),
        ("Sassuolo", "Como")
    ]
    predict_custom_batch(matches)
