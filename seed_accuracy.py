import joblib
import json
import pandas as pd
from sklearn.metrics import accuracy_score
from features.pipeline import FeaturePipeline
from models.train import MODEL_FEATURES, GOAL_FEATURES, LABEL_MAP
from db.database import Database
import logging

logging.basicConfig(level=logging.INFO)

db = Database()

outcome_model = joblib.load("models/outcome_ensemble.joblib")
home_goals_model = joblib.load("models/goals_home_xgb.joblib")
away_goals_model = joblib.load("models/goals_away_xgb.joblib")

pipeline = FeaturePipeline()
df = pipeline.build_dataset()

if df.empty:
    print("No data")
    exit()

df = df.dropna(subset=MODEL_FEATURES)

y = df["winner"].map(LABEL_MAP)
y = y.dropna()
X_base = df[MODEL_FEATURES].loc[y.index]
df_aligned = df.loc[y.index]

pred_home = home_goals_model.predict(X_base)
pred_away = away_goals_model.predict(X_base)

X_full = X_base.copy()
X_full["predicted_home_goals"] = pred_home
X_full["predicted_away_goals"] = pred_away
X_full["predicted_goal_diff"] = pred_home - pred_away

all_features = MODEL_FEATURES + GOAL_FEATURES

n = len(X_full)
test_start = int(n * 0.80)
X_test = X_full.iloc[test_start:][all_features]
y_test = y.iloc[test_start:]
df_test = df_aligned.iloc[test_start:]

y_pred = outcome_model.predict(X_test)
y_proba = outcome_model.predict_proba(X_test)

LABEL_MAP_REV = {0: "AWAY_TEAM", 1: "DRAW", 2: "HOME_TEAM"}
label_to_winner = {v: k for k, v in LABEL_MAP.items()}

predictions = []

for i in range(len(X_test)):
    row_base = X_base.iloc[test_start + i]
    row_df = df_test.iloc[i]
    
    pred_idx = test_start + i
    pred_home_val = pred_home[pred_idx]
    pred_away_val = pred_away[pred_idx]
    
    y_pred_val = int(y_pred[i])
    predicted_winner = LABEL_MAP_REV.get(y_pred_val, "HOME_TEAM")
    actual_winner = label_to_winner.get(y_test.iloc[i])
    correct = 1 if predicted_winner == actual_winner else 0
    
    features_json = {k: float(row_base[k]) for k in MODEL_FEATURES}
    features_json["predicted_home_goals"] = float(pred_home_val)
    features_json["predicted_away_goals"] = float(pred_away_val)
    features_json["predicted_goal_diff"] = float(pred_home_val - pred_away_val)
    
    pred_record = {
        "match_id": int(row_df["match_id"]),
        "home_team": row_df["home_team_name"],
        "away_team": row_df["away_team_name"],
        "match_date": str(row_df["utc_date"]),
        "league_key": row_df.get("league_key", "unknown"),
        "model_name": "xgboost-v2-stacked",
        "model_version": "2.0",
        "pred_away_win": float(y_proba[i][0]),
        "pred_draw": float(y_proba[i][1]),
        "pred_home_win": float(y_proba[i][2]),
        "pred_home_goals": float(pred_home_val),
        "pred_away_goals": float(pred_away_val),
        "confidence": float(max(y_proba[i])),
        "features_json": json.dumps(features_json),
        "actual_winner": actual_winner,
        "correct": correct,
    }
    
    predictions.append(pred_record)

for pred in predictions:
    db.log_prediction(pred)

print(f"Inserted {len(predictions)} historical predictions")

with db.engine.connect() as conn:
    from sqlalchemy import text
    result = conn.execute(text('''
        SELECT COUNT(*) as total,
            SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END) as correct
        FROM predictions_log WHERE actual_winner IS NOT NULL
    ''')).fetchone()
    
    if result.total:
        print(f'=== Model Accuracy ===')
        print(f'Total: {result.total}')
        print(f'Correct: {result.correct}')
        print(f'Accuracy: {result.correct/result.total*100:.1f}%')