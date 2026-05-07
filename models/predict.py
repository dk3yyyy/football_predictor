import json
import logging
import os

import joblib

from db.database import Database
from features.pipeline import FeaturePipeline

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))


def get_predictions() -> list[dict]:
    """Generates predictions for upcoming matches, and saves to the Database log."""
    from models.train import GOAL_FEATURES, MODEL_FEATURES

    # Load models
    out_path = os.path.join(MODELS_DIR, "outcome_ensemble.joblib")
    h_path = os.path.join(MODELS_DIR, "goals_home_xgb.joblib")
    a_path = os.path.join(MODELS_DIR, "goals_away_xgb.joblib")

    if not os.path.exists(out_path):
        logger.error("Ensemble model not found! Train first.")
        return []

    logger.info("Loading trained Ensemble Model...")
    outcome_model = joblib.load(out_path)

    if not os.path.exists(h_path) or not os.path.exists(a_path):
        logger.warning("Goals models not found!")
        return []

    home_goals_model = joblib.load(h_path)
    away_goals_model = joblib.load(a_path)

    # Load data for upcoming games
    pipeline = FeaturePipeline()
    predict_df = pipeline.build_fixtures_predict_set()

    if predict_df.empty:
        logger.info("No upcoming matches available.")
        return []

    predict_df = predict_df.fillna(0.0)
    X_base = predict_df[MODEL_FEATURES]

    if X_base.empty:
        logger.info("No features available.")
        return []

    # Stage 1: Get goal predictions
    logger.info("Generating goal predictions...")
    pred_home = home_goals_model.predict(X_base)
    pred_away = away_goals_model.predict(X_base)

    # Stage 2: Add goal features for stacked model
    X_full = X_base.copy()
    X_full["predicted_home_goals"] = pred_home
    X_full["predicted_away_goals"] = pred_away
    X_full["predicted_goal_diff"] = pred_home - pred_away

    all_features = MODEL_FEATURES + GOAL_FEATURES

    # XGBClassifier returns probabilities for [0: AWAY, 1: DRAW, 2: HOME]
    probs = outcome_model.predict_proba(X_full[all_features])

    db = Database()
    predictions = []

    for idx, row in predict_df.iterrows():
        i = X_full.index.get_loc(idx)

        features_json = row[MODEL_FEATURES].to_dict()
        features_json["predicted_home_goals"] = float(pred_home[i])
        features_json["predicted_away_goals"] = float(pred_away[i])
        features_json["predicted_goal_diff"] = float(pred_home[i] - pred_away[i])

        pred_record = {
            "match_id": row["match_id"],
            "home_team": row["home_team_name"],
            "away_team": row["away_team_name"],
            "match_date": row["utc_date"].isoformat(),
            "league_key": row.get("league_key", "unknown"),
            "model_name": "xgboost-v2-stacked",
            "model_version": "2.0",
            "pred_away_win": float(probs[i][0]),
            "pred_draw": float(probs[i][1]),
            "pred_home_win": float(probs[i][2]),
            "pred_home_goals": float(pred_home[i]),
            "pred_away_goals": float(pred_away[i]),
            "confidence": float(max(probs[i])),
            "features_json": json.dumps(features_json),
            "actual_winner": None,
            "correct": None,
        }

        db.log_prediction(pred_record)
        predictions.append(pred_record)

        logger.info(
            f"PREDICT: {pred_record['home_team']} vs {pred_record['away_team']} | "
            f"Home: {pred_record['pred_home_win']:.1%} | Draw: {pred_record['pred_draw']:.1%} | "
            f"Away: {pred_record['pred_away_win']:.1%}"
        )

    return predictions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    get_predictions()
