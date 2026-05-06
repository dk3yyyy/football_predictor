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
    # Ensure models exist
    out_path = os.path.join(MODELS_DIR, "outcome_xgb.joblib")
    h_path = os.path.join(MODELS_DIR, "goals_home_xgb.joblib")
    a_path = os.path.join(MODELS_DIR, "goals_away_xgb.joblib")

    if not all(os.path.exists(p) for p in [out_path, h_path, a_path]):
        logger.error("Models not found! Train them first using `python -m models.train`")
        return []

    logger.info("Loading trained XGBoost Models...")
    outcome_model = joblib.load(out_path)
    home_goals_model = joblib.load(h_path)
    away_goals_model = joblib.load(a_path)

    # Feature columns used in training
    from models.train import MODEL_FEATURES

    # Load data for upcoming games
    pipeline = FeaturePipeline()
    predict_df = pipeline.build_fixtures_predict_set()

    if predict_df.empty:
        logger.info("No upcoming matched available to predict. Data sync required.")
        return []

    predict_df = predict_df.fillna(0.0)
    X = predict_df[MODEL_FEATURES]

    if X.empty:
        logger.info("No upcoming matched available to predict. Data sync required.")
        return []

    # XGBClassifier returns probabilities for [0: AWAY, 1: DRAW, 2: HOME]
    probs = outcome_model.predict_proba(X)

    # XGBRegressors for count goals
    home_g_preds = home_goals_model.predict(X)
    away_g_preds = away_goals_model.predict(X)

    db = Database()
    predictions = []

    for idx, row in predict_df.iterrows():
        i = X.index.get_loc(idx)

        # Build features json so we can backtest our predictors
        features_json = row[MODEL_FEATURES].to_dict()

        pred_record = {
            "match_id": row["match_id"],
            "home_team": row["home_team_name"],
            "away_team": row["away_team_name"],
            "match_date": row["utc_date"].isoformat(),
            "league_key": row.get("league_key", "unknown"),
            "model_name": "xgboost-v1",
            "model_version": "1.0",
            # Ensure native Python floats instead of Numpy Dtypes for JSON serialisation
            "pred_away_win": float(probs[i][0]),
            "pred_draw": float(probs[i][1]),
            "pred_home_win": float(probs[i][2]),
            "pred_home_goals": float(home_g_preds[i]),
            "pred_away_goals": float(away_g_preds[i]),
            "confidence": float(max(probs[i])),
            "features_json": json.dumps(features_json),
            "actual_winner": None,
            "correct": None,
        }

        # Log to Database safely
        db.log_prediction(pred_record)
        predictions.append(pred_record)

        logger.info(
            f"PREDICT: {pred_record['home_team']} vs {pred_record['away_team']} | "
            f"Home: {pred_record['pred_home_win']:.2f}% | "
            f"Away: {pred_record['pred_away_win']:.2f}%"
        )

    return predictions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    get_predictions()
