import logging
import os

import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from features.pipeline import FeaturePipeline

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FEATURES = [
    "home_rolling_points",
    "home_rolling_gf",
    "home_rolling_ga",
    "home_days_rest",
    "away_rolling_points",
    "away_rolling_gf",
    "away_rolling_ga",
    "away_days_rest",
    "home_xg",
    "home_xga",
    "away_xg",
    "away_xga",
]


def train_outcome_model():
    """Trains an XGBoost multiclass classifier to predict Match Outcomes."""
    pipeline = FeaturePipeline()
    df = pipeline.build_dataset()

    if df.empty or len(df) < 50:
        logger.warning("Not enough data to train models. Skipping.")
        return

    # Drop NaNs carefully (the pipeline drops initial ones, but keep it secure)
    df = df.dropna(subset=MODEL_FEATURES)

    X = df[MODEL_FEATURES]

    # Map target: 0 = AWAY, 1 = DRAW, 2 = HOME
    label_map = {"AWAY_TEAM": 0, "DRAW": 1, "HOME_TEAM": 2}
    y = df["winner"].map(label_map)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=3,
    )

    logger.info("Training outcome classifier...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Save the model
    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, "outcome_xgb.joblib")
    joblib.dump(model, out_path)
    logger.info(f"Outcome model saved to {out_path}")


def train_goals_model():
    """
    Trains XGBoost regressors with Poisson objectives for Home and Away Goals.
    This effectively substitutes for a manual statsmodels Poisson regression
    but utilizes the boosted trees capacity for non-linear interactions.
    """
    pipeline = FeaturePipeline()
    df = pipeline.build_dataset()

    if df.empty or len(df) < 50:
        return

    df = df.dropna(subset=MODEL_FEATURES)
    X = df[MODEL_FEATURES]
    y_home = df["home_goals_ft"]
    y_away = df["away_goals_ft"]

    # We train on full dataset for recent forms
    home_model = XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05, objective="count:poisson"
    )

    away_model = XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05, objective="count:poisson"
    )

    logger.info("Training home/away Poisson goals models...")
    home_model.fit(X, y_home, verbose=False)
    away_model.fit(X, y_away, verbose=False)

    joblib.dump(home_model, os.path.join(MODELS_DIR, "goals_home_xgb.joblib"))
    joblib.dump(away_model, os.path.join(MODELS_DIR, "goals_away_xgb.joblib"))
    logger.info("Goals models saved.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_outcome_model()
    train_goals_model()
