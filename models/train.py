import logging
import os

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier, XGBRegressor

from features.pipeline import FeaturePipeline

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FEATURES = [
    # Core form metrics (5-game rolling)
    "home_rolling_points",
    "home_rolling_gf",
    "home_rolling_ga",
    "home_rolling_gd",
    "home_rolling_ppg",
    "home_rolling_cs",
    "home_rolling_btts",
    "home_days_rest",
    "away_rolling_points",
    "away_rolling_gf",
    "away_rolling_ga",
    "away_rolling_gd",
    "away_rolling_ppg",
    "away_rolling_cs",
    "away_rolling_btts",
    "away_days_rest",
    # Head-to-Head features
    "home_h2h_points",
    "away_h2h_points",
    "h2h_draws",
    # Home/Away split form
    "home_home_form",
    "away_away_form",
    # Elo ratings
    "home_elo",
    "away_elo",
    "elo_diff",
]

# Goal prediction features (will be added after stacking)
GOAL_FEATURES = ["predicted_home_goals", "predicted_away_goals", "predicted_goal_diff"]

LABEL_MAP = {"AWAY_TEAM": 0, "DRAW": 1, "HOME_TEAM": 2}


def train_outcome_model():
    """Trains a stacked model: goals predictions -> outcome prediction."""
    from sklearn.calibration import CalibratedClassifierCV
    import joblib

    pipeline = FeaturePipeline()
    df = pipeline.build_dataset()

    if df.empty or len(df) < 50:
        logger.warning("Not enough data to train models. Skipping.")
        return

    # Load or train goals models first
    try:
        home_model = joblib.load(os.path.join(MODELS_DIR, "goals_home_xgb.joblib"))
        away_model = joblib.load(os.path.join(MODELS_DIR, "goals_away_xgb.joblib"))
        logger.info("Loaded pre-trained goals models")
    except Exception:
        logger.warning("Goals models not found, training now...")
        train_goals_model()
        home_model = joblib.load(os.path.join(MODELS_DIR, "goals_home_xgb.joblib"))
        away_model = joblib.load(os.path.join(MODELS_DIR, "goals_away_xgb.joblib"))

    # Drop NaNs
    df = df.dropna(subset=MODEL_FEATURES)

    X_base = df[MODEL_FEATURES]
    y = df["winner"].map(LABEL_MAP)
    y = y.dropna()
    X_base = X_base.loc[y.index]

    # Stage 1: Get goal predictions (use OOF-style for training data)
    logger.info("Generating goal predictions for stacking...")
    pred_home = home_model.predict(X_base)
    pred_away = away_model.predict(X_base)

    # Add goal features to training data
    X_full = X_base.copy()
    X_full["predicted_home_goals"] = pred_home
    X_full["predicted_away_goals"] = pred_away
    X_full["predicted_goal_diff"] = pred_home - pred_away

    all_features = MODEL_FEATURES + GOAL_FEATURES

    logger.info(f"Training on {len(X_full)} matches with {len(all_features)} features")

    # Time-series split: 70% train, 15% calibrate, 15% test
    n = len(X_full)
    train_end = int(n * 0.70)
    cal_end = int(n * 0.85)

    X_train = X_full.iloc[:train_end][all_features]
    X_cal = X_full.iloc[train_end:cal_end][all_features]
    X_test = X_full.iloc[cal_end:][all_features]

    y_train = y.iloc[:train_end]
    y_cal = y.iloc[train_end:cal_end]
    y_test = y.iloc[cal_end:]

    logger.info(f"Splits: train={len(X_train)}, calibrate={len(X_cal)}, test={len(X_test)}")

    # Tuned XGBoost with regularization
    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.03,
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=3,
        colsample_bytree=0.7,
        subsample=0.7,
        reg_alpha=0.5,
        reg_lambda=1.5,
        verbosity=0,
    )

    logger.info("Training stacked XGBoost classifier...")
    model.fit(X_train, y_train, eval_set=[(X_cal, y_cal)], verbose=False)

    # Calibrate
    logger.info("Calibrating probabilities with isotonic regression...")
    calibrated_model = CalibratedClassifierCV(model, method="isotonic", cv=3)
    calibrated_model.fit(X_cal, y_cal)

    # Evaluate
    y_pred = calibrated_model.predict(X_test)
    y_proba = calibrated_model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)

    logger.info(f"Test accuracy: {acc:.1%}")
    logger.info(f"Test log loss: {ll:.4f}")

    # Feature importance
    importances = dict(zip(all_features, model.feature_importances_, strict=True))
    top = sorted(importances.items(), key=lambda x: -x[1])[:10]
    logger.info("Top features:")
    for feat, imp in top:
        logger.info(f"  {feat:25s}: {imp:.4f}")

    # Save the calibrated model
    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, "outcome_ensemble.joblib")
    joblib.dump(calibrated_model, out_path)
    logger.info(f"Model saved to {out_path}")


def train_goals_model():
    """Trains XGBoost Poisson regressors for Home and Away Goals."""
    pipeline = FeaturePipeline()
    df = pipeline.build_dataset()

    if df.empty or len(df) < 50:
        return

    df = df.dropna(subset=MODEL_FEATURES)
    X = df[MODEL_FEATURES]
    y_home = df["home_goals_ft"]
    y_away = df["away_goals_ft"]

    # Tuned Poisson regressors
    base_params = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.03,
        "objective": "count:poisson",
        "colsample_bytree": 0.7,
        "subsample": 0.7,
        "reg_alpha": 0.3,
        "reg_lambda": 1.0,
        "verbosity": 0,
    }

    home_model = XGBRegressor(**base_params)
    away_model = XGBRegressor(**base_params)

    logger.info("Training home/away Poisson goals models...")
    home_model.fit(X, y_home)
    away_model.fit(X, y_away)

    logger.info(f"Home goals MAE: {np.abs(y_home - home_model.predict(X)).mean():.2f}")
    logger.info(f"Away goals MAE: {np.abs(y_away - away_model.predict(X)).mean():.2f}")

    joblib.dump(home_model, os.path.join(MODELS_DIR, "goals_home_xgb.joblib"))
    joblib.dump(away_model, os.path.join(MODELS_DIR, "goals_away_xgb.joblib"))
    logger.info("Goals models saved.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_outcome_model()
    train_goals_model()
