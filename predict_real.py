import os

import joblib
import pandas as pd

from db.database import Database
from features.pipeline import FeaturePipeline
from models.train import GOAL_FEATURES, MODEL_FEATURES

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))


def predict_real_match(home_query: str, away_query: str):
    db = Database()
    pipe = FeaturePipeline(db)

    # 1. Load all data to get rolling stats
    print("Loading database and computing latest team form...")
    df = pipe.build_dataset()

    if df.empty:
        print("Error: DataFrame is empty. Run backfill first.")
        return

    teams = set(df["home_team_name"]).union(set(df["away_team_name"]))

    # 2. Basic text matching for team names
    home_name = next((t for t in teams if home_query.lower() in t.lower()), None)
    away_name = next((t for t in teams if away_query.lower() in t.lower()), None)

    if not home_name:
        print(f"Error: Could not find team matching '{home_query}' in database.")
        print("Available teams:", sorted(list(teams))[:15], "...")
        return

    if not away_name:
        print(f"Error: Could not find team matching '{away_query}' in database.")
        return

    print(f"\nResolved Teams: {home_name} (Home) vs {away_name} (Away)")

    # 3. Get the latest match for both teams to extract their CURRENT rolling form
    # We sort by date, drop NAs

    # Home team latest stats (as home or away)
    home_matches = df[
        (df["home_team_name"] == home_name) | (df["away_team_name"] == home_name)
    ].sort_values("utc_date")
    if home_matches.empty:
        print("No match history for Home team.")
        return
    home_latest = home_matches.iloc[-1]

    # Is Home team the home or away team in their last match?
    # That determines whether we pull home_rolling_points or away_rolling_points
    if home_latest["home_team_name"] == home_name:
        h_pts = home_latest["home_rolling_points"]
        h_gf = home_latest["home_rolling_gf"]
        h_ga = home_latest["home_rolling_ga"]
    else:
        h_pts = home_latest["away_rolling_points"]
        h_gf = home_latest["away_rolling_gf"]
        h_ga = home_latest["away_rolling_ga"]

    # Same for Away team
    away_matches = df[
        (df["home_team_name"] == away_name) | (df["away_team_name"] == away_name)
    ].sort_values("utc_date")
    away_latest = away_matches.iloc[-1]

    if away_latest["home_team_name"] == away_name:
        a_pts = away_latest["home_rolling_points"]
        a_gf = away_latest["home_rolling_gf"]
        a_ga = away_latest["home_rolling_ga"]
    else:
        a_pts = away_latest["away_rolling_points"]
        a_gf = away_latest["away_rolling_gf"]
        a_ga = away_latest["away_rolling_ga"]

    # Build the feature vector for this match
    features = {
        "home_rolling_points": h_pts,
        "home_rolling_gf": h_gf,
        "home_rolling_ga": h_ga,
        "home_days_rest": 7.0,  # Defaulting to standard week rest
        "away_rolling_points": a_pts,
        "away_rolling_gf": a_gf,
        "away_rolling_ga": a_ga,
        "away_days_rest": 7.0,
        # Because FBRef failed to scrape, Xg stats default to 0 in our DB training set
        "home_xg": 0.0,
        "home_xga": 0.0,
        "away_xg": 0.0,
        "away_xga": 0.0,
    }

    df_predict = pd.DataFrame([features])
    df_predict = df_predict.reindex(columns=MODEL_FEATURES, fill_value=0.0)
    X_base = df_predict[MODEL_FEATURES]

    # Load models
    outcome_model = joblib.load(os.path.join(MODELS_DIR, "outcome_ensemble.joblib"))
    home_model = joblib.load(os.path.join(MODELS_DIR, "goals_home_xgb.joblib"))
    away_model = joblib.load(os.path.join(MODELS_DIR, "goals_away_xgb.joblib"))

    # Get goal predictions for stacking
    hg_raw = home_model.predict(X_base)[0]
    ag_raw = away_model.predict(X_base)[0]
    X_stacked = X_base.copy()
    X_stacked["predicted_home_goals"] = hg_raw
    X_stacked["predicted_away_goals"] = ag_raw
    X_stacked["predicted_goal_diff"] = hg_raw - ag_raw
    all_features = MODEL_FEATURES + GOAL_FEATURES

    probs = outcome_model.predict_proba(X_stacked[all_features])[0]
    hg = hg_raw
    ag = ag_raw

    print("\n" + "=" * 50)
    print(f"🌍 LA LIGA PREDICTION: {home_name} vs {away_name}")
    print("=" * 50)
    print(
        f"📈 Form [{home_name}]: {h_pts:.1f} pts in last 5 ({h_gf:.1f} scored, {h_ga:.1f} conceded)"
    )
    print(
        f"📈 Form [{away_name}]: {a_pts:.1f} pts in last 5 ({a_gf:.1f} scored, {a_ga:.1f} conceded)"
    )
    print("-" * 50)
    print(f"⚽ Projected Scoreline: {home_name} {hg:.1f} - {ag:.1f} {away_name}")
    print(f"Win Probability [{home_name}]: {probs[2] * 100:.1f}%")
    print(f"Win Probability [Draw]: {probs[1] * 100:.1f}%")
    print(f"Win Probability [{away_name}]: {probs[0] * 100:.1f}%")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    predict_real_match("Atlético", "FC Barcelona")
