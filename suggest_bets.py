import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import text

from db.database import Database
from scheduler import job_fixtures

# 1. First, Let's make sure we have upcoming fixtures downloaded!
print("Fetching upcoming match schedules (next 14 days)...")
try:
    job_fixtures()
except Exception as e:
    print(f"Error fetching fixtures: {e}")

# 2. Let's run the predictor to evaluate those matches
print("Running prediction models on upcoming games...")
try:
    subprocess.run([sys.executable, "-m", "models.predict"], check=True)
except Exception as e:
    print(f"Error running predictions: {e}")

# 3. Query the results for the most confident predictions
db = Database()

query = """
    SELECT
        l.match_id,
        m.home_team_name,
        m.away_team_name,
        m.utc_date,
        l.pred_home_win AS home_prob,
        l.pred_draw AS draw_prob,
        l.pred_away_win AS away_prob,
        l.pred_home_goals AS home_goals_proj,
        l.pred_away_goals AS away_goals_proj
    FROM predictions_log l
    JOIN (
        SELECT match_id, MAX(created_at) as latest_pred
        FROM predictions_log
        GROUP BY match_id
    ) latest ON l.match_id = latest.match_id AND l.created_at = latest.latest_pred
    JOIN matches m ON l.match_id = m.match_id
    WHERE m.status IN ('SCHEDULED', 'TIMED')
    AND m.utc_date >= :now
    ORDER BY m.utc_date ASC
"""

now_str = datetime.now(timezone.utc).isoformat()
with db.engine.connect() as conn:
    df = pd.read_sql(text(query), conn, params={"now": now_str})

if df.empty:
    print("No upcoming games found to bet on.")
else:
    # Filter for high-confidence predictions (e.g. over 60% win probability)
    # or clear goal margins

    print("\n" + "=" * 60)
    print("🏟️  TOP RECOMMENDED BETS FOR SPORTYBET")
    print("=" * 60)

    count = 0
    # Determine predicted winner dynamically
    df["predicted_winner"] = df[["home_prob", "draw_prob", "away_prob"]].idxmax(axis=1)
    df["predicted_winner"] = df["predicted_winner"].map(
        {"home_prob": "HOME_TEAM", "draw_prob": "DRAW", "away_prob": "AWAY_TEAM"}
    )

    # Create a unified max probability column to sort by confidence
    df["max_prob"] = df[["home_prob", "draw_prob", "away_prob"]].max(axis=1)
    df = df.sort_values(by="max_prob", ascending=False).head(10)

    for _, row in df.iterrows():
        home = row["home_team_name"]
        away = row["away_team_name"]
        date_str = pd.to_datetime(row["utc_date"]).strftime("%A, %b %d")

        # Determine the safest betting market
        if row["predicted_winner"] == "HOME_TEAM" and row["home_prob"] >= 0.55:
            pick = f"Home Win ({home})"
            prob = row["home_prob"]
        elif row["predicted_winner"] == "AWAY_TEAM" and row["away_prob"] >= 0.55:
            pick = f"Away Win ({away})"
            prob = row["away_prob"]
        elif row["predicted_winner"] == "DRAW" and row["draw_prob"] >= 0.40:
            pick = "Draw"
            prob = row["draw_prob"]
        else:
            # Look at goal totals instead (Over/Under 2.5)
            total_goals = row["home_goals_proj"] + row["away_goals_proj"]
            if total_goals >= 3.0:
                pick = "Over 2.5 Goals"
                prob = -1
            elif total_goals <= 1.5:
                pick = "Under 2.5 Goals"
                prob = -1
            else:
                pick = "Skip/Too Close"
                prob = row["max_prob"]

        if pick != "Skip/Too Close":
            count += 1
            print(f"\n📅 Date: {date_str}")
            print(f"⚽ Match: {home} vs {away}")
            print(f"✅ Recommended Bet: {pick}")
            if prob != -1:
                print(f"📊 ML Confidence: {prob * 100:.1f}%")
            else:
                print(f"📊 Projected Total Goals: {total_goals:.1f}")
            print("-" * 60)

    if count == 0:
        print("No high-confidence games found in the immediate future. Check back tomorrow!")
