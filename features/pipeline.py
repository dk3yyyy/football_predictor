import logging

import pandas as pd
from sqlalchemy import text

from db.database import Database

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Constructs the target modeling dataset.
    Extracts raw match data, combines with stats, and engineeres features:
    - Target: match output
    - Form: last 5 games points, goals scored/conceded
    - Expected Goals Diff (xG)
    - Fatigue (days since last match)
    """

    def __init__(self, db: Database = None):
        if db is None:
            self.db = Database()
        else:
            self.db = db

    def load_finished_matches(self, limit: int = None) -> pd.DataFrame:
        """Loads all completed matches with results."""
        query = "SELECT * FROM matches WHERE status = 'FINISHED' ORDER BY utc_date ASC"
        if limit:
            query += f" LIMIT {limit}"

        with self.db.engine.connect() as conn:
            df = pd.read_sql(text(query), conn)

        # Ensure dates are datetime
        df["utc_date"] = pd.to_datetime(df["utc_date"])

        # Calculate derived targets
        df["home_points"] = df["winner"].map({"HOME_TEAM": 3, "DRAW": 1, "AWAY_TEAM": 0})
        df["away_points"] = df["winner"].map({"AWAY_TEAM": 3, "DRAW": 1, "HOME_TEAM": 0})
        return df

    def compute_rolling_stats(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Computes rolling statistics for each team using their last `window` matches.
        Must respect the current timeline (shift by 1 so we don't leak the outcome of the match being predicted).
        """
        logger.info(f"Computing {window}-game rolling form stats...")

        # Melt dataframe to focus on team-level game logs
        home_logs = df[
            [
                "match_id",
                "utc_date",
                "home_team_name",
                "home_goals_ft",
                "away_goals_ft",
                "home_points",
            ]
        ].copy()
        home_logs.columns = [
            "match_id",
            "date",
            "team_name",
            "goals_scored",
            "goals_conceded",
            "points",
        ]
        home_logs["is_home"] = 1

        away_logs = df[
            [
                "match_id",
                "utc_date",
                "away_team_name",
                "away_goals_ft",
                "home_goals_ft",
                "away_points",
            ]
        ].copy()
        away_logs.columns = [
            "match_id",
            "date",
            "team_name",
            "goals_scored",
            "goals_conceded",
            "points",
        ]
        away_logs["is_home"] = 0

        team_logs = pd.concat([home_logs, away_logs]).sort_values(by=["team_name", "date"])

        # Calculate Rolling Stats (must be shifted to avoid lookahead bias!)
        grouped = team_logs.groupby("team_name")

        team_logs["rolling_points"] = grouped["points"].transform(
            lambda x: x.rolling(window).sum().shift(1)
        )
        team_logs["rolling_gf"] = grouped["goals_scored"].transform(
            lambda x: x.rolling(window).sum().shift(1)
        )
        team_logs["rolling_ga"] = grouped["goals_conceded"].transform(
            lambda x: x.rolling(window).sum().shift(1)
        )

        # Calculate Fatigue (rest days)
        team_logs["days_rest"] = (
            grouped["date"].diff().dt.days.shift(1).fillna(10)
        )  # default 10 days for start of season

        # Format back to attach to matches
        home_features = team_logs[team_logs["is_home"] == 1][
            ["match_id", "rolling_points", "rolling_gf", "rolling_ga", "days_rest"]
        ]
        home_features.columns = [
            "match_id",
            "home_rolling_points",
            "home_rolling_gf",
            "home_rolling_ga",
            "home_days_rest",
        ]

        away_features = team_logs[team_logs["is_home"] == 0][
            ["match_id", "rolling_points", "rolling_gf", "rolling_ga", "days_rest"]
        ]
        away_features.columns = [
            "match_id",
            "away_rolling_points",
            "away_rolling_gf",
            "away_rolling_ga",
            "away_days_rest",
        ]

        df = df.merge(home_features, on="match_id", how="left")
        df = df.merge(away_features, on="match_id", how="left")
        return df

    def attach_xg_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pulls latest team xG stats prior to match date and attaches to dataset.
        Due to SQLite limitations with lateral joins, we pull all xG to memory and merge map.
        """
        logger.info("Attaching Expected Goals (xG) stats...")
        with self.db.engine.connect() as conn:
            xg_df = pd.read_sql(
                text("SELECT team_name, xg, xga, xg_diff, scraped_at FROM xg_stats"), conn
            )

        if xg_df.empty:
            df["home_xg"] = df["away_xg"] = df["home_xga"] = df["away_xga"] = 0.0
            return df

        xg_df["scraped_at"] = pd.to_datetime(xg_df["scraped_at"])

        # Simplified for now: just grab latest xG.
        # In an ideal backtest, we join as-of dates.
        latest_xg = xg_df.sort_values("scraped_at").groupby("team_name").tail(1)
        latest_xg = latest_xg[["team_name", "xg", "xga", "xg_diff"]]

        df = df.merge(latest_xg, left_on="home_team_name", right_on="team_name", how="left")
        df = df.rename(
            columns={"xg": "home_xg", "xga": "home_xga", "xg_diff": "home_xg_diff"}
        ).drop(columns=["team_name"], errors="ignore")

        df = df.merge(latest_xg, left_on="away_team_name", right_on="team_name", how="left")
        df = df.rename(
            columns={"xg": "away_xg", "xga": "away_xga", "xg_diff": "away_xg_diff"}
        ).drop(columns=["team_name"], errors="ignore")

        # Fill NAs
        for col in ["home_xg", "home_xga", "home_xg_diff", "away_xg", "away_xga", "away_xg_diff"]:
            df[col] = df[col].fillna(0.0)

        return df

    def build_dataset(self) -> pd.DataFrame:
        """Runs complete orchestration of model dataset."""
        df = self.load_finished_matches()
        if df.empty:
            return df

        df = self.compute_rolling_stats(df, window=5)
        df = self.attach_xg_stats(df)

        # Drop initial untracked matches (NAs due to rolling window limit)
        df = df.dropna(subset=["home_rolling_points"])

        return df

    def build_fixtures_predict_set(self) -> pd.DataFrame:
        """
        Builds feature frame for *upcoming* scheduled matches to run live predictions.
        """
        with self.db.engine.connect() as conn:
            df = pd.read_sql(
                text(
                    "SELECT * FROM matches WHERE status IN ('SCHEDULED', 'TIMED') ORDER BY utc_date ASC"
                ),
                conn,
            )

        if df.empty:
            return pd.DataFrame()

        df["utc_date"] = pd.to_datetime(df["utc_date"])

        # Pull latest metrics (this is live, so latest DB values are fine, no lookahead risk)
        df = self.attach_xg_stats(df)

        # Calculate Rolling form... To do this properly for future matches, we must pull recent completed matches too
        historical = self.load_finished_matches()
        combined = pd.concat([historical, df], ignore_index=True).sort_values("utc_date")

        combined = self.compute_rolling_stats(combined)

        # Isolate back to the scheduled matches
        predict_df = combined[combined["status"].isin(["SCHEDULED", "TIMED"])].copy()

        return predict_df
