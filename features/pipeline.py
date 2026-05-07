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

    def __init__(self, db: "Database" | None = None):
        if db is None:
            self.db = Database()
        else:
            self.db = db

    def load_finished_matches(self, limit: int | None = None) -> pd.DataFrame:
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

        # Core form metrics
        team_logs["rolling_points"] = grouped["points"].transform(
            lambda x: x.rolling(window).sum().shift(1)
        )
        team_logs["rolling_gf"] = grouped["goals_scored"].transform(
            lambda x: x.rolling(window).sum().shift(1)
        )
        team_logs["rolling_ga"] = grouped["goals_conceded"].transform(
            lambda x: x.rolling(window).sum().shift(1)
        )
        # Goal difference (GF - GA)
        team_logs["rolling_gd"] = grouped.apply(
            lambda x: (x["goals_scored"].rolling(window).sum() - x["goals_conceded"].rolling(window).sum()).shift(1), include_groups=False
        ).reset_index(level=0, drop=True)

        # Points per game
        team_logs["rolling_ppg"] = grouped["points"].transform(
            lambda x: x.rolling(window).mean().shift(1)
        )

        # Clean sheet percentage
        team_logs["rolling_cs"] = grouped.apply(
            lambda x: (x["goals_conceded"].eq(0).rolling(window).mean() * 100).shift(1), include_groups=False
        ).reset_index(level=0, drop=True)

        # Both teams to score (BTTS) tendency
        team_logs["rolling_btts"] = grouped.apply(
            lambda x: ((x["goals_scored"] > 0) & (x["goals_conceded"] > 0)).rolling(window).mean().shift(1), include_groups=False
        ).reset_index(level=0, drop=True)

        # Calculate Fatigue (rest days)
        team_logs["days_rest"] = (
            grouped["date"].diff().dt.days.shift(1).fillna(10)
        )  # default 10 days for start of season

        # ---------- Format back to attach to matches ----------
        base_cols = ["match_id", "rolling_points", "rolling_gf", "rolling_ga",
                     "rolling_gd", "rolling_ppg", "rolling_cs", "rolling_btts", "days_rest"]

        home_features = team_logs[team_logs["is_home"] == 1][base_cols]
        home_features.columns = ["match_id"] + ["home_" + c for c in base_cols[1:]]

        away_features = team_logs[team_logs["is_home"] == 0][base_cols]
        away_features.columns = ["match_id"] + ["away_" + c for c in base_cols[1:]]

        # Compute HOME-only and AWAY-only form separately
        home_only = team_logs[(team_logs["is_home"] == 1) & team_logs["points"].notna()]
        away_only = team_logs[(team_logs["is_home"] == 0) & team_logs["points"].notna()]

        # Per-team home form: average points per home game
        team_home_form = {}
        if not home_only.empty:
            for team in home_only["team_name"].unique():
                team_home_form[team] = home_only[home_only["team_name"] == team]["points"].mean()

        # Per-team away form: average points per away game
        team_away_form = {}
        if not away_only.empty:
            for team in away_only["team_name"].unique():
                team_away_form[team] = away_only[away_only["team_name"] == team]["points"].mean()

        # Attach per-team form to matches
        df["home_home_form"] = df["home_team_name"].map(team_home_form)
        df["away_away_form"] = df["away_team_name"].map(team_away_form)

        # Default: 1.5 pts/game (league average home form)
        df["home_home_form"] = df["home_home_form"].fillna(1.5)
        df["away_away_form"] = df["away_away_form"].fillna(1.2)

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

    def compute_h2h_stats(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Computes head-to-head statistics between teams.
        For each match, looks at last N encounters between the specific home/away team pair.
        """
        logger.info(f"Computing {window}-match head-to-head stats...")

        # Create a unique key for each matchup
        df["matchup"] = df.apply(
            lambda x: "-".join(sorted([str(x["home_team_name"]), str(x["away_team_name"])])),
            axis=1
        )

        h2h_data = []

        for _matchup, group in df.groupby("matchup"):
            group = group.sort_values("utc_date")

            for _idx, match in group.iterrows():
                # Get prior matches (before current match date)
                prior = group[group["utc_date"] < match["utc_date"]].tail(window)

                if prior.empty:
                    h2h_data.append({
                        "match_id": match["match_id"],
                        "home_h2h_points": 7.5,  # neutral default
                        "away_h2h_points": 7.5,
                        "h2h_draws": 0.3,
                    })
                    continue

                # Calculate H2H points
                home_pts = 0
                away_pts = 0
                draws = 0

                for _, m in prior.iterrows():
                    # Check whether current home team was actually home in this prior match
                    current_home = str(match["home_team_name"])
                    prior_home = str(m["home_team_name"])

                    if current_home == prior_home:
                        # Current home team was home in this prior encounter
                        if m["winner"] == "HOME_TEAM":
                            home_pts += 3
                        elif m["winner"] == "AWAY_TEAM":
                            away_pts += 3
                        else:
                            draws += 1
                            home_pts += 1
                            away_pts += 1
                    else:
                        # Current home team was away in this prior encounter
                        if m["winner"] == "AWAY_TEAM":
                            home_pts += 3
                        elif m["winner"] == "HOME_TEAM":
                            away_pts += 3
                        else:
                            draws += 1
                            home_pts += 1
                            away_pts += 1

                # Normalize to 0-15 scale (same as rolling points)
                n = max(len(prior), 1)
                h2h_data.append({
                    "match_id": match["match_id"],
                    "home_h2h_points": min(home_pts * (5 / n), 15.0),
                    "away_h2h_points": min(away_pts * (5 / n), 15.0),
                    "h2h_draws": draws / n,
                })

        h2h_df = pd.DataFrame(h2h_data)

        # Merge back
        df = df.merge(h2h_df, on="match_id", how="left")
        df = df.drop(columns=["matchup"], errors="ignore")

        # Fill defaults for no H2H history
        df["home_h2h_points"] = df["home_h2h_points"].fillna(7.5)
        df["away_h2h_points"] = df["away_h2h_points"].fillna(7.5)
        df["h2h_draws"] = df["h2h_draws"].fillna(0.3)

        return df

    def compute_elo_ratings(self, df: pd.DataFrame, k_factor: int = 20, initial_elo: float = 1500.0) -> pd.DataFrame:
        """
        Computes rolling Elo ratings for each team.
        Uses standard Elo formula: E = 1 / (1 + 10^((Rb - Ra)/400))
        K=20 is conservative for football. New ratings are shifted to not leak outcomes.
        """
        logger.info("Computing Elo ratings...")

        # Create team game logs sorted by date
        home_logs = df[
            ["match_id", "utc_date", "home_team_name", "away_team_name", "home_goals_ft", "away_goals_ft", "winner"]
        ].copy()
        home_logs.columns = ["match_id", "date", "team", "opponent", "goals_for", "goals_against", "winner"]
        home_logs["is_home"] = 1

        away_logs = df[
            ["match_id", "utc_date", "away_team_name", "home_team_name", "away_goals_ft", "home_goals_ft", "winner"]
        ].copy()
        away_logs.columns = ["match_id", "date", "team", "opponent", "goals_for", "goals_against", "winner"]
        away_logs["is_home"] = 0

        team_logs = pd.concat([home_logs, away_logs]).sort_values(by=["team", "date"]).reset_index(drop=True)

        # Initialize Elo tracking dict
        team_elo: dict[str, float] = {}
        ratings = []

        for _, row in team_logs.iterrows():
            team = row["team"]
            opponent = row["opponent"]

            # Get current Elo (default 1500)
            current_elo = team_elo.get(team, initial_elo)
            opponent_elo = team_elo.get(opponent, initial_elo)

            # Store Elo BEFORE the match (this is what we use for features)
            ratings.append({
                "match_id": row["match_id"],
                "team": team,
                "is_home": row["is_home"],
                "elo_before": current_elo,
                "opponent_elo_before": opponent_elo,
            })

            # Calculate actual score
            if row["winner"] == "HOME_TEAM":
                if row["is_home"]:
                    actual = 1.0  # Home win
                else:
                    actual = 0.0  # Away loss
            elif row["winner"] == "AWAY_TEAM":
                if row["is_home"]:
                    actual = 0.0  # Home loss
                else:
                    actual = 1.0  # Away win
            else:
                actual = 0.5  # Draw

            # Calculate expected score
            if row["is_home"]:
                expected = 1.0 / (1.0 + 10 ** ((opponent_elo - current_elo) / 400.0))
            else:
                expected = 1.0 / (1.0 + 10 ** ((current_elo - opponent_elo) / 400.0))

            # Update Elo
            new_elo = current_elo + k_factor * (actual - expected)
            team_elo[team] = new_elo

        # Convert to DataFrame
        elo_df = pd.DataFrame(ratings)

        # Get home team elo for each match_id
        home_ratings = elo_df[elo_df["is_home"] == 1][["match_id", "elo_before", "opponent_elo_before"]].copy()
        home_ratings.columns = ["match_id", "home_elo", "opponent_elo"]

        # Get away team elo for each match_id
        away_ratings = elo_df[elo_df["is_home"] == 0][["match_id", "elo_before", "opponent_elo_before"]].copy()
        away_ratings.columns = ["match_id", "away_elo", "opponent_elo"]

        # Merge together - home_ratings has home team's elo, away_ratings has away team's elo
        elo_features = home_ratings.merge(away_ratings, on="match_id")
        elo_features["elo_diff"] = elo_features["home_elo"] - elo_features["away_elo"]

        # Merge back to main df
        df = df.merge(elo_features, on="match_id", how="left")

        # Fill defaults for missing
        df["home_elo"] = df["home_elo"].fillna(initial_elo)
        df["away_elo"] = df["away_elo"].fillna(initial_elo)
        df["elo_diff"] = df["elo_diff"].fillna(0.0)

        return df

    def build_dataset(self) -> pd.DataFrame:
        """Runs complete orchestration of model dataset."""
        df = self.load_finished_matches()
        if df.empty:
            return df

        df = self.compute_rolling_stats(df, window=5)
        df = self.attach_xg_stats(df)
        df = self.compute_h2h_stats(df, window=5)
        df = self.compute_elo_ratings(df, k_factor=20, initial_elo=1500.0)

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

        # Add H2H features for predictions
        combined = self.compute_h2h_stats(combined, window=5)

        # Add Elo features
        combined = self.compute_elo_ratings(combined, k_factor=20, initial_elo=1500.0)

        # Isolate back to the scheduled matches
        predict_df = combined[combined["status"].isin(["SCHEDULED", "TIMED"])].copy()

        return predict_df
