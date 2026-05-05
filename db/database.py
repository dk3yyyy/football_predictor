"""
Database models and storage layer.

Tables:
  - matches         : fixtures + results (core table)
  - match_details   : lineups, goals, bookings, subs
  - standings       : league table snapshots
  - teams           : club metadata
  - xg_stats        : FBref advanced stats per team per season
  - possession_stats
  - defensive_stats
  - injury_reports  : squad availability
  - odds            : pre-match odds + implied probabilities
  - predictions     : model outputs (append-only log)

We use SQLAlchemy Core (not ORM) for lean, fast inserts.
"""

import json
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    create_engine, text,
    Table, Column, MetaData,
    Integer, Float, String, Boolean, DateTime, Text,
    UniqueConstraint, Index,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from config.settings import DATABASE_URL

logger = logging.getLogger(__name__)

metadata = MetaData()

# ── Table definitions ─────────────────────────────────────────────────────────

matches = Table("matches", metadata,
    Column("id",               Integer, primary_key=True, autoincrement=True),
    Column("match_id",         Integer, nullable=False),
    Column("source",           String(50)),
    Column("competition_id",   Integer),
    Column("competition_name", String(100)),
    Column("season",           Integer),
    Column("matchday",         Integer),
    Column("home_team_id",     Integer),
    Column("home_team_name",   String(100)),
    Column("away_team_id",     Integer),
    Column("away_team_name",   String(100)),
    Column("utc_date",         String(30)),
    Column("status",           String(20)),
    Column("stage",            String(50)),
    Column("home_goals_ft",    Integer),
    Column("away_goals_ft",    Integer),
    Column("home_goals_ht",    Integer),
    Column("away_goals_ht",    Integer),
    Column("winner",           String(20)),    # HOME_TEAM / AWAY_TEAM / DRAW
    Column("referee",          String(100)),
    Column("scraped_at",       String(30)),
    UniqueConstraint("match_id", "source", name="uq_match_source"),
)

match_details = Table("match_details", metadata,
    Column("id",         Integer, primary_key=True, autoincrement=True),
    Column("match_id",   Integer, nullable=False, unique=True),
    Column("lineups",    Text),     # JSON
    Column("goals",      Text),     # JSON
    Column("bookings",   Text),     # JSON
    Column("subs",       Text),     # JSON
    Column("scraped_at", String(30)),
)

standings = Table("standings", metadata,
    Column("id",            Integer, primary_key=True, autoincrement=True),
    Column("league_key",    String(50)),
    Column("season",        Integer),
    Column("position",      Integer),
    Column("team_id",       Integer),
    Column("team_name",     String(100)),
    Column("played",        Integer),
    Column("won",           Integer),
    Column("drawn",         Integer),
    Column("lost",          Integer),
    Column("goals_for",     Integer),
    Column("goals_against", Integer),
    Column("goal_diff",     Integer),
    Column("points",        Integer),
    Column("form",          String(20)),
    Column("scraped_at",    String(30)),
    UniqueConstraint("league_key", "season", "team_id", "scraped_at",
                     name="uq_standing_snapshot"),
)

teams = Table("teams", metadata,
    Column("id",          Integer, primary_key=True, autoincrement=True),
    Column("team_id",     Integer, nullable=False, unique=True),
    Column("name",        String(100)),
    Column("short_name",  String(50)),
    Column("tla",         String(5)),
    Column("crest_url",   String(300)),
    Column("venue",       String(100)),
    Column("founded",     Integer),
    Column("colors",      String(100)),
    Column("website",     String(200)),
    Column("scraped_at",  String(30)),
)

xg_stats = Table("xg_stats", metadata,
    Column("id",           Integer, primary_key=True, autoincrement=True),
    Column("source",       String(50)),
    Column("league_key",   String(50)),
    Column("season",       Integer),
    Column("team_name",    String(100)),
    Column("xg",           Float),
    Column("npxg",         Float),
    Column("xg_per_shot",  Float),
    Column("shots",        Integer),
    Column("shots_on_tgt", Integer),
    Column("goals",        Integer),
    Column("xga",          Float),
    Column("npxga",        Float),
    Column("goals_ag",     Integer),
    Column("shots_ag",     Integer),
    Column("xg_diff",      Float),
    Column("scraped_at",   String(30)),
    UniqueConstraint("league_key", "season", "team_name", "scraped_at",
                     name="uq_xg_snapshot"),
)

possession_stats = Table("possession_stats", metadata,
    Column("id",                  Integer, primary_key=True, autoincrement=True),
    Column("source",              String(50)),
    Column("league_key",          String(50)),
    Column("season",              Integer),
    Column("team_name",           String(100)),
    Column("possession_pct",      Float),
    Column("progressive_passes",  Integer),
    Column("progressive_carries", Integer),
    Column("touches_att_third",   Integer),
    Column("carries_into_box",    Integer),
    Column("scraped_at",          String(30)),
)

defensive_stats = Table("defensive_stats", metadata,
    Column("id",                Integer, primary_key=True, autoincrement=True),
    Column("source",            String(50)),
    Column("league_key",        String(50)),
    Column("season",            Integer),
    Column("team_name",         String(100)),
    Column("tackles",           Integer),
    Column("tackles_won",       Integer),
    Column("interceptions",     Integer),
    Column("blocks",            Integer),
    Column("clearances",        Integer),
    Column("errors",            Integer),
    Column("pressures",         Integer),
    Column("pressure_regains",  Integer),
    Column("scraped_at",        String(30)),
)

injury_reports = Table("injury_reports", metadata,
    Column("id",              Integer, primary_key=True, autoincrement=True),
    Column("team_name",       String(100)),
    Column("player_name",     String(100)),
    Column("player_id",       Integer),
    Column("status",          String(30)),    # injured / suspended / doubt
    Column("reason",          Text),
    Column("expected_return", String(100)),
    Column("source",          String(50)),
    Column("scraped_at",      String(30)),
    Index("ix_injury_team_date", "team_name", "scraped_at"),
)

odds_table = Table("odds", metadata,
    Column("id",               Integer, primary_key=True, autoincrement=True),
    Column("source",           String(50)),
    Column("league_key",       String(50)),
    Column("home_team",        String(100)),
    Column("away_team",        String(100)),
    Column("match_date",       String(30)),
    Column("odds_home",        Float),
    Column("odds_draw",        Float),
    Column("odds_away",        Float),
    Column("prob_home",        Float),
    Column("prob_draw",        Float),
    Column("prob_away",        Float),
    Column("market_overround", Float),
    Column("scraped_at",       String(30)),
    Index("ix_odds_teams_date", "home_team", "away_team", "match_date"),
)

predictions_log = Table("predictions_log", metadata,
    Column("id",            Integer, primary_key=True, autoincrement=True),
    Column("match_id",      Integer),
    Column("home_team",     String(100)),
    Column("away_team",     String(100)),
    Column("match_date",    String(30)),
    Column("league_key",    String(50)),
    Column("model_name",    String(50)),
    Column("model_version", String(20)),
    Column("pred_home_win", Float),
    Column("pred_draw",     Float),
    Column("pred_away_win", Float),
    Column("pred_home_goals",Float),
    Column("pred_away_goals",Float),
    Column("confidence",    Float),
    Column("features_json", Text),    # snapshot of features used
    Column("actual_winner", String(20)),   # filled in after match
    Column("correct",       Boolean),      # filled in after match
    Column("created_at",    String(30)),
)


# ── Database engine and init ──────────────────────────────────────────────────

class Database:
    """Thin wrapper around SQLAlchemy engine for upsert/insert operations."""

    def __init__(self, url: str = DATABASE_URL):
        self.engine = create_engine(url, echo=False, future=True)
        self._init_db()

    def _init_db(self):
        """Create all tables if they don't exist."""
        metadata.create_all(self.engine)
        logger.info("Database initialised at %s", self.engine.url)

    # ── Generic upsert ────────────────────────────────────────────────────────

    def upsert_matches(self, records: list[dict]) -> int:
        """Insert or update matches (keyed on match_id + source)."""
        return self._upsert(matches, records, conflict_cols=["match_id", "source"])

    def upsert_standings(self, records: list[dict]) -> int:
        return self._bulk_insert(standings, records)

    def upsert_teams(self, records: list[dict]) -> int:
        return self._upsert(teams, records, conflict_cols=["team_id"])

    def insert_match_details(self, record: dict) -> int:
        """Serialise JSON fields and insert."""
        row = {**record}
        for field in ("lineups", "goals", "bookings", "subs"):
            if isinstance(row.get(field), (dict, list)):
                row[field] = json.dumps(row[field])
        return self._upsert(match_details, [row], conflict_cols=["match_id"])

    def insert_xg_stats(self, records: list[dict]) -> int:
        return self._bulk_insert(xg_stats, records)

    def insert_possession_stats(self, records: list[dict]) -> int:
        return self._bulk_insert(possession_stats, records)

    def insert_defensive_stats(self, records: list[dict]) -> int:
        return self._bulk_insert(defensive_stats, records)

    def insert_injuries(self, records: list[dict]) -> int:
        return self._bulk_insert(injury_reports, records)

    def insert_odds(self, records: list[dict]) -> int:
        return self._bulk_insert(odds_table, records)

    def log_prediction(self, record: dict) -> int:
        if isinstance(record.get("features_json"), dict):
            record = {**record, "features_json": json.dumps(record["features_json"])}
        record.setdefault("created_at", datetime.utcnow().isoformat())
        return self._bulk_insert(predictions_log, [record])

    def update_prediction_result(self, prediction_id: int, actual_winner: str):
        """Fill in actual result after a match finishes."""
        with self.engine.begin() as conn:
            pred = conn.execute(
                text("SELECT pred_home_win, pred_draw, pred_away_win FROM predictions_log WHERE id = :id"),
                {"id": prediction_id},
            ).fetchone()

            if not pred:
                return

            correct = (
                (actual_winner == "HOME_TEAM" and pred[0] > pred[1] and pred[0] > pred[2]) or
                (actual_winner == "DRAW"      and pred[1] > pred[0] and pred[1] > pred[2]) or
                (actual_winner == "AWAY_TEAM" and pred[2] > pred[0] and pred[2] > pred[1])
            )
            conn.execute(
                text("UPDATE predictions_log SET actual_winner=:w, correct=:c WHERE id=:id"),
                {"w": actual_winner, "c": correct, "id": prediction_id},
            )

    # ── Query helpers ─────────────────────────────────────────────────────────

    def get_recent_matches(self, team_name: str, n: int = 10) -> list[dict]:
        """Last N completed matches for a team (home or away)."""
        with self.engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT * FROM matches
                WHERE (home_team_name = :team OR away_team_name = :team)
                  AND status = 'FINISHED'
                ORDER BY utc_date DESC
                LIMIT :n
            """), {"team": team_name, "n": n}).fetchall()
        return [dict(r._mapping) for r in rows]

    def get_head_to_head(self, home: str, away: str, n: int = 10) -> list[dict]:
        """Historical H2H between two teams."""
        with self.engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT * FROM matches
                WHERE ((home_team_name = :home AND away_team_name = :away)
                    OR (home_team_name = :away AND away_team_name = :home))
                  AND status = 'FINISHED'
                ORDER BY utc_date DESC
                LIMIT :n
            """), {"home": home, "away": away, "n": n}).fetchall()
        return [dict(r._mapping) for r in rows]

    def get_latest_xg(self, team_name: str, league_key: str) -> Optional[dict]:
        """Most recent xG stats snapshot for a team."""
        with self.engine.connect() as conn:
            row = conn.execute(text("""
                SELECT * FROM xg_stats
                WHERE team_name = :team AND league_key = :league
                ORDER BY scraped_at DESC LIMIT 1
            """), {"team": team_name, "league": league_key}).fetchone()
        return dict(row._mapping) if row else None

    def get_current_injuries(self, team_name: str) -> list[dict]:
        """Latest injury report entries for a team."""
        with self.engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT * FROM injury_reports
                WHERE team_name = :team
                ORDER BY scraped_at DESC LIMIT 30
            """), {"team": team_name}).fetchall()
        return [dict(r._mapping) for r in rows]

    def get_prediction_accuracy(self, model_name: str) -> dict:
        """Rolling accuracy stats for a model."""
        with self.engine.connect() as conn:
            row = conn.execute(text("""
                SELECT
                    COUNT(*)                          AS total,
                    SUM(CASE WHEN correct THEN 1 ELSE 0 END) AS correct,
                    ROUND(AVG(CASE WHEN correct THEN 1.0 ELSE 0.0 END) * 100, 2) AS accuracy_pct
                FROM predictions_log
                WHERE model_name = :m AND actual_winner IS NOT NULL
            """), {"m": model_name}).fetchone()
        return dict(row._mapping) if row else {}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _upsert(self, table: Table, records: list[dict], conflict_cols: list[str]) -> int:
        if not records:
            return 0
        clean = [self._filter_cols(table, r) for r in records]
        with self.engine.begin() as conn:
            stmt = sqlite_insert(table)
            stmt = stmt.on_conflict_do_update(
                index_elements=conflict_cols,
                set_={c: stmt.excluded[c] for c in clean[0] if c not in conflict_cols},
            )
            conn.execute(stmt, clean)
        return len(clean)

    def _bulk_insert(self, table: Table, records: list[dict]) -> int:
        if not records:
            return 0
        clean = [self._filter_cols(table, r) for r in records]
        with self.engine.begin() as conn:
            conn.execute(table.insert(), clean)
        return len(clean)

    @staticmethod
    def _filter_cols(table: Table, record: dict) -> dict:
        """Drop keys that don't have a corresponding column."""
        valid = {c.name for c in table.columns}
        return {k: v for k, v in record.items() if k in valid}