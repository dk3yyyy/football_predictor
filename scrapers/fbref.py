"""
FBrefScraper — advanced stats from fbref.com

Covers per-team per-season:
  - xG (expected goals) and xGA (expected goals against)
  - Progressive passes, carries, pressures
  - Possession stats
  - Shooting stats (shots on target, conversion)
  - Defensive stats (tackles, interceptions, blocks)

FBref uses Sports Reference's tables — we parse with pandas read_html
which handles the multi-level headers automatically.

IMPORTANT: FBref has strict rate limits. Always use the delays in settings.py.
Do not run this more than once per day per league.
"""

import re
import logging
import time
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup

from scrapers.base import BaseScraper, ScraperError
from config.settings import (
    FBREF_BASE_URL,
    FBREF_DELAY_MIN,
    FBREF_DELAY_MAX,
    FBREF_COMPETITIONS,
    CURRENT_SEASON,
)

logger = logging.getLogger(__name__)

# Table IDs we want from FBref squad stats pages
STAT_TABLES = {
    "shooting":    "stats_squads_shooting_for",
    "passing":     "stats_squads_passing_for",
    "possession":  "stats_squads_possession_for",
    "defense":     "stats_squads_defense_for",
    "gca":         "stats_squads_gca_for",          # goal-creating actions
    "misc":        "stats_squads_misc_for",          # fouls, cards, aerials
    # Against (allowed by opponent)
    "shooting_ag": "stats_squads_shooting_against",
    "passing_ag":  "stats_squads_passing_against",
}


class FBrefScraper(BaseScraper):
    """
    Scrapes FBref squad-level advanced stats.
    Returns one dict per team with all key metrics flattened.
    """

    RATE_LIMIT_CALLS  = 4     # very conservative
    RATE_LIMIT_PERIOD = 60.0

    def __init__(self):
        super().__init__(delay_min=FBREF_DELAY_MIN, delay_max=FBREF_DELAY_MAX)

    @property
    def source_name(self) -> str:
        return "fbref.com"

    # ── URL builders ──────────────────────────────────────────────────────────

    def _squad_stats_url(self, comp_id: int, season_str: str, stat_type: str) -> str:
        """
        e.g. https://fbref.com/en/comps/9/2023-2024/shooting/2023-2024-Premier-League-Stats
        stat_type: shooting | passing | possession | defense | gca | misc
        """
        return (
            f"{FBREF_BASE_URL}/comps/{comp_id}/{season_str}/{stat_type}/"
            f"{season_str}-Stats"
        )

    @staticmethod
    def _season_str(season_year: int) -> str:
        """Convert 2024 → '2024-2025'"""
        return f"{season_year}-{season_year + 1}"

    # ── Core page fetch + table parse ─────────────────────────────────────────

    def _fetch_table(self, url: str, table_id: str) -> list[dict]:
        """
        Fetch a page and extract a specific HTML table by id.
        Returns rows as list of dicts with flattened column names.
        """
        try:
            resp = self.get(url)
        except requests.RequestException as exc:
            raise ScraperError(f"Failed to fetch {url}: {exc}") from exc

        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"id": table_id})

        if table is None:
            logger.warning("Table '%s' not found at %s", table_id, url)
            return []

        # Parse header (FBref uses nested <thead> rows)
        headers = self._parse_headers(table)
        rows    = []

        tbody = table.find("tbody")
        if not tbody:
            return []

        for tr in tbody.find_all("tr"):
            # Skip separator rows (class="thead" rows inside tbody)
            if tr.get("class") and "thead" in tr.get("class"):
                continue

            cells = tr.find_all(["td", "th"])
            if len(cells) < 2:
                continue

            row = {}
            for i, cell in enumerate(cells):
                key = headers[i] if i < len(headers) else f"col_{i}"
                val = cell.get_text(strip=True)
                # data-stat attribute is more reliable than positional header
                stat = cell.get("data-stat", "")
                if stat:
                    row[stat] = val
                else:
                    row[key] = val

            # Skip rows with no team name
            if not row.get("team") and not row.get("squad"):
                continue
            rows.append(row)

        return rows

    @staticmethod
    def _parse_headers(table) -> list[str]:
        """Flatten multi-level <thead> into single list of column names."""
        thead = table.find("thead")
        if not thead:
            return []

        rows = thead.find_all("tr")
        if not rows:
            return []

        # Use the last header row for column names
        last = rows[-1]
        return [
            th.get("data-stat") or th.get_text(strip=True).lower().replace(" ", "_")
            for th in last.find_all(["th", "td"])
        ]

    # ── Stat-specific scrapers ────────────────────────────────────────────────

    def scrape_xg_stats(
        self,
        league_key: str,
        season: int = CURRENT_SEASON,
    ) -> list[dict]:
        """
        xG, xGA, npxG (non-penalty expected goals), xG per shot.
        This is the single most predictive stat category.
        """
        comp_id    = FBREF_COMPETITIONS.get(league_key)
        if not comp_id:
            raise ScraperError(f"No FBref comp ID for league: {league_key}")

        season_str = self._season_str(season)
        url        = self._squad_stats_url(comp_id, season_str, "shooting")

        raw_for = self._fetch_table(url, STAT_TABLES["shooting"])
        raw_ag  = self._fetch_table(url, STAT_TABLES["shooting_ag"])

        # Index against-stats by team name for merge
        ag_by_team = {r.get("squad", r.get("team", "")): r for r in raw_ag}

        records = []
        for row in raw_for:
            team_name = row.get("squad") or row.get("team", "")
            ag        = ag_by_team.get(team_name, {})
            records.append({
                "source":       self.source_name,
                "league_key":   league_key,
                "season":       season,
                "team_name":    team_name,
                # Attacking
                "xg":           self._to_float(row.get("xg")),
                "npxg":         self._to_float(row.get("npxg")),
                "xg_per_shot":  self._to_float(row.get("xg_per_shot")),
                "shots":        self._to_int(row.get("shots")),
                "shots_on_tgt": self._to_int(row.get("shots_on_target")),
                "goals":        self._to_int(row.get("goals")),
                # Defensive
                "xga":          self._to_float(ag.get("xg")),
                "npxga":        self._to_float(ag.get("npxg")),
                "goals_ag":     self._to_int(ag.get("goals")),
                "shots_ag":     self._to_int(ag.get("shots")),
                # Derived
                "xg_diff":      self._safe_diff(row.get("xg"), ag.get("xg")),
                "scraped_at":   datetime.utcnow().isoformat(),
            })

        self.log_result(len(records), f"{league_key} xG stats")
        return records

    def scrape_possession_stats(
        self,
        league_key: str,
        season: int = CURRENT_SEASON,
    ) -> list[dict]:
        """
        Possession %, progressive passes, progressive carries, pressures.
        """
        comp_id    = FBREF_COMPETITIONS.get(league_key)
        if not comp_id:
            raise ScraperError(f"No FBref comp ID for league: {league_key}")

        season_str = self._season_str(season)
        url        = self._squad_stats_url(comp_id, season_str, "possession")
        raw        = self._fetch_table(url, STAT_TABLES["possession"])

        records = []
        for row in raw:
            records.append({
                "source":             self.source_name,
                "league_key":         league_key,
                "season":             season,
                "team_name":          row.get("squad") or row.get("team", ""),
                "possession_pct":     self._to_float(row.get("possession")),
                "progressive_passes": self._to_int(row.get("progressive_passes")),
                "progressive_carries":self._to_int(row.get("progressive_carries")),
                "touches_att_third":  self._to_int(row.get("touches_att_3rd")),
                "carries_into_box":   self._to_int(row.get("carries_into_penalty_area")),
                "scraped_at":         datetime.utcnow().isoformat(),
            })

        self.log_result(len(records), f"{league_key} possession stats")
        return records

    def scrape_defensive_stats(
        self,
        league_key: str,
        season: int = CURRENT_SEASON,
    ) -> list[dict]:
        """
        Tackles, interceptions, blocks, pressures.
        """
        comp_id    = FBREF_COMPETITIONS.get(league_key)
        if not comp_id:
            raise ScraperError(f"No FBref comp ID for league: {league_key}")

        season_str = self._season_str(season)
        url        = self._squad_stats_url(comp_id, season_str, "defense")
        raw        = self._fetch_table(url, STAT_TABLES["defense"])

        records = []
        for row in raw:
            records.append({
                "source":           self.source_name,
                "league_key":       league_key,
                "season":           season,
                "team_name":        row.get("squad") or row.get("team", ""),
                "tackles":          self._to_int(row.get("tackles")),
                "tackles_won":      self._to_int(row.get("tackles_won")),
                "interceptions":    self._to_int(row.get("interceptions")),
                "blocks":           self._to_int(row.get("blocks")),
                "clearances":       self._to_int(row.get("clearances")),
                "errors":           self._to_int(row.get("errors")),
                "pressures":        self._to_int(row.get("pressures")),
                "pressure_regains": self._to_int(row.get("pressure_regains")),
                "scraped_at":       datetime.utcnow().isoformat(),
            })

        self.log_result(len(records), f"{league_key} defensive stats")
        return records

    # ── All stats for a league ────────────────────────────────────────────────

    def scrape_all_stats(
        self,
        league_key: str,
        season: int = CURRENT_SEASON,
    ) -> dict[str, list]:
        """
        Scrape all stat categories for one league.
        Returns dict keyed by category.
        """
        logger.info("Scraping all FBref stats for %s %s", league_key, season)
        return {
            "xg":         self.scrape_xg_stats(league_key, season),
            "possession": self.scrape_possession_stats(league_key, season),
            "defense":    self.scrape_defensive_stats(league_key, season),
        }

    # ── Type coercions ────────────────────────────────────────────────────────

    @staticmethod
    def _to_float(val) -> Optional[float]:
        try:
            return float(str(val).replace(",", "").strip())
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _to_int(val) -> Optional[int]:
        try:
            return int(str(val).replace(",", "").strip())
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_diff(a, b) -> Optional[float]:
        try:
            return round(float(a) - float(b), 3)
        except (ValueError, TypeError):
            return None

    # ── Required abstract ─────────────────────────────────────────────────────

    def scrape(self, league_key: str = "premier_league", season: int = CURRENT_SEASON, **kwargs):
        return self.scrape_all_stats(league_key, season)