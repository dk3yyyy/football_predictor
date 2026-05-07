"""
OddsScraper — match odds from Oddsportal

Odds data is extremely valuable for prediction:
  - Market consensus on win probability (implied probability)
  - Odds movement (sharp money signals)
  - Over/Under lines (market-estimated total goals)

We scrape 1X2 (home/draw/away) and Over/Under 2.5 odds.
These are then converted to implied probabilities with margin removal.

Why odds matter: bookmakers aggregate enormous information.
Odds often move before public news breaks — sharp bettors know
about injuries, weather, and form before the public.
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional

import requests
from bs4 import BeautifulSoup

from config.settings import (
    ODDSPORTAL_BASE_URL,
    ODDSPORTAL_DELAY_MAX,
    ODDSPORTAL_DELAY_MIN,
)
from scrapers.base import BaseScraper, ScraperError

logger = logging.getLogger(__name__)

# Oddsportal URL slugs per league
ODDSPORTAL_LEAGUES = {
    "premier_league": "/football/england/premier-league/",
    "la_liga": "/football/spain/laliga/",
    "bundesliga": "/football/germany/bundesliga/",
    "serie_a": "/football/italy/serie-a/",
    "ligue_1": "/football/france/ligue-1/",
    "champions_league": "/football/europe/champions-league/",
}


class OddsScraper(BaseScraper):
    """
    Scrapes 1X2 and Over/Under odds from Oddsportal.
    Converts raw decimal odds to calibrated implied probabilities.
    """

    RATE_LIMIT_CALLS = 6
    RATE_LIMIT_PERIOD = 60.0

    def __init__(self):
        super().__init__(delay_min=ODDSPORTAL_DELAY_MIN, delay_max=ODDSPORTAL_DELAY_MAX)

    @property
    def source_name(self) -> str:
        return "oddsportal.com"

    # ── Main scrape ───────────────────────────────────────────────────────────

    def scrape_league_odds(self, league_key: str) -> list[dict]:
        """
        Scrape upcoming fixture odds for a league.
        Returns list of match-odds records.
        """
        path = ODDSPORTAL_LEAGUES.get(league_key)
        if not path:
            raise ScraperError(f"No Oddsportal path for league: {league_key}")

        url = ODDSPORTAL_BASE_URL + path
        logger.info("Scraping odds: %s", url)

        try:
            resp = self.get(url)
        except requests.RequestException as exc:
            raise ScraperError(f"Failed to fetch odds page: {exc}") from exc

        soup = BeautifulSoup(resp.text, "html.parser")
        records = self._parse_odds_table(soup, league_key)

        # Also try JSON embedded in page scripts (Oddsportal sometimes serves this)
        if not records:
            records = self._parse_json_data(resp.text, league_key)

        self.log_result(len(records), f"{league_key} odds")
        return records

    # ── HTML table parser ─────────────────────────────────────────────────────

    def _parse_odds_table(self, soup: BeautifulSoup, league_key: str) -> list[dict]:
        """Parse the main odds table from Oddsportal HTML."""
        records = []

        # Oddsportal table has class "table-main" or similar
        table = soup.find("table", {"class": re.compile(r"table-main", re.I)}) or soup.find(
            "table", {"id": re.compile(r"odds", re.I)}
        )
        if not table:
            logger.debug("No standard odds table found, will try JSON fallback")
            return []

        for row in table.find_all("tr", {"class": re.compile(r"deactivate|odd|even", re.I)}):
            record = self._parse_odds_row(row, league_key)
            if record:
                records.append(record)

        return records

    def _parse_odds_row(self, row: BeautifulSoup, league_key: str) -> Optional[dict]:
        """Extract odds from a single table row."""
        cells = row.find_all("td")
        if len(cells) < 5:
            return None

        # Teams are usually in the first meaningful cell
        teams_cell = row.find("td", {"class": re.compile(r"name|teams|event", re.I)})
        if not teams_cell:
            teams_cell = cells[1] if len(cells) > 1 else cells[0]

        teams_text = teams_cell.get_text(strip=True)
        # Format: "Home Team - Away Team" or "Home Team vs Away Team"
        sep = " - " if " - " in teams_text else " vs "
        parts = teams_text.split(sep, 1)
        if len(parts) != 2:
            return None

        home_team = parts[0].strip()
        away_team = parts[1].strip()

        # Date/time cell
        date_cell = row.find("td", {"class": re.compile(r"date|time", re.I)})
        match_date = date_cell.get_text(strip=True) if date_cell else None

        # Odds cells — look for cells with numeric content (decimal odds)
        odds_cells = [c for c in cells if re.match(r"^\d+\.\d+$", c.get_text(strip=True))]

        home_odds = self._to_float(odds_cells[0].get_text()) if len(odds_cells) > 0 else None
        draw_odds = self._to_float(odds_cells[1].get_text()) if len(odds_cells) > 1 else None
        away_odds = self._to_float(odds_cells[2].get_text()) if len(odds_cells) > 2 else None

        if not any([home_odds, draw_odds, away_odds]):
            return None

        probs = self._remove_margin(home_odds, draw_odds, away_odds)

        return {
            "source": self.source_name,
            "league_key": league_key,
            "home_team": home_team,
            "away_team": away_team,
            "match_date": match_date,
            # Raw decimal odds
            "odds_home": home_odds,
            "odds_draw": draw_odds,
            "odds_away": away_odds,
            # Implied probabilities (margin removed)
            "prob_home": probs.get("home"),
            "prob_draw": probs.get("draw"),
            "prob_away": probs.get("away"),
            "market_overround": probs.get("overround"),
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── JSON data fallback ────────────────────────────────────────────────────

    def _parse_json_data(self, html: str, league_key: str) -> list[dict]:
        """
        Oddsportal sometimes embeds data in a script tag as JSON.
        Try to extract it as a fallback.
        """
        # Look for window.pageProps or similar
        pattern = re.compile(r"window\.__INITIAL_STATE__\s*=\s*(\{.*?\});", re.DOTALL)
        m = pattern.search(html)
        if not m:
            pattern2 = re.compile(r'"events"\s*:\s*(\[.*?\])', re.DOTALL)
            m = pattern2.search(html)
            if not m:
                logger.debug("No JSON data found in page")
                return []

        try:
            data = json.loads(m.group(1))
            events = data if isinstance(data, list) else data.get("events", [])
            records = []
            for ev in events:
                record = self._normalise_json_event(ev, league_key)
                if record:
                    records.append(record)
            return records
        except (json.JSONDecodeError, AttributeError) as exc:
            logger.warning("JSON parse error in odds page: %s", exc)
            return []

    def _normalise_json_event(self, ev: dict, league_key: str) -> Optional[dict]:
        """Normalise a JSON event object from Oddsportal's embedded data."""
        home_odds = self._to_float(ev.get("odds_home") or ev.get("home"))
        draw_odds = self._to_float(ev.get("odds_draw") or ev.get("draw"))
        away_odds = self._to_float(ev.get("odds_away") or ev.get("away"))

        if not any([home_odds, draw_odds, away_odds]):
            return None

        probs = self._remove_margin(home_odds, draw_odds, away_odds)

        return {
            "source": self.source_name,
            "league_key": league_key,
            "home_team": ev.get("home_team") or ev.get("home_name", ""),
            "away_team": ev.get("away_team") or ev.get("away_name", ""),
            "match_date": ev.get("date") or ev.get("start_time"),
            "odds_home": home_odds,
            "odds_draw": draw_odds,
            "odds_away": away_odds,
            "prob_home": probs.get("home"),
            "prob_draw": probs.get("draw"),
            "prob_away": probs.get("away"),
            "market_overround": probs.get("overround"),
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Probability maths ─────────────────────────────────────────────────────

    @staticmethod
    def _remove_margin(
        home: Optional[float],
        draw: Optional[float],
        away: Optional[float],
    ) -> dict:
        """
        Convert decimal odds → fair implied probabilities by removing the
        bookmaker margin (overround).

        Raw implied prob = 1 / decimal_odds
        Sum of raw probs > 1 (that's the margin / vigorish).
        Normalise by dividing each by the total.

        Example:
          home=2.10, draw=3.40, away=3.20
          raw: 0.476, 0.294, 0.313  → total=1.083 (8.3% margin)
          fair: 0.440, 0.272, 0.289
        """
        result: dict = {"home": None, "draw": None, "away": None, "overround": None}

        odds = {"home": home, "draw": draw, "away": away}
        valid = {k: v for k, v in odds.items() if v and v > 1.0}

        if not valid:
            return result

        raw_probs = {k: 1.0 / v for k, v in valid.items()}
        total = sum(raw_probs.values())
        overround = round(total - 1.0, 4)

        for k, raw in raw_probs.items():
            result[k] = round(raw / total, 4)

        result["overround"] = overround
        return result

    # ── Type helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _to_float(val) -> Optional[float]:
        try:
            return float(str(val).strip())
        except (ValueError, TypeError):
            return None

    # ── Batch ─────────────────────────────────────────────────────────────────

    def scrape_all_leagues(self) -> dict[str, list]:
        results = {}
        for league_key in ODDSPORTAL_LEAGUES:
            try:
                results[league_key] = self.scrape_league_odds(league_key)
            except ScraperError as exc:
                logger.error("Odds scrape failed for %s: %s", league_key, exc)
                results[league_key] = []
        return results

    def scrape(self, **kwargs):
        return self.scrape_all_leagues()
