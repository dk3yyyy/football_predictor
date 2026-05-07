"""
FootballDataScraper — Football-Data.org v4 API

Covers:
  - Fixtures (upcoming matches)
  - Results (completed matches with scores)
  - Standings / league tables
  - Match detail (lineups, goals, cards, substitutions)

Free tier: 10 requests/minute, current season only.
Paid tier: historical seasons, more competitions.

Docs: https://docs.football-data.org/general/v4/
"""

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from config.settings import (
    CURRENT_SEASON,
    FOOTBALL_DATA_API_KEY,
    FOOTBALL_DATA_BASE_URL,
    FOOTBALL_DATA_RATE_LIMIT,
    FOOTBALL_DATA_RATE_WINDOW,
    LEAGUES,
)
from scrapers.base import BaseScraper, ScraperError

logger = logging.getLogger(__name__)


class FootballDataScraper(BaseScraper):
    """
    Scrapes Football-Data.org API.
    All methods return clean, normalised dicts ready for DB insert.
    """

    RATE_LIMIT_CALLS = FOOTBALL_DATA_RATE_LIMIT
    RATE_LIMIT_PERIOD = FOOTBALL_DATA_RATE_WINDOW

    def __init__(self):
        super().__init__(delay_min=0, delay_max=0)  # rate limiter handles pacing
        if not FOOTBALL_DATA_API_KEY:
            raise ScraperError(
                "FOOTBALL_DATA_API_KEY not set. "
                "Get a free key at https://www.football-data.org/client/register"
            )
        self._api_headers = {
            "X-Auth-Token": FOOTBALL_DATA_API_KEY,
            "Accept": "application/json",
        }

    @property
    def source_name(self) -> str:
        return "football-data.org"

    # ── Internal API call ─────────────────────────────────────────────────────

    def _api_get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        url = f"{FOOTBALL_DATA_BASE_URL}/{endpoint.lstrip('/')}"
        resp = self.get(url, params=params, extra_headers=self._api_headers, skip_delay=True)
        return resp.json()

    # ── Fixtures ──────────────────────────────────────────────────────────────

    def scrape_fixtures(
        self,
        league_key: str,
        days_ahead: int = 14,
        season: int = CURRENT_SEASON,
    ) -> list[dict]:
        """
        Fetch upcoming fixtures for a league within the next N days.
        Returns list of fixture dicts.
        """
        comp_code = LEAGUES.get(league_key)
        if not comp_code:
            raise ScraperError(f"No API code for league: {league_key}")

        date_from = date.today().isoformat()
        date_to = (date.today() + timedelta(days=days_ahead)).isoformat()

        data = self._api_get(
            f"competitions/{comp_code}/matches",
            params={
                "season": season,
                "dateFrom": date_from,
                "dateTo": date_to,
                "status": "SCHEDULED",
            },
        )
        fixtures = [self._normalise_match(m) for m in data.get("matches", [])]
        self.log_result(len(fixtures), "fixtures")
        return fixtures

    # ── Results ───────────────────────────────────────────────────────────────

    def scrape_results(
        self,
        league_key: str,
        days_back: int = 7,
        season: int = CURRENT_SEASON,
    ) -> list[dict]:
        """
        Fetch completed match results for the last N days.
        """
        comp_code = LEAGUES.get(league_key)
        if not comp_code:
            raise ScraperError(f"No API code for league: {league_key}")

        date_from = (date.today() - timedelta(days=days_back)).isoformat()
        date_to = date.today().isoformat()

        data = self._api_get(
            f"competitions/{comp_code}/matches",
            params={
                "season": season,
                "dateFrom": date_from,
                "dateTo": date_to,
                "status": "FINISHED",
            },
        )
        results = [self._normalise_match(m) for m in data.get("matches", [])]
        self.log_result(len(results), "results")
        return results

    def scrape_full_season_results(
        self,
        league_key: str,
        season: int,
    ) -> list[dict]:
        """
        Backfill: pull all finished matches for an entire season.
        Use for historical data collection on first run.
        """
        comp_code = LEAGUES.get(league_key)
        if not comp_code:
            raise ScraperError(f"No API code for league: {league_key}")

        data = self._api_get(
            f"competitions/{comp_code}/matches",
            params={"season": season, "status": "FINISHED"},
        )
        results = [self._normalise_match(m) for m in data.get("matches", [])]
        self.log_result(len(results), f"season-{season} results")
        return results

    # ── Match detail (lineups, events) ────────────────────────────────────────

    def scrape_match_detail(self, match_id: int) -> dict:
        """
        Full match detail: lineups, scorers, bookings, substitutions.
        Requires match_id from fixtures/results.
        """
        data = self._api_get(f"matches/{match_id}")
        return self._normalise_match_detail(data)

    # ── Standings ─────────────────────────────────────────────────────────────

    def scrape_standings(
        self,
        league_key: str,
        season: int = CURRENT_SEASON,
    ) -> list[dict]:
        """
        Current league table standings.
        """
        comp_code = LEAGUES.get(league_key)
        if not comp_code:
            raise ScraperError(f"No API code for league: {league_key}")

        data = self._api_get(
            f"competitions/{comp_code}/standings",
            params={"season": season},
        )
        rows = []
        for table in data.get("standings", []):
            if table.get("type") == "TOTAL":
                for entry in table.get("table", []):
                    rows.append(self._normalise_standing(entry, league_key, season))
        self.log_result(len(rows), "standings rows")
        return rows

    # ── Teams ─────────────────────────────────────────────────────────────────

    def scrape_teams(
        self,
        league_key: str,
        season: int = CURRENT_SEASON,
    ) -> list[dict]:
        """
        All teams in a competition with squad info.
        """
        comp_code = LEAGUES.get(league_key)
        if not comp_code:
            raise ScraperError(f"No API code for league: {league_key}")

        data = self._api_get(
            f"competitions/{comp_code}/teams",
            params={"season": season},
        )
        teams = [self._normalise_team(t) for t in data.get("teams", [])]
        self.log_result(len(teams), "teams")
        return teams

    # ── Normalisers ───────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_match(raw: dict) -> dict:
        """Flatten a raw match dict into a clean DB-ready record."""
        home = raw.get("homeTeam", {})
        away = raw.get("awayTeam", {})
        score = raw.get("score", {})
        full = score.get("fullTime", {})
        half = score.get("halfTime", {})

        return {
            # Identifiers
            "match_id": raw.get("id"),
            "source": "football-data.org",
            # Competition
            "competition_id": raw.get("competition", {}).get("id"),
            "competition_name": raw.get("competition", {}).get("name"),
            "season": raw.get("season", {}).get("startDate", "")[:4],
            "matchday": raw.get("matchday"),
            # Teams
            "home_team_id": home.get("id"),
            "home_team_name": home.get("name"),
            "away_team_id": away.get("id"),
            "away_team_name": away.get("name"),
            # Schedule
            "utc_date": raw.get("utcDate"),
            "status": raw.get("status"),
            "stage": raw.get("stage"),
            # Score
            "home_goals_ft": full.get("home"),
            "away_goals_ft": full.get("away"),
            "home_goals_ht": half.get("home"),
            "away_goals_ht": half.get("away"),
            "winner": score.get("winner"),  # HOME_TEAM / AWAY_TEAM / DRAW
            # Referee
            "referee": (raw.get("referees") or [{}])[0].get("name"),
            # Timestamps
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _normalise_match_detail(raw: dict) -> dict:
        """Extract lineups, goals, bookings from a /matches/{id} response."""
        match = raw.get("match", raw)  # API returns wrapped or unwrapped
        match.get("homeTeam", {}).get("id")
        match.get("awayTeam", {}).get("id")

        lineups = {}
        for team_key in ("homeTeam", "awayTeam"):
            team = match.get(team_key, {})
            lineups[team_key] = {
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "formation": team.get("formation"),
                "starting": [
                    {"player_id": p.get("id"), "name": p.get("name"), "position": p.get("position")}
                    for p in team.get("lineup", [])
                ],
                "bench": [
                    {"player_id": p.get("id"), "name": p.get("name"), "position": p.get("position")}
                    for p in team.get("bench", [])
                ],
            }

        goals = [
            {
                "minute": g.get("minute"),
                "extra_min": g.get("extraTime"),
                "type": g.get("type"),  # REGULAR / OWN_GOAL / PENALTY
                "scorer_id": g.get("scorer", {}).get("id"),
                "scorer": g.get("scorer", {}).get("name"),
                "assist_id": (g.get("assist") or {}).get("id"),
                "assist": (g.get("assist") or {}).get("name"),
                "team_id": g.get("team", {}).get("id"),
            }
            for g in match.get("goals", [])
        ]

        bookings = [
            {
                "minute": b.get("minute"),
                "type": b.get("card"),  # YELLOW / RED / YELLOW_RED
                "player_id": b.get("player", {}).get("id"),
                "player": b.get("player", {}).get("name"),
                "team_id": b.get("team", {}).get("id"),
            }
            for b in match.get("bookings", [])
        ]

        subs = [
            {
                "minute": s.get("minute"),
                "player_in_id": s.get("playerIn", {}).get("id"),
                "player_in": s.get("playerIn", {}).get("name"),
                "player_out_id": s.get("playerOut", {}).get("id"),
                "player_out": s.get("playerOut", {}).get("name"),
                "team_id": s.get("team", {}).get("id"),
            }
            for s in match.get("substitutions", [])
        ]

        return {
            "match_id": match.get("id"),
            "lineups": lineups,
            "goals": goals,
            "bookings": bookings,
            "subs": subs,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _normalise_standing(row: dict, league_key: str, season: int) -> dict:
        team = row.get("team", {})
        return {
            "league_key": league_key,
            "season": season,
            "position": row.get("position"),
            "team_id": team.get("id"),
            "team_name": team.get("name"),
            "played": row.get("playedGames"),
            "won": row.get("won"),
            "drawn": row.get("draw"),
            "lost": row.get("lost"),
            "goals_for": row.get("goalsFor"),
            "goals_against": row.get("goalsAgainst"),
            "goal_diff": row.get("goalDifference"),
            "points": row.get("points"),
            "form": row.get("form"),  # e.g. "W,D,L,W,W"
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _normalise_team(raw: dict) -> dict:
        return {
            "team_id": raw.get("id"),
            "name": raw.get("name"),
            "short_name": raw.get("shortName"),
            "tla": raw.get("tla"),  # three-letter abbreviation
            "crest_url": raw.get("crest"),
            "venue": raw.get("venue"),
            "founded": raw.get("founded"),
            "colors": raw.get("clubColors"),
            "website": raw.get("website"),
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Convenience: scrape all leagues ───────────────────────────────────────

    def scrape_all_fixtures(self, days_ahead: int = 14) -> dict[str, list]:
        """Scrape upcoming fixtures for every configured API league."""
        results = {}
        for league_key, code in LEAGUES.items():
            if code is None:
                continue
            try:
                results[league_key] = self.scrape_fixtures(league_key, days_ahead)
            except ScraperError as exc:
                logger.error("Failed %s fixtures: %s", league_key, exc)
                results[league_key] = []
        return results

    def scrape_all_results(self, days_back: int = 7) -> dict[str, list]:
        """Scrape recent results for every configured API league."""
        results = {}
        for league_key, code in LEAGUES.items():
            if code is None:
                continue
            try:
                results[league_key] = self.scrape_results(league_key, days_back)
            except ScraperError as exc:
                logger.error("Failed %s results: %s", league_key, exc)
                results[league_key] = []
        return results

    # ── Required abstract ─────────────────────────────────────────────────────

    def scrape(self, **kwargs):
        """
        Default scrape: fixtures + results for all leagues.
        Called by the scheduler with no arguments.
        """
        return {
            "fixtures": self.scrape_all_fixtures(),
            "results": self.scrape_all_results(),
        }
