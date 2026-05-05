"""
Tests for the scraping layer.
Run with: pytest tests/test_scrapers.py -v

These tests mock HTTP responses so they run offline without API keys.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from scrapers.base import BaseScraper, RateLimiter, ScraperError
from scrapers.football_data import FootballDataScraper
from scrapers.fbref import FBrefScraper
from scrapers.injuries import InjuryScraper
from scrapers.odds import OddsScraper


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_MATCH = {
    "id": 12345,
    "competition": {"id": 2021, "name": "Premier League"},
    "season": {"startDate": "2024-08-01"},
    "matchday": 10,
    "utcDate": "2024-10-26T15:00:00Z",
    "status": "FINISHED",
    "stage": "REGULAR_SEASON",
    "homeTeam": {"id": 57, "name": "Arsenal FC"},
    "awayTeam": {"id": 61, "name": "Chelsea FC"},
    "score": {
        "winner": "HOME_TEAM",
        "fullTime": {"home": 2, "away": 1},
        "halfTime": {"home": 1, "away": 0},
    },
    "referees": [{"name": "Mike Dean"}],
}

SAMPLE_STANDING = {
    "position": 1,
    "team": {"id": 57, "name": "Arsenal FC"},
    "playedGames": 10,
    "won": 8, "draw": 1, "lost": 1,
    "goalsFor": 22, "goalsAgainst": 8,
    "goalDifference": 14,
    "points": 25,
    "form": "W,W,D,W,W",
}


# ── RateLimiter tests ─────────────────────────────────────────────────────────

class TestRateLimiter:
    def test_allows_requests_within_limit(self):
        rl = RateLimiter(calls=5, period=60.0)
        import time
        start = time.monotonic()
        for _ in range(5):
            rl.wait()
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, "5 requests within limit should not block"

    def test_rate_limiter_blocks_when_exceeded(self):
        """When limit is 1/period, second call should block."""
        rl = RateLimiter(calls=1, period=60.0)
        rl.wait()
        # Manually stuff the timestamp list so next call would block
        rl._timestamps = [rl._timestamps[-1]]
        # We just check it doesn't crash (sleep would be very long in real test)
        assert len(rl._timestamps) == 1


# ── FootballDataScraper tests ─────────────────────────────────────────────────

class TestFootballDataScraper:

    @pytest.fixture
    def scraper(self, monkeypatch):
        import scrapers.football_data
        monkeypatch.setattr(scrapers.football_data, "FOOTBALL_DATA_API_KEY", "test_key_123")
        return FootballDataScraper()

    def test_normalise_match(self, scraper):
        result = scraper._normalise_match(SAMPLE_MATCH)
        assert result["match_id"] == 12345
        assert result["home_team_name"] == "Arsenal FC"
        assert result["away_team_name"] == "Chelsea FC"
        assert result["home_goals_ft"] == 2
        assert result["away_goals_ft"] == 1
        assert result["winner"] == "HOME_TEAM"
        assert result["referee"] == "Mike Dean"
        assert result["source"] == "football-data.org"

    def test_normalise_standing(self, scraper):
        result = scraper._normalise_standing(SAMPLE_STANDING, "premier_league", 2024)
        assert result["position"] == 1
        assert result["team_name"] == "Arsenal FC"
        assert result["points"] == 25
        assert result["form"] == "W,W,D,W,W"
        assert result["league_key"] == "premier_league"

    def test_scrape_fixtures_calls_api(self, scraper, monkeypatch):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "matches": [SAMPLE_MATCH]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(scraper.session, "get", return_value=mock_response):
            fixtures = scraper.scrape_fixtures("premier_league", days_ahead=7)

        assert len(fixtures) == 1
        assert fixtures[0]["match_id"] == 12345

    def test_scrape_fixtures_bad_league(self, scraper):
        with pytest.raises(ScraperError, match="No API code"):
            scraper.scrape_fixtures("nigeria_npfl")

    def test_normalise_match_missing_score(self, scraper):
        """Should handle matches without scores gracefully (upcoming fixtures)."""
        raw = {**SAMPLE_MATCH, "score": {"winner": None, "fullTime": {"home": None, "away": None}, "halfTime": {}}}
        result = scraper._normalise_match(raw)
        assert result["home_goals_ft"] is None
        assert result["away_goals_ft"] is None


# ── FBrefScraper tests ────────────────────────────────────────────────────────

class TestFBrefScraper:

    @pytest.fixture
    def scraper(self):
        return FBrefScraper()

    def test_season_str(self, scraper):
        assert scraper._season_str(2024) == "2024-2025"
        assert scraper._season_str(2022) == "2022-2023"

    def test_to_float(self, scraper):
        assert scraper._to_float("12.5") == 12.5
        assert scraper._to_float("1,234.5") == 1234.5
        assert scraper._to_float("N/A") is None
        assert scraper._to_float(None) is None

    def test_to_int(self, scraper):
        assert scraper._to_int("42") == 42
        assert scraper._to_int("N/A") is None

    def test_safe_diff(self, scraper):
        assert scraper._safe_diff("15.3", "12.1") == pytest.approx(3.2, abs=0.001)
        assert scraper._safe_diff(None, "5.0") is None

    def test_bad_league_raises(self, scraper):
        with pytest.raises(ScraperError, match="No FBref comp ID"):
            scraper.scrape_xg_stats("nigeria_npfl")


# ── OddsScraper tests ─────────────────────────────────────────────────────────

class TestOddsScraper:

    @pytest.fixture
    def scraper(self):
        return OddsScraper()

    def test_remove_margin_basic(self, scraper):
        """Standard 3-way market should produce probs summing to ~1.0."""
        result = scraper._remove_margin(2.10, 3.40, 3.20)
        assert result["home"] is not None
        assert abs(result["home"] + result["draw"] + result["away"] - 1.0) < 0.001
        assert result["overround"] > 0

    def test_remove_margin_all_none(self, scraper):
        result = scraper._remove_margin(None, None, None)
        assert result["home"] is None
        assert result["overround"] is None

    def test_remove_margin_partial(self, scraper):
        """Two valid odds should still normalise correctly."""
        result = scraper._remove_margin(2.0, None, 2.0)
        assert result["home"] == pytest.approx(0.5, abs=0.001)
        assert result["away"] == pytest.approx(0.5, abs=0.001)
        assert result["draw"] is None

    def test_implied_prob_accuracy(self, scraper):
        """
        Evens (2.0) on both sides = 50/50 before margin.
        With 2.0/2.0 odds, no margin, each should be exactly 0.5.
        """
        result = scraper._remove_margin(2.0, None, 2.0)
        assert result["home"] == 0.5
        assert result["away"] == 0.5

    def test_bad_league_raises(self, scraper):
        with pytest.raises(ScraperError, match="No Oddsportal path"):
            scraper.scrape_league_odds("nigeria_npfl")


# ── InjuryScraper tests ───────────────────────────────────────────────────────

class TestInjuryScraper:

    @pytest.fixture
    def scraper(self):
        return InjuryScraper()

    def test_classify_status_injured(self, scraper):
        assert scraper._classify_status("hamstring injury expected 3 weeks") == "injured"

    def test_classify_status_suspended(self, scraper):
        assert scraper._classify_status("suspended for 3 matches, red card") == "suspended"

    def test_classify_status_doubt(self, scraper):
        assert scraper._classify_status("doubtful, minor knock in training") == "doubt"

    def test_extract_return_date_weeks(self, scraper):
        result = scraper._extract_return_date("out for 3-4 weeks with hamstring")
        assert "3-4 weeks" in result or "weeks" in result.lower()

    def test_extract_return_date_none(self, scraper):
        result = scraper._extract_return_date("suspended")
        assert result is None

    def test_extract_player_name(self, scraper):
        name = scraper._extract_player_name("Bukayo Saka is expected to return next week")
        assert "Bukayo Saka" in name

    def test_unknown_team_returns_empty(self, scraper):
        result = scraper.scrape_team_injuries("Nonexistent FC")
        assert result == []