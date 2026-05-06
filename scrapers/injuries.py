"""
InjuryScraper — squad availability data

Sources:
  1. Soccerway team pages  → injury/suspension lists
  2. BBC Sport team news   → manager quotes, confirmed absences

This is some of the highest-value contextual data for match prediction.
A team missing its first-choice goalkeeper or striker is a significant signal.

Output schema per player record:
  team_name, player_name, player_id (if available), status, reason,
  expected_return, source, scraped_at
"""

import logging
import re
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup

from config.settings import (
    SOCCERWAY_BASE_URL,
    SOCCERWAY_DELAY_MAX,
    SOCCERWAY_DELAY_MIN,
)
from scrapers.base import BaseScraper

logger = logging.getLogger(__name__)

# Soccerway team page slugs for major clubs (extend as needed)
# Format: team_name → soccerway path
SOCCERWAY_TEAMS = {
    # Premier League
    "Arsenal": "/teams/england/arsenal-football-club/683/",
    "Chelsea": "/teams/england/chelsea-fc/631/",
    "Liverpool": "/teams/england/liverpool-fc/663/",
    "Manchester City": "/teams/england/manchester-city-fc/664/",
    "Manchester United": "/teams/england/manchester-united-fc/665/",
    "Tottenham Hotspur": "/teams/england/tottenham-hotspur/674/",
    "Newcastle United": "/teams/england/newcastle-united-fc/667/",
    "Aston Villa": "/teams/england/aston-villa-fc/625/",
    # La Liga
    "Real Madrid": "/teams/spain/real-madrid-cf/2832/",
    "FC Barcelona": "/teams/spain/futbol-club-barcelona/2817/",
    "Atletico Madrid": "/teams/spain/atletico-de-madrid/2813/",
    # Bundesliga
    "Bayern Munich": "/teams/germany/fc-bayern-munchen/2364/",
    "Borussia Dortmund": "/teams/germany/borussia-dortmund/2369/",
    # Serie A
    "Inter Milan": "/teams/italy/fc-internazionale-milano/2736/",
    "AC Milan": "/teams/italy/associazione-calcio-milan/2719/",
    "Juventus": "/teams/italy/juventus-fc/2741/",
    # Ligue 1
    "Paris Saint-Germain": "/teams/france/paris-saint-germain-fc/3004/",
}

# Keywords that indicate unavailability
INJURY_KEYWORDS = [
    "injury",
    "injured",
    "hamstring",
    "knee",
    "muscle",
    "fracture",
    "strain",
    "torn",
    "ligament",
    "concussion",
    "foot",
    "ankle",
    "back",
]
SUSPENSION_KEYWORDS = [
    "suspended",
    "suspension",
    "ban",
    "banned",
    "red card",
    "accumulated",
    "yellow cards",
]
DOUBT_KEYWORDS = ["doubt", "doubtful", "fitness", "knock", "minor"]


class InjuryScraper(BaseScraper):
    """Scrapes injury and suspension data for club squads."""

    RATE_LIMIT_CALLS = 10
    RATE_LIMIT_PERIOD = 60.0

    def __init__(self):
        super().__init__(delay_min=SOCCERWAY_DELAY_MIN, delay_max=SOCCERWAY_DELAY_MAX)

    @property
    def source_name(self) -> str:
        return "soccerway.com"

    # ── Soccerway ─────────────────────────────────────────────────────────────

    def scrape_team_injuries(self, team_name: str) -> list[dict]:
        """
        Scrape the injury/suspension list for a single team from Soccerway.
        """
        path = SOCCERWAY_TEAMS.get(team_name)
        if not path:
            logger.warning("No Soccerway path configured for: %s", team_name)
            return []

        url = SOCCERWAY_BASE_URL + path
        try:
            resp = self.get(url)
        except requests.RequestException as exc:
            logger.error("Failed to fetch Soccerway page for %s: %s", team_name, exc)
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        records = []

        # Soccerway lists injuries in a section with class "injuries" or similar
        injury_section = (
            soup.find("div", {"class": "injuries"})
            or soup.find("section", {"class": re.compile(r"injur", re.I)})
            or soup.find("div", {"id": re.compile(r"injur", re.I)})
        )

        if injury_section:
            records.extend(self._parse_injury_section(injury_section, team_name, "soccerway.com"))
        else:
            # Fallback: scan all player rows for injury indicators
            records.extend(self._scan_squad_table(soup, team_name))

        self.log_result(len(records), f"{team_name} injury records")
        return records

    def _parse_injury_section(
        self,
        section: BeautifulSoup,
        team_name: str,
        source: str,
    ) -> list[dict]:
        records = []
        for row in section.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) < 2:
                continue

            player_cell = cells[0]
            info_cell = cells[1] if len(cells) > 1 else None

            player_name = player_cell.get_text(strip=True)
            player_link = player_cell.find("a")
            player_id = None
            if player_link:
                href = player_link.get("href", "")
                m = re.search(r"/(\d+)/?$", href)
                if m:
                    player_id = int(m.group(1))

            info_text = info_cell.get_text(strip=True) if info_cell else ""
            status = self._classify_status(info_text)

            if not player_name or player_name.lower() in ("player", "name", ""):
                continue

            records.append(
                {
                    "team_name": team_name,
                    "player_name": player_name,
                    "player_id": player_id,
                    "status": status,
                    "reason": info_text,
                    "expected_return": self._extract_return_date(info_text),
                    "source": source,
                    "scraped_at": datetime.utcnow().isoformat(),
                }
            )
        return records

    def _scan_squad_table(self, soup: BeautifulSoup, team_name: str) -> list[dict]:
        """
        Fallback: scan the full squad table and pick out players
        whose status cell contains injury/suspension keywords.
        """
        records = []
        for table in soup.find_all("table"):
            for row in table.find_all("tr"):
                cells = row.find_all(["td", "th"])
                row_text = " ".join(c.get_text(strip=True) for c in cells).lower()

                if not any(kw in row_text for kw in INJURY_KEYWORDS + SUSPENSION_KEYWORDS):
                    continue

                # First cell is usually the player name
                player_name = cells[0].get_text(strip=True) if cells else ""
                status = self._classify_status(row_text)

                if not player_name or len(player_name) < 3:
                    continue

                records.append(
                    {
                        "team_name": team_name,
                        "player_name": player_name,
                        "player_id": None,
                        "status": status,
                        "reason": row_text[:200],
                        "expected_return": self._extract_return_date(row_text),
                        "source": self.source_name,
                        "scraped_at": datetime.utcnow().isoformat(),
                    }
                )
        return records

    # ── BBC Sport team news ───────────────────────────────────────────────────

    def scrape_bbc_team_news(self, bbc_team_path: str, team_name: str) -> list[dict]:
        """
        Scrape BBC Sport team news page for confirmed injury/suspension quotes.
        bbc_team_path e.g. '/sport/football/teams/arsenal'
        """
        url = f"https://www.bbc.co.uk{bbc_team_path}"
        try:
            resp = self.get(url)
        except requests.RequestException as exc:
            logger.error("Failed BBC Sport page for %s: %s", team_name, exc)
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        records = []

        # BBC structures team news in <li> or <p> elements inside an "injuries" section
        for tag in soup.find_all(["li", "p"]):
            text = tag.get_text(strip=True)
            text_lower = text.lower()

            if not any(kw in text_lower for kw in INJURY_KEYWORDS + SUSPENSION_KEYWORDS):
                continue
            if len(text) < 10 or len(text) > 400:
                continue

            records.append(
                {
                    "team_name": team_name,
                    "player_name": self._extract_player_name(text),
                    "player_id": None,
                    "status": self._classify_status(text_lower),
                    "reason": text[:300],
                    "expected_return": self._extract_return_date(text),
                    "source": "bbc-sport.co.uk",
                    "scraped_at": datetime.utcnow().isoformat(),
                }
            )

        self.log_result(len(records), f"{team_name} BBC news records")
        return records

    # ── Batch scrape ─────────────────────────────────────────────────────────

    def scrape_all_teams(self, team_names: Optional[list[str]] = None) -> list[dict]:
        """
        Scrape injury/suspension data for all configured teams (or a subset).
        """
        teams = team_names or list(SOCCERWAY_TEAMS.keys())
        all_recs = []
        for name in teams:
            recs = self.scrape_team_injuries(name)
            all_recs.extend(recs)
        return all_recs

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _classify_status(text: str) -> str:
        t = text.lower()
        if any(kw in t for kw in SUSPENSION_KEYWORDS):
            return "suspended"
        if any(kw in t for kw in DOUBT_KEYWORDS):
            return "doubt"
        if any(kw in t for kw in INJURY_KEYWORDS):
            return "injured"
        return "unavailable"

    @staticmethod
    def _extract_return_date(text: str) -> Optional[str]:
        """Try to extract a return date or timeframe from free text."""
        # Match patterns like "3-4 weeks", "until January", "2-3 months"
        patterns = [
            r"(\d+[-–]\d+\s+(?:weeks?|months?))",
            r"(until\s+\w+(?:\s+\d{1,2})?)",
            r"(returns?\s+\w+(?:\s+\d{1,2})?)",
            r"(\d{1,2}\s+\w+\s+\d{4})",  # e.g. "15 March 2025"
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None

    @staticmethod
    def _extract_player_name(text: str) -> str:
        """
        Naive extraction: take first proper-noun-like token(s) from text.
        Works for sentences starting with the player's name.
        """
        # Split on common separators after the name
        for sep in [" is ", " has ", " will ", " was ", " (", " -"]:
            if sep in text:
                candidate = text.split(sep)[0].strip()
                # Reasonable name: 2-5 words, each capitalised
                words = candidate.split()
                if 1 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
                    return candidate
        return text[:40]  # fallback: first 40 chars

    def scrape(self, **kwargs):
        return self.scrape_all_teams()
