import os
from dotenv import load_dotenv

# Load any variables from .env file securely
load_dotenv()

# ====================
# Global Configurations
# ====================

CURRENT_SEASON = 2024
SEASONS = [2022, 2023, 2024]
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///db/football_predictor.db")

# Dictionary of Custom League Keys (used internally) mapped to external API codes.
LEAGUES = {
    "premier_league": "PL",
    "la_liga": "PD",
    "serie_a": "SA",
    "bundesliga": "BL1",
    "ligue_1": "FL1"
}

# ====================
# Base Scraper Settings
# ====================

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0",
]
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0
USE_PROXY = False
PROXY_URL = os.getenv("PROXY_URL", "")
LOG_LEVEL = "INFO"

# ====================
# Football-Data API Settings
# ====================

FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "")
FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"
FOOTBALL_DATA_RATE_LIMIT = 10
FOOTBALL_DATA_RATE_WINDOW = 60.0

# ====================
# FBref Settings
# ====================

FBREF_BASE_URL = "https://fbref.com/en"
# FBref restricts scraping heavily
FBREF_DELAY_MIN = 3.0
FBREF_DELAY_MAX = 6.0

FBREF_COMPETITIONS = {
    "premier_league": 9,
    "la_liga": 12,
    "serie_a": 11,
    "bundesliga": 20,
    "ligue_1": 13
}

# ====================
# Soccerway & Injuries Settings
# ====================

SOCCERWAY_BASE_URL = "https://int.soccerway.com"
SOCCERWAY_DELAY_MIN = 2.0
SOCCERWAY_DELAY_MAX = 5.0

# ====================
# Oddsportal Settings
# ====================

ODDSPORTAL_BASE_URL = "https://www.oddsportal.com"
ODDSPORTAL_DELAY_MIN = 3.0
ODDSPORTAL_DELAY_MAX = 6.0

# ====================
# Scheduler Settings
# ====================

SCHEDULE = {
    "fixtures":    "0 2 * * *",     # Every day at 02:00 UTC
    "results":     "0 4 * * *",     # Every day at 04:00 UTC
    "squad_stats": "0 6 * * 1",     # Every Monday at 06:00 UTC
    "injuries":    "0 8 * * *",     # Every day at 08:00 UTC
    "odds":        "0 10 * * *",    # Every day at 10:00 UTC
}
