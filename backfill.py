"""
backfill.py — one-time historical data loader.

Run this once after setup to populate the DB with past seasons:
    python backfill.py

This gives the ML models enough historical data to train on.
Recommended: run with at least 3 seasons (2022, 2023, 2024).
"""

import logging

from config.settings import LEAGUES, SEASONS
from db.database import Database
from scrapers.fbref import FBrefScraper
from scrapers.football_data import FootballDataScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("backfill")


def backfill_results(db: Database):
    """Pull all historical results from Football-Data.org API."""
    scraper = FootballDataScraper()
    for season in SEASONS:
        for league_key, code in LEAGUES.items():
            if code is None:
                continue
            logger.info("Backfilling %s season %d results...", league_key, season)
            try:
                results = scraper.scrape_full_season_results(league_key, season)
                n = db.upsert_matches(results)
                logger.info("  stored %d matches", n)
            except Exception as exc:
                logger.error("  failed: %s", exc)


def backfill_advanced_stats(db: Database):
    """Pull FBref advanced stats for historical seasons."""
    scraper = FBrefScraper()
    for season in SEASONS:
        for league_key in LEAGUES:
            if LEAGUES[league_key] is None:
                continue
            logger.info("Backfilling %s season %d advanced stats...", league_key, season)
            try:
                stats = scraper.scrape_all_stats(league_key, season)
                db.insert_xg_stats(stats.get("xg", []))
                db.insert_possession_stats(stats.get("possession", []))
                db.insert_defensive_stats(stats.get("defense", []))
                logger.info("  stored xg, possession, defense stats")
            except Exception as exc:
                logger.error("  failed: %s", exc)


def main():
    logger.info("Starting historical data backfill...")
    db = Database()

    logger.info("=== Phase 1: Match results ===")
    backfill_results(db)

    logger.info("=== Phase 2: Advanced stats ===")
    backfill_advanced_stats(db)

    logger.info("Backfill complete.")


if __name__ == "__main__":
    main()
