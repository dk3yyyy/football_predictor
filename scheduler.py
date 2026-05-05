"""
Scheduler — orchestrates all scraping jobs.

Uses APScheduler with cron triggers.
Each job catches its own exceptions so one failure doesn't kill the others.

Run this as a long-running process:
    python -m scheduler.runner

Or trigger individual jobs manually:
    from scheduler.runner import run_job
    run_job("fixtures")
"""

import logging
import signal
import sys
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED

from config.settings import SCHEDULE, LEAGUES, CURRENT_SEASON
from db.database import Database
from scrapers.football_data import FootballDataScraper
from scrapers.fbref import FBrefScraper
from scrapers.injuries import InjuryScraper
from scrapers.odds import OddsScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Shared DB instance (thread-safe with SQLAlchemy connection pool)
db = Database()


# ── Individual job functions ──────────────────────────────────────────────────

def job_fixtures():
    """Scrape and store upcoming fixtures for all API leagues."""
    logger.info("JOB fixtures — start")
    scraper = FootballDataScraper()
    all_fixtures = scraper.scrape_all_fixtures(days_ahead=14)
    total = 0
    for league_key, fixtures in all_fixtures.items():
        n = db.upsert_matches(fixtures)
        total += n
        logger.info("  %s: %d fixtures stored", league_key, n)
    logger.info("JOB fixtures — done (%d total)", total)


def job_results():
    """Scrape and store recent results, then trigger detail scrape for new matches."""
    logger.info("JOB results — start")
    scraper = FootballDataScraper()
    all_results = scraper.scrape_all_results(days_back=3)
    total = 0
    for league_key, results in all_results.items():
        n = db.upsert_matches(results)
        total += n
    logger.info("JOB results — done (%d total)", total)

    # Scrape match details for any finished match still missing lineups
    _backfill_match_details(scraper)


def _backfill_match_details(scraper: FootballDataScraper):
    """Fetch lineups + events for recently finished matches lacking detail."""
    with db.engine.connect() as conn:
        from sqlalchemy import text
        rows = conn.execute(text("""
            SELECT m.match_id FROM matches m
            LEFT JOIN match_details md ON m.match_id = md.match_id
            WHERE m.status = 'FINISHED'
              AND md.match_id IS NULL
            ORDER BY m.utc_date DESC
            LIMIT 20
        """)).fetchall()

    for (match_id,) in rows:
        try:
            detail = scraper.scrape_match_detail(match_id)
            db.insert_match_details(detail)
            logger.debug("  detail stored for match %d", match_id)
        except Exception as exc:
            logger.warning("  detail scrape failed for %d: %s", match_id, exc)


def job_squad_stats():
    """Scrape FBref advanced stats for all configured leagues."""
    logger.info("JOB squad_stats — start")
    scraper = FBrefScraper()
    for league_key in LEAGUES:
        if LEAGUES[league_key] is None:
            continue
        try:
            stats = scraper.scrape_all_stats(league_key, CURRENT_SEASON)
            db.insert_xg_stats(stats.get("xg", []))
            db.insert_possession_stats(stats.get("possession", []))
            db.insert_defensive_stats(stats.get("defense", []))
            logger.info("  %s stats stored", league_key)
        except Exception as exc:
            logger.error("  squad_stats failed for %s: %s", league_key, exc)
    logger.info("JOB squad_stats — done")


def job_injuries():
    """Scrape current injury and suspension reports."""
    logger.info("JOB injuries — start")
    scraper = InjuryScraper()
    try:
        records = scraper.scrape_all_teams()
        n = db.insert_injuries(records)
        logger.info("JOB injuries — done (%d records)", n)
    except Exception as exc:
        logger.error("JOB injuries failed: %s", exc)


def job_odds():
    """Scrape upcoming match odds from Oddsportal."""
    logger.info("JOB odds — start")
    scraper = OddsScraper()
    try:
        all_odds = scraper.scrape_all_leagues()
        total = 0
        for league_key, records in all_odds.items():
            n = db.insert_odds(records)
            total += n
        logger.info("JOB odds — done (%d records)", total)
    except Exception as exc:
        logger.error("JOB odds failed: %s", exc)


# ── Job registry ──────────────────────────────────────────────────────────────

JOBS = {
    "fixtures":    job_fixtures,
    "results":     job_results,
    "squad_stats": job_squad_stats,
    "injuries":    job_injuries,
    "odds":        job_odds,
}


def run_job(name: str):
    """Manually trigger a job by name. Useful for backfill and testing."""
    fn = JOBS.get(name)
    if not fn:
        raise ValueError(f"Unknown job: {name}. Available: {list(JOBS)}")
    logger.info("Manual trigger: %s", name)
    fn()


# ── Scheduler setup ───────────────────────────────────────────────────────────

def on_job_executed(event):
    logger.info("Job '%s' completed at %s", event.job_id, datetime.now().isoformat())


def on_job_error(event):
    logger.error("Job '%s' crashed: %s", event.job_id, event.exception)


def build_scheduler() -> BlockingScheduler:
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_listener(on_job_executed, EVENT_JOB_EXECUTED)
    scheduler.add_listener(on_job_error,    EVENT_JOB_ERROR)

    for job_name, cron_expr in SCHEDULE.items():
        fn = JOBS.get(job_name)
        if not fn:
            logger.warning("No function for scheduled job: %s", job_name)
            continue
        scheduler.add_job(
            fn,
            trigger=CronTrigger.from_crontab(cron_expr, timezone="UTC"),
            id=job_name,
            name=job_name,
            misfire_grace_time=3600,    # run even if up to 1hr late
            coalesce=True,              # skip duplicates if delayed
        )
        logger.info("Scheduled '%s' → %s (UTC)", job_name, cron_expr)

    return scheduler


def main():
    logger.info("Starting football predictor scheduler...")

    # Run fixtures immediately on start to populate the DB
    logger.info("Initial data load on startup...")
    try:
        job_fixtures()
    except Exception as exc:
        logger.error("Initial fixtures load failed: %s", exc)

    scheduler = build_scheduler()

    def handle_shutdown(signum, frame):
        logger.info("Shutdown signal received — stopping scheduler")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT,  handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    logger.info("Scheduler running. Press Ctrl+C to stop.")
    scheduler.start()


if __name__ == "__main__":
    main()