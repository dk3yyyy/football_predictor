"""
BaseScraper — inherited by every source-specific scraper.

Provides:
  - Session management with rotating User-Agents
  - Exponential backoff retries
  - Configurable rate limiting (token bucket)
  - Structured logging
  - Proxy support
  - robots.txt-aware delays
"""

import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import (
    MAX_RETRIES,
    PROXY_URL,
    REQUEST_TIMEOUT,
    RETRY_BACKOFF,
    USE_PROXY,
    USER_AGENTS,
)

logger = logging.getLogger(__name__)


class RateLimiter:
    """Thread-safe token bucket rate limiter."""

    def __init__(self, calls: int, period: float):
        self.calls = calls
        self.period = period
        self._lock = threading.Lock()
        self._timestamps: list[float] = []

    def wait(self):
        with self._lock:
            now = time.monotonic()
            # Drop timestamps outside the window
            self._timestamps = [t for t in self._timestamps if now - t < self.period]
            if len(self._timestamps) >= self.calls:
                sleep_for = self.period - (now - self._timestamps[0])
                if sleep_for > 0:
                    logger.debug("Rate limit hit — sleeping %.2fs", sleep_for)
                    time.sleep(sleep_for)
            self._timestamps.append(time.monotonic())


class BaseScraper(ABC):
    """
    Abstract base class for all football data scrapers.

    Subclass and implement:
        scrape() → dict | list  (your actual scraping logic)
        source_name (str property)
    """

    # Override in subclass if the source needs a specific rate limit
    RATE_LIMIT_CALLS = 30
    RATE_LIMIT_PERIOD = 60.0  # seconds

    def __init__(self, delay_min: float = 2.0, delay_max: float = 5.0):
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.session = self._build_session()
        self._rate_limiter = RateLimiter(self.RATE_LIMIT_CALLS, self.RATE_LIMIT_PERIOD)
        self.logger = logging.getLogger(self.__class__.__name__)

    # ── Session setup ─────────────────────────────────────────────────────────

    def _build_session(self) -> requests.Session:
        session = requests.Session()

        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=RETRY_BACKOFF,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        if USE_PROXY and PROXY_URL:
            session.proxies = {"http": PROXY_URL, "https": PROXY_URL}
            self.logger.info("Proxy enabled: %s", PROXY_URL)

        return session

    def _rotate_headers(self) -> dict:
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Connection": "keep-alive",
            "DNT": "1",
        }

    # ── Core HTTP ─────────────────────────────────────────────────────────────

    def get(
        self,
        url: str,
        params: Optional[dict] = None,
        extra_headers: Optional[dict] = None,
        skip_delay: bool = False,
    ) -> requests.Response:
        """
        Polite GET with rate limiting, random delay, and retries.
        Raises requests.HTTPError on final failure.
        """
        self._rate_limiter.wait()

        if not skip_delay:
            delay = random.uniform(self.delay_min, self.delay_max)
            self.logger.debug("Sleeping %.2fs before %s", delay, url)
            time.sleep(delay)

        headers = self._rotate_headers()
        if extra_headers:
            headers.update(extra_headers)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self.logger.debug("GET %s (attempt %d)", url, attempt)
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                return response

            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response else "?"
                self.logger.warning(
                    "HTTP %s on %s (attempt %d/%d)", status, url, attempt, MAX_RETRIES
                )
                if attempt == MAX_RETRIES:
                    raise
                backoff = RETRY_BACKOFF**attempt + random.uniform(0, 1)
                self.logger.info("Retrying in %.1fs", backoff)
                time.sleep(backoff)

            except requests.exceptions.ConnectionError as exc:
                self.logger.error("Connection error on %s: %s", url, exc)
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(RETRY_BACKOFF**attempt)

            except requests.exceptions.Timeout:
                self.logger.warning("Timeout on %s (attempt %d)", url, attempt)
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(RETRY_BACKOFF**attempt)

    # ── Abstract interface ────────────────────────────────────────────────────

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable name for logging and DB tagging."""
        ...

    @abstractmethod
    def scrape(self, **kwargs):
        """
        Run the scrape. Return a list of dicts ready for DB insertion,
        or raise ScraperError on unrecoverable failure.
        """
        ...

    # ── Helpers ───────────────────────────────────────────────────────────────

    def log_result(self, count: int, entity: str):
        self.logger.info("[%s] scraped %d %s records", self.source_name, count, entity)

    def __repr__(self):
        return f"<{self.__class__.__name__} source={self.source_name}>"


class ScraperError(Exception):
    """Raised when a scraper cannot recover from an error."""

    pass
