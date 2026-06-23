import logging
import re
import threading
import time
import xml.etree.ElementTree as ET

import requests
from langchain_core.tools import tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2
MIN_SECONDS_BETWEEN_REQUESTS = 4
RATE_LIMIT_COOLDOWN_SECONDS = 20
CACHE_TTL_SECONDS = 1800
REQUEST_HEADERS = {
    "User-Agent": "AI-Researcher/1.0 (Streamlit academic research assistant)"
}

_cache_lock = threading.Lock()
_arxiv_cache: dict[tuple[str, int], dict] = {}
_last_request_started_at = 0.0
_rate_limited_until = 0.0


def sanitize_topic_query(topic: str) -> str:
    cleaned = re.sub(r"[\"()]+", " ", topic)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    if not cleaned:
        raise ValueError("Please provide a topic to search on arXiv.")
    return "+".join(cleaned.split())


def clear_arxiv_cache() -> None:
    global _last_request_started_at, _rate_limited_until
    with _cache_lock:
        _arxiv_cache.clear()
        _last_request_started_at = 0.0
        _rate_limited_until = 0.0


def _copy_result(result: dict, *, cached: bool, stale_cache: bool = False, notice: str | None = None) -> dict:
    copied = {
        "entries": list(result.get("entries", [])),
        "cached": cached,
        "stale_cache": stale_cache,
    }
    if notice:
        copied["notice"] = notice
    return copied


def _get_cached_result(cache_key: tuple[str, int], *, allow_stale: bool = False) -> dict | None:
    with _cache_lock:
        cached_entry = _arxiv_cache.get(cache_key)
        if not cached_entry:
            return None

        age_seconds = time.monotonic() - cached_entry["timestamp"]
        if age_seconds <= CACHE_TTL_SECONDS:
            return _copy_result(cached_entry["result"], cached=True)

        if allow_stale:
            return _copy_result(
                cached_entry["result"],
                cached=True,
                stale_cache=True,
                notice="Showing cached arXiv results because live search is temporarily limited.",
            )

    return None


def _store_cached_result(cache_key: tuple[str, int], result: dict) -> dict:
    with _cache_lock:
        _arxiv_cache[cache_key] = {
            "timestamp": time.monotonic(),
            "result": {"entries": list(result.get("entries", []))},
        }
    return _copy_result(result, cached=False)


def _wait_for_request_slot() -> None:
    global _last_request_started_at

    while True:
        with _cache_lock:
            now = time.monotonic()
            cooldown_wait = max(_rate_limited_until - now, 0)
            spacing_wait = max(MIN_SECONDS_BETWEEN_REQUESTS - (now - _last_request_started_at), 0)
            wait_seconds = max(cooldown_wait, spacing_wait)

            if wait_seconds <= 0:
                _last_request_started_at = now
                return

        logger.info("Waiting %.1f seconds before the next arXiv request.", wait_seconds)
        time.sleep(wait_seconds)


def _mark_rate_limited() -> None:
    global _rate_limited_until
    with _cache_lock:
        _rate_limited_until = max(
            _rate_limited_until,
            time.monotonic() + RATE_LIMIT_COOLDOWN_SECONDS,
        )


def search_arxiv_papers(topic: str, max_results: int = 5) -> dict:
    """
    Search for arXiv papers with exponential backoff retry logic.
    Returns empty results on persistent network failures instead of raising errors.
    """
    query = sanitize_topic_query(topic)
    cache_key = (query, max_results)
    cached_result = _get_cached_result(cache_key)
    if cached_result:
        logger.info("Serving cached arXiv results for topic: %s", topic)
        return cached_result

    url = (
        f"{ARXIV_API_URL}?search_query=all:{query}"
        f"&max_results={max_results}"
        "&sortBy=submittedDate"
        "&sortOrder=descending"
    )

    last_exception = None
    backoff_time = INITIAL_BACKOFF_SECONDS

    for attempt in range(MAX_RETRIES):
        try:
            _wait_for_request_slot()
            logger.info(f"Attempting arXiv search (attempt {attempt + 1}/{MAX_RETRIES}): {topic}")
            response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS, headers=REQUEST_HEADERS)
            response.raise_for_status()
            logger.info(f"Successfully retrieved arXiv papers for topic: {topic}")
            parsed_result = parse_arxiv_xml(response.text)
            return _store_cached_result(cache_key, parsed_result)

        except requests.exceptions.Timeout:
            last_exception = "Request timed out. arXiv service may be slow or unavailable."
            logger.warning(f"Timeout on attempt {attempt + 1}: {last_exception}")

        except requests.exceptions.ConnectionError as e:
            error_msg = str(e)
            if "Failed to resolve" in error_msg or "NameResolutionError" in error_msg:
                last_exception = "Network error: Unable to reach arXiv. Check your internet connection."
            elif "Connection refused" in error_msg:
                last_exception = "Connection error: arXiv service may be temporarily unavailable."
            else:
                last_exception = "Network connection failed. Please check your internet connection."
            logger.warning(f"Connection error on attempt {attempt + 1}: {e}")

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if hasattr(e, 'response') else "unknown"
            if status_code == 429:
                _mark_rate_limited()
                stale_result = _get_cached_result(cache_key, allow_stale=True)
                if stale_result:
                    logger.warning("Rate limited by arXiv. Returning stale cached results for topic: %s", topic)
                    return stale_result

                last_exception = (
                    "Rate limited by arXiv. Please wait a little and try the paper search again."
                )
                logger.warning(f"Rate limited on attempt {attempt + 1}")
                backoff_time = max(backoff_time, RATE_LIMIT_COOLDOWN_SECONDS)
            elif status_code >= 500:
                last_exception = "arXiv service error. Please try again in a few moments."
                logger.warning(f"Server error ({status_code}) on attempt {attempt + 1}")
            else:
                last_exception = f"HTTP error {status_code}. Please try again."
                logger.warning(f"HTTP error on attempt {attempt + 1}: {e}")

        except requests.exceptions.RequestException as e:
            last_exception = "An unexpected network error occurred. Please try again."
            logger.warning(f"Request exception on attempt {attempt + 1}: {e}")

        except Exception as e:
            last_exception = "An unexpected error occurred while searching arXiv."
            logger.error(f"Unexpected error on attempt {attempt + 1}: {e}", exc_info=True)

        # Exponential backoff before retry (skip on last attempt)
        if attempt < MAX_RETRIES - 1:
            logger.info(f"Retrying in {backoff_time} seconds...")
            time.sleep(backoff_time)
            backoff_time *= 2

    # All retries exhausted - return empty results with logged error
    logger.error(f"All {MAX_RETRIES} attempts failed for topic: {topic}. Last error: {last_exception}")
    return {"entries": [], "error": last_exception}


def parse_arxiv_xml(xml_content: str) -> dict:
    entries = []
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(xml_content)

    for entry in root.findall("atom:entry", ns):
        authors = [
            author.findtext("atom:name", namespaces=ns)
            for author in entry.findall("atom:author", ns)
        ]
        categories = [
            category.attrib.get("term")
            for category in entry.findall("atom:category", ns)
        ]

        abstract_url = entry.findtext("atom:id", namespaces=ns)
        pdf_link = None
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("type") == "application/pdf":
                pdf_link = link.attrib.get("href")
                break

        entries.append(
            {
                "id": (abstract_url or "").rsplit("/", 1)[-1],
                "title": (entry.findtext("atom:title", namespaces=ns) or "").strip(),
                "summary": (entry.findtext("atom:summary", namespaces=ns) or "").strip(),
                "authors": [author for author in authors if author],
                "categories": [category for category in categories if category],
                "pdf": pdf_link,
                "abstract_url": abstract_url,
                "published": entry.findtext("atom:published", namespaces=ns),
            }
        )

    return {"entries": entries}


@tool
def arxiv_search(topic: str) -> list[dict]:
    """Search for recently uploaded arXiv papers."""
    papers = search_arxiv_papers(topic)
    if not papers["entries"]:
        raise ValueError(f"No papers found for topic: {topic}")
    return papers
