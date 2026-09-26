"""HTTP helpers with retries, exponential backoff and an explicit offline switch."""

from __future__ import annotations

import time
from typing import Any, Mapping

import requests

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)
USER_AGENT = "Namma-Flow/1.0 (open-source urban micro-flood research)"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class NetworkUnavailable(RuntimeError):
    """Raised when a remote resource cannot be fetched (offline mode, HTTP or transport failure)."""


def _request(
    url: str,
    params: Mapping[str, Any] | None,
    timeout_s: float,
    max_retries: int,
    backoff_s: float,
    offline: bool,
) -> requests.Response:
    if offline:
        raise NetworkUnavailable(f"Offline mode: refusing to fetch {url}")
    attempts = max(1, int(max_retries))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(
                url, params=params, timeout=timeout_s, headers={"User-Agent": USER_AGENT}
            )
            if response.status_code in RETRYABLE_STATUS:
                raise requests.HTTPError(f"HTTP {response.status_code}", response=response)
            if response.status_code >= 400:
                # Client errors (bad params, 404 tile) are not worth retrying.
                raise NetworkUnavailable(
                    f"{url} returned HTTP {response.status_code}: {response.text[:200]}"
                )
            return response
        except NetworkUnavailable:
            raise
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            if attempt < attempts:
                delay = backoff_s * (2 ** (attempt - 1))
                LOGGER.warning("Request to %s failed (%s); retry %d/%d in %.1fs", url, exc, attempt, attempts - 1, delay)
                time.sleep(delay)
    raise NetworkUnavailable(f"Failed to fetch {url} after {attempts} attempts: {last_error}")


def get_json(
    url: str,
    params: Mapping[str, Any] | None = None,
    *,
    timeout_s: float = 30.0,
    max_retries: int = 3,
    backoff_s: float = 2.0,
    offline: bool = False,
) -> Any:
    response = _request(url, params, timeout_s, max_retries, backoff_s, offline)
    try:
        return response.json()
    except ValueError as exc:
        raise NetworkUnavailable(f"{url} returned invalid JSON: {exc}") from exc


def get_bytes(
    url: str,
    params: Mapping[str, Any] | None = None,
    *,
    timeout_s: float = 60.0,
    max_retries: int = 3,
    backoff_s: float = 2.0,
    offline: bool = False,
) -> bytes:
    return _request(url, params, timeout_s, max_retries, backoff_s, offline).content
