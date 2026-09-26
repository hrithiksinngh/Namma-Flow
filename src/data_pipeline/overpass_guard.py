"""Bounded, logged osmnx requests: a busy Overpass / Nominatim server never blocks a stage.

osmnx 2.x answers HTTP 429 / 504 by sleeping 55-60 s and calling itself again, with no attempt
limit, no exception and (with ``log_console`` off) no visible message, so a mirror that keeps
answering 504 blocks stage 01 / 02 for hours and the next ``network.overpass_urls`` mirror is
never tried. :func:`bounded_osmnx_requests` (entered inside the osmnx settings context of
:mod:`~src.data_pipeline.network` and :mod:`~src.data_pipeline.drains`) temporarily replaces

* ``osmnx._overpass.requests`` / ``osmnx._nominatim.requests`` with a proxy that turns a
  429 / 504 answer into :class:`ServerBusy` before osmnx can sleep and recurse, and
* ``osmnx._overpass._overpass_request`` / ``osmnx._nominatim._nominatim_request`` (module
  globals looked up at call time, so osmnx's own recursive retry also goes through the wrapper)
  with a wrapper that makes at most ``network.overpass_max_attempts`` attempts per request and
  mirror, waits ``network.overpass_retry_pause_s`` between them with a WARNING, and then raises
  :class:`ServerBusy` (a :class:`~src.utils.http.NetworkUnavailable`) so the caller moves on to
  the next mirror.

Everything is restored on exit. The wrapper also records the ``osm3s.timestamp_osm_base`` of
every Overpass answer (including answers replayed from the osmnx cache), i.e. the OpenStreetMap
snapshot a graph was built from (:class:`OverpassLog`).
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping

from src.utils.config import ConfigError, get_section
from src.utils.http import NetworkUnavailable
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

BUSY_STATUSES = frozenset({429, 504})
RETRY_DEFAULTS: dict[str, Any] = {"overpass_max_attempts": 2, "overpass_retry_pause_s": 30.0}
# (osmnx submodule, request function, settings attribute holding the endpoint, label)
_TARGETS = (
    ("_overpass", "_overpass_request", "overpass_url", "Overpass"),
    ("_nominatim", "_nominatim_request", "nominatim_url", "Nominatim"),
)


class ServerBusy(NetworkUnavailable):
    """An Overpass / Nominatim server answered HTTP 429 or 504 (rate-limited or overloaded)."""

    def __init__(self, message: str, status: int | None = None, exhausted: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.exhausted = exhausted


@dataclass(frozen=True)
class RetryPolicy:
    """How often a busy server is retried per request and mirror, and how long to wait in between."""

    max_attempts: int = 2
    retry_pause_s: float = 30.0

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "RetryPolicy":
        section = get_section(cfg, "network", RETRY_DEFAULTS)
        attempts, pause = section["overpass_max_attempts"], section["overpass_retry_pause_s"]
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
            raise ConfigError(f"network.overpass_max_attempts must be an integer >= 1, got {attempts!r}")
        try:
            pause_s = float(pause)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"network.overpass_retry_pause_s must be a number >= 0, got {pause!r}") from exc
        if isinstance(pause, bool) or not math.isfinite(pause_s) or pause_s < 0:
            raise ConfigError(f"network.overpass_retry_pause_s must be a number >= 0, got {pause!r}")
        return cls(max_attempts=attempts, retry_pause_s=pause_s)


@dataclass
class OverpassLog:
    """What the guarded requests saw: mirrors answering, attempts and OSM snapshot timestamps."""

    attempts: int = 0
    requests_sent: int = 0
    endpoints: list[str] = field(default_factory=list)
    osm_base: list[str] = field(default_factory=list)

    def record(self, url: str, response: Any) -> None:
        if url not in self.endpoints:
            self.endpoints.append(url)
        meta = response.get("osm3s") if isinstance(response, Mapping) else None
        stamp = meta.get("timestamp_osm_base") if isinstance(meta, Mapping) else None
        if isinstance(stamp, str) and stamp and stamp not in self.osm_base:
            self.osm_base.append(stamp)

    @property
    def latest_osm_base(self) -> str | None:
        """Most recent OSM snapshot timestamp seen (ISO 8601 UTC strings sort chronologically)."""
        return max(self.osm_base) if self.osm_base else None


class _BusyAwareRequests:
    """Proxy of the ``requests`` module whose ``get`` / ``post`` raise :class:`ServerBusy` on 429 / 504."""

    def __init__(self, real: Any, label: str, log: OverpassLog) -> None:
        self._real = real
        self._label = label
        self._log = log

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def _checked(self, response: Any, url: Any) -> Any:
        status = getattr(response, "status_code", None)
        if status in BUSY_STATUSES:
            reason = getattr(response, "reason", "") or ""
            raise ServerBusy(f"{self._label} server {url} answered HTTP {status} {reason}".strip(), status=status)
        return response

    def post(self, url: Any, *args: Any, **kwargs: Any) -> Any:
        self._log.requests_sent += 1  # a real HTTP query (cache hits never reach requests)
        return self._checked(self._real.post(url, *args, **kwargs), url)

    def get(self, url: Any, *args: Any, **kwargs: Any) -> Any:
        return self._checked(self._real.get(url, *args, **kwargs), url)


def _bounded(original: Callable[..., Any], ox: Any, setting: str, label: str, policy: RetryPolicy,
             log: OverpassLog, sleep: Callable[[float], None]) -> Callable[..., Any]:
    """Wrap an osmnx request function: at most ``policy.max_attempts`` attempts, each logged."""
    depth = [0]

    def endpoint() -> str:
        return str(getattr(ox.settings, setting, "?"))

    def nested(*args: Any, **kwargs: Any) -> Any:
        # osmnx's own recursive 429/504 retry (only reached if the requests proxy was bypassed).
        if depth[0] >= policy.max_attempts:
            raise ServerBusy(f"{label} server {endpoint()} is busy (HTTP 429/504); gave up after "
                             f"{policy.max_attempts} attempt(s)", exhausted=True)
        LOGGER.warning("%s server %s is busy; osmnx retry %d of %d", label, endpoint(), depth[0] + 1,
                       policy.max_attempts)
        depth[0] += 1
        return original(*args, **kwargs)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if depth[0]:
            return nested(*args, **kwargs)
        url = endpoint()
        for attempt in range(1, policy.max_attempts + 1):
            depth[0] = 1
            log.attempts += 1
            try:
                response = original(*args, **kwargs)
            except ServerBusy as exc:
                if exc.exhausted:
                    raise
                if attempt >= policy.max_attempts:
                    raise ServerBusy(f"{exc}; gave up on {url} after {attempt} attempt(s) "
                                     f"(network.overpass_max_attempts={policy.max_attempts})",
                                     status=exc.status, exhausted=True) from exc
                LOGGER.warning("%s; retrying in %.3g s (attempt %d of %d)", exc, policy.retry_pause_s,
                               attempt + 1, policy.max_attempts)
                sleep(policy.retry_pause_s)
                continue
            finally:
                depth[0] = 0
            if label == "Overpass":
                log.record(url, response)
            return response
        raise AssertionError("unreachable")  # pragma: no cover - the loop always returns or raises

    return wrapper


@contextmanager
def bounded_osmnx_requests(ox: Any, policy: RetryPolicy,
                           sleep: Callable[[float], None] = time.sleep) -> Iterator[OverpassLog]:
    """Bound osmnx's 429/504 retries (see the module docstring); yields the :class:`OverpassLog`.

    A no-op (except for the yielded, empty log) for osmnx builds or test doubles without the
    private request functions.
    """
    log = OverpassLog()
    restore: list[tuple[Any, str, Any]] = []
    for module_name, func_name, setting, label in _TARGETS:
        module = getattr(ox, module_name, None)
        original = getattr(module, func_name, None) if module is not None else None
        if not callable(original):
            LOGGER.debug("osmnx has no %s.%s; %s retries are not bounded", module_name, func_name, label)
            continue
        real_requests = getattr(module, "requests", None)
        if any(callable(getattr(real_requests, verb, None)) for verb in ("get", "post")):
            restore.append((module, "requests", real_requests))
            module.requests = _BusyAwareRequests(real_requests, label, log)
        restore.append((module, func_name, original))
        setattr(module, func_name, _bounded(original, ox, setting, label, policy, log, sleep))
    try:
        yield log
    finally:
        for module, name, value in reversed(restore):
            setattr(module, name, value)
