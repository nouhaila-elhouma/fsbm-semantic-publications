"""Retry avec backoff exponentiel, délais polis et exceptions de scraping.

Le projet ne contourne JAMAIS les protections de Google Scholar : un blocage
(HTTP 429, page "unusual traffic", CAPTCHA) lève ``BlockedError`` qui n'est
pas rejouée automatiquement — l'appelant sauvegarde puis arrête proprement.
"""
from __future__ import annotations

import functools
import logging
import random
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)
F = TypeVar("F", bound=Callable[..., Any])


class BlockedError(RuntimeError):
    """La source a bloqué / limité les requêtes (429, CAPTCHA, trafic inhabituel)."""


class TransientError(RuntimeError):
    """Erreur temporaire (réseau, 5xx) : peut être rejouée."""


def backoff_delay(attempt: int, base: float = 2.0, cap: float = 120.0, jitter: float = 0.25) -> float:
    """Délai exponentiel : ``base * 2**attempt`` (plafonné) avec gigue aléatoire."""
    delay = min(cap, base * (2 ** attempt))
    return delay * (1 + random.uniform(-jitter, jitter))


def polite_sleep(min_delay: float, max_delay: float) -> float:
    """Attend un délai aléatoire uniforme dans ``[min_delay, max_delay]`` ; retourne la durée."""
    delay = random.uniform(min_delay, max(min_delay, max_delay))
    time.sleep(delay)
    return delay


def retry(
    retries: int = 3,
    base_delay: float = 2.0,
    retry_on: tuple[type[BaseException], ...] = (TransientError,),
    never_retry: tuple[type[BaseException], ...] = (BlockedError,),
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[[F], F]:
    """Décorateur : rejoue la fonction sur ``retry_on`` (max ``retries`` rejeux).

    Les exceptions de ``never_retry`` sont propagées immédiatement.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except never_retry:
                    raise
                except retry_on as exc:
                    if attempt >= retries:
                        logger.error("%s: échec définitif après %d tentatives (%s)",
                                     func.__name__, attempt + 1, exc)
                        raise
                    delay = backoff_delay(attempt, base_delay)
                    logger.warning("%s: %s — nouvelle tentative %d/%d dans %.1fs",
                                   func.__name__, exc, attempt + 1, retries, delay)
                    sleep(delay)
            raise RuntimeError("unreachable")  # pragma: no cover

        return wrapper  # type: ignore[return-value]

    return decorator
