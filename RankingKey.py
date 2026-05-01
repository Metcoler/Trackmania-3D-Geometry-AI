from __future__ import annotations

import math
from functools import lru_cache
from typing import Mapping, Tuple


RANKING_METRIC_NAMES = (
    "finished",
    "crashes",
    "progress",
    "progress_norm",
    "ranking_progress",
    "ranking_progress_norm",
    "discrete_progress",
    "discrete_progress_norm",
    "dense_progress",
    "dense_progress_norm",
    "time",
    "distance",
)


def canonical_ranking_key_expression(value: str) -> str:
    expression = str(value).strip()
    if not expression:
        raise ValueError("Ranking key expression cannot be empty.")
    if (
        "_" in expression
        and expression not in RANKING_METRIC_NAMES
        and not any(char in expression for char in "(),")
    ):
        raise ValueError(
            "Legacy ranking key names are no longer supported. "
            "Use an explicit tuple, for example "
            "'(dense_progress, finished, -time, -crashes, -distance)'."
        )
    return expression


@lru_cache(maxsize=128)
def parse_ranking_key_expression(value: str) -> Tuple[Tuple[float, str], ...]:
    expression = canonical_ranking_key_expression(value)
    if expression.startswith("(") and expression.endswith(")"):
        expression = expression[1:-1]

    items: list[Tuple[float, str]] = []
    for raw_part in expression.split(","):
        part = raw_part.strip()
        if not part:
            continue

        sign = 1.0
        if part.startswith("+"):
            part = part[1:].strip()
        elif part.startswith("-"):
            sign = -1.0
            part = part[1:].strip()

        if part not in RANKING_METRIC_NAMES:
            allowed = ", ".join(RANKING_METRIC_NAMES)
            raise ValueError(
                f"Unknown ranking metric '{part}' in '{value}'. Allowed metrics: {allowed}."
            )
        items.append((sign, part))

    if not items:
        raise ValueError(f"Ranking key expression '{value}' produced no metrics.")
    return tuple(items)


def finite_or_large(value: float) -> float:
    value = float(value)
    if math.isfinite(value):
        return value
    return 1e9


def evaluate_ranking_key(
    expression: str,
    metrics: Mapping[str, float],
) -> Tuple[float, ...]:
    parsed = parse_ranking_key_expression(expression)
    return tuple(sign * float(metrics[name]) for sign, name in parsed)
