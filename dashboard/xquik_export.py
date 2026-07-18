from __future__ import annotations

from typing import Any


TEXT_COLUMNS = (
    "text",
    "tweet_text",
    "full_text",
    "content",
    "body",
    "message",
    "comment",
)

USER_COLUMNS = (
    "username",
    "user",
    "author_username",
    "screen_name",
    "handle",
    "name",
)

DATE_COLUMNS = (
    "created_at",
    "date",
    "timestamp",
    "time",
)

SENTIMENT_COLUMNS = (
    "sentiment",
    "label",
    "prediction",
    "polarity",
)

ID_COLUMNS = (
    "id",
    "tweet_id",
    "post_id",
    "status_id",
    "source_id",
    "url",
)

SENTIMENT_ALIASES = {
    "pos": "positive",
    "positive": "positive",
    "1": "positive",
    "neg": "negative",
    "negative": "negative",
    "-1": "negative",
    "neu": "neutral",
    "neutral": "neutral",
    "0": "neutral",
}


def _value(row: dict[str, Any], aliases: tuple[str, ...]) -> str:
    for alias in aliases:
        value = row.get(alias)
        if value is None:
            continue
        cleaned = str(value).strip()
        if cleaned and cleaned.lower() not in {"<na>", "nan", "nat"}:
            return cleaned
    return ""


def _normalize_sentiment(value: str) -> str:
    return SENTIMENT_ALIASES.get(value.strip().lower(), "neutral")


def normalize_xquik_records(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []

    for index, row in enumerate(records, start=1):
        normalized_row = {str(key).strip().lower(): value for key, value in row.items()}
        text = _value(normalized_row, TEXT_COLUMNS)
        if not text:
            continue

        normalized.append(
            {
                "text": text,
                "cleaned_text": text.lower(),
                "user": _value(normalized_row, USER_COLUMNS) or "xquik_export",
                "date": _value(normalized_row, DATE_COLUMNS),
                "sentiment": _normalize_sentiment(
                    _value(normalized_row, SENTIMENT_COLUMNS)
                ),
                "source_id": _value(normalized_row, ID_COLUMNS) or str(index),
            }
        )

    return normalized
