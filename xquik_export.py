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
    lookup = {str(key).strip().lower(): str(key) for key in row}
    for alias in aliases:
        key = lookup.get(alias)
        if key is None:
            continue
        value = row.get(key)
        if value is None:
            continue
        cleaned = str(value).strip()
        if cleaned and cleaned.lower() != "nan":
            return cleaned
    return ""


def _normalize_sentiment(value: str) -> str:
    return SENTIMENT_ALIASES.get(value.strip().lower(), "neutral")


def normalize_xquik_records(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []

    for index, row in enumerate(records, start=1):
        text = _value(row, TEXT_COLUMNS)
        if not text:
            continue

        normalized.append(
            {
                "text": text,
                "cleaned_text": text.lower(),
                "user": _value(row, USER_COLUMNS) or "xquik_export",
                "date": _value(row, DATE_COLUMNS) or str(index),
                "sentiment": _normalize_sentiment(_value(row, SENTIMENT_COLUMNS)),
                "source_id": _value(row, ID_COLUMNS) or str(index),
            }
        )

    return normalized
