from datetime import datetime, timezone


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def format_for_display(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def to_iso(dt: datetime) -> str:
    return dt.isoformat()
