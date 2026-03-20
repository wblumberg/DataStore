"""SQLite-backed observation store utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Sequence


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def to_iso_utc(value: datetime) -> str:
    """Serialize datetimes in canonical UTC format."""
    return _ensure_utc(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time_like(value: Any) -> datetime | None:
    """Parse common time representations used by ingest workflows."""
    if value is None:
        return None

    if isinstance(value, datetime):
        return _ensure_utc(value)

    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None

    text = str(value).strip()
    if not text:
        return None

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(text)
        return _ensure_utc(parsed)
    except ValueError:
        pass

    formats = (
        "%Y%m%d%H%M%S",
        "%Y%m%d%H%M",
        "%Y%m%d%H",
        "%Y%m%d_%H%M%S",
        "%Y%m%d_%H%M",
        "%Y%m%d_%H",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H",
    )
    compact = text.replace("T", " ")
    for fmt in formats:
        try:
            return datetime.strptime(compact, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    return None


@dataclass(frozen=True)
class ObservationInsert:
    obs_type: str
    obs_time: datetime
    platform_id: str | None
    lat: float | None
    lon: float | None
    elevation_m: float | None
    source_name: str | None
    payload: dict[str, Any]


class ObservationSQLStore:
    """Simple SQLite storage layer for point observations."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser().resolve()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    obs_type TEXT NOT NULL,
                    obs_time TEXT NOT NULL,
                    platform_id TEXT,
                    lat REAL,
                    lon REAL,
                    elevation_m REAL,
                    source_name TEXT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_observations_type_time ON observations (obs_type, obs_time)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_observations_source_time ON observations (source_name, obs_time)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_observations_time ON observations (obs_time)"
            )
            conn.commit()

    def insert_many(self, rows: Sequence[ObservationInsert]) -> dict[str, int]:
        attempted = len(rows)
        if attempted == 0:
            return {"attempted": 0, "inserted": 0, "skipped": 0}

        payloads = []
        skipped = 0
        for row in rows:
            obs_time = parse_time_like(row.obs_time)
            if obs_time is None:
                skipped += 1
                continue

            payloads.append(
                (
                    row.obs_type.strip().upper(),
                    to_iso_utc(obs_time),
                    row.platform_id,
                    row.lat,
                    row.lon,
                    row.elevation_m,
                    row.source_name,
                    json.dumps(row.payload, separators=(",", ":"), default=str),
                )
            )

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO observations (
                    obs_type, obs_time, platform_id, lat, lon, elevation_m, source_name, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payloads,
            )
            conn.commit()

        inserted = len(payloads)
        return {
            "attempted": attempted,
            "inserted": inserted,
            "skipped": (attempted - inserted) + skipped,
        }

    def list_unique_times(self, obs_types: Sequence[str] | None = None, limit: int = 200) -> list[str]:
        where, params = self._build_filters(obs_types=obs_types)
        sql = (
            "SELECT DISTINCT obs_time FROM observations"
            f" {where} ORDER BY obs_time DESC LIMIT ?"
        )
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [str(row["obs_time"]) for row in rows]

    def query_observations(
        self,
        *,
        obs_types: Sequence[str] | None = None,
        center_time: datetime | None = None,
        minutes_before: int = 0,
        minutes_after: int = 0,
        latest_only: bool = False,
        prefer_most_data: bool = True,
        parameter_names: Sequence[str] | None = None,
        bin_minutes: int = 0,
        max_rows: int = 50000,
    ) -> dict[str, Any]:
        center_dt = parse_time_like(center_time) if center_time is not None else None
        start_time = None
        end_time = None
        if center_dt is not None:
            start_time = center_dt - timedelta(minutes=max(0, minutes_before))
            end_time = center_dt + timedelta(minutes=max(0, minutes_after))

        where, params = self._build_filters(
            obs_types=obs_types,
            after=start_time,
            before=end_time,
        )

        sql = (
            "SELECT obs_type, obs_time, platform_id, lat, lon, elevation_m, source_name, payload_json "
            "FROM observations"
            f" {where} ORDER BY obs_time DESC LIMIT ?"
        )
        params.append(int(max_rows))

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        records = []
        for row in rows:
            payload = json.loads(row["payload_json"]) if row["payload_json"] else {}
            if parameter_names:
                requested = [name for name in parameter_names if name]
                payload = {key: payload.get(key) for key in requested if key in payload}

            record = {
                "obs_type": row["obs_type"],
                "obs_time": row["obs_time"],
                "platform_id": row["platform_id"],
                "lat": row["lat"],
                "lon": row["lon"],
                "elevation_m": row["elevation_m"],
                "source_name": row["source_name"],
                "payload": payload,
            }
            if bin_minutes > 0:
                bucket = _bucket_time(parse_time_like(row["obs_time"]), bin_minutes)
                record["time_bin"] = to_iso_utc(bucket) if bucket is not None else None
            records.append(record)

        selected_time = None
        if latest_only and records:
            if prefer_most_data:
                counts: dict[str, int] = {}
                for record in records:
                    obs_time = str(record["obs_time"])
                    counts[obs_time] = counts.get(obs_time, 0) + 1
                selected_time = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
            else:
                selected_time = max(str(record["obs_time"]) for record in records)
            records = [record for record in records if str(record["obs_time"]) == selected_time]

        return {
            "count": len(records),
            "center_time": to_iso_utc(center_dt) if center_dt is not None else None,
            "selected_time": selected_time,
            "records": records,
        }

    def delete_rows(
        self,
        *,
        obs_types: Sequence[str] | None = None,
        source_names: Sequence[str] | None = None,
        before: datetime | None = None,
        after: datetime | None = None,
    ) -> int:
        where_parts = []
        params: list[Any] = []

        if obs_types:
            values = [item.strip().upper() for item in obs_types if item and item.strip()]
            if values:
                where_parts.append("obs_type IN (%s)" % ",".join("?" for _ in values))
                params.extend(values)

        if source_names:
            values = [item.strip() for item in source_names if item and item.strip()]
            if values:
                where_parts.append("source_name IN (%s)" % ",".join("?" for _ in values))
                params.extend(values)

        if before is not None:
            where_parts.append("obs_time <= ?")
            params.append(to_iso_utc(_ensure_utc(before)))

        if after is not None:
            where_parts.append("obs_time >= ?")
            params.append(to_iso_utc(_ensure_utc(after)))

        sql = "DELETE FROM observations"
        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)

        with self._connect() as conn:
            cursor = conn.execute(sql, params)
            conn.commit()
            return int(cursor.rowcount if cursor.rowcount is not None else 0)

    def prune_older_than(self, *, keep_days: int, obs_types: Sequence[str] | None = None) -> int:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=max(0, int(keep_days)))

        where_parts = ["obs_time < ?"]
        params: list[Any] = [to_iso_utc(cutoff)]

        if obs_types:
            values = [item.strip().upper() for item in obs_types if item and item.strip()]
            if values:
                where_parts.append("obs_type IN (%s)" % ",".join("?" for _ in values))
                params.extend(values)

        sql = "DELETE FROM observations WHERE " + " AND ".join(where_parts)

        with self._connect() as conn:
            cursor = conn.execute(sql, params)
            conn.commit()
            return int(cursor.rowcount if cursor.rowcount is not None else 0)

    def vacuum(self) -> None:
        with self._connect() as conn:
            conn.execute("VACUUM")
            conn.commit()

    def _build_filters(
        self,
        *,
        obs_types: Sequence[str] | None = None,
        before: datetime | None = None,
        after: datetime | None = None,
    ) -> tuple[str, list[Any]]:
        clauses = []
        params: list[Any] = []

        if obs_types:
            values = [item.strip().upper() for item in obs_types if item and item.strip()]
            if values:
                clauses.append("obs_type IN (%s)" % ",".join("?" for _ in values))
                params.extend(values)

        if after is not None:
            clauses.append("obs_time >= ?")
            params.append(to_iso_utc(_ensure_utc(after)))

        if before is not None:
            clauses.append("obs_time <= ?")
            params.append(to_iso_utc(_ensure_utc(before)))

        if not clauses:
            return "", params

        return "WHERE " + " AND ".join(clauses), params


def _bucket_time(value: datetime | None, interval_minutes: int) -> datetime | None:
    if value is None:
        return None
    if interval_minutes <= 0:
        return value

    utc = _ensure_utc(value)
    interval = interval_minutes * 60
    epoch = int(utc.timestamp())
    bucket_epoch = epoch - (epoch % interval)
    return datetime.fromtimestamp(bucket_epoch, tz=timezone.utc)


__all__ = [
    "ObservationInsert",
    "ObservationSQLStore",
    "parse_time_like",
    "to_iso_utc",
]
