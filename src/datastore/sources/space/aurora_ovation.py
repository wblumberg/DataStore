"""Aurora ovation source helpers."""

from __future__ import annotations

import argparse

import requests

AURORA_OVATION_URL = "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"


def fetch_latest_aurora_json(url: str = AURORA_OVATION_URL, timeout: int = 20) -> dict:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch latest SWPC ovation aurora JSON")
    parser.add_argument("--url", default=AURORA_OVATION_URL)
    parser.add_argument("--timeout", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = fetch_latest_aurora_json(url=args.url, timeout=args.timeout)
    print(payload)
    return 0


__all__ = ["AURORA_OVATION_URL", "fetch_latest_aurora_json", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
