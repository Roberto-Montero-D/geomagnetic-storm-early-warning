"""Audit CDAW LASCO CME height-time records.

Default mode
------------
Audits seven representative months across the 1996–2025 experiment.

Full mode
---------
Audits every month from 1996 through 2025.

The script reuses locally cached monthly HTML pages and .yht files.

Examples
--------
Sample audit:

    python -m scripts.audit_cdaw_yht

Full audit:

    python -m scripts.audit_cdaw_yht --full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import requests

from src.data.cme_cdaw import (
    audit_yht_record,
    discover_month,
    download_yht,
    summarize_audit,
)


CACHE_DIR = Path(
    "data/raw/cdaw"
)

OUTPUT_DIR = Path(
    "results/temp/cdaw_audit"
)


SAMPLE_MONTHS = [
    (1996, 1),
    (2000, 1),
    (2005, 1),
    (2010, 1),
    (2015, 1),
    (2020, 1),
    (2025, 1),
]


def month_sequence(
    *,
    full: bool,
) -> list[tuple[int, int]]:
    """Return months included in the audit."""

    if not full:
        return SAMPLE_MONTHS

    return [
        (year, month)
        for year in range(
            1996,
            2026,
        )
        for month in range(
            1,
            13,
        )
    ]


def run_audit(
    *,
    full: bool,
    delay_seconds: float,
) -> pd.DataFrame:
    """Run the requested CDAW audit."""

    months = month_sequence(
        full=full,
    )

    records: list[dict] = []

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "geomagnetic-storm-"
                "early-warning/"
                "phase-0-cdaw-audit"
            )
        }
    )

    try:
        for year, month in months:
            print(
                "\n"
                "================================"
            )
            print(
                f"{year:04d}-{month:02d}"
            )
            print(
                "================================"
            )

            try:
                events = discover_month(
                    year,
                    month,
                    cache_dir=CACHE_DIR,
                    session=session,
                    delay_seconds=(
                        delay_seconds
                    ),
                )

            except Exception as exc:
                print(
                    "MONTH FAILED:",
                    repr(exc),
                )

                records.append(
                    {
                        "year": year,
                        "month": month,
                        "status": (
                            "monthly_page_failed"
                        ),
                        "error": repr(exc),
                    }
                )

                continue

            print(
                f"Catalog events: "
                f"{len(events):,}"
            )

            for index, event in enumerate(
                events,
                start=1,
            ):
                print(
                    (
                        f"\r"
                        f"  {index:,}/"
                        f"{len(events):,}"
                    ),
                    end="",
                    flush=True,
                )

                base = {
                    "year": year,
                    "month": month,
                    "status": "ok",
                    "error": None,
                }

                if event.yht_url is None:
                    records.append(
                        {
                            **base,
                            "status": (
                                "missing_yht_url"
                            ),
                            "first_appearance": (
                                event.first_appearance
                            ),
                            "remarks": (
                                event.remarks
                            ),
                            "retrospective_insert": (
                                event.retrospective_insert
                            ),
                            "primary_candidate": (
                                event.primary_candidate
                            ),
                        }
                    )

                    continue

                try:
                    yht = download_yht(
                        event,
                        cache_dir=CACHE_DIR,
                        session=session,
                        delay_seconds=(
                            delay_seconds
                        ),
                    )

                    audit = audit_yht_record(
                        event,
                        yht,
                    )

                    records.append(
                        {
                            **base,
                            **audit,
                        }
                    )

                except Exception as exc:
                    records.append(
                        {
                            **base,
                            "status": (
                                "yht_failed"
                            ),
                            "error": repr(exc),
                            "first_appearance": (
                                event.first_appearance
                            ),
                            "yht_url": (
                                event.yht_url
                            ),
                            "remarks": (
                                event.remarks
                            ),
                            "retrospective_insert": (
                                event.retrospective_insert
                            ),
                            "primary_candidate": (
                                event.primary_candidate
                            ),
                        }
                    )

            print()

    finally:
        session.close()

    return pd.DataFrame(
        records
    )


def print_summary(
    df: pd.DataFrame,
) -> dict:
    """Print and return aggregate audit statistics."""

    ok = df[
        df["status"] == "ok"
    ].copy()

    summary = summarize_audit(
        ok
    )

    summary[
        "total_rows"
    ] = len(df)

    summary[
        "successful_yht_records"
    ] = len(ok)

    summary[
        "failed_records"
    ] = int(
        (
            df["status"] != "ok"
        ).sum()
    )

    print(
        "\n\n"
        "======================================"
    )

    print(
        "CDAW HEIGHT-TIME AUDIT SUMMARY"
    )

    print(
        "======================================"
    )

    for key, value in summary.items():
        if isinstance(
            value,
            float,
        ):
            print(
                f"{key:40s}: "
                f"{value:.6f}"
            )

        else:
            print(
                f"{key:40s}: "
                f"{value}"
            )

    return summary

def summarize_existing_audit(
    audit_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """Recalculate a summary from an existing audit CSV."""

    df = pd.read_csv(
        audit_path,
        low_memory=False,
    )

    summary = print_summary(
        df
    )

    return df, summary


def save_results(
    df: pd.DataFrame,
    summary: dict,
    *,
    full: bool,
) -> None:
    """Save row-level and aggregate audit output."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = (
        "full"
        if full
        else "sample"
    )

    csv_path = (
        OUTPUT_DIR
        / f"{prefix}_audit.csv"
    )

    json_path = (
        OUTPUT_DIR
        / f"{prefix}_summary.json"
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    json_path.write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        f"\nSaved audit rows: "
        f"{csv_path}"
    )

    print(
        f"Saved summary: "
        f"{json_path}"
    )


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Audit all months from "
            "1996 through 2025."
        ),
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help=(
            "Delay in seconds after each "
            "uncached CDAW request."
        ),
    )

    parser.add_argument(
    "--summarize-existing",
    type=Path,
    default=None,
    help=(
        "Recalculate summary from an existing audit CSV "
        "without downloading any CDAW data."
    ),
)

    args = parser.parse_args()

    if args.summarize_existing is not None:
        df, summary = summarize_existing_audit(
            args.summarize_existing
        )

        json_path = (
            OUTPUT_DIR
            / "full_summary.json"
        )

        json_path.write_text(
            json.dumps(
                summary,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        print(
            f"\nUpdated summary: "
            f"{json_path}"
        )

        return

    print(
        "CDAW LASCO HEIGHT-TIME AUDIT"
    )

    print(
        "Mode:",
        (
            "FULL"
            if args.full
            else "SEVEN-MONTH SAMPLE"
        ),
    )

    print(
        f"Cache: {CACHE_DIR}"
    )

    df = run_audit(
        full=args.full,
        delay_seconds=args.delay,
    )

    summary = print_summary(
        df
    )

    save_results(
        df,
        summary,
        full=args.full,
    )


if __name__ == "__main__":
    main()