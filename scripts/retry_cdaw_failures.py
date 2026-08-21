"""Retry transient CDAW .yht failures from an existing audit.

This script does NOT rerun the complete CDAW audit.

It retries only records with:

    status == "yht_failed"

Monthly-page failures, including known LASCO outage periods, are
preserved and are not retried here.

Important
---------
Updates are applied atomically at the row level:

- the original failed row is preserved until the retry and audit both
  succeed;
- status is changed to ``ok`` only after every replacement value has
  been prepared successfully;
- a failed DataFrame update cannot silently convert a failed row into
  a successful row.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pandas as pd
import requests

from src.data.cme_cdaw import (
    CdawCatalogEvent,
    audit_yht_record,
    download_yht,
)


CACHE_DIR = Path(
    "data/raw/cdaw"
)

DEFAULT_AUDIT_PATH = Path(
    "results/temp/cdaw_audit/full_audit.csv"
)


def _optional_string(
    value,
) -> str:
    """Convert a possibly missing CSV value to a safe string."""

    if pd.isna(value):
        return ""

    return str(value)


def event_from_failed_row(
    row: pd.Series,
) -> CdawCatalogEvent:
    """Reconstruct the event object required for a .yht retry."""

    yht_url = _optional_string(
        row.get(
            "yht_url",
            "",
        )
    )

    if not yht_url:
        raise ValueError(
            "Failed audit row has no .yht URL."
        )

    return CdawCatalogEvent(
        first_appearance=pd.Timestamp(
            row["first_appearance"]
        ),
        central_pa_raw=_optional_string(
            row.get(
                "central_pa_raw",
                "",
            )
        ),
        width_raw=_optional_string(
            row.get(
                "width_raw",
                "",
            )
        ),
        linear_speed_raw=_optional_string(
            row.get(
                "linear_speed_raw",
                "",
            )
        ),
        remarks=_optional_string(
            row.get(
                "remarks",
                "",
            )
        ),
        yht_url=yht_url,
    )


def _prepare_row_update(
    event: CdawCatalogEvent,
    audit: dict,
) -> dict:
    """Prepare a complete successful row update.

    Timestamp values are serialized before insertion so pandas does not
    reject them when the CSV-backed column currently has string dtype.
    """

    update = {
        "year": event.first_appearance.year,
        "month": event.first_appearance.month,
        "status": "ok",
        "error": None,
    }

    for key, value in audit.items():

        if isinstance(
            value,
            pd.Timestamp,
        ):
            value = value.isoformat(
                sep=" "
            )

        elif value is pd.NaT:
            value = None

        update[key] = value

    return update


def retry_yht_failures(
    audit_path: Path,
    *,
    delay_seconds: float,
) -> pd.DataFrame:
    """Retry only rows previously marked ``yht_failed``."""

    # Object dtype prevents strict string columns from rejecting
    # Timestamp/bool/numeric values during row replacement.
    df = pd.read_csv(
        audit_path,
        low_memory=False,
        dtype=object,
    )

    failed_mask = (
        df["status"]
        == "yht_failed"
    )

    failed_indices = df.index[
        failed_mask
    ].tolist()

    print(
        f"Transient .yht failures found: "
        f"{len(failed_indices):,}"
    )

    if not failed_indices:
        print(
            "Nothing to retry."
        )

        return df

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
        for index in failed_indices:

            row = df.loc[index].copy()

            print(
                "\nRetrying:"
            )

            print(
                "  first appearance: "
                f"{row['first_appearance']}"
            )

            print(
                "  URL: "
                f"{row['yht_url']}"
            )

            try:
                event = event_from_failed_row(
                    row
                )

                record = download_yht(
                    event,
                    cache_dir=CACHE_DIR,
                    session=session,
                    delay_seconds=(
                        delay_seconds
                    ),
                )

                audit = audit_yht_record(
                    event,
                    record,
                )

                # Build the complete replacement BEFORE mutating df.
                update = _prepare_row_update(
                    event,
                    audit,
                )

                # Add any columns introduced by the successful audit.
                for column in update:
                    if column not in df.columns:
                        df[column] = pd.NA

                # Only now mutate the existing row.
                #
                # We do not clear the row first: fields not replaced by
                # the audit remain preserved from the original record.
                for column, value in update.items():
                    df.at[
                        index,
                        column,
                    ] = value

                print(
                    "  RETRY SUCCESSFUL"
                )

            except Exception as exc:

                # Preserve the original failure status.
                df.at[
                    index,
                    "status",
                ] = "yht_failed"

                df.at[
                    index,
                    "error",
                ] = repr(
                    exc
                )

                print(
                    "  RETRY FAILED:"
                )

                print(
                    f"  {exc!r}"
                )

    finally:
        session.close()

    return df


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audit",
        type=Path,
        default=DEFAULT_AUDIT_PATH,
        help=(
            "Existing full audit CSV."
        ),
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help=(
            "Delay after an uncached request."
        ),
    )

    args = parser.parse_args()

    print(
        "CDAW TRANSIENT FAILURE RETRY"
    )

    print(
        f"Audit: {args.audit}"
    )

    if not args.audit.exists():
        raise FileNotFoundError(
            f"Audit file not found: "
            f"{args.audit}"
        )

    backup_path = (
        args.audit.parent
        / (
            args.audit.stem
            + "_before_retry.csv"
        )
    )

    # Create the backup BEFORE any processing/writing.
    if not backup_path.exists():
        shutil.copy2(
            args.audit,
            backup_path,
        )

        print(
            f"Backup saved: "
            f"{backup_path}"
        )

    updated = retry_yht_failures(
        args.audit,
        delay_seconds=args.delay,
    )

    # Write only after all retry processing has completed.
    updated.to_csv(
        args.audit,
        index=False,
    )

    print(
        f"\nUpdated audit saved: "
        f"{args.audit}"
    )

    remaining = int(
        (
            updated["status"]
            == "yht_failed"
        ).sum()
    )

    print(
        f"Remaining .yht failures: "
        f"{remaining}"
    )


if __name__ == "__main__":
    main()