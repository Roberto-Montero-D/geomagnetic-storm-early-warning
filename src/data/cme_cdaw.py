"""CDAW SOHO/LASCO CME catalog acquisition and audit utilities.

This module is intentionally an audit/acquisition layer.

It does NOT construct model features.

Responsibilities
----------------
1. Discover CME records from CDAW monthly catalog pages.
2. Preserve the exact .yht URL associated with each CME.
3. Parse raw timestamped height-time measurements.
4. Preserve retrospective CDAW metadata separately.
5. Flag explicitly retrospectively inserted catalog events.
6. Cache downloaded source files.
7. Provide reproducible audit summaries.

Causality
---------
The final CDAW catalog contains retrospective information and MUST NOT
be treated as historically available at first CME appearance.

Only timestamped height-time measurements may later be considered as
candidate causal inputs, and only when their individual measurement
timestamps satisfy the project's temporal cutoff.

Final catalog quantities such as:

- SPEED
- ACCEL
- WIDTH
- ONSET1 / ONSET2
- mass
- kinetic energy

are retained only for audit/validation and MUST NOT be exposed as
primary causal predictors.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup


CDAW_BASE_URL = (
    "https://cdaw.gsfc.nasa.gov/"
    "CME_list/UNIVERSAL_ver2"
)

DEFAULT_USER_AGENT = (
    "geomagnetic-storm-early-warning/"
    "phase-0-cdaw-audit"
)


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class CdawCatalogEvent:
    """One CME row discovered from a monthly CDAW catalog page."""

    first_appearance: pd.Timestamp
    central_pa_raw: str
    width_raw: str
    linear_speed_raw: str
    remarks: str
    yht_url: str | None

    @property
    def retrospective_insert(self) -> bool:
        """Whether CDAW explicitly marks the CME as later inserted."""

        return bool(
            re.search(
                r"\bnewly\s+inserted\b",
                self.remarks,
                flags=re.IGNORECASE,
            )
        )

    @property
    def insertion_date(self) -> pd.Timestamp | None:
        """Extract an explicit retrospective insertion date."""

        match = re.search(
            r"newly\s+inserted\s+on\s+"
            r"(\d{4}/\d{2}/\d{2})",
            self.remarks,
            flags=re.IGNORECASE,
        )

        if match is None:
            return None

        return pd.Timestamp(
            datetime.strptime(
                match.group(1),
                "%Y/%m/%d",
            )
        )

    @property
    def primary_candidate(self) -> bool:
        """Return the provisional primary-candidate flag.

        IMPORTANT
        ---------
        ``True`` means only that no explicit retrospective-insertion
        marker was found.

        It does NOT prove historical real-time operator availability.
        """

        return not self.retrospective_insert


@dataclass
class CdawYhtRecord:
    """Parsed CDAW .yht record."""

    source_url: str | None
    metadata: dict[str, str]
    measurements: pd.DataFrame

    @property
    def n_measurements(self) -> int:
        """Number of parsed height-time measurements."""

        return len(self.measurements)

    @property
    def first_measurement_time(self) -> pd.Timestamp | None:
        """Earliest measurement timestamp, independent of source order."""

        if self.measurements.empty:
            return None

        return self.measurements["timestamp"].min()

    @property
    def last_measurement_time(self) -> pd.Timestamp | None:
        """Latest measurement timestamp, independent of source order."""

        if self.measurements.empty:
            return None

        return self.measurements["timestamp"].max()

    @property
    def telescope_set(self) -> set[str]:
        """Set of telescopes represented in the trajectory."""

        if self.measurements.empty:
            return set()

        return set(
            self.measurements["telescope"]
            .dropna()
            .astype(str)
        )


# ---------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------


def monthly_catalog_url(
    year: int,
    month: int,
) -> str:
    """Return the CDAW monthly catalog URL."""

    if not 1 <= month <= 12:
        raise ValueError(
            f"Invalid month: {month}"
        )

    return (
        f"{CDAW_BASE_URL}/"
        f"{year:04d}_{month:02d}/"
        f"univ{year:04d}_{month:02d}.html"
    )


# ---------------------------------------------------------------------
# HTTP / caching
# ---------------------------------------------------------------------


def _session() -> requests.Session:
    """Create a requests session with a project-specific user agent."""

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": DEFAULT_USER_AGENT,
        }
    )

    return session


def fetch_text(
    url: str,
    *,
    timeout: float = 30.0,
    session: requests.Session | None = None,
) -> str:
    """Download text from CDAW."""

    own_session = session is None

    if session is None:
        session = _session()

    try:
        response = session.get(
            url,
            timeout=timeout,
        )

        response.raise_for_status()

        response.encoding = (
            response.apparent_encoding
            or response.encoding
            or "utf-8"
        )

        return response.text

    finally:
        if own_session:
            session.close()


def fetch_cached_text(
    url: str,
    cache_path: str | Path,
    *,
    timeout: float = 30.0,
    delay_seconds: float = 0.0,
    session: requests.Session | None = None,
    overwrite: bool = False,
) -> str:
    """Fetch text with a persistent local cache."""

    cache_path = Path(cache_path)

    if cache_path.exists() and not overwrite:
        return cache_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

    text = fetch_text(
        url,
        timeout=timeout,
        session=session,
    )

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path.write_text(
        text,
        encoding="utf-8",
    )

    if delay_seconds > 0:
        time.sleep(delay_seconds)

    return text


# ---------------------------------------------------------------------
# Monthly HTML parsing
# ---------------------------------------------------------------------


def _clean_cell_text(cell) -> str:
    """Return normalized text from one HTML table cell."""

    return " ".join(
        cell.stripped_strings
    )


def parse_monthly_catalog(
    html: str,
    *,
    page_url: str,
) -> list[CdawCatalogEvent]:
    """Parse one CDAW monthly catalog page.

    The exact HTML hyperlink associated with the CME time is retained.

    This is important because distinct CME events may share the same
    first-appearance timestamp.
    """

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    events: list[CdawCatalogEvent] = []

    for tr in soup.find_all("tr"):
        cells = tr.find_all("td")

        if len(cells) < 12:
            continue

        date_text = _clean_cell_text(
            cells[0]
        )

        time_text = _clean_cell_text(
            cells[1]
        )

        if not re.fullmatch(
            r"\d{4}/\d{2}/\d{2}",
            date_text,
        ):
            continue

        if not re.fullmatch(
            r"\d{2}:\d{2}:\d{2}",
            time_text,
        ):
            continue

        first_appearance = pd.Timestamp(
            datetime.strptime(
                f"{date_text} {time_text}",
                "%Y/%m/%d %H:%M:%S",
            )
        )

        central_pa_raw = _clean_cell_text(
            cells[2]
        )

        width_raw = _clean_cell_text(
            cells[3]
        )

        linear_speed_raw = _clean_cell_text(
            cells[4]
        )

        remarks = _clean_cell_text(
            cells[-1]
        )

        time_link = cells[1].find(
            "a",
            href=True,
        )

        yht_url: str | None = None

        if time_link is not None:
            yht_url = requests.compat.urljoin(
                page_url,
                time_link["href"],
            )

        events.append(
            CdawCatalogEvent(
                first_appearance=first_appearance,
                central_pa_raw=central_pa_raw,
                width_raw=width_raw,
                linear_speed_raw=linear_speed_raw,
                remarks=remarks,
                yht_url=yht_url,
            )
        )

    return events


def discover_month(
    year: int,
    month: int,
    *,
    cache_dir: str | Path,
    session: requests.Session | None = None,
    delay_seconds: float = 0.25,
) -> list[CdawCatalogEvent]:
    """Download/cache and parse one monthly CDAW page."""

    url = monthly_catalog_url(
        year,
        month,
    )

    cache_path = (
        Path(cache_dir)
        / "monthly_html"
        / f"{year:04d}_{month:02d}.html"
    )

    html = fetch_cached_text(
        url,
        cache_path,
        session=session,
        delay_seconds=delay_seconds,
    )

    return parse_monthly_catalog(
        html,
        page_url=url,
    )


# ---------------------------------------------------------------------
# .yht parsing
# ---------------------------------------------------------------------


def parse_yht(
    text: str,
    *,
    source_url: str | None = None,
) -> CdawYhtRecord:
    """Parse a CDAW height-time (.yht) record.

    CDAW responses are not assumed to preserve physical line breaks.
    Parsing therefore uses structural markers rather than newline
    layout.

    Retrospective metadata and raw timestamped measurements are stored
    separately.

    Expected measurement structure:

        HEIGHT DATE TIME ANGLE TEL FC COL ROW

    Raw measurement order is preserved exactly as encountered in the
    source. Sorting is intentionally deferred to downstream code after
    source-order validation.
    """

    normalized = re.sub(
        r"\s+",
        " ",
        text.replace(
            "\x00",
            " ",
        ),
    ).strip()

    metadata: dict[str, str] = {}

    # Metadata can use either:
    #
    #   #KEY: value
    #   #KEY=value
    #
    metadata_pattern = re.compile(
        r"#([A-Z0-9_-]+)\s*[:=]\s*"
        r"(.*?)"
        r"(?="
        r"\s+#[A-Z0-9_-]+\s*[:=]"
        r"|\s+#\s*HEIGHT\b"
        r"|$"
        r")",
        flags=re.IGNORECASE,
    )

    for match in metadata_pattern.finditer(
        normalized
    ):
        key = match.group(1).upper()
        value = match.group(2).strip()

        metadata[key] = value

    header_pattern = re.compile(
        r"#\s*HEIGHT\s+"
        r"DATE\s+"
        r"TIME\s+"
        r"ANGLE\s+"
        r"TEL\s+"
        r"FC\s+"
        r"COL\s+"
        r"ROW",
        flags=re.IGNORECASE,
    )

    header_match = header_pattern.search(
        normalized
    )

    measurement_rows: list[dict] = []

    if header_match is None:
        measurements = pd.DataFrame(
            columns=[
                "timestamp",
                "height_rsun",
                "angle_deg",
                "telescope",
                "feature_code",
                "col",
                "row",
            ]
        )

        return CdawYhtRecord(
            source_url=source_url,
            metadata=metadata,
            measurements=measurements,
        )

    measurement_text = normalized[
        header_match.end():
    ].strip()

    measurement_pattern = re.compile(
        r"(?P<height>[+-]?\d+(?:\.\d+)?)\s+"
        r"(?P<date>\d{4}/\d{2}/\d{2})\s+"
        r"(?P<time>\d{2}:\d{2}:\d{2})\s+"
        r"(?P<angle>[+-]?\d+(?:\.\d+)?)\s+"
        r"(?P<telescope>C2|C3)\s+"
        r"(?P<feature_code>[+-]?\d+)\s+"
        r"(?P<col>[+-]?\d+(?:\.\d+)?)\s+"
        r"(?P<row>[+-]?\d+(?:\.\d+)?)",
        flags=re.IGNORECASE,
    )

    for match in measurement_pattern.finditer(
        measurement_text
    ):
        timestamp = pd.Timestamp(
            datetime.strptime(
                (
                    f"{match.group('date')} "
                    f"{match.group('time')}"
                ),
                "%Y/%m/%d %H:%M:%S",
            )
        )

        measurement_rows.append(
            {
                "timestamp": timestamp,
                "height_rsun": float(
                    match.group("height")
                ),
                "angle_deg": float(
                    match.group("angle")
                ),
                "telescope": (
                    match.group(
                        "telescope"
                    ).upper()
                ),
                "feature_code": int(
                    match.group(
                        "feature_code"
                    )
                ),
                "col": float(
                    match.group("col")
                ),
                "row": float(
                    match.group("row")
                ),
            }
        )

    measurements = pd.DataFrame(
        measurement_rows,
        columns=[
            "timestamp",
            "height_rsun",
            "angle_deg",
            "telescope",
            "feature_code",
            "col",
            "row",
        ],
    )

    # Preserve raw source ordering.
    if not measurements.empty:
        measurements = measurements.reset_index(
            drop=True
        )

    # Fail loudly if a measurement section exists but parsing fails.
    if (
        header_match is not None
        and measurement_text
        and measurements.empty
    ):
        raise ValueError(
            "CDAW .yht measurement header was found, "
            "but no valid measurements could be parsed."
        )

    return CdawYhtRecord(
        source_url=source_url,
        metadata=metadata,
        measurements=measurements,
    )


def download_yht(
    event: CdawCatalogEvent,
    *,
    cache_dir: str | Path,
    session: requests.Session | None = None,
    delay_seconds: float = 0.25,
    timeout: float = 30.0,
) -> CdawYhtRecord:
    """Download/cache one CME's .yht record."""

    if event.yht_url is None:
        raise ValueError(
            "CDAW event has no .yht URL."
        )

    filename = event.yht_url.rsplit(
        "/",
        1,
    )[-1]

    cache_path = (
        Path(cache_dir)
        / "yht"
        / f"{event.first_appearance.year:04d}"
        / f"{event.first_appearance.month:02d}"
        / filename
    )

    text = fetch_cached_text(
        event.yht_url,
        cache_path,
        session=session,
        delay_seconds=delay_seconds,
        timeout=timeout,
    )

    return parse_yht(
        text,
        source_url=event.yht_url,
    )


# ---------------------------------------------------------------------
# Record-level audit
# ---------------------------------------------------------------------


def audit_yht_record(
    event: CdawCatalogEvent,
    record: CdawYhtRecord,
) -> dict:
    """Return audit metrics for one CME/.yht pair."""

    df = record.measurements

    n = len(df)

    duplicate_times = (
        int(
            df["timestamp"]
            .duplicated()
            .sum()
        )
        if n
        else 0
    )

    timestamps_monotonic = (
        bool(
            df["timestamp"]
            .is_monotonic_increasing
        )
        if n
        else True
    )

    invalid_height_count = (
        int(
            (
                ~pd.to_numeric(
                    df["height_rsun"],
                    errors="coerce",
                ).gt(0)
            ).sum()
        )
        if n
        else 0
    )

    telescopes = (
        set(
            df["telescope"]
            .dropna()
            .astype(str)
        )
        if n
        else set()
    )

    c2_only = telescopes == {"C2"}
    c3_only = telescopes == {"C3"}

    c2_c3 = (
        "C2" in telescopes
        and "C3" in telescopes
    )

    # Preserve raw ordering for integrity checks above, but calculate
    # latency statistics from chronological measurement order.
    chronological = (
        df.sort_values(
            "timestamp",
            kind="stable",
        )
        .reset_index(
            drop=True
        )
        if n
        else df.copy()
    )

    first_time = (
        chronological.iloc[0][
            "timestamp"
        ]
        if n >= 1
        else pd.NaT
    )

    second_time = (
        chronological.iloc[1][
            "timestamp"
        ]
        if n >= 2
        else pd.NaT
    )

    third_time = (
        chronological.iloc[2][
            "timestamp"
        ]
        if n >= 3
        else pd.NaT
    )

    first_to_second_hours = (
        (
            second_time
            - first_time
        ).total_seconds()
        / 3600.0
        if n >= 2
        else float("nan")
    )

    first_to_third_hours = (
        (
            third_time
            - first_time
        ).total_seconds()
        / 3600.0
        if n >= 3
        else float("nan")
    )

    return {
        "first_appearance": (
            event.first_appearance
        ),
        "yht_url": (
            event.yht_url
        ),
        "retrospective_insert": (
            event.retrospective_insert
        ),
        "insertion_date": (
            event.insertion_date
        ),
        "primary_candidate": (
            event.primary_candidate
        ),
        "remarks": (
            event.remarks
        ),
        "n_measurements": (
            n
        ),
        "n_ge_2": (
            n >= 2
        ),
        "n_ge_3": (
            n >= 3
        ),
        "n_ge_4": (
            n >= 4
        ),
        "n_ge_5": (
            n >= 5
        ),
        "duplicate_measurement_times": (
            duplicate_times
        ),
        "timestamps_monotonic": (
            timestamps_monotonic
        ),
        "invalid_height_count": (
            invalid_height_count
        ),
        "c2_only": (
            c2_only
        ),
        "c3_only": (
            c3_only
        ),
        "c2_c3": (
            c2_c3
        ),
        "first_measurement_time": (
            first_time
        ),
        "second_measurement_time": (
            second_time
        ),
        "third_measurement_time": (
            third_time
        ),
        "first_to_second_hours": (
            first_to_second_hours
        ),
        "first_to_third_hours": (
            first_to_third_hours
        ),

        # Retrospective CDAW metadata.
        # Audit/validation only.
        "catalog_speed": (
            record.metadata.get(
                "SPEED"
            )
        ),
        "catalog_accel": (
            record.metadata.get(
                "ACCEL"
            )
        ),
        "catalog_width": (
            record.metadata.get(
                "WIDTH"
            )
        ),
        "catalog_quality_index": (
            record.metadata.get(
                "QUALITY_INDEX"
            )
        ),
    }


# ---------------------------------------------------------------------
# Aggregate audit
# ---------------------------------------------------------------------


def summarize_audit(
    audit_df: pd.DataFrame,
) -> dict:
    """Generate aggregate CDAW .yht audit statistics."""

    if audit_df.empty:
        return {
            "n_records": 0,
        }

    valid_third = (
        audit_df[
            "first_to_third_hours"
        ]
        .dropna()
    )

    summary = {
        "n_records": (
            len(audit_df)
        ),
        "n_retrospective_insert": int(
            audit_df[
                "retrospective_insert"
            ].sum()
        ),
        "n_primary_candidates": int(
            audit_df[
                "primary_candidate"
            ].sum()
        ),
        "fraction_primary_candidates": float(
            audit_df[
                "primary_candidate"
            ].mean()
        ),
        "n_ge_2": int(
            audit_df[
                "n_ge_2"
            ].sum()
        ),
        "n_ge_3": int(
            audit_df[
                "n_ge_3"
            ].sum()
        ),
        "n_ge_4": int(
            audit_df[
                "n_ge_4"
            ].sum()
        ),
        "n_ge_5": int(
            audit_df[
                "n_ge_5"
            ].sum()
        ),
        "fraction_ge_3": float(
            audit_df[
                "n_ge_3"
            ].mean()
        ),
        "n_c2_only": int(
            audit_df[
                "c2_only"
            ].sum()
        ),
        "n_c3_only": int(
            audit_df[
                "c3_only"
            ].sum()
        ),
        "n_c2_c3": int(
            audit_df[
                "c2_c3"
            ].sum()
        ),
        "n_duplicate_measurement_times": int(
            (
                audit_df[
                    "duplicate_measurement_times"
                ] > 0
            ).sum()
        ),
        "n_nonmonotonic": int(
            (
                audit_df[
                    "timestamps_monotonic"
                ] == False
            ).sum()
        ),
        "n_invalid_heights": int(
            audit_df[
                "invalid_height_count"
            ].sum()
        ),
    }

    if not valid_third.empty:
        summary.update(
            {
                "third_point_median_hours": float(
                    valid_third.median()
                ),
                "third_point_p75_hours": float(
                    valid_third.quantile(
                        0.75
                    )
                ),
                "third_point_p90_hours": float(
                    valid_third.quantile(
                        0.90
                    )
                ),
                "third_point_p95_hours": float(
                    valid_third.quantile(
                        0.95
                    )
                ),
                "fraction_third_point_within_1h": float(
                    (
                        valid_third <= 1
                    ).mean()
                ),
                "fraction_third_point_within_3h": float(
                    (
                        valid_third <= 3
                    ).mean()
                ),
                "fraction_third_point_within_6h": float(
                    (
                        valid_third <= 6
                    ).mean()
                ),
                "fraction_third_point_within_12h": float(
                    (
                        valid_third <= 12
                    ).mean()
                ),
            }
        )

    return summary