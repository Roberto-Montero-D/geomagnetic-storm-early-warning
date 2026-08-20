"""OMNIWeb raw-data loader.

This module loads a headerless OMNIWeb ``.lst`` file using the schema
declared in its companion ``.fmt`` file.

Responsibilities
----------------
1. Parse the OMNIWeb format file.
2. Validate that the format contains the expected source fields.
3. Load the headerless numeric data file.
4. Construct an hourly DatetimeIndex from YEAR + DOY + Hour.
5. Validate timestamp uniqueness, monotonicity, and hourly continuity.
6. Rename source columns to stable internal names.

This module intentionally does NOT:

- replace OMNI fill values;
- apply the project's causal cutoff;
- construct features;
- construct targets;
- normalize Kp into causal predictor form;
- exclude retrospective indices such as Dst or AE.

Those operations belong to downstream modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


@dataclass(frozen=True)
class OmniField:
    """One field declared in an OMNIWeb .fmt file."""

    position: int
    source_name: str
    fortran_format: str


# Stable internal names for the OMNI subset used by this project.
#
# These mappings correspond to the expanded 17-column subset:
#
#  1 YEAR
#  2 DOY
#  3 Hour
#  4 Scalar B, nT
#  5 BX, nT (GSE, GSM)
#  6 BY, nT (GSM)
#  7 BZ, nT (GSM)
#  8 SW Plasma Temperature, K
#  9 SW Proton Density, N/cm^3
# 10 SW Plasma Speed, km/s
# 11 Flow pressure
# 12 E elecrtric field
# 13 Plasma Beta
# 14 Alfen mach number
# 15 Kp index
# 16 Dst-index, nT
# 17 AE-index, nT

EXPECTED_INTERNAL_NAMES = (
    "year",
    "doy",
    "hour",
    "bt",
    "bx_gse_gsm",
    "by_gsm",
    "bz_gsm",
    "temperature",
    "density",
    "speed",
    "flow_pressure",
    "electric_field",
    "plasma_beta",
    "alfven_mach",
    "kp_raw",
    "dst",
    "ae",
)


EXPECTED_SOURCE_FIELDS = (
    "YEAR",
    "DOY",
    "Hour",
    "Scalar B, nT",
    "BX, nT (GSE, GSM)",
    "BY, nT (GSM)",
    "BZ, nT (GSM)",
    "SW Plasma Temperature, K",
    "SW Proton Density, N/cm^3",
    "SW Plasma Speed, km/s",
    "Flow pressure",
    "E elecrtric field",
    "Plasma Beta",
    "Alfen mach number",
    "Kp index",
    "Dst-index, nT",
    "AE-index, nT",
)


_FMT_LINE_PATTERN = re.compile(
    r"^\s*(\d+)\s+(.+?)\s+([AIFED]\d+(?:\.\d+)?)\s*$"
)


def parse_omni_format(path: str | Path) -> list[OmniField]:
    """Parse an OMNIWeb ``.fmt`` schema file."""

    path = Path(path)

    fields: list[OmniField] = []

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for raw_line in handle:
            line = raw_line.rstrip()

            match = _FMT_LINE_PATTERN.match(line)

            if match is None:
                continue

            position = int(match.group(1))
            source_name = match.group(2).strip()
            fortran_format = match.group(3).strip()

            fields.append(
                OmniField(
                    position=position,
                    source_name=source_name,
                    fortran_format=fortran_format,
                )
            )

    if not fields:
        raise ValueError(
            f"No OMNI field definitions were found in {path}."
        )

    expected_positions = list(
        range(1, len(fields) + 1)
    )
    actual_positions = [
        field.position
        for field in fields
    ]

    if actual_positions != expected_positions:
        raise ValueError(
            "OMNI format positions must be sequential starting at 1. "
            f"Got: {actual_positions}"
        )

    return fields


def validate_project_omni_schema(
    fields: list[OmniField],
) -> None:
    """Validate that the format file matches the project OMNI subset."""

    source_names = tuple(
        field.source_name
        for field in fields
    )

    if len(source_names) != len(EXPECTED_SOURCE_FIELDS):
        raise ValueError(
            "Unexpected OMNI column count. "
            f"Expected {len(EXPECTED_SOURCE_FIELDS)}, "
            f"got {len(source_names)}."
        )

    if source_names != EXPECTED_SOURCE_FIELDS:
        differences = []

        for position, (actual, expected) in enumerate(
            zip(
                source_names,
                EXPECTED_SOURCE_FIELDS,
                strict=True,
            ),
            start=1,
        ):
            if actual != expected:
                differences.append(
                    f"{position}: expected {expected!r}, got {actual!r}"
                )

        raise ValueError(
            "OMNI format does not match the expected project schema. "
            + "; ".join(differences)
        )


def _construct_timestamp(
    frame: pd.DataFrame,
) -> pd.DatetimeIndex:
    """Construct timestamps from YEAR + DOY + Hour."""

    required = {
        "year",
        "doy",
        "hour",
    }

    missing = required - set(frame.columns)

    if missing:
        raise KeyError(
            f"Missing timestamp column(s): {sorted(missing)}"
        )

    year = pd.to_numeric(
        frame["year"],
        errors="raise",
    ).astype(int)

    doy = pd.to_numeric(
        frame["doy"],
        errors="raise",
    ).astype(int)

    hour = pd.to_numeric(
        frame["hour"],
        errors="raise",
    ).astype(int)

    if ((hour < 0) | (hour > 23)).any():
        invalid = sorted(
            hour[
                (hour < 0)
                | (hour > 23)
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Invalid OMNI hour value(s): {invalid}"
        )

    day = pd.to_datetime(
        year.astype(str)
        + doy.astype(str).str.zfill(3),
        format="%Y%j",
        errors="raise",
    )

    timestamp = (
        day
        + pd.to_timedelta(
            hour,
            unit="h",
        )
    )

    return pd.DatetimeIndex(
        timestamp,
        name="timestamp",
    )


def _validate_timestamp_index(
    index: pd.DatetimeIndex,
) -> None:
    """Validate chronological integrity of the OMNI timeline."""

    if index.has_duplicates:
        duplicated = index[
            index.duplicated()
        ].unique()

        raise ValueError(
            "OMNI data contain duplicate timestamps: "
            f"{duplicated.tolist()}"
        )

    if not index.is_monotonic_increasing:
        raise ValueError(
            "OMNI timestamps are not monotonically increasing."
        )

    if len(index) <= 1:
        return

    expected = pd.date_range(
        start=index[0],
        end=index[-1],
        freq="h",
    )

    if not index.equals(expected):
        missing = expected.difference(index)

        raise ValueError(
            "OMNI data are not a continuous hourly time series. "
            f"Missing timestamp(s): {missing.tolist()}"
        )


def load_omni(
    fmt_path: str | Path,
    lst_path: str | Path,
    *,
    validate_schema: bool = True,
    validate_continuity: bool = True,
) -> pd.DataFrame:
    """Load the project's raw OMNIWeb subset.

    Parameters
    ----------
    fmt_path
        Companion OMNIWeb ``.fmt`` file.
    lst_path
        Headerless OMNIWeb numeric ``.lst`` file.
    validate_schema
        Require the exact 17-column schema used by this project.
    validate_continuity
        Require a continuous hourly timeline.

    Returns
    -------
    pandas.DataFrame
        Raw OMNI values with a DatetimeIndex named ``timestamp``.

        Columns:

        bt
        bx_gse_gsm
        by_gsm
        bz_gsm
        temperature
        density
        speed
        flow_pressure
        electric_field
        plasma_beta
        alfven_mach
        kp_raw
        dst
        ae

    Notes
    -----
    ``YEAR``, ``DOY`` and ``Hour`` are consumed to build the index and
    are not retained as ordinary data columns.

    Raw OMNI fill/sentinel values are deliberately preserved here.
    """

    fields = parse_omni_format(fmt_path)

    if validate_schema:
        validate_project_omni_schema(fields)

    source_column_count = len(fields)

    if source_column_count != len(EXPECTED_INTERNAL_NAMES):
        raise ValueError(
            "Cannot assign project internal names because the OMNI "
            f"schema has {source_column_count} fields but "
            f"{len(EXPECTED_INTERNAL_NAMES)} are expected."
        )

    # Load raw source columns BEFORE assigning names.
    #
    # This is important: assigning ``names=...`` during read_csv can
    # mask malformed source rows by forcing pandas into the expected
    # schema shape.
    frame = pd.read_csv(
        lst_path,
        sep=r"\s+",
        header=None,
        engine="python",
    )

    if frame.shape[1] != source_column_count:
        raise ValueError(
            "OMNI data column count does not match the .fmt schema. "
            f"Schema fields: {source_column_count}; "
            f"loaded columns: {frame.shape[1]}."
        )

    frame.columns = EXPECTED_INTERNAL_NAMES

    timestamp = _construct_timestamp(frame)

    if validate_continuity:
        _validate_timestamp_index(timestamp)

    frame.index = timestamp

    frame = frame.drop(
        columns=[
            "year",
            "doy",
            "hour",
        ]
    )

    return frame