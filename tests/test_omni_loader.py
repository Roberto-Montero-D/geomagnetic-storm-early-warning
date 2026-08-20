from pathlib import Path

import pandas as pd
import pytest

from src.data.omni import (
    EXPECTED_INTERNAL_NAMES,
    EXPECTED_SOURCE_FIELDS,
    load_omni,
    parse_omni_format,
    validate_project_omni_schema,
)


FMT_TEXT = """\
  FORMAT OF THE SUBSETTED FILE

    ITEMS                      FORMAT

 1 YEAR                          I4
 2 DOY                           I4
 3 Hour                          I3
 4 Scalar B, nT                  F6.1
 5 BX, nT (GSE, GSM)             F6.1
 6 BY, nT (GSM)                  F6.1
 7 BZ, nT (GSM)                  F6.1
 8 SW Plasma Temperature, K      F9.0
 9 SW Proton Density, N/cm^3     F6.1
10 SW Plasma Speed, km/s         F6.0
11 Flow pressure                 F6.2
12 E elecrtric field             F7.2
13 Plasma Beta                   F7.2
14 Alfen mach number             F6.1
15 Kp index                      I3
16 Dst-index, nT                 I6
17 AE-index, nT                  I5

This file was produced by OMNIweb service
"""


LST_TEXT = """\
1996   1  0   5.7  -0.4  -3.6   4.1   92006.   7.2  423.  2.39  -1.73   2.03  10.0 10     2   15
1996   1  1   4.8   2.4  -3.1   1.1   86550.   8.0  397.  2.34  -0.44   3.10  11.7 10     3   24
1996   1  2   4.8   3.5  -2.1   2.1   80149.   7.8  390.  2.18  -0.82   2.94  11.3 10    -2   32
"""


def write_file(
    path: Path,
    content: str,
) -> Path:
    path.write_text(
        content,
        encoding="utf-8",
    )

    return path


def test_parse_omni_format(tmp_path):
    fmt_path = write_file(
        tmp_path / "omni.fmt",
        FMT_TEXT,
    )

    fields = parse_omni_format(fmt_path)

    assert len(fields) == 17

    assert fields[0].position == 1
    assert fields[0].source_name == "YEAR"
    assert fields[0].fortran_format == "I4"

    assert fields[-1].position == 17
    assert fields[-1].source_name == "AE-index, nT"
    assert fields[-1].fortran_format == "I5"


def test_project_schema_matches_expected_format(tmp_path):
    fmt_path = write_file(
        tmp_path / "omni.fmt",
        FMT_TEXT,
    )

    fields = parse_omni_format(fmt_path)

    validate_project_omni_schema(fields)

    assert tuple(
        field.source_name
        for field in fields
    ) == EXPECTED_SOURCE_FIELDS


def test_load_omni_sample(tmp_path):
    fmt_path = write_file(
        tmp_path / "omni.fmt",
        FMT_TEXT,
    )

    lst_path = write_file(
        tmp_path / "omni.lst",
        LST_TEXT,
    )

    df = load_omni(
        fmt_path,
        lst_path,
    )

    assert len(df) == 3

    assert df.index.equals(
        pd.date_range(
            "1996-01-01 00:00",
            periods=3,
            freq="h",
            name="timestamp",
        )
    )

    assert list(df.columns) == list(
        EXPECTED_INTERNAL_NAMES[3:]
    )

    assert df.loc[
        pd.Timestamp("1996-01-01 00:00"),
        "bt",
    ] == 5.7

    assert df.loc[
        pd.Timestamp("1996-01-01 00:00"),
        "bz_gsm",
    ] == 4.1

    assert df.loc[
        pd.Timestamp("1996-01-01 00:00"),
        "speed",
    ] == 423.0

    assert df.loc[
        pd.Timestamp("1996-01-01 00:00"),
        "kp_raw",
    ] == 10

    assert df.loc[
        pd.Timestamp("1996-01-01 00:00"),
        "dst",
    ] == 2

    assert df.loc[
        pd.Timestamp("1996-01-01 00:00"),
        "ae",
    ] == 15


def test_missing_hour_raises(tmp_path):
    fmt_path = write_file(
        tmp_path / "omni.fmt",
        FMT_TEXT,
    )

    lst = "\n".join(
        [
            LST_TEXT.splitlines()[0],
            LST_TEXT.splitlines()[2],
        ]
    )

    lst_path = write_file(
        tmp_path / "omni.lst",
        lst,
    )

    with pytest.raises(
        ValueError,
        match="continuous hourly time series",
    ):
        load_omni(
            fmt_path,
            lst_path,
        )


def test_duplicate_timestamp_raises(tmp_path):
    fmt_path = write_file(
        tmp_path / "omni.fmt",
        FMT_TEXT,
    )

    first_row = LST_TEXT.splitlines()[0]

    lst_path = write_file(
        tmp_path / "omni.lst",
        first_row + "\n" + first_row + "\n",
    )

    with pytest.raises(
        ValueError,
        match="duplicate timestamps",
    ):
        load_omni(
            fmt_path,
            lst_path,
        )


def test_schema_mismatch_raises(tmp_path):
    bad_fmt = FMT_TEXT.replace(
        "AE-index, nT",
        "Some other field",
    )

    fmt_path = write_file(
        tmp_path / "omni.fmt",
        bad_fmt,
    )

    fields = parse_omni_format(fmt_path)

    with pytest.raises(
        ValueError,
        match="does not match the expected project schema",
    ):
        validate_project_omni_schema(fields)


def test_missing_data_column_raises(tmp_path):
    fmt_path = write_file(
        tmp_path / "omni.fmt",
        FMT_TEXT,
    )

    malformed_rows = []

    for row in LST_TEXT.splitlines():
        parts = row.split()

        malformed_rows.append(
            " ".join(parts[:-1])
        )

    lst_path = write_file(
        tmp_path / "omni.lst",
        "\n".join(malformed_rows),
    )

    with pytest.raises(
        ValueError,
        match="column count does not match",
    ):
        load_omni(
            fmt_path,
            lst_path,
        )


def test_extra_data_column_raises(tmp_path):
    fmt_path = write_file(
        tmp_path / "omni.fmt",
        FMT_TEXT,
    )

    malformed_rows = [
        row + " 999"
        for row in LST_TEXT.splitlines()
    ]

    lst_path = write_file(
        tmp_path / "omni.lst",
        "\n".join(malformed_rows),
    )

    with pytest.raises(
        ValueError,
        match="column count does not match",
    ):
        load_omni(
            fmt_path,
            lst_path,
        )