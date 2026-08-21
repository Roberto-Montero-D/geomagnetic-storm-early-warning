import pandas as pd
import pytest

from src.data.cme_cdaw import (
    CdawCatalogEvent,
    audit_yht_record,
    parse_monthly_catalog,
    parse_yht,
    summarize_audit,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


MONTH_HTML = """
<html>
<body>
<table>

<tr>
<td><a href="movie.html">1996/01/11</a></td>
<td>
<a href="yht/19960111.001436.w018n.v0499.p272s.yht">
00:14:36
</a>
</td>
<td>267</td>
<td>18</td>
<td>499</td>
<td>426</td>
<td>0</td>
<td>-64.3*</td>
<td>----</td>
<td>----</td>
<td>272</td>
<td>Only C3</td>
</tr>

<tr>
<td>1996/01/30</td>
<td>
<a href="yht/inserted.yht">
05:56:07
</a>
</td>
<td>79</td>
<td>24</td>
<td>30</td>
<td>19</td>
<td>0</td>
<td>-0.9*</td>
<td>1.0e+14</td>
<td>4.7e+26</td>
<td>88</td>
<td>
Newly inserted on 2024/05/01.;
Very Poor Event; Only C2
</td>
</tr>

</table>
</body>
</html>
"""


YHT_TEXT = """\
#VERSION=3
#DATE-OBS: 1996/01/11
#TIME-OBS: 00:14:36
#DETECTOR: C3
#SPEED: 499
#ACCEL: -64.3
#WIDTH: 18
#QUALITY_INDEX: 2 (Fair)
#REMARK: Only C3
# HEIGHT DATE TIME ANGLE TEL FC COL ROW
5.06 1996/01/11 00:25:22 270.3 C3 1 303.0 266.0
5.40 1996/01/11 00:31:15 271.5 C3 1 306.0 267.0
5.87 1996/01/11 00:43:37 272.5 C3 1 310.0 268.0
6.22 1996/01/11 00:50:14 273.4 C3 1 313.0 269.0
6.44 1996/01/11 00:56:07 272.2 C3 1 315.0 268.0
6.67 1996/01/11 01:02:40 272.1 C3 1 317.0 268.0
"""


# ---------------------------------------------------------------------
# Monthly catalog parsing
# ---------------------------------------------------------------------


def test_parse_monthly_catalog():
    events = parse_monthly_catalog(
        MONTH_HTML,
        page_url=(
            "https://example.test/"
            "1996_01/univ1996_01.html"
        ),
    )

    assert len(events) == 2

    event = events[0]

    assert event.first_appearance == pd.Timestamp(
        "1996-01-11 00:14:36"
    )

    assert event.central_pa_raw == "267"
    assert event.width_raw == "18"
    assert event.linear_speed_raw == "499"

    assert event.yht_url.endswith(
        "yht/"
        "19960111.001436."
        "w018n.v0499.p272s.yht"
    )

    assert not event.retrospective_insert
    assert event.primary_candidate


def test_retrospective_insert_detection():
    events = parse_monthly_catalog(
        MONTH_HTML,
        page_url=(
            "https://example.test/"
            "1996_01/univ1996_01.html"
        ),
    )

    event = events[1]

    assert event.retrospective_insert
    assert not event.primary_candidate

    assert event.insertion_date == pd.Timestamp(
        "2024-05-01"
    )


# ---------------------------------------------------------------------
# YHT parsing
# ---------------------------------------------------------------------


def test_parse_yht_measurements():
    record = parse_yht(
        YHT_TEXT,
    )

    assert record.n_measurements == 6

    assert (
        record.first_measurement_time
        == pd.Timestamp(
            "1996-01-11 00:25:22"
        )
    )

    assert (
        record.last_measurement_time
        == pd.Timestamp(
            "1996-01-11 01:02:40"
        )
    )

    assert record.telescope_set == {
        "C3"
    }

    first = record.measurements.iloc[0]

    assert first["height_rsun"] == 5.06
    assert first["angle_deg"] == 270.3
    assert first["telescope"] == "C3"
    assert first["feature_code"] == 1


def test_retrospective_metadata_is_separate():
    record = parse_yht(
        YHT_TEXT,
    )

    assert (
        record.metadata["SPEED"]
        == "499"
    )

    assert (
        record.metadata["ACCEL"]
        == "-64.3"
    )

    assert (
        record.metadata["WIDTH"]
        == "18"
    )

    assert (
        "catalog_speed"
        not in record.measurements.columns
    )


def test_parse_flattened_realistic_yht():
    text = (
        "#VERSION=3 "
        "#DATE-OBS: 1996/01/11 "
        "#TIME-OBS: 00:14:36 "
        "#DETECTOR: C3 "
        "#SPEED: 499 "
        "#ACCEL: -64.3 "
        "#WIDTH: 18 "
        "#QUALITY_INDEX: 2 (Fair) "
        "#REMARK: Only C3 "
        "#COMMENT: "
        "# HEIGHT DATE TIME ANGLE TEL FC COL ROW "
        "5.06 1996/01/11 00:25:22 270.3 C3 1 303.0 266.0 "
        "5.40 1996/01/11 00:31:15 271.5 C3 1 306.0 267.0 "
        "5.87 1996/01/11 00:43:37 272.5 C3 1 310.0 268.0 "
        "6.22 1996/01/11 00:50:14 273.4 C3 1 313.0 269.0 "
        "6.44 1996/01/11 00:56:07 272.2 C3 1 315.0 268.0 "
        "6.67 1996/01/11 01:02:40 272.1 C3 1 317.0 268.0"
    )

    record = parse_yht(
        text
    )

    assert record.n_measurements == 6

    assert (
        record.first_measurement_time
        == pd.Timestamp(
            "1996-01-11 00:25:22"
        )
    )

    assert (
        record.last_measurement_time
        == pd.Timestamp(
            "1996-01-11 01:02:40"
        )
    )

    assert record.telescope_set == {
        "C3"
    }

    assert (
        record.metadata["SPEED"]
        == "499"
    )

    assert (
        record.metadata["ACCEL"]
        == "-64.3"
    )

    assert (
        record.metadata["WIDTH"]
        == "18"
    )

    assert (
        record.metadata["VERSION"]
        == "3"
    )


def test_flattened_and_multiline_yht_are_equivalent():
    multiline = """\
#VERSION=3
#SPEED: 499
#ACCEL: -64.3
# HEIGHT DATE TIME ANGLE TEL FC COL ROW
5.06 1996/01/11 00:25:22 270.3 C3 1 303.0 266.0
5.40 1996/01/11 00:31:15 271.5 C3 1 306.0 267.0
5.87 1996/01/11 00:43:37 272.5 C3 1 310.0 268.0
"""

    flattened = " ".join(
        multiline.splitlines()
    )

    record_multiline = parse_yht(
        multiline
    )

    record_flattened = parse_yht(
        flattened
    )

    pd.testing.assert_frame_equal(
        record_multiline.measurements,
        record_flattened.measurements,
    )

    assert (
        record_multiline.metadata
        == record_flattened.metadata
    )


def test_yht_header_with_unparseable_measurements_raises():
    text = """\
#VERSION=3
#SPEED: 499
# HEIGHT DATE TIME ANGLE TEL FC COL ROW
THIS IS NOT A VALID MEASUREMENT
"""

    with pytest.raises(
        ValueError,
        match="measurement header",
    ):
        parse_yht(
            text
        )


# ---------------------------------------------------------------------
# Source-order preservation
# ---------------------------------------------------------------------


def test_yht_parser_preserves_source_order():
    text = """\
# HEIGHT DATE TIME ANGLE TEL FC COL ROW
6.0 1996/01/11 00:30:00 270 C3 1 1 1
5.0 1996/01/11 00:20:00 270 C3 1 1 1
"""

    record = parse_yht(
        text
    )

    assert (
        record.measurements.iloc[0][
            "timestamp"
        ]
        == pd.Timestamp(
            "1996-01-11 00:30:00"
        )
    )

    assert (
        record.measurements.iloc[1][
            "timestamp"
        ]
        == pd.Timestamp(
            "1996-01-11 00:20:00"
        )
    )


def test_audit_detects_nonmonotonic_source():
    text = """\
# HEIGHT DATE TIME ANGLE TEL FC COL ROW
6.0 1996/01/11 00:30:00 270 C3 1 1 1
5.0 1996/01/11 00:20:00 270 C3 1 1 1
"""

    event = CdawCatalogEvent(
        first_appearance=pd.Timestamp(
            "1996-01-11 00:14:36"
        ),
        central_pa_raw="267",
        width_raw="18",
        linear_speed_raw="499",
        remarks="",
        yht_url=(
            "https://example.test/a.yht"
        ),
    )

    record = parse_yht(
        text
    )

    result = audit_yht_record(
        event,
        record,
    )

    assert not result[
        "timestamps_monotonic"
    ]


def test_audit_latency_uses_chronological_order():
    text = """\
# HEIGHT DATE TIME ANGLE TEL FC COL ROW
6.0 1996/01/11 00:30:00 270 C3 1 1 1
5.0 1996/01/11 00:20:00 270 C3 1 1 1
7.0 1996/01/11 00:40:00 270 C3 1 1 1
"""

    event = CdawCatalogEvent(
        first_appearance=pd.Timestamp(
            "1996-01-11 00:14:36"
        ),
        central_pa_raw="267",
        width_raw="18",
        linear_speed_raw="499",
        remarks="",
        yht_url=(
            "https://example.test/a.yht"
        ),
    )

    record = parse_yht(
        text
    )

    result = audit_yht_record(
        event,
        record,
    )

    assert (
        result["first_measurement_time"]
        == pd.Timestamp(
            "1996-01-11 00:20:00"
        )
    )

    assert (
        result["third_measurement_time"]
        == pd.Timestamp(
            "1996-01-11 00:40:00"
        )
    )

    assert result[
        "first_to_third_hours"
    ] == pytest.approx(
        20 / 60
    )


# ---------------------------------------------------------------------
# Record-level audit
# ---------------------------------------------------------------------


def test_audit_yht_record():
    event = CdawCatalogEvent(
        first_appearance=pd.Timestamp(
            "1996-01-11 00:14:36"
        ),
        central_pa_raw="267",
        width_raw="18",
        linear_speed_raw="499",
        remarks="Only C3",
        yht_url=(
            "https://example.test/a.yht"
        ),
    )

    record = parse_yht(
        YHT_TEXT,
    )

    result = audit_yht_record(
        event,
        record,
    )

    assert result[
        "n_measurements"
    ] == 6

    assert result[
        "n_ge_3"
    ]

    assert result[
        "c3_only"
    ]

    assert not result[
        "c2_only"
    ]

    assert result[
        "timestamps_monotonic"
    ]

    assert result[
        "duplicate_measurement_times"
    ] == 0

    assert result[
        "invalid_height_count"
    ] == 0

    expected_hours = (
        pd.Timestamp(
            "1996-01-11 00:43:37"
        )
        - pd.Timestamp(
            "1996-01-11 00:25:22"
        )
    ).total_seconds() / 3600

    assert result[
        "first_to_third_hours"
    ] == pytest.approx(
        expected_hours
    )


# ---------------------------------------------------------------------
# Aggregate audit
# ---------------------------------------------------------------------


def test_summarize_audit():
    df = pd.DataFrame(
        {
            "retrospective_insert": [
                False,
                True,
            ],
            "primary_candidate": [
                True,
                False,
            ],
            "n_ge_2": [
                True,
                True,
            ],
            "n_ge_3": [
                True,
                False,
            ],
            "n_ge_4": [
                True,
                False,
            ],
            "n_ge_5": [
                False,
                False,
            ],
            "c2_only": [
                False,
                True,
            ],
            "c3_only": [
                True,
                False,
            ],
            "c2_c3": [
                False,
                False,
            ],
            "duplicate_measurement_times": [
                0,
                0,
            ],
            "timestamps_monotonic": [
                True,
                True,
            ],
            "invalid_height_count": [
                0,
                0,
            ],
            "first_to_third_hours": [
                0.5,
                float("nan"),
            ],
        }
    )

    summary = summarize_audit(
        df
    )

    assert summary[
        "n_records"
    ] == 2

    assert summary[
        "n_retrospective_insert"
    ] == 1

    assert summary[
        "n_primary_candidates"
    ] == 1

    assert summary[
        "fraction_primary_candidates"
    ] == pytest.approx(
        0.5
    )

    assert summary[
        "n_ge_2"
    ] == 2

    assert summary[
        "n_ge_3"
    ] == 1

    assert summary[
        "n_ge_4"
    ] == 1

    assert summary[
        "n_ge_5"
    ] == 0

    assert summary[
        "fraction_ge_3"
    ] == pytest.approx(
        0.5
    )

    assert summary[
        "n_c2_only"
    ] == 1

    assert summary[
        "n_c3_only"
    ] == 1

    assert summary[
        "n_c2_c3"
    ] == 0

    assert summary[
        "n_duplicate_measurement_times"
    ] == 0

    assert summary[
        "n_nonmonotonic"
    ] == 0

    assert summary[
        "n_invalid_heights"
    ] == 0

    assert summary[
        "third_point_median_hours"
    ] == pytest.approx(
        0.5
    )

    assert summary[
        "fraction_third_point_within_1h"
    ] == pytest.approx(
        1.0
    )


def test_summarize_audit_counts_nonmonotonic_correctly():
    """Regression test for the n_nonmonotonic summary bug."""

    df = pd.DataFrame(
        {
            "retrospective_insert": [
                False,
                False,
                False,
            ],
            "primary_candidate": [
                True,
                True,
                True,
            ],
            "n_ge_2": [
                True,
                True,
                True,
            ],
            "n_ge_3": [
                True,
                True,
                True,
            ],
            "n_ge_4": [
                True,
                True,
                True,
            ],
            "n_ge_5": [
                True,
                True,
                True,
            ],
            "c2_only": [
                True,
                True,
                True,
            ],
            "c3_only": [
                False,
                False,
                False,
            ],
            "c2_c3": [
                False,
                False,
                False,
            ],
            "duplicate_measurement_times": [
                0,
                0,
                0,
            ],
            "timestamps_monotonic": [
                True,
                False,
                True,
            ],
            "invalid_height_count": [
                0,
                0,
                0,
            ],
            "first_to_third_hours": [
                0.5,
                0.6,
                0.7,
            ],
        }
    )

    summary = summarize_audit(
        df
    )

    assert summary[
        "n_records"
    ] == 3

    assert summary[
        "n_nonmonotonic"
    ] == 1