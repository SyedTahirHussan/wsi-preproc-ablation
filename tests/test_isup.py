"""The grading table is domain knowledge, so it is pinned rather than derived."""

from __future__ import annotations

import pytest

from wsi_ablation.isup import (
    crosses_treatment_threshold,
    is_treatment_indicated,
    isup_grade_group,
    label_grid,
)


@pytest.mark.parametrize(
    ("primary", "secondary", "expected"),
    [
        (0, 0, 0),
        (3, 3, 1),
        (3, 4, 2),
        (4, 3, 3),
        (4, 4, 4),
        (3, 5, 4),
        (5, 3, 4),
        (4, 5, 5),
        (5, 4, 5),
        (5, 5, 5),
    ],
)
def test_consensus_table(primary: int, secondary: int, expected: int) -> None:
    assert isup_grade_group(primary, secondary) == expected


def test_three_plus_four_and_four_plus_three_are_not_the_same_grade() -> None:
    """Gleason 7 splits into two grade groups, and the order of the patterns is why."""
    assert isup_grade_group(3, 4) != isup_grade_group(4, 3)


def test_patterns_one_and_two_are_rejected() -> None:
    with pytest.raises(ValueError, match="no longer assigned"):
        isup_grade_group(2, 3)


def test_treatment_line_sits_between_grade_groups_one_and_two() -> None:
    assert not is_treatment_indicated(1)
    assert is_treatment_indicated(2)


def test_crossings_count_the_errors_that_change_management() -> None:
    assert crosses_treatment_threshold(1, 2)
    assert crosses_treatment_threshold(0, 1)
    assert not crosses_treatment_threshold(2, 5)
    assert not crosses_treatment_threshold(4, 3)


def test_label_grid_covers_benign_through_grade_group_five() -> None:
    assert label_grid() == [0, 1, 2, 3, 4, 5]
