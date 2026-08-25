"""ISUP Grade Groups and the clinical threshold that matters.

The mapping is the 2014 ISUP consensus, carried into WHO 2016 and unchanged in
WHO 2022:

    Grade Group 1  Gleason 3+3 = 6
    Grade Group 2  Gleason 3+4 = 7
    Grade Group 3  Gleason 4+3 = 7
    Grade Group 4  Gleason 4+4, 3+5, 5+3 = 8
    Grade Group 5  Gleason 4+5, 5+4, 5+5 = 9-10

Grade Group 0 is not part of the consensus; it is used here for a benign core,
which is the label an end-to-end model has to produce for most biopsy material.

The clinical threshold is Grade Group 1 against Grade Group 2 or above. Below it
active surveillance is the usual offer; at or above it, treatment is discussed.
A grading error that stays on one side of that line costs a patient nothing. An
error that crosses it changes what happens to them, which is why this module
exposes `crosses_treatment_threshold` and the ablation counts those separately
from agreement.
"""

from __future__ import annotations

_CONSENSUS: dict[tuple[int, int], int] = {
    (3, 3): 1,
    (3, 4): 2,
    (4, 3): 3,
    (4, 4): 4,
    (3, 5): 4,
    (5, 3): 4,
    (4, 5): 5,
    (5, 4): 5,
    (5, 5): 5,
}

BENIGN = 0
MAX_GRADE_GROUP = 5
TREATMENT_THRESHOLD = 2


def isup_grade_group(primary: int, secondary: int) -> int:
    """Grade Group for a Gleason pattern pair; 0 for benign (0, 0)."""
    if primary == 0 and secondary == 0:
        return BENIGN
    try:
        return _CONSENSUS[(primary, secondary)]
    except KeyError:
        raise ValueError(
            f"Gleason {primary}+{secondary} is not a recognised pattern pair; "
            "patterns 1 and 2 are no longer assigned on biopsy"
        ) from None


def is_treatment_indicated(grade_group: int) -> bool:
    """True where the grade sits at or above the surveillance/treatment line."""
    return grade_group >= TREATMENT_THRESHOLD


def crosses_treatment_threshold(true_gg: int, pred_gg: int) -> bool:
    """True where the error moves a case across the surveillance/treatment line.

    Benign against Grade Group 1 is also a crossing: it is the difference
    between a patient having cancer on the report and not.
    """
    if (true_gg == BENIGN) != (pred_gg == BENIGN):
        return True
    return is_treatment_indicated(true_gg) != is_treatment_indicated(pred_gg)


def label_grid() -> list[int]:
    """The ordered label set the graders predict over."""
    return list(range(BENIGN, MAX_GRADE_GROUP + 1))
