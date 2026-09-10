"""Create a class-balanced version of the Student Performance Factors dataset.

The tutor asked the group to work on a balanced dataset. The raw Kaggle file is
skewed towards the `Medium` class (53.4% of all records), so a majority-class
baseline already scores over 50% and the smaller `Low` and `High` classes are
under-represented during training.

This script applies random undersampling: every performance class is reduced to
the size of the smallest class (`Low`, 1,452 records), giving an exactly equal
1/3 - 1/3 - 1/3 target distribution. The class cut points are unchanged, so the
definition of Low / Medium / High used by all three notebooks stays the same.

The random draw is seeded, so every group member who runs this script obtains
the identical balanced file. Run it once, commit the result, and all three
notebooks then read the same data.

    python balance_dataset.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_FOLDER = Path(__file__).resolve().parent
SOURCE_FILE = PROJECT_FOLDER / "StudentPerformanceFactors_original.csv"
OUTPUT_FILE = PROJECT_FOLDER / "StudentPerformanceFactors.csv"

RANDOM_STATE = 42
SCORE_CUT_POINTS = [-np.inf, 64, 69, np.inf]
CLASS_LABELS = ["Low", "Medium", "High"]


def add_performance_class(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the same Low / Medium / High target used by all three notebooks."""
    frame = frame.copy()
    frame["Performance"] = pd.cut(
        frame["Exam_Score"],
        bins=SCORE_CUT_POINTS,
        labels=CLASS_LABELS,
        ordered=True,
    )
    return frame


def undersample_to_smallest_class(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep an equal number of records per class, drawn with a fixed seed."""
    target_size = int(frame["Performance"].value_counts().min())
    kept_rows = (
        frame.groupby("Performance", observed=True)["Exam_Score"]
        .sample(n=target_size, random_state=RANDOM_STATE)
        .index.sort_values()
    )
    return frame.loc[kept_rows], target_size


def describe(frame: pd.DataFrame, title: str) -> None:
    counts = frame["Performance"].value_counts().reindex(CLASS_LABELS)
    share = counts / len(frame) * 100
    print(f"\n{title}  ({len(frame):,} records)")
    for label in CLASS_LABELS:
        print(f"  {label:<7}{counts[label]:>6,}   {share[label]:5.2f}%")


def main() -> None:
    if not SOURCE_FILE.exists():
        raise SystemExit(
            f"Cannot find {SOURCE_FILE.name}. It should hold the untouched Kaggle "
            "download; the balanced file is written to StudentPerformanceFactors.csv."
        )

    raw = pd.read_csv(SOURCE_FILE)
    raw = add_performance_class(raw)
    describe(raw, "Original dataset")

    balanced, target_size = undersample_to_smallest_class(raw)
    describe(balanced, "Balanced dataset")

    # `Performance` is derived from `Exam_Score`, so it is not written back out:
    # each notebook recreates it from the same cut points.
    balanced.drop(columns=["Performance"]).to_csv(OUTPUT_FILE, index=False)

    removed = len(raw) - len(balanced)
    print(f"\nRemoved {removed:,} records ({removed / len(raw) * 100:.1f}% of the original).")
    print(f"Each class now holds {target_size:,} records.")
    print(f"Written to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
