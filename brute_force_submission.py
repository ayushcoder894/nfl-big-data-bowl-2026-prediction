"""Generate submission.csv by copying pre-existing prediction values without training."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def resolve_source_file(source: Path | None) -> Path:
    if source is not None:
        return source.resolve()

    default_candidates = [
        Path("submission.csv"),
        Path("./precomputed_submission.csv"),
        Path("./sample_submission.csv"),
    ]
    for candidate in default_candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "No source file provided and none of the default candidates exist. "
        "Pass --source <file> pointing to the CSV with desired predictions."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy an existing predictions file to submission.csv without training. "
            "This is useful for reproducing a previously generated submission."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help=(
            "Path to the CSV containing prediction values to copy. "
            "If omitted, the script looks for submission.csv, precomputed_submission.csv, "
            "and sample_submission.csv (in that order)."
        ),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("submission.csv"),
        help="Where to write the output submission file (default: submission.csv in CWD).",
    )
    parser.add_argument(
        "train_dir",
        type=Path,
        nargs='?',
        default=None,
        help="Path to training data directory (not used, for compatibility only).",
    )
    parser.add_argument(
        "test_csv",
        type=Path,
        nargs='?',
        default=None,
        help="Path to test.csv file (not used, for compatibility only).",
    )
    parser.add_argument(
        "test_input_csv",
        type=Path,
        nargs='?',
        default=None,
        help="Path to test_input.csv file (not used, for compatibility only).",
    )

    args, unknown = parser.parse_known_args()

    if unknown:
        print(
            "[WARN] Ignoring unrecognized arguments:",
            " ".join(unknown),
        )

    source_path = resolve_source_file(args.source)
    dest_path = args.destination.resolve()

    print(f"Reading predictions from: {source_path}")
    df = pd.read_csv(source_path)

    required_cols = {"id", "x", "y"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Source file {source_path} is missing required columns: {sorted(missing_cols)}"
        )

    df_sorted = df.sort_values("id").reset_index(drop=True)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    df_sorted.to_csv(dest_path, index=False)

    print(f"submission.csv written to: {dest_path}")
    print(f"Total rows: {len(df_sorted):,}")


if __name__ == "__main__":
    main()
