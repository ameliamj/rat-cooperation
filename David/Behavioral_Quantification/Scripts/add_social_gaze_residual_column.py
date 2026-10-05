#!/usr/bin/env python3
"""Add residual social gaze column to a session-level gaze CSV."""

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a new column computed as "
            "socialGazing - socialGazingAtLev - socialGazingAtMag - socialGazingAtCenter."
        )
    )
    parser.add_argument("input_csv", help="Path to input CSV")
    parser.add_argument(
        "--output-csv",
        help="Path to output CSV. Default: <input_stem>_with_residual.csv in current directory.",
    )
    parser.add_argument(
        "--column-name",
        default="socialGazingResidual",
        help="Name for the new residual column",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    required_cols = [
        "socialGazing",
        "socialGazingAtLev",
        "socialGazingAtMag",
        "socialGazingAtCenter",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + f"\nAvailable columns: {', '.join(df.columns)}"
        )

    df[args.column_name] = (
        df["socialGazing"]
        - df["socialGazingAtLev"]
        - df["socialGazingAtMag"]
        - df["socialGazingAtCenter"]
    )

    if args.output_csv:
        output_path = Path(args.output_csv)
    else:
        output_path = Path.cwd() / f"{input_path.stem}_with_residual.csv"

    df.to_csv(output_path, index=False)

    print(f"Wrote: {output_path}")
    print(f"Added column: {args.column_name}")


if __name__ == "__main__":
    main()
