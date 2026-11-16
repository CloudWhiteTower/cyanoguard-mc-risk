"""Portable command-line entry points; prediction never starts training."""

from __future__ import annotations

import argparse
import csv
from importlib.resources import files
from pathlib import Path
import tempfile

from mod_mc import __version__


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict target-day total microcystin risk.")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--input", required=True, type=Path, help="39-feature CSV")
    parser.add_argument("--output", required=True, type=Path, help="New output CSV; never overwritten")
    args = parser.parse_args()
    from mod_mc.inference.predict import predict_csv
    try:
        result = predict_csv(args.input, args.output)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(2, f"Error: {error}\n")
    print(f"Predicted {len(result)} rows -> {args.output}")


def build_main() -> None:
    parser = argparse.ArgumentParser(description="Build seven-day features from source observations.")
    parser.add_argument("--stonelab", type=Path, required=True, help="Normalized SL Buoy CSV")
    parser.add_argument("--osugi-root", type=Path, required=True, help="Source OSUGI NetCDF folder")
    parser.add_argument("--output", type=Path, required=True, help="New directory; must not exist")
    parser.add_argument("--start-year", type=int, default=2015, help="First target year (default: 2015)")
    parser.add_argument("--end-year", type=int, default=2021, help="Last target year (default: 2021)")
    args = parser.parse_args()
    from mod_mc.data.final_dataset import build_dataset_for_years
    try:
        if not args.stonelab.is_file() or not args.osugi_root.is_dir():
            raise ValueError("Both the Stone Lab CSV and OSUGI directory must exist")
        args.output.mkdir(parents=True, exist_ok=False)
        output = build_dataset_for_years(args.stonelab, args.osugi_root, args.output,
                                         args.start_year, args.end_year)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(output.dataset_path)


def prepare_main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the original Stone Lab workbook for feature construction.")
    parser.add_argument("--input", type=Path, required=True, help="Original 2013-2022 XLSX workbook")
    parser.add_argument("--output", type=Path, required=True, help="New normalized SL Buoy CSV")
    args = parser.parse_args()
    from mod_mc.data.prepare_source import prepare_stonelab
    try:
        rows = prepare_stonelab(args.input, args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(f"Prepared {rows} rows -> {args.output}")


def reproduce_main() -> None:
    parser = argparse.ArgumentParser(description="Explicitly rerun temporal CV on user-supplied data.")
    parser.add_argument("--dataset", type=Path, required=True, help="Labeled feature CSV")
    parser.add_argument("--output", type=Path, required=True, help="New directory; must not exist")
    args = parser.parse_args()
    import yaml
    from mod_mc.training.tune_temporal_cv_2021 import run_temporal_cv_experiment
    try:
        dataset = args.dataset.resolve(strict=True)
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError("Output directory already exists")
        config = yaml.safe_load(files("mod_mc.resources").joinpath("training.yaml").read_text("utf-8"))
        config["data_path"] = str(dataset)
        config["output_dir"] = str(args.output.resolve())
        args.output.mkdir(parents=True, exist_ok=False)
        with tempfile.TemporaryDirectory(prefix="mc-risk-config-") as temp:
            config_path = Path(temp) / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            output = run_temporal_cv_experiment(config_path, Path.cwd())
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(output)


if __name__ == "__main__":
    main()
