"""Run the v3 two-day inference bundle. This entry point never trains models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from mod_mc.inference.two_day_bundle import TwoDayPredictor

DEFAULT_BUNDLE = Path(__file__).resolve().parent / "resources/v3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--features-csv", type=Path)
    mode.add_argument("--stonelab-daily", type=Path)
    parser.add_argument("--osugi-daily", type=Path)
    parser.add_argument("--mc-daily", type=Path)
    parser.add_argument("--window-end", action="append", help="YYYY-MM-DD; repeat for several windows")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    predictor = TwoDayPredictor(args.bundle)
    if args.features_csv:
        result = predictor.predict_features(pd.read_csv(args.features_csv))
    else:
        if not (args.osugi_daily and args.mc_daily and args.window_end):
            parser.error("Daily mode requires --osugi-daily, --mc-daily and --window-end")
        result = predictor.predict_daily(pd.read_csv(args.stonelab_daily), pd.read_csv(args.osugi_daily),
                                        pd.read_csv(args.mc_daily), args.window_end)
    protected = args.bundle.resolve()
    inputs = {p.resolve() for p in [args.features_csv, args.stonelab_daily, args.osugi_daily, args.mc_daily] if p is not None}
    outputs = [p.resolve() for p in (args.output_csv, args.output_json) if p is not None]
    if len(set(outputs)) != len(outputs):
        parser.error("Choose distinct CSV and JSON output paths")
    for output in outputs:
        if output.is_relative_to(protected):
            parser.error("Write predictions outside the frozen bundle")
        if output in inputs:
            parser.error("Predictions cannot overwrite an input file")
        if output.exists():
            parser.error(f"Output already exists: {output}")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False, date_format="%Y-%m-%d")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        records = json.loads(result.to_json(orient="records", date_format="iso", force_ascii=False))
        args.output_json.write_text(json.dumps({"model_version":"v3","predictions":records}, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
