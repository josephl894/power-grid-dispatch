from __future__ import annotations

import argparse
from pathlib import Path

from nyiso_cost_processing import CostBuildConfig, build_cost_input_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and validate NYISO cost-aware research input tables.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed/cost_inputs")
    parser.add_argument("--p27-chunksize", type=int, default=250_000)
    parser.add_argument(
        "--max-p27-gen-months",
        type=int,
        default=0,
        help="Optional cap for P-27 Gen monthly files. Leave at 0 to process all available months.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CostBuildConfig(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        p27_chunksize=args.p27_chunksize,
        max_p27_gen_months=args.max_p27_gen_months,
    )
    outputs, manifest, validation = build_cost_input_datasets(config)

    print("Cost-input raw manifest:")
    print(manifest.to_string(index=False))

    print("\nProcessed cost-input tables:")
    for name, frame in outputs.items():
        print(f"{name}\trows={frame.shape[0]}\tcols={frame.shape[1]}")

    print("\nValidation checks:")
    print(validation.to_string(index=False))

    warnings = validation[validation["status"] != "PASS"]
    if not warnings.empty:
        print("\nWarnings require review:")
        print(warnings.to_string(index=False))


if __name__ == "__main__":
    main()
