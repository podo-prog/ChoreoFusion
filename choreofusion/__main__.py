from __future__ import annotations

import argparse
import json

from .annotations import DEFAULT_LAYER
from .cluster import classify_one, fit
from .boundary import add_boundary_cli, run_boundary_cli


def main() -> None:
    parser = argparse.ArgumentParser(prog="choreofusion")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="fit an unlabeled motion clustering model")
    fit_parser.add_argument("--data-dir", default="data/raw")
    fit_parser.add_argument("--output-dir", default="outputs")
    fit_parser.add_argument("--layer", default=DEFAULT_LAYER)
    fit_parser.add_argument("--clusters", default="auto", help="auto or a fixed positive integer")

    classify_parser = subparsers.add_parser("classify", help="assign clusters to a segmented BVH clip")
    classify_parser.add_argument("--model", default="outputs/model.joblib")
    classify_parser.add_argument("--bvh", required=True)
    classify_parser.add_argument("--annotations", required=True)
    classify_parser.add_argument("--output", default="outputs/predictions.json")
    classify_parser.add_argument("--layer")

    add_boundary_cli(subparsers)
    args = parser.parse_args()
    if args.command in ("train-boundary", "segment", "segment-tcn"):
        run_boundary_cli(args)
    elif args.command == "fit":
        report = fit(args.data_dir, args.output_dir, args.layer, args.clusters)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        predictions = classify_one(args.model, args.bvh, args.annotations, args.output, args.layer)
        print(f"Wrote {len(predictions)} segment cluster assignments to {args.output}")


if __name__ == "__main__":
    main()
