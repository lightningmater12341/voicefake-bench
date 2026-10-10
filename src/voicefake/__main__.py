"""Command-line entry point for the incrementally built benchmark."""

import argparse
import json

from .manifest import validate_manifest
from .protocols import build_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="VoiceFake Bench utilities")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-manifest", help="Audit dataset CSV metadata")
    validate.add_argument("path", help="Path to a CSV dataset manifest")
    importer = commands.add_parser("import-protocols", help="Import ASVspoof 2019 LA CM keys")
    for split in ("train", "dev", "eval"):
        importer.add_argument(f"--{split}", required=True, help=f"Official {split} CM key file")
    importer.add_argument("--output", required=True, help="New manifest CSV path")
    selection = importer.add_mutually_exclusive_group()
    selection.add_argument("--per-stratum", type=int, default=100,
                           help="Cap per split/label/attack (default: 100)")
    selection.add_argument("--all", action="store_true", help="Include all protocol records")
    importer.add_argument("--seed", type=int, default=42, help="Subset ranking seed (default: 42)")
    args = parser.parse_args()
    try:
        if args.command == "validate-manifest":
            result = validate_manifest(args.path)
        else:
            result = build_manifest(
                {split: getattr(args, split) for split in ("train", "dev", "eval")},
                args.output, None if args.all else args.per_stratum, args.seed,
            )
    except (OSError, ValueError) as exc:
        parser.exit(status=2, message=f"Dataset metadata check failed: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
