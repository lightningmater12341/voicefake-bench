"""Command-line entry point for the incrementally built benchmark."""

import argparse
import json

from .manifest import validate_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="VoiceFake Bench utilities")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-manifest", help="Audit dataset CSV metadata")
    validate.add_argument("path", help="Path to a CSV dataset manifest")
    args = parser.parse_args()
    try:
        result = validate_manifest(args.path)
    except (OSError, ValueError) as exc:
        parser.exit(status=2, message=f"Manifest validation failed: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
