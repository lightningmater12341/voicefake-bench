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
    inspector = commands.add_parser("inspect-audio", help="Decode and audit a local recording")
    inspector.add_argument("path", help="Path to a local audio file")
    inspector.add_argument("--config", help="Audio configuration JSON; otherwise use documented defaults")
    extractor = commands.add_parser("extract-features", help="Cache MFCC summaries and index a manifest")
    extractor.add_argument("--manifest", required=True)
    extractor.add_argument("--audio-root", required=True)
    extractor.add_argument("--cache-dir", required=True)
    extractor.add_argument("--output", required=True, help="New JSON feature-index path")
    extractor.add_argument("--audio-config", help="Audio processing JSON")
    extractor.add_argument("--feature-config", help="Feature recipe JSON")
    extractor.add_argument("--allow-eval", action="store_true", help="Allow evaluation audio after experiment freeze")
    args = parser.parse_args()
    try:
        if args.command == "validate-manifest":
            result = validate_manifest(args.path)
        elif args.command == "import-protocols":
            result = build_manifest(
                {split: getattr(args, split) for split in ("train", "dev", "eval")},
                args.output, None if args.all else args.per_stratum, args.seed,
            )
        elif args.command == "inspect-audio":
            from .audio import AudioConfig, load_audio
            config = AudioConfig.from_json(args.config) if args.config else AudioConfig()
            _, result = load_audio(args.path, config)
        else:
            from .audio import AudioConfig
            from .features import FeatureConfig
            from .feature_cache import extract_manifest
            audio_config = AudioConfig.from_json(args.audio_config) if args.audio_config else AudioConfig()
            feature_config = FeatureConfig.from_json(args.feature_config) if args.feature_config else FeatureConfig()
            result = extract_manifest(args.manifest, args.audio_root, args.cache_dir, args.output,
                                      audio_config, feature_config, args.allow_eval)
    except ImportError:
        parser.exit(status=2, message="Audio support requires: python -m pip install -e \".[audio]\"\n")
    except (OSError, ValueError) as exc:
        parser.exit(status=2, message=f"Input validation failed: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
