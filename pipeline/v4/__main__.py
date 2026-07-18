"""Command line entry points for the isolated v4 DEV pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .acceptance import run_acceptance
from .analysis import aggregate, fit_mouse
from .cache import materialize_container, verify_cache


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pipeline.v4")
    commands = parser.add_subparsers(dest="command", required=True)

    materialize = commands.add_parser("cache-materialize")
    materialize.add_argument("--source", required=True, type=_path)
    materialize.add_argument("--manifest", required=True, type=_path)
    materialize.add_argument("--out", required=True, type=_path)
    materialize.add_argument("--container", required=True, type=int)
    materialize.add_argument("--neural-release", required=True)
    materialize.add_argument("--data-manifest-sha256", required=True)

    verify = commands.add_parser("cache-verify")
    verify.add_argument("--cache", required=True, type=_path)
    verify.add_argument("--manifest", required=True, type=_path)
    verify.add_argument("--report", required=True, type=_path)

    acceptance = commands.add_parser("acceptance")
    acceptance.add_argument("--output", required=True, type=_path)
    acceptance.add_argument(
        "--profile", choices=("registered", "fast"), default="registered"
    )

    mouse = commands.add_parser("fit-mouse")
    mouse.add_argument("--cache", required=True, type=_path)
    mouse.add_argument("--manifest", required=True, type=_path)
    mouse.add_argument("--out", required=True, type=_path)
    mouse.add_argument("--mouse-id", required=True, type=int)
    mouse.add_argument("--cache-release", required=True)
    mouse.add_argument("--cache-manifest-sha256", required=True)
    mouse.add_argument("--prereg-sha256", required=True)
    mouse.add_argument("--environment-sha256", required=True)

    combined = commands.add_parser("aggregate")
    combined.add_argument("--mouse-results", required=True, type=_path)
    combined.add_argument("--manifest", required=True, type=_path)
    combined.add_argument("--out", required=True, type=_path)
    combined.add_argument("--cache-release", required=True)
    combined.add_argument("--cache-manifest-sha256", required=True)
    combined.add_argument("--prereg-sha256", required=True)
    combined.add_argument("--environment-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "cache-materialize":
        result = materialize_container(
            args.source,
            args.manifest,
            args.out,
            args.container,
            neural_release=args.neural_release,
            data_manifest_sha256=args.data_manifest_sha256,
        )
    elif args.command == "cache-verify":
        result = verify_cache(args.cache, args.manifest, args.report)
    elif args.command == "acceptance":
        result = run_acceptance(args.output, profile=args.profile)
    elif args.command == "fit-mouse":
        result = fit_mouse(
            args.cache,
            args.manifest,
            args.out,
            mouse_id=args.mouse_id,
            cache_release=args.cache_release,
            cache_manifest_sha256=args.cache_manifest_sha256,
            prereg_sha256=args.prereg_sha256,
            environment_sha256=args.environment_sha256,
        )
    else:
        result = aggregate(
            args.mouse_results,
            args.manifest,
            args.out,
            cache_release=args.cache_release,
            cache_manifest_sha256=args.cache_manifest_sha256,
            prereg_sha256=args.prereg_sha256,
            environment_sha256=args.environment_sha256,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
