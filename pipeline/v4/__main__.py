"""Command line entry points for the isolated v4 DEV pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .acceptance import run_acceptance
from .analysis import aggregate_targets
from .cache import materialize_container, verify_cache
from .hmm_checkpoint import fit_chunk, plan_chunks, verify_release
from .target import fit_mouse_targets, fit_target, hazard_plan


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

    hmm_plan = commands.add_parser("hmm-plan")
    hmm_plan.add_argument("--manifest", required=True, type=_path)
    hmm_plan.add_argument("--max-fit-keys", default=5, type=int)

    hmm_chunk = commands.add_parser("hmm-fit-chunk")
    hmm_chunk.add_argument("--cache", required=True, type=_path)
    hmm_chunk.add_argument("--manifest", required=True, type=_path)
    hmm_chunk.add_argument("--out", required=True, type=_path)
    hmm_chunk.add_argument("--mouse-id", required=True, type=int)
    hmm_chunk.add_argument("--fit-ids", required=True)
    hmm_chunk.add_argument("--cache-release", required=True)
    hmm_chunk.add_argument("--cache-manifest-sha256", required=True)
    hmm_chunk.add_argument("--prereg-sha256", required=True)
    hmm_chunk.add_argument("--environment-sha256", required=True)
    hmm_chunk.add_argument("--code-commit", required=True)

    hmm_verify = commands.add_parser("hmm-verify")
    hmm_verify.add_argument("--checkpoints", required=True, type=_path)
    hmm_verify.add_argument("--manifest", required=True, type=_path)
    hmm_verify.add_argument("--out", required=True, type=_path)
    hmm_verify.add_argument("--cache-release", required=True)
    hmm_verify.add_argument("--cache-manifest-sha256", required=True)
    hmm_verify.add_argument("--prereg-sha256", required=True)
    hmm_verify.add_argument("--environment-sha256", required=True)
    hmm_verify.add_argument("--code-commit", required=True)

    hazard = commands.add_parser("hazard-plan")
    hazard.add_argument("--manifest", required=True, type=_path)

    target = commands.add_parser("fit-target")
    target.add_argument("--cache", required=True, type=_path)
    target.add_argument("--manifest", required=True, type=_path)
    target.add_argument("--out", required=True, type=_path)
    target.add_argument("--target-session", required=True, type=int)
    target.add_argument("--cache-release", required=True)
    target.add_argument("--cache-manifest-sha256", required=True)
    target.add_argument("--hmm-prereg-sha256", required=True)
    target.add_argument("--hazard-prereg-sha256", required=True)
    target.add_argument("--environment-sha256", required=True)
    target.add_argument("--code-commit", required=True)
    target.add_argument("--hmm-checkpoints", required=True, type=_path)
    target.add_argument("--hmm-release", required=True)
    target.add_argument("--hmm-manifest-sha256", required=True)

    mouse = commands.add_parser("fit-mouse")
    mouse.add_argument("--cache", required=True, type=_path)
    mouse.add_argument("--manifest", required=True, type=_path)
    mouse.add_argument("--out", required=True, type=_path)
    mouse.add_argument("--mouse-id", required=True, type=int)
    mouse.add_argument("--cache-release", required=True)
    mouse.add_argument("--cache-manifest-sha256", required=True)
    mouse.add_argument("--hmm-prereg-sha256", required=True)
    mouse.add_argument("--hazard-prereg-sha256", required=True)
    mouse.add_argument("--environment-sha256", required=True)
    mouse.add_argument("--code-commit", required=True)
    mouse.add_argument("--hmm-checkpoints", required=True, type=_path)
    mouse.add_argument("--hmm-release", required=True)
    mouse.add_argument("--hmm-manifest-sha256", required=True)

    combined = commands.add_parser("aggregate")
    combined.add_argument("--target-results", required=True, type=_path)
    combined.add_argument("--manifest", required=True, type=_path)
    combined.add_argument("--out", required=True, type=_path)
    combined.add_argument("--cache-release", required=True)
    combined.add_argument("--cache-manifest-sha256", required=True)
    combined.add_argument("--hmm-prereg-sha256", required=True)
    combined.add_argument("--hazard-prereg-sha256", required=True)
    combined.add_argument("--environment-sha256", required=True)
    combined.add_argument("--code-commit", required=True)
    combined.add_argument("--hmm-release", required=True)
    combined.add_argument("--hmm-manifest-sha256", required=True)
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
        result = run_acceptance(args.output)
    elif args.command == "hmm-plan":
        result = plan_chunks(
            args.manifest, max_fit_keys=args.max_fit_keys
        )
    elif args.command == "hmm-fit-chunk":
        result = fit_chunk(
            args.cache,
            args.manifest,
            args.out,
            mouse_id=args.mouse_id,
            fit_ids=args.fit_ids.split(","),
            cache_release=args.cache_release,
            cache_manifest_sha256=args.cache_manifest_sha256,
            prereg_sha256=args.prereg_sha256,
            environment_sha256=args.environment_sha256,
            code_commit=args.code_commit,
        )
    elif args.command == "hmm-verify":
        result = verify_release(
            args.checkpoints,
            args.manifest,
            args.out,
            cache_release=args.cache_release,
            cache_manifest_sha256=args.cache_manifest_sha256,
            prereg_sha256=args.prereg_sha256,
            environment_sha256=args.environment_sha256,
            code_commit=args.code_commit,
        )
    elif args.command == "hazard-plan":
        result = hazard_plan(args.manifest)
    elif args.command == "fit-target":
        result = fit_target(
            args.cache,
            args.manifest,
            args.out,
            target_session=args.target_session,
            cache_release=args.cache_release,
            cache_manifest_sha256=args.cache_manifest_sha256,
            hmm_prereg_sha256=args.hmm_prereg_sha256,
            hazard_prereg_sha256=args.hazard_prereg_sha256,
            environment_sha256=args.environment_sha256,
            code_commit=args.code_commit,
            hmm_checkpoints=args.hmm_checkpoints,
            hmm_release=args.hmm_release,
            hmm_manifest_sha256=args.hmm_manifest_sha256,
        )
    elif args.command == "fit-mouse":
        result = fit_mouse_targets(
            args.cache,
            args.manifest,
            args.out,
            mouse_id=args.mouse_id,
            cache_release=args.cache_release,
            cache_manifest_sha256=args.cache_manifest_sha256,
            hmm_prereg_sha256=args.hmm_prereg_sha256,
            hazard_prereg_sha256=args.hazard_prereg_sha256,
            environment_sha256=args.environment_sha256,
            code_commit=args.code_commit,
            hmm_checkpoints=args.hmm_checkpoints,
            hmm_release=args.hmm_release,
            hmm_manifest_sha256=args.hmm_manifest_sha256,
        )
    else:
        result = aggregate_targets(
            args.target_results,
            args.manifest,
            args.out,
            cache_release=args.cache_release,
            cache_manifest_sha256=args.cache_manifest_sha256,
            hmm_prereg_sha256=args.hmm_prereg_sha256,
            hazard_prereg_sha256=args.hazard_prereg_sha256,
            environment_sha256=args.environment_sha256,
            code_commit=args.code_commit,
            hmm_release=args.hmm_release,
            hmm_manifest_sha256=args.hmm_manifest_sha256,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
