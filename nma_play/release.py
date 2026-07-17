"""Verified, cache-aware readers for the immutable public DEV releases.

The notebook helpers intentionally avoid AllenSDK and the 16 GB neural bundle.
They consume only the compact behavioral scan and analysis-ready feature cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd

REPOSITORY = "c-lin-chunyi/nma-project-data-analysis"
BEHAVIORAL_TAG = "behavioral-v3.1-29482141350"
FEATURE_TAG = "neural-dev-features-v1-29482249873"
BEHAVIORAL_ARCHIVE = "behavioral-v3.1-scan.tar.gz"
FEATURE_MANIFEST = "feature-cache-manifest.json"
FEATURE_NAMES = (
    "events_baselined_post",
    "events_unbaselined_pre",
    "events_unbaselined_post",
    "events_baselined_full_pre",
    "dff_baselined_post",
)

BEHAVIORAL_REQUIRED_COLUMNS = {
    "trial_labels": {
        "behavior_session_id",
        "mouse_id",
        "trial_id",
        "trial_index",
        "late_hit",
        "early_hit",
        "miss",
        "aborted",
        "engaged_A",
        "engaged_B",
        "engaged_A_hysteretic",
        "keep_A",
        "keep_B",
        "keep_A_hysteretic",
    },
    "session_scan": {
        "behavior_session_id",
        "mouse_id",
        "n_trials",
        "late_hit_B",
        "miss_B",
        "behavioral_eligible",
    },
}


class ReleaseDataError(RuntimeError):
    """Raised when a release asset is unavailable, unsafe, or invalid."""


def default_cache_dir() -> Path:
    override = os.environ.get("NMA_RELEASE_CACHE")
    if override:
        return Path(override).expanduser()
    if "google.colab" in sys.modules or Path("/content").is_dir():
        return Path("/content/nma-release-cache")
    return Path.home() / ".cache" / "nma-project-data-analysis"


def release_url(tag: str, asset: str) -> str:
    return f"https://github.com/{REPOSITORY}/releases/download/{tag}/{asset}"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sha256sums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and len(parts[0]) == 64:
            result[parts[-1].lstrip("*")] = parts[0].lower()
    if not result:
        raise ReleaseDataError(f"No SHA-256 entries found in {path}")
    return result


def _progress(name: str, block: int, block_size: int, total: int) -> None:
    if total <= 0:
        return
    downloaded = min(block * block_size, total)
    pct = 100 * downloaded / total
    print(f"\r{name}: {downloaded / 2**20:.1f}/{total / 2**20:.1f} MiB ({pct:5.1f}%)",
          end="", flush=True)
    if downloaded >= total:
        print()


def _copy_or_download(
    tag: str,
    asset: str,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    source_dir: Path | None = None,
    show_progress: bool = True,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and (
        expected_sha256 is None or sha256_file(destination) == expected_sha256.lower()
    ):
        return destination
    if destination.exists():
        destination.unlink()

    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        partial.unlink()
    try:
        if source_dir is not None:
            source = Path(source_dir) / asset
            if not source.is_file():
                raise ReleaseDataError(f"Missing local release asset: {source}")
            shutil.copyfile(source, partial)
        else:
            callback = (
                (lambda block, size, total: _progress(asset, block, size, total))
                if show_progress else None
            )
            urllib.request.urlretrieve(release_url(tag, asset), partial, callback)
        if expected_sha256 is not None:
            actual = sha256_file(partial)
            if actual != expected_sha256.lower():
                raise ReleaseDataError(
                    f"SHA-256 mismatch for {asset}: expected {expected_sha256}, got {actual}"
                )
        os.replace(partial, destination)
    except (OSError, urllib.error.URLError) as exc:
        if partial.exists():
            partial.unlink()
        if isinstance(exc, ReleaseDataError):
            raise
        raise ReleaseDataError(f"Could not obtain {asset}: {exc}") from exc
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    return destination


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    """Extract regular files/directories without trusting archive paths or links."""
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ReleaseDataError(f"Unsafe path in {archive.name}: {member.name}")
            target = (destination / relative).resolve()
            try:
                target.relative_to(base)
            except ValueError as exc:
                raise ReleaseDataError(
                    f"Archive member escapes destination: {member.name}"
                ) from exc
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ReleaseDataError(
                    f"Unsupported link/device in {archive.name}: {member.name}"
                )
            source = tar.extractfile(member)
            if source is None:
                raise ReleaseDataError(f"Could not read {member.name} from {archive.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _extract_once(archive: Path, destination: Path, expected_sha256: str) -> None:
    marker = destination / ".archive-sha256"
    if marker.is_file() and marker.read_text().strip() == expected_sha256:
        return
    if destination.exists():
        shutil.rmtree(destination)
    _safe_extract_tar(archive, destination)
    marker.write_text(expected_sha256 + "\n")


@dataclass(frozen=True)
class BehaviorScan:
    tag: str
    root: Path
    tables: Mapping[str, pd.DataFrame]
    manifest: Mapping[str, Any]

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.tables[name]

    @property
    def trial_labels(self) -> pd.DataFrame:
        return self.tables["trial_labels"]

    @property
    def session_scan(self) -> pd.DataFrame:
        return self.tables["session_scan"]


def _behavior_table_name(path: Path) -> str:
    return path.stem.lstrip("_")


def _validate_behavior(scan: BehaviorScan, *, strict_public: bool) -> None:
    if scan.manifest.get("schema") != "behavioral-v3.1":
        raise ReleaseDataError("Unexpected behavioral manifest schema")
    for name, required in BEHAVIORAL_REQUIRED_COLUMNS.items():
        if name not in scan.tables:
            raise ReleaseDataError(f"Behavioral scan is missing {name}.parquet")
        missing = required - set(scan.tables[name].columns)
        if missing:
            raise ReleaseDataError(f"{name}.parquet is missing columns: {sorted(missing)}")
    if strict_public:
        if int(scan.manifest.get("n_dev_sessions", -1)) != 50:
            raise ReleaseDataError("Public behavioral release must contain 50 DEV sessions")
        if scan.session_scan["behavior_session_id"].nunique() != 50:
            raise ReleaseDataError("Behavioral session table does not contain 50 sessions")


def load_behavioral_scan(
    cache_dir: str | Path | None = None,
    *,
    source_dir: str | Path | None = None,
    show_progress: bool = True,
) -> BehaviorScan:
    """Download, verify, extract, and read the compact behavioral DEV scan."""
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    source_value = source_dir or os.environ.get("NMA_BEHAVIORAL_SOURCE_DIR")
    local_source = Path(source_value) if source_value is not None else None
    root = cache / "behavioral" / BEHAVIORAL_TAG
    sums_path = _copy_or_download(
        BEHAVIORAL_TAG, "SHA256SUMS", root / "SHA256SUMS",
        source_dir=local_source, show_progress=show_progress,
    )
    sums = parse_sha256sums(sums_path)
    expected = sums.get(BEHAVIORAL_ARCHIVE)
    if expected is None:
        raise ReleaseDataError(f"SHA256SUMS does not cover {BEHAVIORAL_ARCHIVE}")
    archive = _copy_or_download(
        BEHAVIORAL_TAG, BEHAVIORAL_ARCHIVE, root / BEHAVIORAL_ARCHIVE,
        expected_sha256=expected, source_dir=local_source, show_progress=show_progress,
    )
    manifest_sha = sums.get("behavioral-manifest.json")
    manifest_path = _copy_or_download(
        BEHAVIORAL_TAG, "behavioral-manifest.json", root / "behavioral-manifest.json",
        expected_sha256=manifest_sha, source_dir=local_source, show_progress=show_progress,
    )
    extracted = root / "scan"
    _extract_once(archive, extracted, expected)
    try:
        tables = {
            _behavior_table_name(path): pd.read_parquet(path)
            for path in sorted(extracted.glob("*.parquet"))
        }
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        raise ReleaseDataError(f"Could not read behavioral scan files: {exc}") from exc
    scan = BehaviorScan(BEHAVIORAL_TAG, extracted, tables, manifest)
    _validate_behavior(scan, strict_public=local_source is None)
    return scan


@dataclass(frozen=True)
class FeatureMatrix:
    values: np.ndarray
    trial_ids: np.ndarray
    cell_ids: np.ndarray
    name: str
    experiment_id: int


class FeatureCache:
    """Lazy reader over the extracted, experiment-level feature cache."""

    feature_names = FEATURE_NAMES

    def __init__(
        self,
        root: Path,
        manifest: Mapping[str, Any],
        validation: Mapping[str, Any],
        experiments: pd.DataFrame,
        *,
        strict_public: bool = True,
    ) -> None:
        self.root = Path(root)
        self.manifest = dict(manifest)
        self.validation = dict(validation)
        self.source_experiments = experiments.copy()
        meta_rows = []
        self._meta: dict[int, dict[str, Any]] = {}
        for path in sorted(self.root.glob("*.feature-meta.json")):
            meta = json.loads(path.read_text())
            identity = dict(meta["identity"])
            oeid = int(identity["ophys_experiment_id"])
            self._meta[oeid] = meta
            meta_rows.append({
                **identity,
                "n_trials": int(meta["n_trials"]),
                "n_cells": int(meta["n_cells"]),
            })
        self.index = pd.DataFrame(meta_rows).sort_values(
            ["mouse_id", "ophys_container_id", "ophys_experiment_id"]
        ).reset_index(drop=True)
        self._validate(strict_public=strict_public)

    @property
    def experiment_ids(self) -> list[int]:
        return self.index["ophys_experiment_id"].astype(int).tolist()

    def meta(self, experiment_id: int) -> dict[str, Any]:
        try:
            return self._meta[int(experiment_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown feature-cache experiment: {experiment_id}") from exc

    def labels(self, experiment_id: int) -> pd.DataFrame:
        path = self.root / f"{int(experiment_id)}.labels.parquet"
        return pd.read_parquet(path)

    def q2(self, experiment_id: int) -> pd.DataFrame:
        path = self.root / f"{int(experiment_id)}.q2.parquet"
        return pd.read_parquet(path)

    def matrix(self, experiment_id: int, name: str = FEATURE_NAMES[0]) -> FeatureMatrix:
        if name not in FEATURE_NAMES:
            raise KeyError(f"Unknown feature {name!r}; choose one of {FEATURE_NAMES}")
        oeid = int(experiment_id)
        with h5py.File(self.root / f"{oeid}.features.h5", "r") as h5:
            values = h5[name][:]
            trial_ids = h5["trial_id"][:]
            cell_ids = h5["cell_specimen_id"][:]
        return FeatureMatrix(values, trial_ids, cell_ids, name, oeid)

    def _validate(self, *, strict_public: bool) -> None:
        if self.manifest.get("schema") != "neural-dev-feature-cache-v1":
            raise ReleaseDataError("Unexpected feature-cache manifest schema")
        if not bool(self.validation.get("complete")):
            raise ReleaseDataError(
                f"Feature-cache validation is not complete: {self.validation.get('failures')}"
            )
        if self.index.empty:
            raise ReleaseDataError("No feature experiments were extracted")
        expected = int(self.manifest.get("n_active_experiments", -1))
        if len(self.index) != expected:
            raise ReleaseDataError(
                f"Expected {expected} active experiments, found {len(self.index)}"
            )
        if strict_public and (
            expected != 50 or int(self.manifest.get("n_containers", -1)) != 10
        ):
            raise ReleaseDataError("Public feature cache must contain 50 experiments/10 containers")
        for oeid in self.experiment_ids:
            files = [
                self.root / f"{oeid}.features.h5",
                self.root / f"{oeid}.labels.parquet",
                self.root / f"{oeid}.q2.parquet",
            ]
            if not all(path.is_file() for path in files):
                missing = [path.name for path in files if not path.is_file()]
                raise ReleaseDataError(f"Experiment {oeid} is missing files: {missing}")
            labels = self.labels(oeid)
            q2 = self.q2(oeid)
            with h5py.File(files[0], "r") as h5:
                missing_features = set(FEATURE_NAMES) - set(h5.keys())
                if missing_features:
                    raise ReleaseDataError(
                        f"Experiment {oeid} is missing HDF5 datasets: {sorted(missing_features)}"
                    )
                trial_ids = h5["trial_id"][:]
                n_cells = len(h5["cell_specimen_id"])
                for feature in FEATURE_NAMES:
                    if h5[feature].shape != (len(trial_ids), n_cells):
                        raise ReleaseDataError(
                            f"Experiment {oeid} has invalid shape for {feature}"
                        )
            if not np.array_equal(labels["trial_id"].to_numpy(), trial_ids):
                raise ReleaseDataError(f"Experiment {oeid} label/HDF5 trial IDs do not align")
            if not np.array_equal(q2["trial_id"].to_numpy(), trial_ids):
                raise ReleaseDataError(f"Experiment {oeid} Q2/HDF5 trial IDs do not align")


def load_feature_cache(
    cache_dir: str | Path | None = None,
    *,
    source_dir: str | Path | None = None,
    show_progress: bool = True,
) -> FeatureCache:
    """Download and verify all compact feature shards, then return a lazy reader."""
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    source_value = source_dir or os.environ.get("NMA_FEATURE_SOURCE_DIR")
    local_source = Path(source_value) if source_value is not None else None
    root = cache / "features" / FEATURE_TAG
    sums_path = _copy_or_download(
        FEATURE_TAG, "SHA256SUMS", root / "SHA256SUMS",
        source_dir=local_source, show_progress=show_progress,
    )
    sums = parse_sha256sums(sums_path)
    metadata_names = [
        FEATURE_MANIFEST,
        "feature-cache-validation.json",
        "dev_experiments.csv",
    ]
    metadata: dict[str, Path] = {}
    for name in metadata_names:
        expected = sums.get(name)
        if expected is None:
            raise ReleaseDataError(f"SHA256SUMS does not cover {name}")
        metadata[name] = _copy_or_download(
            FEATURE_TAG, name, root / name, expected_sha256=expected,
            source_dir=local_source, show_progress=show_progress,
        )
    manifest = json.loads(metadata[FEATURE_MANIFEST].read_text())
    validation = json.loads(metadata["feature-cache-validation.json"].read_text())
    extracted = root / "cache"
    parts = manifest.get("parts", [])
    if not parts:
        raise ReleaseDataError("Feature manifest lists no shards")
    for part in parts:
        name = str(part["name"])
        expected = str(part["sha256"]).lower()
        archive = _copy_or_download(
            FEATURE_TAG, name, root / "archives" / name,
            expected_sha256=expected, source_dir=local_source, show_progress=show_progress,
        )
        marker = extracted / f".{name}.sha256"
        if not marker.is_file() or marker.read_text().strip() != expected:
            _safe_extract_tar(archive, extracted)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(expected + "\n")
    experiments = pd.read_csv(metadata["dev_experiments.csv"])
    try:
        return FeatureCache(
            extracted,
            manifest,
            validation,
            experiments,
            strict_public=local_source is None,
        )
    except ReleaseDataError:
        raise
    except Exception as exc:
        raise ReleaseDataError(f"Could not read feature-cache files: {exc}") from exc
