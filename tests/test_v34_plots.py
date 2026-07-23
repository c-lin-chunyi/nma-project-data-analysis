import importlib.util
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_TAG = "neural-confirm-v3.4-" + "a" * 64


def load_module():
    spec = importlib.util.spec_from_file_location(
        "v34_plots", ROOT / "pipeline/draw/v34_plots.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plots = load_module()


def interval(values):
    values = np.asarray(values, float)
    return {
        "mean": float(values.mean()),
        "low": float(np.quantile(values, 0.1)),
        "high": float(np.quantile(values, 0.9)),
        "n_mice": int(len(values)),
    }


def synthetic_inputs(root: Path, n_mice: int = 29):
    mice = np.arange(1000, 1000 + n_mice)
    q1_mice = pd.DataFrame(
        {"mouse_id": mice, "auc": np.linspace(0.54, 0.76, n_mice)}
    )
    session_rows = []
    fold_rows = []
    coverage_rows = []
    selection_rows = []
    for mouse_index, mouse in enumerate(mice):
        for session in range(2):
            behavior_session_id = int(mouse * 100 + session)
            auc = 0.54 + 0.006 * mouse_index + 0.004 * session
            session_rows.append(
                {
                    "mouse_id": mouse,
                    "behavior_session_id": behavior_session_id,
                    "auc": auc,
                    "conditional_auc": auc - 0.005,
                    "random_auc": auc + 0.008,
                    "dff_auc": auc - 0.012,
                    "miss_B": 20 + session * 5,
                    "behavioral_eligible": True,
                }
            )
            for seed in range(10):
                for fold in range(1, 6):
                    fold_rows.append(
                        {
                            "mouse_id": mouse,
                            "behavior_session_id": behavior_session_id,
                            "seed": seed,
                            "fold": fold,
                            "auc": auc + (fold - 3) * 0.003 + seed * 0.0001,
                            "miss_B": 20 + session * 5,
                            "estimability": "estimable",
                        }
                    )
            coverage_rows.append(
                {
                    "mouse_id": mouse,
                    "behavior_session_id": behavior_session_id,
                    "behavioral_eligible": not (
                        mouse_index == n_mice - 1 and session == 1
                    ),
                    "q1_estimability": (
                        "estimable"
                        if not (mouse_index == n_mice - 2 and session == 1)
                        else "temporal_support_nonestimable"
                    ),
                    "q2_estimability": (
                        "estimable"
                        if not (mouse_index == n_mice - 3 and session == 1)
                        else "q2_class_nonestimable"
                    ),
                }
            )
            for seed in range(10):
                for outer_fold in range(1, 6):
                    for model in ("m0", "m1"):
                        selected_power = -2 if model == "m0" else -1
                        for power in range(-4, 3):
                            selection_rows.append(
                                {
                                    "mouse_id": mouse,
                                    "behavior_session_id": behavior_session_id,
                                    "seed": seed,
                                    "outer_fold": outer_fold,
                                    "model": model,
                                    "C": float(10**power),
                                    "mean_log_loss": 0.6 + abs(power - selected_power) * 0.01,
                                    "se_log_loss": 0.01,
                                    "selected": power == selected_power,
                                }
                            )
    q1_sessions = pd.DataFrame(session_rows)
    q1_folds = pd.DataFrame(fold_rows)
    q2_mice = pd.DataFrame(
        {
            "mouse_id": mice,
            "delta_log_loss": np.linspace(-0.01, 0.04, n_mice),
            "delta_brier": np.linspace(-0.005, 0.02, n_mice),
            "delta_auc": np.linspace(-0.02, 0.08, n_mice),
            "m0_log_loss": np.linspace(0.60, 0.68, n_mice),
            "m1_log_loss": np.linspace(0.59, 0.64, n_mice),
            "m0_brier": np.linspace(0.20, 0.24, n_mice),
            "m1_brier": np.linspace(0.19, 0.22, n_mice),
            "m0_auc": np.linspace(0.58, 0.68, n_mice),
            "m1_auc": np.linspace(0.61, 0.74, n_mice),
        }
    )
    frames = {
        "q1_sessions": q1_sessions,
        "q1_mice": q1_mice,
        "q1_folds": q1_folds,
        "q2_mice": q2_mice,
        "q2_selection": pd.DataFrame(selection_rows),
        "coverage": pd.DataFrame(coverage_rows),
    }
    manifest = {
        "schema": "neural-confirm-v3.4",
        "confirm_data_accessed": True,
        "outcome": "confirmatory_supported",
        "primary": {
            "SESOI": 0.55,
            "mouse_interval": interval(q1_mice.auc),
        },
        "secondary": {
            "mouse_intervals": {
                metric: interval(q2_mice[metric])
                for metric in ("delta_log_loss", "delta_brier", "delta_auc")
            }
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "analysis-manifest.json").write_text(
        json.dumps(manifest, allow_nan=False) + "\n"
    )
    frames["q1_sessions"].to_parquet(root / "q1_sessions.parquet", index=False)
    frames["q1_mice"].to_parquet(root / "q1_mice.parquet", index=False)
    frames["q1_folds"].to_parquet(
        root / "q1_fold_diagnostics.parquet", index=False
    )
    frames["q2_mice"].to_parquet(root / "q2_mice.parquet", index=False)
    frames["q2_selection"].to_parquet(
        root / "q2_C_selection.parquet", index=False
    )
    frames["coverage"].to_parquet(
        root / "coverage_failures.parquet", index=False
    )
    return manifest, frames


class V34PlotDataTests(unittest.TestCase):
    def test_requirements_and_font_provenance_are_exact(self):
        requirements = (ROOT / "requirements-plot.txt").read_text()
        for dependency in (
            "numpy==2.3.5",
            "pandas==3.0.3",
            "pyarrow==25.0.0",
            "matplotlib==3.11.1",
            "seaborn==0.13.2",
        ):
            self.assertIn(dependency, requirements)
        source = plots.validate_fonts(ROOT / "assets/fonts/stix2")
        self.assertEqual(source["tag"], "v2.13b171")
        self.assertEqual(source["version"], "2.13 b171")

    def test_mouse_mapping_is_auc_ordered_and_private(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, frames = synthetic_inputs(Path(tmp))
            mapping, digest = plots.mouse_mapping(frames)
        self.assertEqual(mapping[1000], "M01")
        self.assertEqual(mapping[1028], "M29")
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertNotIn("1000", json.dumps({"mapping_sha256": digest}))

    def test_registered_weighting_and_fold_aggregation(self):
        frame = pd.DataFrame(
            {
                "mouse_id": [1, 1],
                "auc": [0.5, 0.9],
                "miss_B": [1, 3],
            }
        )
        result = plots.weighted_mouse_metric(frame, "auc")
        self.assertAlmostEqual(result.auc.iloc[0], 0.8)
        folds = pd.DataFrame(
            {
                "mouse_id": [1, 1, 1, 1],
                "behavior_session_id": [10, 10, 11, 11],
                "seed": [0, 1, 0, 1],
                "fold": [1, 1, 1, 1],
                "auc": [0.5, 0.7, 0.8, 1.0],
                "miss_B": [1, 1, 3, 3],
                "estimability": ["estimable"] * 4,
            }
        )
        folded = plots.fold_mouse_metrics(folds)
        self.assertAlmostEqual(folded.auc.iloc[0], 0.825)

    def test_coverage_categories_are_mutually_exclusive(self):
        frame = pd.DataFrame(
            {
                "mouse_id": [1, 2, 3],
                "behavioral_eligible": [True, True, False],
                "q1_estimability": [
                    "estimable",
                    "temporal_support_nonestimable",
                    "behavioral_ineligible",
                ],
                "q2_estimability": [
                    "estimable",
                    "q2_class_nonestimable",
                    "behavioral_ineligible",
                ],
            }
        )
        counts = plots.coverage_counts(frame)
        for analysis in ("Q1", "Q2"):
            self.assertEqual(
                counts.loc[counts.analysis.eq(analysis), "count"].sum(), 3
            )
            categories = set(counts.loc[counts.analysis.eq(analysis), "category"])
            self.assertEqual(
                categories,
                {
                    "Estimable",
                    "Eligible, not estimable",
                    "Behaviorally ineligible",
                },
            )

    def test_input_contract_rejects_bad_tag_schema_columns_and_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic_inputs(root)
            with self.assertRaisesRegex(ValueError, "source release"):
                plots.load_inputs(root, "latest")

            manifest_path = root / "analysis-manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["schema"] = "wrong"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "schema"):
                plots.load_inputs(root, SOURCE_TAG)

            synthetic_inputs(root)
            manifest = json.loads(manifest_path.read_text())
            manifest["primary"]["mouse_interval"]["low"] = 0.9
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "ordered"):
                plots.load_inputs(root, SOURCE_TAG)

            synthetic_inputs(root)
            broken = pd.read_parquet(root / "q1_mice.parquet").drop(columns="auc")
            broken.to_parquet(root / "q1_mice.parquet", index=False)
            with self.assertRaisesRegex(ValueError, "missing columns"):
                plots.load_inputs(root, SOURCE_TAG)

    def test_empty_statistical_tables_are_not_pipeline_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, _ = synthetic_inputs(root)
            pd.DataFrame().to_parquet(root / "q1_mice.parquet", index=False)
            manifest["primary"]["mouse_interval"] = {
                "mean": None,
                "low": None,
                "high": None,
                "n_mice": 0,
            }
            manifest["outcome"] = "nonestimable_coverage"
            (root / "analysis-manifest.json").write_text(json.dumps(manifest))
            loaded, frames = plots.load_inputs(root, SOURCE_TAG)
        self.assertEqual(loaded["outcome"], "nonestimable_coverage")
        self.assertTrue(frames["q1_mice"].empty)

    def test_workflow_reads_only_public_aggregate_release(self):
        workflow = (
            ROOT / ".github/workflows/neural-confirm-v3.4-plots.yml"
        ).read_text()
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("requirements-plot.txt", workflow)
        self.assertIn("texlive-xetex", workflow)
        self.assertIn("pdffonts", workflow)
        self.assertIn("neural-confirm-v3.4-plots-${SOURCE_SHA}-${GITHUB_SHA}", workflow)
        self.assertNotIn("environment: confirm-v3.4", workflow)
        self.assertNotIn("allensdk", workflow.lower())
        self.assertNotIn("features.h5", workflow)
        self.assertNotIn("--clobber", workflow)
        self.assertIn('.removeprefix("./")', workflow)
        for name in plots.SOURCE_FILES:
            self.assertIn(name, workflow)


@unittest.skipUnless(
    shutil.which("xelatex")
    and shutil.which("pdftocairo")
    and shutil.which("pdffonts")
    and importlib.util.find_spec("matplotlib")
    and importlib.util.find_spec("seaborn"),
    "full plot stack is not installed",
)
class V34PlotRenderTests(unittest.TestCase):
    def test_all_eight_figures_render_from_one_pgf_pdf_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "source", root / "plots"
            expected_manifest, _ = synthetic_inputs(source, n_mice=4)
            old_mpl = os.environ.get("MPLCONFIGDIR")
            os.environ["MPLCONFIGDIR"] = str(root / "mpl")
            os.environ.setdefault("SOURCE_DATE_EPOCH", "1622037715")
            try:
                result = plots.render(
                    source,
                    output,
                    SOURCE_TAG,
                    ROOT / "assets/fonts/stix2",
                )
            finally:
                if old_mpl is None:
                    os.environ.pop("MPLCONFIGDIR", None)
                else:
                    os.environ["MPLCONFIGDIR"] = old_mpl

            self.assertEqual(result["schema"], plots.PLOT_SCHEMA)
            self.assertEqual(len(result["figures"]), 8)
            self.assertEqual(
                result["registered_intervals_used_without_recomputation"]["q1_auc"],
                expected_manifest["primary"]["mouse_interval"],
            )
            self.assertFalse(result["confirm_raw_data_accessed"])
            self.assertFalse(result["models_refit"])
            for stem, _number, _title in plots.FIGURES:
                for suffix in ("pgf", "pdf", "svg", "png"):
                    path = output / f"{stem}.{suffix}"
                    self.assertTrue(path.is_file() and path.stat().st_size > 0, path)
                ET.parse(output / f"{stem}.svg")
                pages = subprocess.check_output(
                    ["pdfinfo", output / f"{stem}.pdf"], text=True
                )
                self.assertRegex(pages, r"(?m)^Pages:\s+1$")
                fonts = subprocess.check_output(
                    ["pdffonts", output / f"{stem}.pdf"], text=True
                )
                self.assertIn("STIXTwoText", fonts)

            all_fonts = "\n".join(
                subprocess.check_output(["pdffonts", path], text=True)
                for path in output.glob("*.pdf")
            )
            self.assertIn("STIXTwoMath", all_fonts)

            png = next(output.glob("*.png")).read_bytes()
            offset = 8
            dpi = None
            while offset < len(png):
                length = struct.unpack(">I", png[offset : offset + 4])[0]
                kind = png[offset + 4 : offset + 8]
                payload = png[offset + 8 : offset + 8 + length]
                if kind == b"pHYs":
                    xppm, yppm, unit = struct.unpack(">IIB", payload)
                    self.assertEqual((xppm, unit), (yppm, 1))
                    dpi = xppm * 0.0254
                    break
                offset += 12 + length
            self.assertIsNotNone(dpi)
            self.assertAlmostEqual(dpi, 600, delta=1)


if __name__ == "__main__":
    unittest.main()
