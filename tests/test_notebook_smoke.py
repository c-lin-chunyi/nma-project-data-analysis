from __future__ import annotations

import os
from pathlib import Path

import pytest

from test_playground import make_behavior_assets, make_feature_assets

nbformat = pytest.importorskip("nbformat")
pytest.importorskip("nbclient")
pytest.importorskip("plotly")
pytest.importorskip("ipywidgets")

from nbclient import NotebookClient

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("name", "source_env", "source_factory"),
    [
        (
            "01_behavioral_playground.ipynb",
            "NMA_BEHAVIORAL_SOURCE_DIR",
            make_behavior_assets,
        ),
        (
            "02_neural_feature_explorer.ipynb",
            "NMA_FEATURE_SOURCE_DIR",
            make_feature_assets,
        ),
    ],
)
def test_notebook_executes_with_local_release_fixture(
    tmp_path, monkeypatch, name, source_env, source_factory
):
    source = source_factory(tmp_path)
    monkeypatch.setenv(source_env, str(source))
    monkeypatch.setenv("NMA_RELEASE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("NMA_NOTEBOOK_TEST", "1")
    document = nbformat.read(ROOT / "notebooks" / name, as_version=4)
    client = NotebookClient(
        document,
        timeout=600,
        kernel_name=os.environ.get("NMA_NOTEBOOK_KERNEL", "python3"),
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
