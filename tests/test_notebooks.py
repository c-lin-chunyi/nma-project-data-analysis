from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
NOTEBOOKS = [
    ROOT / "notebooks" / "01_behavioral_playground.ipynb",
    ROOT / "notebooks" / "02_neural_feature_explorer.ipynb",
]


@pytest.mark.parametrize("path", NOTEBOOKS)
def test_notebook_is_self_contained_output_free_and_compilable(path):
    document = json.loads(path.read_text())
    assert document["nbformat"] == 4
    code_cells = [cell for cell in document["cells"] if cell["cell_type"] == "code"]
    joined = "\n".join("".join(cell["source"]) for cell in code_cells)
    assert "git clone" not in joined
    assert "requirements-notebooks.txt" not in joined
    assert "from nma_play" not in joined
    assert "import nma_play" not in joined
    assert '"pip", "install"' in "".join(code_cells[0]["source"])
    assert "SHA-256 mismatch" in joined
    for index, cell in enumerate(code_cells):
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"{path.name}:code-{index}", "exec")


def test_behavior_notebook_uses_only_compact_scan():
    text = NOTEBOOKS[0].read_text()
    assert "behavioral-v3.1-29482141350" in text
    assert "behavioral-v3.1-scan.tar.gz" in text
    assert "dev-bundles.tar.gz" not in text


def test_neural_notebook_contains_visual_and_decoder_contracts():
    document = json.loads(NOTEBOOKS[1].read_text())
    text = NOTEBOOKS[1].read_text()
    setup = "".join(document["cells"][2]["source"])
    final_cell = "".join(document["cells"][-1]["source"])
    assert "neural-dev-features-v1-29482249873" in text
    assert "Trial × cell response matrix" in text
    assert "Representation comparisons" in text
    assert "Trial geometry" in text
    assert "Behavior–neural relationships" in text
    assert "Q2 covariate missingness" in text
    assert "render_decoder()" in final_cell
    assert 'DECODER_CV = "blocked"' in final_cell
    assert "EXPLORATORY" in text
    assert "reset_index(names=" not in text
    assert '("matplotlib", "matplotlib", "matplotlib")' in setup
    assert '("seaborn", "seaborn", "seaborn")' in setup
    assert '("plotly", "plotly", "plotly")' not in setup
    assert '("ipywidgets", "ipywidgets", "ipywidgets")' not in setup
    assert "display_matplotlib_figure" in final_cell
    assert "widgets." not in final_cell
    assert ".observe(" not in final_cell
    assert "px." not in final_cell
    assert "go." not in final_cell
    assert "kaleido" not in text


def test_neural_static_analysis_runs_in_a_clear_order():
    document = json.loads(NOTEBOOKS[1].read_text())
    source = "".join(document["cells"][-1]["source"])
    assert source.rindex("render_matrix()") < source.rindex("render_geometry()")
    assert source.rindex("render_geometry()") < source.rindex("render_decoder()")
    assert "SELECTED_EXPERIMENT_ID" in source
    assert "widgets" not in source
    assert "interactive" not in source.lower()
