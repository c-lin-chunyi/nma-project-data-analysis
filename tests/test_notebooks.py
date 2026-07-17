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
    assert "Run decoder" in text
    assert "Blocked + purge · registered" in text
    assert "EXPLORATORY" in text
    assert "reset_index(names=" not in text
    assert 'kaleido==0.2.1' in setup
    assert "display_widget_figure" in final_cell
    assert "figure.to_image" not in final_cell


def test_neural_widget_initialization_avoids_colab_output_races():
    document = json.loads(NOTEBOOKS[1].read_text())
    source = "".join(document["cells"][-1]["source"])
    assert source.index("sync_containers()") < source.index(
        'mouse_dd.observe(sync_containers, names="value")'
    )
    assert source.index("display(widgets.VBox") < source.rindex("render_matrix()")
    assert "matrix_out.clear_output" not in source
    assert "geometry_out.clear_output" not in source
