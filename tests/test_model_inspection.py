"""V.6: независимый от UI экспорт графа модели и безопасных тензоров."""

import json

import pytest
import torch

from bioplast.runner import (
    inspect_model,
    inspect_tensor,
    load_model_manifest,
    load_training_checkpoint,
    run_config,
)
from experiments.xor_backprop import MLP


def test_tensor_inspection_copies_small_finite_values():
    tensor = torch.tensor([[1.0, -2.0], [0.0, 3.0]], requires_grad=True)

    inspected = inspect_tensor("weight", "parameter", tensor)
    with torch.no_grad():
        tensor.add_(10.0)

    assert inspected.value_mode == "full"
    assert inspected.values == [[1.0, -2.0], [0.0, 3.0]]
    assert inspected.requires_grad is True
    assert inspected.summary is not None
    assert inspected.summary.element_count == 4
    assert inspected.summary.l2_norm == pytest.approx(14**0.5)
    assert inspected.summary.sparsity == pytest.approx(0.25)


def test_tensor_inspection_omits_large_values_but_keeps_summary():
    inspected = inspect_tensor(
        "weight",
        "parameter",
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
        full_values_max_elements=4,
    )

    assert inspected.value_mode == "summary"
    assert inspected.values is None
    assert inspected.values_omitted_reason == "size_limit"
    assert inspected.summary is not None
    assert inspected.summary.minimum == 0.0
    assert inspected.summary.maximum == 5.0


def test_tensor_inspection_never_embeds_nan_or_infinity():
    inspected = inspect_tensor(
        "unstable",
        "activation",
        torch.tensor([0.0, -2.0, float("inf"), float("nan")]),
    )

    assert inspected.value_mode == "summary"
    assert inspected.values_omitted_reason == "non_finite"
    assert inspected.summary is not None
    assert inspected.summary.finite_count == 2
    assert inspected.summary.non_finite_count == 2
    assert inspected.summary.mean == pytest.approx(-1.0)
    assert inspected.summary.std == pytest.approx(1.0)
    json.dumps(inspected.to_dict(), allow_nan=False)


def test_model_inspector_exports_xor_execution_graph_and_restores_mode():
    model = MLP([2, 8, 1])
    model.train()
    model.layers[1].eval()
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

    manifest = inspect_model(
        model,
        run_id="xor-run",
        model_name="mlp-2-8-1",
        example_args=(x,),
        layer_ids={"layers.0": "hidden", "layers.1": "output"},
        activations={"layers.0": "relu", "layers.1": "identity"},
        capture_phase="completed",
        step=2000,
    )

    assert model.training is True
    assert model.layers[0].training is True
    assert model.layers[1].training is False
    assert [layer.layer_id for layer in manifest.layers] == ["hidden", "output"]
    assert manifest.layers[0].input_shape == (None, 2)
    assert manifest.layers[0].output_shape == (None, 8)
    assert manifest.layers[0].activation == "relu"
    assert manifest.layers[0].parameter_count == 24
    assert manifest.layers[1].parameter_count == 9
    assert manifest.connections[0].to_dict() == {
        "source": "hidden",
        "target": "output",
        "kind": "forward",
    }
    assert {tensor.name for tensor in manifest.layers[0].tensors} == {"weight", "bias"}
    assert all(tensor.value_mode == "full" for tensor in manifest.layers[0].tensors)
    assert manifest.capture_phase == "completed"
    assert manifest.step == 2000


def test_xor_run_writes_loadable_model_manifest(tmp_path):
    config = {
        "session": "V.6",
        "dataset": "xor",
        "model": "mlp-2-3-1",
        "experiment": "xor_backprop",
        "device": "cpu",
        "seed": 0,
        "hidden": 3,
        "steps": 1,
        "log_every": 1,
    }

    run_dir = run_config(config, runs_dir=tmp_path)
    manifest = load_model_manifest(run_dir)
    checkpoint = load_training_checkpoint(run_dir / "checkpoint.pt")
    raw = (run_dir / "model.json").read_text(encoding="utf-8")

    assert manifest.run_id == run_dir.name
    assert manifest.model_name == "mlp-2-3-1"
    assert [layer.layer_id for layer in manifest.layers] == ["hidden", "output"]
    assert manifest.layers[0].tensors[0].shape == (3, 2)
    assert checkpoint["step"] == 1
    assert set(checkpoint["model_state_dict"]) == {
        "layers.0.weight",
        "layers.0.bias",
        "layers.1.weight",
        "layers.1.bias",
    }
    assert checkpoint["optimizer_state_dict"] is not None
    assert '"schema_version": 1' in raw
    assert "NaN" not in raw and "Infinity" not in raw
