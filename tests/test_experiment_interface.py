"""Единый результат эксперимента и одинаковый экспорт PyTorch-моделей."""

import json

import pytest
import torch

from bioplast.data import Dataset
from bioplast.runner import (
    ContractError,
    ExperimentResult,
    load_model_manifest,
    load_training_checkpoint,
    run_config,
)
from bioplast.runner.experiment import finalize_experiment


def test_experiment_result_rejects_non_finite_final_metrics():
    with pytest.raises(ContractError, match="конечные JSON"):
        ExperimentResult(final={"loss": float("nan")})


def test_runner_rejects_legacy_dict_result(tmp_path):
    with pytest.raises(ContractError, match="ExperimentResult"):
        finalize_experiment(
            {"loss": 1.0},  # type: ignore[arg-type]
            config={"experiment": "legacy"},
            run_id="run",
            run_dir=tmp_path,
            logger=None,
        )


def test_mnist_and_xor_use_same_model_artifact_format(tmp_path, monkeypatch):
    import experiments.mnist_mlp_backprop as mnist_experiment

    dataset = Dataset(
        train_x=torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.2, 0.2, 0.8, 0.8],
                [0.8, 0.8, 0.2, 0.2],
            ]
        ),
        train_y=torch.tensor([0, 0, 1, 1, 0, 1, 0, 1]),
        test_x=torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]),
        test_y=torch.tensor([0, 1]),
    )
    monkeypatch.setattr(mnist_experiment, "load_mnist", lambda **_kwargs: dataset)

    mnist_dir = run_config(
        {
            "session": "test",
            "dataset": "mnist",
            "model": "mlp-4-3-2",
            "experiment": "mnist_mlp_backprop",
            "device": "cpu",
            "seed": 0,
            "hidden": [3],
            "epochs": 1,
            "batch_size": 4,
            "lr": 0.01,
        },
        runs_dir=tmp_path / "runs",
    )
    xor_dir = run_config(
        {
            "session": "test",
            "dataset": "xor",
            "model": "mlp-2-3-1",
            "experiment": "xor_backprop",
            "device": "cpu",
            "seed": 0,
            "hidden": 3,
            "steps": 1,
            "log_every": 1,
        },
        runs_dir=tmp_path / "runs",
    )

    for run_dir in (mnist_dir, xor_dir):
        assert (run_dir / "model.json").is_file()
        assert (run_dir / "checkpoint.pt").is_file()
        assert set(json.loads((run_dir / "model.json").read_text(encoding="utf-8"))) == {
            "schema_version",
            "kind",
            "run_id",
            "model_name",
            "captured_at",
            "capture_phase",
            "step",
            "capture_batch_size",
            "layers",
            "connections",
        }
        assert set(load_training_checkpoint(run_dir / "checkpoint.pt")) == {
            "schema_version",
            "kind",
            "run_id",
            "experiment",
            "model_name",
            "created_at",
            "step",
            "model_state_dict",
            "optimizer_state_dict",
        }

    mnist_model = load_model_manifest(mnist_dir)
    assert [layer.layer_id for layer in mnist_model.layers] == ["hidden", "output"]
    assert mnist_model.layers[0].tensors[0].shape == (3, 4)
    assert mnist_model.layers[1].tensors[0].shape == (2, 3)
    assert mnist_model.capture_batch_size == 4
    assert load_training_checkpoint(mnist_dir / "checkpoint.pt")["optimizer_state_dict"]
