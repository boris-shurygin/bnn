"""Бинарный checkpoint отделён от JSON-контракта визуализатора."""

import torch
from torch import nn

from bioplast.runner import load_training_checkpoint, write_training_checkpoint
from experiments.xor_backprop import MLP


def test_training_checkpoint_round_trip_restores_model_and_optimizer(tmp_path):
    run_dir = tmp_path / "xor-run"
    run_dir.mkdir()
    model = MLP([2, 3, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    x = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    y = torch.ones((2, 1))

    loss = nn.BCEWithLogitsLoss()(model(x), y)
    loss.backward()
    optimizer.step()
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}

    path = write_training_checkpoint(
        run_dir,
        run_id="xor-run",
        experiment="xor_backprop",
        model_name="mlp-2-3-1",
        model=model,
        optimizer=optimizer,
        step=17,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    checkpoint = load_training_checkpoint(path)
    restored_model = MLP([2, 3, 1])
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.05)
    restored_model.load_state_dict(checkpoint["model_state_dict"])
    restored_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    assert checkpoint["schema_version"] == 1
    assert checkpoint["kind"] == "training_checkpoint"
    assert checkpoint["step"] == 17
    assert checkpoint["experiment"] == "xor_backprop"
    assert checkpoint["model_name"] == "mlp-2-3-1"
    assert restored_optimizer.state_dict()["state"]
    for name, value in restored_model.state_dict().items():
        assert torch.equal(value, expected[name])
    assert not path.with_suffix(".pt.tmp").exists()
