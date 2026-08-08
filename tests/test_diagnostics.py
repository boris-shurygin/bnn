"""Диагностика v0: регистратор метрик и замеры состояния сети."""

import torch
from torch import nn

from bioplast.diagnostics import MetricsRecorder, activation_stats, sparsity, weight_norms
from bioplast.diagnostics.metrics import group_series


def test_recorder_collects_rows():
    rec = MetricsRecorder()
    rec.update(0, {"loss/train": 1.0, "acc/test": 0.5})
    rec.update(1, {"loss/train": torch.tensor(0.25), "acc/test": 0.9})

    assert rec.rows[1]["loss/train"] == 0.25  # тензор приведён к числу
    assert rec.series("acc/test") == ([0, 1], [0.5, 0.9])
    assert len(rec) == 2


def test_recorder_tolerates_missing_keys():
    rec = MetricsRecorder()
    rec.update(0, {"a": 1.0})
    rec.update(1, {"b": 2.0})

    assert rec.series("a") == ([0], [1.0])
    assert set(rec.keys()) == {"epoch", "a", "b"}


def test_group_series_splits_by_prefix():
    rows = [{"epoch": 0, "loss/train": 1.0, "w_norm/fc1": 2.0, "plain": 3.0}]

    grouped = group_series(rows)

    assert set(grouped) == {"loss", "w_norm", "scalar"}
    assert grouped["loss"]["loss/train"] == ([0], [1.0])
    assert grouped["scalar"]["plain"] == ([0], [3.0])


def test_sparsity_counts_zeros():
    assert sparsity(torch.tensor([0.0, 0.0, 1.0, 2.0])) == 0.5
    assert sparsity(torch.zeros(4)) == 1.0


def test_weight_norms_skip_bias():
    model = nn.Sequential(nn.Linear(3, 2))

    norms = weight_norms(model)

    assert list(norms) == ["w_norm/0"]
    assert norms["w_norm/0"] > 0


def test_activation_stats_expose_runaway_signal():
    """`act_max` — сторож runaway: без вычитания baseline он должен расти."""
    stats = activation_stats("h1", torch.tensor([[0.0, 3.0], [0.0, 1.0]]))

    assert stats["act_max/h1"] == 3.0
    assert stats["act_sparsity/h1"] == 0.5
    assert stats["act_rms/h1"] > 0
