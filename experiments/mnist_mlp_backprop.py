"""Backprop-эталон на MNIST: 784 → 256 → 10. Ориентир ~98% на тесте.

Датасет целиком лежит на устройстве, батчи — срезы тензора; `DataLoader` не
используется сознательно (CLAUDE.md). Диагностика та же, что на XOR, — чтобы
локальное правило потом сравнивалось по одинаковым графикам.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from bioplast.data import load_mnist
from bioplast.diagnostics.probes import log_module_state
from bioplast.runner import ExperimentResult, ModelArtifacts

# Дублирование MLP с xor_backprop.py — намеренное: правило трёх, второй раз
# ещё не повод заводить общий модуль.


class MLP(nn.Module):
    def __init__(self, dims: list[int]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))

    def forward(self, x: torch.Tensor, collect: bool = False):
        acts: dict[str, torch.Tensor] = {}
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < last:
                x = torch.relu(x)
                if collect:
                    acts[f"h{i + 1}"] = x
        if collect:
            acts["out"] = x
            return x, acts
        return x


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch: int = 2048) -> float:
    correct = 0
    for start in range(0, x.shape[0], batch):
        logits = model(x[start : start + batch])
        correct += (logits.argmax(dim=1) == y[start : start + batch]).sum().item()
    return correct / x.shape[0]


def run(config: dict[str, Any], ctx) -> ExperimentResult:
    hidden = list(config.get("hidden", [256]))
    epochs = int(config.get("epochs", 10))
    batch_size = int(config.get("batch_size", 128))
    lr = float(config.get("lr", 1e-3))
    data_root = Path(config.get("data_root", "data/mnist"))
    device = ctx.device

    data = load_mnist(
        root=data_root,
        device=device,
        flatten=True,
        train_size=config.get("train_size"),
    )
    ctx.log.info(
        "MNIST: train %s, test %s, всё на %s",
        tuple(data.train_x.shape),
        tuple(data.test_x.shape),
        device,
    )

    dims = [data.input_dim, *hidden, data.num_classes]
    model = MLP(dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    ctx.log.info(
        "архитектура %s, веса на %s, batch=%d, lr=%g, epochs=%d",
        dims,
        next(model.parameters()).device,
        batch_size,
        lr,
        epochs,
    )

    generator = torch.Generator(device=device).manual_seed(ctx.seed)
    test_acc = 0.0
    train_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, seen, correct = 0.0, 0, 0
        acts: dict[str, torch.Tensor] = {}

        for xb, yb in data.batches(batch_size, shuffle=True, generator=generator):
            ctx.control.checkpoint(step=train_step, phase="train_batch")
            logits, acts = model(xb, collect=True)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.shape[0]
            correct += (logits.argmax(dim=1) == yb).sum().item()
            seen += xb.shape[0]
            train_step += 1

        model.eval()
        test_acc = evaluate(model, data.test_x, data.test_y)
        train_loss = total_loss / seen

        # градиенты последнего батча ещё в .grad — снимаем состояние после эпохи
        ctx.metrics.update(
            epoch,
            {
                "loss/train": train_loss,
                "acc/train": correct / seen,
                "acc/test": test_acc,
                **log_module_state(model, acts),
            },
        )
        ctx.log.info(
            "эпоха %2d: loss=%.4f, train=%.4f, test=%.4f", epoch, train_loss, correct / seen, test_acc
        )

    last_layer = len(model.layers) - 1
    single_hidden = last_layer == 1
    layer_ids = {}
    for index in range(len(model.layers)):
        if index == last_layer:
            layer_id = "output"
        elif single_hidden:
            layer_id = "hidden"
        else:
            layer_id = f"hidden_{index + 1}"
        layer_ids[f"layers.{index}"] = layer_id
    activations = {
        f"layers.{index}": "identity" if index == last_layer else "relu"
        for index in range(len(model.layers))
    }
    return ExperimentResult(
        final={
            "test_acc": test_acc,
            "train_acc": correct / seen,
            "train_loss": train_loss,
            "params": sum(p.numel() for p in model.parameters()),
        },
        model_artifacts=ModelArtifacts(
            model=model,
            optimizer=optimizer,
            example_args=(data.train_x[:batch_size],),
            layer_ids=layer_ids,
            activations=activations,
            step=epochs,
        ),
    )
