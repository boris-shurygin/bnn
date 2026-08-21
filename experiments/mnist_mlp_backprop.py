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
from bioplast.runner import (
    ContractError,
    ExperimentResult,
    ModelArtifacts,
    load_training_recovery,
    recovery_interval,
    write_training_recovery,
)

# Дублирование MLP с xor_backprop.py — намеренное: правило трёх, второй раз
# ещё не повод заводить общий модуль.

RECOVERY_ADAPTER = "mnist_mlp_backprop_v1"


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


def _gradient_state(model: nn.Module) -> dict[str, torch.Tensor | None]:
    return {
        name: parameter.grad.detach().cpu().clone() if parameter.grad is not None else None
        for name, parameter in model.named_parameters()
    }


def _restore_gradients(model: nn.Module, state: dict[str, Any]) -> None:
    parameters = dict(model.named_parameters())
    if set(state) != set(parameters):
        raise ContractError("MNIST recovery содержит несовместимый набор gradients")
    for name, value in state.items():
        if value is None:
            parameters[name].grad = None
        elif isinstance(value, torch.Tensor):
            parameters[name].grad = value.to(
                device=parameters[name].device,
                dtype=parameters[name].dtype,
            )
        else:
            raise ContractError(f"gradient {name!r} в MNIST recovery не является tensor")


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
    epoch = 1
    batch_start = 0
    order: torch.Tensor | None = None
    total_loss, seen, correct = 0.0, 0, 0
    acts: dict[str, torch.Tensor] = {}
    train_loss = float("nan")
    train_acc = 0.0
    every = recovery_interval(config, default=50)

    recovery_path = ctx.run_dir / "recovery" / "state.json"
    if recovery_path.is_file():
        recovery_state, training = load_training_recovery(
            ctx, config, adapter=RECOVERY_ADAPTER
        )
        try:
            model.load_state_dict(training["model_state_dict"], strict=True)
            optimizer.load_state_dict(training["optimizer_state_dict"])
            generator_state = training["generator_state"]
            if not isinstance(generator_state, torch.Tensor):
                raise TypeError("generator_state is not a tensor")
            generator.set_state(generator_state.detach().cpu())
            epoch = int(training["epoch"])
            batch_start = int(training["batch_start"])
            train_step = int(training["global_step"])
            raw_order = training.get("permutation")
            order = raw_order.to(device=device) if isinstance(raw_order, torch.Tensor) else None
            total_loss = float(training["total_loss"])
            seen = int(training["seen"])
            correct = int(training["correct"])
            test_acc = float(training["test_acc"])
            train_loss = float(training["train_loss"])
            train_acc = float(training["train_acc"])
            raw_acts = training["last_acts"]
            gradients = training["gradients"]
            metrics_rows = training["metrics_rows"]
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise ContractError(f"некорректный MNIST training recovery: {exc}") from exc
        batches_per_epoch = (data.train_x.shape[0] + batch_size - 1) // batch_size
        completed_batches = (
            (batch_start + batch_size - 1) // batch_size if order is not None else 0
        )
        expected_step = (epoch - 1) * batches_per_epoch + completed_batches
        permutation_valid = order is None or (
            order.ndim == 1
            and order.numel() == data.train_x.shape[0]
            and torch.equal(
                torch.sort(order.detach().cpu()).values,
                torch.arange(data.train_x.shape[0]),
            )
        )
        if (
            not isinstance(raw_acts, dict)
            or not isinstance(gradients, dict)
            or not isinstance(metrics_rows, list)
            or not 1 <= epoch <= epochs + 1
            or not 0 <= batch_start <= data.train_x.shape[0]
            or (order is None and batch_start != 0)
            or (
                order is not None
                and batch_start != data.train_x.shape[0]
                and batch_start % batch_size != 0
            )
            or not permutation_valid
            or len(metrics_rows) != epoch - 1
            or train_step != expected_step
            or (order is not None and seen != batch_start)
        ):
            raise ContractError("MNIST training recovery содержит неверный progress cursor")
        acts = {
            str(name): value.to(device=device)
            for name, value in raw_acts.items()
            if isinstance(value, torch.Tensor)
        }
        if len(acts) != len(raw_acts):
            raise ContractError("MNIST recovery содержит не-tensor activation")
        _restore_gradients(model, gradients)
        ctx.metrics.rows = list(metrics_rows)
        ctx.log.info(
            "MNIST training recovery #%s: epoch=%d, batch_start=%d, global_step=%d",
            recovery_state.get("generation"),
            epoch,
            batch_start,
            train_step,
        )
    else:
        write_training_recovery(
            ctx,
            config,
            adapter=RECOVERY_ADAPTER,
            cursor="before_epoch:1",
            progress={"epoch": 1, "batch": 0, "global_step": 0},
            training={
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "generator_state": generator.get_state(),
                "epoch": 1,
                "batch_start": 0,
                "global_step": 0,
                "permutation": None,
                "total_loss": 0.0,
                "seen": 0,
                "correct": 0,
                "test_acc": test_acc,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "last_acts": {},
                "gradients": _gradient_state(model),
                "metrics_rows": [],
            },
        )

    def save_recovery(cursor: str) -> None:
        write_training_recovery(
            ctx,
            config,
            adapter=RECOVERY_ADAPTER,
            cursor=cursor,
            progress={
                "epoch": epoch,
                "batch": batch_start // batch_size,
                "global_step": train_step,
            },
            training={
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "generator_state": generator.get_state(),
                "epoch": epoch,
                "batch_start": batch_start,
                "global_step": train_step,
                "permutation": order.detach().cpu() if order is not None else None,
                "total_loss": total_loss,
                "seen": seen,
                "correct": correct,
                "test_acc": test_acc,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "last_acts": {
                    name: value.detach().cpu().clone() for name, value in acts.items()
                },
                "gradients": _gradient_state(model),
                "metrics_rows": list(ctx.metrics.rows),
            },
        )

    n = data.train_x.shape[0]
    while epoch <= epochs:
        model.train()
        if order is None:
            order = torch.randperm(n, device=data.train_x.device, generator=generator)
            batch_start = 0
            total_loss, seen, correct = 0.0, 0, 0
            acts = {}

        for start in range(batch_start, n, batch_size):
            ctx.control.checkpoint(step=train_step, phase="train_batch")
            idx = order[start : start + batch_size]
            xb, yb = data.train_x[idx], data.train_y[idx]
            logits, acts = model(xb, collect=True)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.shape[0]
            correct += (logits.argmax(dim=1) == yb).sum().item()
            seen += xb.shape[0]
            train_step += 1
            batch_start = min(start + batch_size, n)
            if train_step % every == 0 or batch_start >= n:
                save_recovery(f"after_train_batch:{train_step}")

        model.eval()
        test_acc = evaluate(model, data.test_x, data.test_y)
        train_loss = total_loss / seen
        train_acc = correct / seen

        # градиенты последнего батча ещё в .grad — снимаем состояние после эпохи
        ctx.metrics.update(
            epoch,
            {
                "loss/train": train_loss,
                "acc/train": train_acc,
                "acc/test": test_acc,
                **log_module_state(model, acts),
            },
        )
        ctx.log.info(
            "эпоха %2d: loss=%.4f, train=%.4f, test=%.4f", epoch, train_loss, train_acc, test_acc
        )
        completed_epoch = epoch
        epoch += 1
        batch_start = 0
        order = None
        save_recovery(f"after_epoch:{completed_epoch}")

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
            "train_acc": train_acc,
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
