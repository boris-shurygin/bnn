"""Shared payload for resumable training experiments.

The runner owns durable generations, while an experiment owns its model and
loop-local cursor.  This module only standardises the pieces every training
adapter must preserve: config identity, cooperative-control state and all RNGs.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from bioplast.runner.contracts import ContractError
from bioplast.runner.lifecycle import load_recovery, write_recovery

TRAINING_RECOVERY_VERSION = 1


def config_fingerprint(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(config),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def recovery_interval(config: Mapping[str, Any], *, default: int) -> int:
    """Resolve adapter checkpoint frequency from config or the shared env var."""
    raw = config.get("recovery_every_steps")
    if raw is None:
        raw = os.environ.get("BIOPLAST_RECOVERY_EVERY_STEPS", default)
    if isinstance(raw, bool):
        raise ContractError("recovery_every_steps должен быть положительным целым")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ContractError("recovery_every_steps должен быть положительным целым") from exc
    if value < 1:
        raise ContractError("recovery_every_steps должен быть положительным целым")
    return value


def capture_control_state(control: Any) -> dict[str, Any]:
    return {
        "last_seq": int(control.last_seq),
        "mode": control.mode.value,
        "delay_ms": int(control.delay_ms),
        "input_seq": int(control.input_seq),
        "input_values": (
            list(control.input_values) if control.input_values is not None else None
        ),
    }


def capture_rng_state(*, include_cuda: bool = True) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    cuda_states = (
        torch.cuda.get_rng_state_all()
        if include_cuda and torch.cuda.is_available()
        else []
    )
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_states,
        "torch_cuda_device_count": len(cuda_states),
    }


def restore_rng_state(value: Mapping[str, Any]) -> None:
    try:
        random.setstate(value["python"])
        numpy_state = value["numpy"]
        if not isinstance(numpy_state, Mapping):
            raise TypeError("numpy state is not a mapping")
        keys = numpy_state["keys"]
        if not isinstance(keys, torch.Tensor):
            raise TypeError("numpy keys are not a tensor")
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                keys.detach().cpu().numpy().astype(np.uint32, copy=False),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        cpu_state = value["torch_cpu"]
        if not isinstance(cpu_state, torch.Tensor):
            raise TypeError("torch CPU state is not a tensor")
        torch.set_rng_state(cpu_state.detach().cpu())

        cuda_states = value.get("torch_cuda", [])
        expected_cuda = int(value.get("torch_cuda_device_count", len(cuda_states)))
        if not isinstance(cuda_states, list) or len(cuda_states) != expected_cuda:
            raise TypeError("torch CUDA state list is inconsistent")
        if cuda_states:
            if not torch.cuda.is_available():
                raise ContractError(
                    "recovery содержит CUDA RNG, но CUDA недоступна при возобновлении"
                )
            if torch.cuda.device_count() != expected_cuda:
                raise ContractError(
                    "число CUDA-устройств не совпадает с training recovery"
                )
            torch.cuda.set_rng_state_all([state.detach().cpu() for state in cuda_states])
    except ContractError:
        raise
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ContractError(f"некорректное состояние RNG в training recovery: {exc}") from exc


def write_training_recovery(
    ctx: Any,
    config: Mapping[str, Any],
    *,
    adapter: str,
    cursor: str,
    progress: Mapping[str, Any],
    training: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": TRAINING_RECOVERY_VERSION,
        "kind": "training_recovery",
        "adapter": adapter,
        "config_sha256": config_fingerprint(config),
        "device": str(ctx.device),
        "training": dict(training),
        # Reading CUDA RNG eagerly initialises CUDA. CPU experiments deliberately
        # stay on CPU; CUDA experiments preserve every device state.
        "rng": capture_rng_state(include_cuda=str(ctx.device).startswith("cuda")),
        "control": capture_control_state(ctx.control),
    }
    return write_recovery(
        ctx.run_dir,
        payload,
        adapter=adapter,
        attempt=ctx.attempt,
        cursor=cursor,
        command_seq=ctx.control.last_seq,
        event_seq=0,
        progress=progress,
        retain_generations=3,
    )


def load_training_recovery(
    ctx: Any,
    config: Mapping[str, Any],
    *,
    adapter: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state, payload = load_recovery(ctx.run_dir, expected_adapter=adapter)
    if (
        payload.get("schema_version") != TRAINING_RECOVERY_VERSION
        or payload.get("kind") != "training_recovery"
        or payload.get("adapter") != adapter
    ):
        raise ContractError("неподдерживаемый training recovery payload")
    if payload.get("config_sha256") != config_fingerprint(config):
        raise ContractError("config.json изменился после записи training recovery")
    if payload.get("device") != str(ctx.device):
        raise ContractError(
            f"training recovery создан для device={payload.get('device')!r}, "
            f"получен {ctx.device!r}"
        )
    training = payload.get("training")
    rng = payload.get("rng")
    if not isinstance(training, dict) or not isinstance(rng, Mapping):
        raise ContractError("training recovery не содержит training/RNG state")
    restore_rng_state(rng)
    return state, training
