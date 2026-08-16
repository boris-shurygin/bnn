"""PyTorch-инспектор для сериализуемого контракта модели.

Инспектор вызывается внутри worker-процесса в согласованной точке. Он не
возвращает ссылок на модель или тензоры: результат состоит только из frozen
dataclass контракта и JSON-совместимых значений. Тот же ``inspect_tensor``
можно переиспользовать в будущих live-событиях и файловых snapshots.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from bioplast.runner.contracts import (
    ConnectionSpec,
    ContractError,
    LayerSpec,
    ModelManifest,
    TensorSpec,
    TensorSummary,
    utc_offset_iso,
)


class InspectionError(ContractError):
    """Модель или тензор нельзя безопасно представить текущим контрактом."""


def inspect_tensor(
    name: str,
    role: str,
    tensor: Any,
    *,
    full_values_max_elements: int = 256,
) -> TensorSpec:
    """Скопировать безопасное представление одного torch-тензора.

    Полные значения встраиваются только для маленького конечного dense-тензора.
    Для больших или содержащих NaN/Inf тензоров остаётся конечная статистика и
    явная причина, почему values не были включены.
    """
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise InspectionError(f"{name!r} не является torch.Tensor")
    if full_values_max_elements < 0:
        raise InspectionError("full_values_max_elements не может быть отрицательным")
    if tensor.layout != torch.strided:
        raise InspectionError(f"тензор {name!r} имеет неподдерживаемый layout {tensor.layout}")
    if tensor.device.type == "meta":
        raise InspectionError(f"тензор {name!r} на meta device не содержит значений")

    detached = tensor.detach()
    summary = _summarize_tensor(detached)
    dtype = str(detached.dtype).removeprefix("torch.")
    values: Any = None
    omitted_reason: str | None = None

    if summary.element_count == 0:
        omitted_reason = "empty_tensor"
    elif summary.non_finite_count:
        omitted_reason = "non_finite"
    elif summary.element_count > full_values_max_elements:
        omitted_reason = "size_limit"
    elif detached.is_complex():
        omitted_reason = "unsupported_json_dtype"
    else:
        # Копия разрывает связь с изменяемым storage модели. Для CUDA операция
        # синхронизируется до возврата, поэтому наружу выходит законченный снимок.
        values = detached.to(device="cpu", copy=True).tolist()

    return TensorSpec(
        name=name,
        role=role,
        shape=tuple(int(size) for size in detached.shape),
        dtype=dtype,
        requires_grad=bool(tensor.requires_grad),
        value_mode="full" if values is not None else "summary",
        summary=summary,
        values=values,
        values_omitted_reason=omitted_reason,
    )


def inspect_model(
    model: Any,
    *,
    run_id: str,
    model_name: str,
    example_args: tuple[Any, ...],
    example_kwargs: Mapping[str, Any] | None = None,
    layer_ids: Mapping[str, str] | None = None,
    activations: Mapping[str, str | None] | None = None,
    full_values_max_elements: int = 256,
    capture_phase: str | None = None,
    step: int | None = None,
) -> ModelManifest:
    """Снять граф исполненных leaf-модулей и состояние их тензоров.

    Формы получаются отдельным observation-forward в ``eval`` и
    ``inference_mode``. Исходные training-флаги всех модулей восстанавливаются.
    Связи v1 следуют фактическому порядку вызовов leaf-модулей; для backprop-XOR
    это точный линейный граф. Функциональная активация задаётся через
    ``activations`` и относится к выходу соответствующего слоя.
    """
    import torch
    from torch import nn

    if not isinstance(model, nn.Module):
        raise InspectionError("model должен быть torch.nn.Module")
    if not example_args:
        raise InspectionError("для определения форм нужен хотя бы один example arg")

    aliases = dict(layer_ids or {})
    activation_names = dict(activations or {})
    kwargs = dict(example_kwargs or {})
    observed: list[tuple[str, nn.Module, tuple[int | None, ...], tuple[int | None, ...]]] = []
    calls: defaultdict[str, int] = defaultdict(int)
    handles = []

    modules = [
        (name, module)
        for name, module in model.named_modules()
        if not tuple(module.children())
    ]
    if not modules:
        raise InspectionError("в модели нет leaf-модулей для инспекции")

    for module_name, module in modules:
        base_id = aliases.get(module_name, module_name or "model")

        def capture(_module, inputs, output, *, name=module_name, base=base_id):
            input_tensor = _first_tensor(inputs)
            output_tensor = _first_tensor(output)
            if input_tensor is None or output_tensor is None:
                raise InspectionError(f"слой {name or '<root>'!r} не имеет tensor input/output")
            calls[base] += 1
            layer_id = base if calls[base] == 1 else f"{base}#{calls[base]}"
            observed.append(
                (
                    layer_id,
                    _module,
                    _shape_with_dynamic_batch(input_tensor),
                    _shape_with_dynamic_batch(output_tensor),
                )
            )

        handles.append(module.register_forward_hook(capture))

    training_flags = {module: module.training for module in model.modules()}
    try:
        model.eval()
        with torch.inference_mode():
            model(*example_args, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
        for module, training in training_flags.items():
            module.training = training

    if not observed:
        raise InspectionError("observation-forward не вызвал ни одного leaf-модуля")

    layers: list[LayerSpec] = []
    for layer_id, module, input_shape, output_shape in observed:
        original_name = _original_module_name(module, modules)
        tensors = tuple(
            [
                inspect_tensor(
                    name,
                    "parameter",
                    parameter,
                    full_values_max_elements=full_values_max_elements,
                )
                for name, parameter in module.named_parameters(recurse=False)
            ]
            + [
                inspect_tensor(
                    name,
                    "buffer",
                    buffer,
                    full_values_max_elements=full_values_max_elements,
                )
                for name, buffer in module.named_buffers(recurse=False)
            ]
        )
        layers.append(
            LayerSpec(
                layer_id=layer_id,
                layer_type=type(module).__name__,
                input_shape=input_shape,
                output_shape=output_shape,
                activation=activation_names.get(
                    original_name,
                    activation_names.get(layer_id),
                ),
                parameter_count=sum(
                    parameter.numel() for parameter in module.parameters(recurse=False)
                ),
                tensors=tensors,
            )
        )

    connections = tuple(
        ConnectionSpec(source=source.layer_id, target=target.layer_id)
        for source, target in zip(layers, layers[1:])
    )
    return ModelManifest(
        run_id=run_id,
        model_name=model_name,
        layers=tuple(layers),
        connections=connections,
        captured_at=utc_offset_iso(),
        capture_phase=capture_phase,
        step=step,
    )


def _summarize_tensor(tensor: Any) -> TensorSummary:
    import torch

    element_count = tensor.numel()
    if element_count == 0:
        return TensorSummary(0, 0, 0, None, None, None, None, None, None, None)

    finite_mask = torch.isfinite(tensor)
    finite_count = int(finite_mask.sum().item())
    non_finite_count = element_count - finite_count
    sparsity = float((tensor == 0).sum().item() / element_count)
    if finite_count == 0:
        return TensorSummary(
            element_count,
            0,
            non_finite_count,
            None,
            None,
            None,
            None,
            None,
            None,
            sparsity,
        )

    finite = tensor[finite_mask]
    numeric = finite.abs().to(torch.float64) if finite.is_complex() else finite.to(torch.float64)
    return TensorSummary(
        element_count=element_count,
        finite_count=finite_count,
        non_finite_count=non_finite_count,
        minimum=_finite_float(numeric.min().item()),
        maximum=_finite_float(numeric.max().item()),
        mean=_finite_float(numeric.mean().item()),
        std=_finite_float(numeric.std(correction=0).item()),
        l1_norm=_finite_float(numeric.abs().sum().item()),
        l2_norm=_finite_float(torch.linalg.vector_norm(numeric).item()),
        sparsity=sparsity,
    )


def _finite_float(value: float) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise InspectionError("агрегат тензора переполнился до NaN/Inf")
    return converted


def _first_tensor(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    elif isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _shape_with_dynamic_batch(tensor: Any) -> tuple[int | None, ...]:
    shape: list[int | None] = [int(size) for size in tensor.shape]
    if shape:
        shape[0] = None
    return tuple(shape)


def _original_module_name(module: Any, modules: list[tuple[str, Any]]) -> str:
    for name, candidate in modules:
        if candidate is module:
            return name
    raise InspectionError("внутренняя ошибка: исполненный модуль не найден")
