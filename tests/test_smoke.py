"""Проверка, что окружение собрано: пакет импортируется, torch на месте."""

import bioplast


def test_package_imports():
    assert bioplast.__version__


def test_torch_is_fp32_by_default():
    """Локальные правила считаются только в fp32 (см. CLAUDE.md)."""
    import torch

    assert torch.get_default_dtype() is torch.float32
