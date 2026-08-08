"""MNIST: скачивание, разбор IDX, кэш, выгрузка целиком на устройство.

Датасет качается скриптом и не лежит в репозитории — это условие того, что
прогон на арендованной машине не потребует ручных шагов.
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}


@dataclass
class Dataset:
    """Датасет целиком на устройстве. Батчи — срезы, не итератор загрузчика."""

    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor

    @property
    def input_dim(self) -> int:
        return int(np.prod(self.train_x.shape[1:]))

    @property
    def num_classes(self) -> int:
        return int(self.train_y.max().item()) + 1

    def batches(self, batch_size: int, shuffle: bool = True, generator=None):
        """Итератор по обучающей выборке срезами тензора, уже лежащего на GPU."""
        n = self.train_x.shape[0]
        order = (
            torch.randperm(n, device=self.train_x.device, generator=generator)
            if shuffle
            else torch.arange(n, device=self.train_x.device)
        )
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            yield self.train_x[idx], self.train_y[idx]


def download(root: Path | str = "data/mnist") -> Path:
    """Скачать сырые архивы, если их ещё нет. Возвращает папку."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for filename in FILES.values():
        target = root / filename
        if target.exists():
            continue
        url = MIRROR + filename
        tmp = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(target)  # атомарно: недокачанный файл не выдаст себя за готовый
    return root


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        zero, dtype_code, ndim = struct.unpack(">HBB", handle.read(4))
        if zero != 0 or dtype_code != 0x08:  # 0x08 — unsigned byte
            raise ValueError(f"{path.name}: не похоже на IDX-файл с uint8")
        shape = struct.unpack(f">{ndim}I", handle.read(4 * ndim))
        # copy(): frombuffer отдаёт read-only массив, torch на такой ругается
        return np.frombuffer(handle.read(), dtype=np.uint8).reshape(shape).copy()


def load_mnist(
    root: Path | str = "data/mnist",
    device: str | torch.device = "cpu",
    flatten: bool = True,
    normalize: bool = True,
    train_size: int | None = None,
) -> Dataset:
    """Загрузить MNIST целиком на `device`.

    `normalize` делит на 255; центрирование входа сюда сознательно не входит —
    это исследуемый фактор блока 3, а не деталь загрузчика.
    """
    root = Path(root)
    cache = root / "mnist.npz"

    if cache.exists():
        with np.load(cache) as blob:
            arrays = {key: blob[key] for key in FILES}
    else:
        download(root)
        arrays = {key: _read_idx(root / name) for key, name in FILES.items()}
        np.savez_compressed(cache, **arrays)

    def to_tensor(key: str) -> torch.Tensor:
        array = np.ascontiguousarray(arrays[key])
        if key.endswith("_y"):
            return torch.from_numpy(array).to(device=device, dtype=torch.long)
        # fp32 всегда: для локальных правил fp16 — тихая ловушка (см. CLAUDE.md)
        tensor = torch.from_numpy(array).to(device=device, dtype=torch.float32)
        if normalize:
            tensor = tensor / 255.0
        return tensor.reshape(tensor.shape[0], -1) if flatten else tensor.unsqueeze(1)

    data = Dataset(
        train_x=to_tensor("train_x"),
        train_y=to_tensor("train_y"),
        test_x=to_tensor("test_x"),
        test_y=to_tensor("test_y"),
    )
    if train_size is not None:
        data.train_x = data.train_x[:train_size]
        data.train_y = data.train_y[:train_size]
    return data
