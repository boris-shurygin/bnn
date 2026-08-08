"""Загрузка датасетов целиком в память устройства.

`DataLoader` не используется: MNIST — 188 МБ во float32, CIFAR-10 — 614 МБ,
оба помещаются в 12 ГБ с запасом, а на таком масштабе оверхед загрузчика
доминирует над вычислениями. На Windows это тем более важно: `num_workers > 0`
порождает процессы через `spawn`.
"""

from bioplast.data.mnist import Dataset, load_mnist

__all__ = ["Dataset", "load_mnist"]
