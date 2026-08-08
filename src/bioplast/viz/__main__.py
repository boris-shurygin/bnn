"""CLI визуализации: `uv run python -m bioplast.viz runs --all`.

Отдельный `__main__` нужен потому, что `bioplast.viz` реэкспортирует `plot_run`,
и запуск `-m bioplast.viz.plot` импортировал бы модуль дважды.
"""

import sys

from bioplast.viz.plot import main

if __name__ == "__main__":
    sys.exit(main())
