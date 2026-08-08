"""Визуализация, уровень 0: `metrics.json` → набор PNG. Никакого UI.

HTML-отчёт (jinja2 + plotly) появится в блоке 2, веб-приложение — не раньше
блока 4 и только если будет за чем наблюдать в динамике.
"""

from bioplast.viz.plot import plot_run

__all__ = ["plot_run"]
