"""Раннер: конфиг-файл → прогон → `runs/<id>/`.

Прогон полностью описывается JSON-конфигом; ручных шагов нет ни одного —
это условие того, что вечерние свипы и (позже) аренда облака вообще окупятся.
"""

from bioplast.runner.run import RunContext, config_slug, make_run_id, run_config
from bioplast.runner.queue import run_queue

__all__ = ["RunContext", "config_slug", "make_run_id", "run_config", "run_queue"]
