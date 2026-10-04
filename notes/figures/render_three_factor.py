"""Векторные учебные схемы к ../three-factor-formulas.md.

Из корня проекта: uv run python notes/figures/render_three_factor.py
Для визуальной проверки: добавить --preview-dir runs/three-factor-figures
Зависимость: matplotlib из основного окружения проекта.
"""

from argparse import ArgumentParser
from math import cos, radians, sin
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as CurvePath


INK = "#203044"
MUTED = "#526274"
BLUE = "#096b8b"
ORANGE = "#a44812"
GREEN = "#277044"
PALE_BLUE = "#edf6fa"
PALE_ORANGE = "#fff3e9"
PALE_GREEN = "#edf7ef"
LINE = "#c8d2dc"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "mathtext.fontset": "dejavusans",
    "svg.fonttype": "path",  # Формулы и кириллица не зависят от шрифтов читателя.
    "svg.hashsalt": "bioplast-three-factor-v1",
    "axes.unicode_minus": True,
})


def text(ax, x, y, value, size=12, color=INK, ha="center", **kwargs):
    return ax.text(
        x, y, value, fontsize=size, color=color, ha=ha, va="center",
        linespacing=1.45, zorder=5, **kwargs,
    )


def box(ax, x, y, width, height, color=LINE, fill="white", radius=0.12):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.4, edgecolor=color, facecolor=fill, zorder=2,
    )
    ax.add_patch(patch)
    return patch


def circle(ax, x, y, value, radius=0.36, color=BLUE, fill=PALE_BLUE, size=15):
    ax.add_patch(Circle((x, y), radius, edgecolor=color, facecolor=fill,
                        linewidth=1.7, zorder=3))
    text(ax, x, y, value, size, color)


def port(x, y, radius, angle):
    """Точка на окружности: отдельный порт для каждой связи."""
    angle = radians(angle)
    return x + radius * cos(angle), y + radius * sin(angle)


def arrow(ax, points, color=BLUE, dashed=False, width=1.8, *, halo=False):
    """Прямая (2 точки) или кубический сплайн Безье (1 + 3n точек).

    Для каждого сегмента после начальной точки идут две управляющие и
    конечная точки. Общая стрелка сохраняет непрерывный штрих вдоль кривой.
    Светлая обводка отделяет пересекающиеся, но не соединённые маршруты.
    """
    if len(points) == 2:
        codes = [CurvePath.MOVETO, CurvePath.LINETO]
    elif len(points) >= 4 and (len(points) - 1) % 3 == 0:
        codes = [CurvePath.MOVETO] + [CurvePath.CURVE4] * (len(points) - 1)
    else:
        raise ValueError("Стрелке нужны 2 точки или 1 + 3n точек сплайна.")
    style = (0, (4, 3)) if dashed else "-"
    patch = FancyArrowPatch(
        path=CurvePath(points, codes), arrowstyle="-|>", mutation_scale=13,
        linewidth=width, color=color, linestyle=style,
        capstyle="round", joinstyle="round", zorder=1,
    )
    if halo:
        patch.set_path_effects([
            path_effects.Stroke(linewidth=width + 3.5, foreground="white"),
            path_effects.Normal(),
        ])
    ax.add_patch(patch)


def canvas(title, subtitle):
    fig = plt.figure(figsize=(12, 8), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1), xlim=(0, 12), ylim=(0, 8))
    ax.set_axis_off()
    text(ax, 0.45, 7.65, title, 20, ha="left", weight="bold")
    text(ax, 0.45, 7.24, subtitle, 11.5, MUTED, ha="left")
    ax.plot([0.45, 11.55], [0.65, 0.65], color=LINE, lw=1)
    for x, color, dashed, label in (
        (0.5, BLUE, False, "forward / активность"),
        (4.35, ORANGE, True, "learning / модуляция"),
        (8.4, GREEN, False, "изменение веса"),
    ):
        arrow(ax, [(x, 0.35), (x + 0.45, 0.35)], color, dashed)
        text(ax, x + 0.58, 0.35, label, 10.5, color, ha="left")
    return fig, ax


def neuron():
    fig, ax = canvas(
        "1. Один нейрон: два входных канала",
        "Один пример, один скрытый ReLU-нейрон j слоя l. Индекс l на рисунке опущен.",
    )
    background = box(ax, 4.05, 2.5, 4.4, 3.9, fill="#f8fafc")
    background.set_zorder(0)
    text(ax, 4.28, 6.12, "Нейрон j", 12, MUTED, ha="left")
    box(ax, 4.5, 5.1, 3.5, 0.75, ORANGE, PALE_ORANGE)
    text(ax, 6.25, 5.47, r"$a_j^{\mathrm{dev}}=a_j^{\mathrm{raw}}-\beta_j$", 17, ORANGE)
    text(ax, 5.1, 6.72, r"$a_j^{\mathrm{raw}}$", 17, ORANGE)
    arrow(ax, [(5.1, 6.48), (5.1, 5.88)], ORANGE, True)
    text(ax, 7.45, 6.72, r"$\beta_j$", 17, ORANGE)
    arrow(ax, [(7.45, 6.48), (7.45, 5.88)], ORANGE, True)
    box(ax, 9, 5.05, 2.35, 0.85, ORANGE, PALE_ORANGE)
    text(ax, 10.175, 5.48, r"$e_j=a_j^{\mathrm{dev}}\psi(z_j)$", 15, ORANGE)
    arrow(ax, [(8, 5.47), (9, 5.47)], ORANGE, True)
    circle(ax, 5.25, 3.65, r"$z_j$", 0.6, size=21)
    box(ax, 6.6, 3.25, 1.4, 0.8, BLUE, PALE_BLUE)
    text(ax, 7.3, 3.65, r"$f(z_j)$", 19, BLUE)
    arrow(ax, [(5.85, 3.65), (6.6, 3.65)])
    arrow(ax, [(8, 3.65), (10.15, 3.65)])
    text(ax, 10.75, 3.65, r"$\mathrm{post}_j$", 20, BLUE)
    text(ax, 10.0, 3.18, "выход вперёд", 11, BLUE)
    box(ax, 6.6, 4.25, 1.4, 0.55, ORANGE, PALE_ORANGE)
    text(ax, 7.3, 4.53, r"$\psi(z_j)$", 17, ORANGE)
    arrow(ax, [port(5.25, 3.65, 0.62, 60), (5.75, 4.53),
               (6.05, 4.53), (6.6, 4.53)])
    arrow(ax, [(8, 4.53), (8.95, 4.53), (10.175, 4.53), (10.175, 5.05)],
          ORANGE, True)
    text(ax, 6.2, 2.86, r"$z_j=\sum_i w_{ji}^{\mathrm{ff}}\,\mathrm{pre}_i+b_j$", 16)

    for y, label, angle in [(4.7, "1", 145), (3.65, "i", 180), (2.6, "n", 215)]:
        circle(ax, 0.95, y, rf"$\mathrm{{pre}}_{label}$", 0.48, size=15)
        end = port(5.25, 3.65, 0.62, angle)
        if label == "i":
            arrow(ax, [(1.43, y), end], width=2)
        else:
            # Горизонтальный участок проходит через синапс; далее — плавный
            # вход в собственный порт сомы, без общего наконечника.
            arrow(ax, [(1.43, y), (1.95, y), (2.55, y), (2.95, y),
                       (3.65, y), port(5.25, 3.65, 1.27, angle), end], width=1.3)
        ax.add_patch(Circle((2.95, y), 0.095, color=BLUE, zorder=4))
    text(ax, 2.85, 4.02, r"$w_{ji}^{\mathrm{ff}}$", 17, BLUE)
    text(ax, 1.9, 5.28, "Базальные входы", 12, BLUE)
    text(ax, 2.3, 3.27, "синапс i → j", 10.5, BLUE)

    box(ax, 3.8, 1.08, 5.15, 0.87, GREEN, PALE_GREEN)
    text(ax, 6.375, 1.52,
         r"$\Delta w_{ji}=\eta\,\mathrm{pre}_i\,\psi(z_j)\,a_j^{\mathrm{dev}}$",
         18, GREEN)
    arrow(ax, [port(0.95, 3.65, 0.5, -35), (1.95, 2.95),
               (1.4, 1.52), (3.8, 1.52)], halo=True)
    text(ax, 2.55, 1.14, "локальный вход", 10.5, BLUE)
    arrow(ax, [(11.35, 5.48), (11.65, 5.48), (11.7, 5.1), (11.7, 4.4),
               (11.7, 2.8), (11.5, 1.52), (8.95, 1.52)], ORANGE, True)
    arrow(ax, [(4.35, 1.95), (4.15, 2.6), (3.55, 3.24), (3.06, 3.56)],
          GREEN, halo=True)
    text(ax, 9.55, 2.29, "Новый вес действует\nсо следующего forward.", 11, GREEN)
    return fig


def outer_product():
    fig, ax = canvas(
        "2. Из сигналов нейронов — в обновления синапсов",
        "Учебный слой: 3 входа, 2 нейрона, B = 1. Строка j — получатель, столбец i — источник.",
    )
    text(ax, 1.9, 6.6, "После локального gate", 12, ORANGE)
    text(ax, 1.9, 6.12, r"$z=(2,-1),\quad \psi(z)=(1,0)$", 14)
    text(ax, 7.8, 6.57, r"$\mathrm{pre}^{\mathsf{T}}=(1,\ 2,\ -1)$", 20, BLUE)
    text(ax, 4.1, 5.58, r"$e$", 19, ORANGE)
    x0, top, cw, ch = 5.2, 5.4, 1.9, 1.03
    for i, value in enumerate([1, 2, -1]):
        x = x0 + (i + 0.5) * cw
        circle(ax, x, 5.85, rf"${value}$", 0.3, size=16)
        arrow(ax, [(x, 5.54), (x, 5.41)])
    for j, (e, vals) in enumerate([(0.4, [0.4, 0.8, -0.4]), (0, [0, 0, 0])]):
        y = top - (j + 0.5) * ch
        circle(ax, 1.25, y, rf"$j={j+1}$", 0.42, size=13)
        text(ax, 2.8, y, "gate = 1" if j == 0 else "gate = 0", 12, ORANGE)
        circle(ax, 4.1, y, rf"${e:g}$", 0.35, ORANGE, PALE_ORANGE, 17)
        arrow(ax, [(4.47, y), (5.18, y)], ORANGE, True)
        for i, v in enumerate(vals):
            fill = PALE_GREEN if (j, i) == (0, 1) else ("#f6f8fa" if j else "white")
            ax.add_patch(Rectangle((x0+i*cw, top-(j+1)*ch), cw, ch,
                                   facecolor=fill, edgecolor=LINE, lw=1.3, zorder=2))
            text(ax, x0+(i+0.5)*cw, y, rf"${v:g}$", 22,
                 GREEN if (j, i) == (0, 1) else INK)
    text(ax, 7.95, 2.98, r"$U^W=e\,\mathrm{pre}^{\mathsf{T}}\quad(2\times3)$", 19)
    box(ax, 0.7, 1.75, 10.6, 0.86, GREEN, PALE_GREEN)
    text(ax, 6, 2.18,
         r"$\Delta w_{12}=0.1\times0.4\times2=0.08"
         r"\qquad w_{12}: 0.50\ \longrightarrow\ 0.58$", 19, GREEN)
    text(ax, 6, 1.17, "Вторая строка нулевая: ReLU-gate закрыт. "
         "Отрицательный вход меняет знак в третьем столбце.", 12)
    return fig


def network():
    fig, ax = canvas(
        "3. От ошибки выхода — к апикали скрытых нейронов",
        "XOR: два входа, один logit. Для наглядности показан скрытый слой H = 3.",
    )
    xs, xh, xo = 1.0, 4.5, 8.0
    ins, hidden = [3.1, 4.9], [2.35, 4.0, 5.65]
    for i, y in enumerate(ins):
        circle(ax, xs, y, rf"$x_{i+1}$", 0.37)
    for j, y in enumerate(hidden):
        background = box(ax, xh-1.22, y-0.49, 1.75, 1.51,
                         fill="#f8fafc")
        background.set_zorder(0)
        for index, iy in enumerate(ins):
            start_angle = [-32, 8, 40][j] if index == 0 else [-40, -8, 32][j]
            end_angle = 207 if index == 0 else 153
            approach_y = y + (-0.34 if index == 0 else 0.34)
            # Выравниваем путь до входа в серую рамку: кривая не должна
            # прятаться за апикальной площадкой соседнего канала.
            arrow(ax, [port(xs, iy, 0.39, start_angle),
                       port(xs, iy, 1.04, start_angle),
                       (2.5, approach_y), (3.2, approach_y),
                       (3.55, approach_y), port(xh, y, 0.7, end_angle),
                       port(xh, y, 0.405, end_angle)], width=1.1, halo=True)
        end_angle = [212, 180, 148][j]
        arrow(ax, [(xh+0.4, y), (xh+1.4, y),
                   port(xo, 4, 1.17, end_angle),
                   port(xo, 4, 0.425, end_angle)], width=1.3, halo=True)
        circle(ax, xh, y, rf"$h_{j+1}$", 0.38)
        # Обратная связь заканчивается в отдельной апикальной площадке.
        box(ax, xh-1.1, y+0.49, 1.22, 0.45, ORANGE, PALE_ORANGE)
        text(ax, xh-0.49, y+0.715, rf"$a_{{1,{j+1}}}^{{\mathrm{{dev}}}}$", 13, ORANGE)
    circle(ax, xo, 4.0, r"$z_2$", 0.4)
    box(ax, 9.0, 3.6, 1.1, 0.8, BLUE, PALE_BLUE)
    text(ax, 9.55, 4, r"$\sigma$", 21, BLUE)
    arrow(ax, [(8.41, 4), (9.0, 4)])
    arrow(ax, [(10.1, 4), (10.9, 4)])
    text(ax, 11.2, 4, r"$p$", 22, BLUE)
    text(ax, 8.0, 3.27, "logit", 11, BLUE)
    text(ax, 10.5, 3.27, "вероятность", 11, BLUE)
    text(ax, 2.5, 6.12, r"$W_1^{\mathrm{ff}}$", 19, BLUE)
    text(ax, 7.0, 5.18, r"$W_2^{\mathrm{ff}}$", 19, BLUE)
    text(ax, 1.0, 2.34, r"$\mathrm{pre}_1=x$", 13, BLUE)
    text(ax, 6.0, 1.65, r"$\mathrm{pre}_2=h=\mathrm{post}_1$", 15, BLUE)

    box(ax, 8.2, 5.67, 3.25, 0.81, ORANGE, PALE_ORANGE)
    text(ax, 9.825, 6.075, r"$e_2=y-p$", 21, ORANGE)
    text(ax, 9.25, 6.91, r"$y$", 17, ORANGE)
    arrow(ax, [(9.25, 6.71), (9.25, 6.49)], ORANGE, True)
    arrow(ax, [(11.2, 4.27), (11.2, 5.67)])
    box(ax, 5.6, 5.68, 1.85, 0.78, ORANGE, PALE_ORANGE)
    text(ax, 6.525, 6.075, r"$W_1^{\mathrm{fb}}$", 19, ORANGE)
    arrow(ax, [(8.2, 6.075), (7.45, 6.075)], ORANGE, True)
    # Три независимых маршрута из разных портов W_fb вместо общей шины.
    # Пересечения с forward имеют светлый зазор, а не точку соединения.
    feedback_routes = [
        [(5.6, 6.3), (5.2, 6.3), (5.12, 6.365), (4.65, 6.365)],
        [(5.6, 6.02), (5.3, 6.02), (5.25, 5.78), (5.25, 5.5),
         (5.25, 4.95), (5.15, 4.715), (4.65, 4.715)],
        [(6.3, 5.68), (6.3, 4.75), (6.0, 3.065), (4.65, 3.065)],
    ]
    for points in feedback_routes:
        arrow(ax, points, ORANGE, True, 1.4, halo=True)
    text(ax, 8.9, 2.45, "Апикальные площадки\nпередают сигнал в пластичность.", 11, ORANGE)
    text(ax, 5.95, 1.02,
         r"$a_1^{\mathrm{raw}}=\beta_1+e_2(W_1^{\mathrm{fb}})^{\mathsf{T}}"
         r"\qquad e_1=(a_1^{\mathrm{raw}}-\beta_1)\odot\psi(z_1)$",
         16, ORANGE)
    return fig


def trace():
    fig, ax = canvas(
        "4. Eligibility trace: память синапса во времени",
        "Один синапс, B = 1. Начальный след равен нулю; затухание λ = 0.5.",
    )
    xvals = [4.15, 7.0, 9.85]
    labels = [
        (5.42, r"$\psi(z_j)\,\mathrm{pre}_i$", "новая eligibility", BLUE),
        (4.28, r"$T_{ji}$", "накопленный след", INK),
        (3.14, r"$a_j^{\mathrm{dev}}$", "модуляция сейчас", ORANGE),
        (2.0, r"$U_{ji}=a_j^{\mathrm{dev}}T_{ji}$", "направление update", GREEN),
    ]
    for y, math, label, color in labels:
        text(ax, 0.6, y+0.13, math, 17, color, ha="left")
        text(ax, 0.6, y-0.21, label, 10.5, MUTED, ha="left")
    for x, label in zip(xvals, [r"$t_0$", r"$t_1$", r"$t_2$"]):
        text(ax, x, 6.7, label, 19)
    arrow(ax, [(3.5, 6.25), (10.8, 6.25)], MUTED)
    rows = [(5.42, [2, 0, 0], BLUE, PALE_BLUE),
            (4.28, [2, 1, 0.5], INK, "#f3f6f9"),
            (3.14, [0, 0, 0.4], ORANGE, PALE_ORANGE),
            (2.0, [0, 0, 0.2], GREEN, PALE_GREEN)]
    for y, vals, color, fill in rows:
        for x, val in zip(xvals, vals):
            box(ax, x-0.65, y-0.37, 1.3, 0.74, color, fill)
            text(ax, x, y, rf"${val:g}$", 22, color)
    for start, end in zip(xvals, xvals[1:]):
        arrow(ax, [(start+0.65, 4.28), (end-0.65, 4.28)], MUTED)
        text(ax, (start+end)/2, 4.6, r"$\times\lambda$", 12, MUTED)
    arrow(ax, [(10.5, 4.28), (11.7, 4.28), (11.7, 2.0), (10.5, 2.0)], MUTED)
    arrow(ax, [(9.85, 2.77), (9.85, 2.38)], ORANGE, True)
    text(ax, 6.0, 1.04,
         "В t₂ нового вклада нет, но память сохранилась: 0.4 × 0.5 = 0.2.",
         13, GREEN)
    return fig


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    output = Path(__file__).resolve().parent
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
    for name, draw in [
        ("three-factor-neuron", neuron),
        ("three-factor-outer-product", outer_product),
        ("three-factor-network", network),
        ("three-factor-trace", trace),
    ]:
        fig = draw()
        fig.savefig(output / f"{name}.svg", metadata={"Date": None, "Title": name})
        if args.preview_dir:
            fig.savefig(args.preview_dir / f"{name}.png", dpi=140)
        plt.close(fig)
        print(f"{name}.svg")


if __name__ == "__main__":
    main()
