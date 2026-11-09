"""Draw the data figures of the README from the saved outcomes and the scenario banks.

Usage from the project root, with the ``plots`` extra installed::

    python tools/readme_figures.py                 # writes docs/figures/*.svg
    python tools/readme_figures.py --png           # also writes PNG copies next to them

The order margins, the new laws and the transfer figure read the first orders in
``paper/outcomes`` through ``paper/rebuild_tables.py``, so they show the stored results of
the article. The degradation laws are rebuilt from the seeds of the synthetic figure
runner. The colours follow the roles of the article's figures: blue for PPO, graphite for
the Time rule, grey for a reference design and burgundy only for late orders.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import io
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT / "paper"), str(PROJECT / "experiments" / "synthetic")]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

import rebuild_tables as rt  # noqa: E402

INK, TEXT2, MUTED, GRID = "#34434B", "#5B6A72", "#9AA6AE", "#DDE4E8"
BLUE, BURGUNDY, LATE_FILL, PALE = "#2B779E", "#982B45", "#F7ECEF", "#EDF4F7"
CONTEXT = "#D5DCE0"
WIDTH = 8.0
L = rt.PREPARATION
SOURCE_LAWS = ("linear_reference", "concave_running_in", "convex_accelerating_fatigue", "bearing_late_knee")
NEW_LAWS = ("paris_inspired", "two_knee", "gompertz", "power_law")
NAMES = {"linear_reference": "Linear", "concave_running_in": "Concave", "convex_accelerating_fatigue": "Convex",
         "bearing_late_knee": "Late-knee", "paris_inspired": "Paris-inspired", "two_knee": "Two-knee",
         "gompertz": "Gompertz", "power_law": "Power-law"}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9.5, "text.color": INK, "axes.labelcolor": INK,
    "axes.labelsize": 9.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK,
    "ytick.labelcolor": INK, "xtick.labelsize": 9, "ytick.labelsize": 9, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3, "legend.frameon": False,
    "legend.fontsize": 9, "figure.facecolor": "white", "savefig.facecolor": "white",
    "svg.fonttype": "path", "svg.hashsalt": "maintenancearena", "axes.unicode_minus": True,
})


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(figure, output: Path, stem: str, png: bool) -> None:
    # Through a buffer, so the SVG keeps LF line endings on every platform (.gitattributes)
    # and loses the trailing spaces that matplotlib leaves after path commands.
    buffer = io.StringIO()
    figure.savefig(buffer, format="svg", metadata={"Date": None})
    text = "\n".join(line.rstrip() for line in buffer.getvalue().split("\n"))
    (output / f"{stem}.svg").write_bytes(text.encode("utf-8"))
    if png:
        figure.savefig(output / f"{stem}.png", dpi=150)
    plt.close(figure)
    print(f"wrote {output / stem}.svg")


def margin(row) -> int | None:
    """Steps between the first order and the last safe order; negative when the order came too late."""
    return None if row.order_step is None else row.last_productive_step - L - row.order_step


# --- Degradation laws ---------------------------------------------------------------------------

def degradation_laws(output: Path, png: bool) -> None:
    import common
    import figures as synthetic
    laws = synthetic.curves(common.load_config(PROJECT / "experiments" / "synthetic" / "figures.json"))["laws"]
    damage = lambda health: -np.log(np.asarray(health, dtype=float)) / np.log(5.0)
    figure, axes = plt.subplots(2, 4, figsize=(WIDTH, 3.9), sharex=True, sharey=True)
    for ax, law in zip(axes.ravel(), (*SOURCE_LAWS, *NEW_LAWS)):
        for other in SOURCE_LAWS:
            for path in laws[other]:
                # Every fourth point of the grey context keeps its shape and the file small.
                ages = np.unique(np.r_[0:len(path["health"]):4, len(path["health"]) - 1])
                ax.plot(ages, damage(path["health"])[ages], color=CONTEXT, lw=0.7, zorder=1)
        for path in laws[law]:
            ax.plot(damage(path["health"]), color=INK, lw=1.1, zorder=2)
            ax.plot(path["last_safe_order_age"], damage(path["health_at_last_safe_order"]), "o", ms=4.6,
                    color=BURGUNDY, mec="white", mew=0.9, zorder=3)
        ax.axhline(1.0, color=TEXT2, lw=0.7, ls=(0, (3, 2)), zorder=1.5)
        ax.set_title(NAMES[law], loc="left", fontsize=9.5, color=INK, pad=3)
        ax.set(xlim=(0, 165), ylim=(-0.04, 1.06), xticks=(0, 50, 100, 150), yticks=(0, 0.5, 1),
               yticklabels=("0", "0.5", "1"))
        ax.tick_params(length=2.5)
    axes[0, 0].set_ylabel("Damage")
    axes[1, 0].set_ylabel("Damage")
    figure.text(0.5, 0.015, "Operating age (steps)", ha="center", fontsize=9.5)
    key = [Line2D([], [], color=INK, lw=1.1), Line2D([], [], color=CONTEXT, lw=1.6),
           Line2D([], [], ls="", marker="o", ms=4.6, color=BURGUNDY, mec="white", mew=0.9),
           Line2D([], [], color=TEXT2, lw=0.7, ls=(0, (3, 2)))]
    figure.legend(key, ["Five lives per law", "Lives of the source laws", "Last safe order", "Failure"],
                  loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), handlelength=1.8, columnspacing=1.6)
    figure.subplots_adjust(left=0.075, right=0.99, top=0.85, bottom=0.14, wspace=0.1, hspace=0.32)
    save(figure, output, "fig_degradation_laws", png)


# --- Order margins on the source laws -----------------------------------------------------------

def order_margins(output: Path, png: bool) -> None:
    rows = rt.read_study("source_comparison")
    groups = (("PPO, ten policies", "PPO", BLUE), ("Time rule, τ = 15", "Rule", INK),
              ("Time rule, τ = 13", "Naive projection rule", INK))
    steps = np.arange(-2, 4)
    figure, axes = plt.subplots(3, 1, figsize=(WIDTH, 3.6), sharex=True, sharey=True)
    for ax, (label, method, colour) in zip(axes, groups):
        margins = [margin(row) for row in rows if row.method == method]
        share = [100 * margins.count(int(step)) / len(margins) for step in steps]
        late = 100 * sum(1 for m in margins if m is None or m < 0) / len(margins)
        ax.axvspan(-2.6, -0.5, color=LATE_FILL, lw=0, zorder=0)
        ax.axvline(-0.5, color=BURGUNDY, lw=0.8, zorder=1)
        ax.bar(steps, share, width=0.56, color=colour, zorder=2)
        grid(ax)
        ax.set(xlim=(-2.6, 3.6), ylim=(0, 72), yticks=(0, 30, 60))
        ax.yaxis.tick_right()
        ax.text(-0.02, 0.5, label, transform=ax.transAxes, ha="right", va="center", fontsize=9.5)
        summary = f"{len(margins):,} orders, none late" if late == 0 else f"{len(margins):,} orders, {late:.0f}% late"
        ax.text(0.985, 0.62, summary, transform=ax.transAxes, ha="right", va="center", fontsize=9, color=TEXT2)
        ax.tick_params(axis="x", length=0)
    axes[0].text(-1.55, 66, "late", ha="center", va="top", fontsize=9, color=BURGUNDY)
    axes[0].text(-0.42, 66, "last safe order", ha="left", va="top", fontsize=9, color=TEXT2)
    axes[1].yaxis.set_label_position("right")
    axes[1].set_ylabel("Share of first orders (%)", rotation=270, labelpad=13)
    axes[-1].set_xticks(steps)
    axes[-1].set_xlabel("Order margin: steps between the first order and the last safe order")
    figure.subplots_adjust(left=0.17, right=0.93, top=0.97, bottom=0.15, hspace=0.25)
    save(figure, output, "fig_order_margins", png)


# --- Adaptation to the new laws -----------------------------------------------------------------

def new_laws(output: Path, png: bool) -> None:
    rows = rt.read_study("degradation_change")
    laws = ("Paris-inspired", "Two-knee", "Gompertz", "Power-law")
    designs = (("Source PPO, unchanged", "Source PPO", dict(marker="o", mfc="white", mec=BLUE, mew=1.3)),
               ("Fresh PPO", "Fresh PPO", dict(marker="^", color=MUTED, mec="white", mew=0.6, ms=7.2)),
               ("Fine-tuned PPO", "Fine-tuned PPO", dict(marker="o", color=BLUE, mec="white", mew=0.6)),
               ("Fine-tuned PPO + calibration", "Fine-tuned PPO, calibrated cut",
                dict(marker="D", color=BLUE, mec="white", mew=0.6, ms=5.6)),
               ("Time rule, adapted", "Recalibrated Rule", dict(marker="s", color=INK, mec="white", mew=0.6, ms=5.8)))
    values = {}
    for _, method, _ in designs:
        for law in (*laws, None):
            selected = [row for row in rows if row.method == method and (law is None or row.group == law)]
            values[method, law] = rt.summarize(selected)
    figure, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9), sharey=True, gridspec_kw={"width_ratios": (1.35, 1)})
    positions = {law: index for index, law in enumerate((*laws, None))}
    for ax, metric, limit, label in ((axes[0], "failures", 80, "Failures (%)"),
                                     (axes[1], "unused_life", 4.5, "Unused life (%)")):
        for offset, (_, method, style) in zip((-0.28, -0.14, 0.0, 0.14, 0.28), designs):
            ys = [positions[law] + offset for law in (*laws, None)]
            xs = [values[method, law][metric] for law in (*laws, None)]
            ax.plot(xs, ys, ls="", **{"ms": 6.4, **style}, zorder=3)
        ax.axhline(len(laws) - 0.5, color=MUTED, lw=0.6)
        grid(ax, "x")
        ax.set(xlim=(-0.02 * limit, limit), xlabel=label)
        ax.tick_params(axis="y", length=0)
    axes[0].set(yticks=range(len(laws) + 1), yticklabels=[*laws, "All four"], ylim=(len(laws) + 0.5, -0.6))
    handles = [Line2D([], [], ls="", **{"ms": 6.4, **style}) for _, _, style in designs]
    figure.legend(handles, [name for name, _, _ in designs], loc="upper center", ncol=5,
                  bbox_to_anchor=(0.5, 1.0), handletextpad=0.2, columnspacing=0.9)
    figure.subplots_adjust(left=0.13, right=0.985, top=0.86, bottom=0.17, wspace=0.08)
    save(figure, output, "fig_new_laws", png)


# --- Transfer to XJTU-SY ------------------------------------------------------------------------

def median_order(orders):
    """Median first order over the seeds, a missing order ranking after every order."""
    ranked = sorted(orders, key=lambda order: (order is None, order or 0))
    return ranked[len(ranked) // 2]


def transfer(output: Path, png: bool) -> None:
    rows = [row for row in rt.read_study("bearings") if row.group == "XJTU-SY"]
    orders = defaultdict(list)
    for row in rows:
        orders[row.method, row.unit].append(row)
    bearings = sorted({row.unit for row in rows}, key=lambda unit: (orders["PPO: Full", unit][0].last_productive_step,
                                                                    unit))
    late_x = -9

    def position(method, bearing):
        life = orders[method, bearing][0].last_productive_step
        order = median_order([row.order_step for row in orders[method, bearing]])
        if order is None or order > life - L:
            return late_x, False
        return 100 * (life - L - order) / (life - L), True

    figure, ax = plt.subplots(figsize=(WIDTH, 4.6))
    ax.axvspan(-15, -3.5, color=LATE_FILL, lw=0, zorder=0)
    femto_longest = 468   # minutes; every FEMTO test ended within it
    longest = [index for index, bearing in enumerate(bearings)
               if orders["PPO: Full", bearing][0].last_productive_step > femto_longest]
    ax.axhspan(min(longest) - 0.5, max(longest) + 0.5, xmin=(-3.5 + 15) / 115, color=PALE, lw=0, zorder=0)
    for index, bearing in enumerate(bearings):
        (x_full, full_timely), (x_no_age, no_age_timely) = (position("PPO: Full", bearing),
                                                            position("PPO: No age", bearing))
        if full_timely and no_age_timely:
            ax.plot((x_full, x_no_age), (index, index), color=MUTED, lw=1.0, zorder=1)
        for x, timely, colour in ((x_full, full_timely, MUTED), (x_no_age, no_age_timely, BLUE)):
            if timely:
                ax.plot(x, index, "o", ms=7, color=colour, mec="white", mew=0.9, zorder=3)
        if not (full_timely and no_age_timely):
            ax.plot(late_x, index, "x", ms=6, color=BURGUNDY, mew=1.4, zorder=3)
    labels = [f"{bearing}   {orders['PPO: Full', bearing][0].last_productive_step:,} min" for bearing in bearings]
    ax.set(yticks=range(len(bearings)), yticklabels=labels, ylim=(len(bearings) - 0.4, -0.6),
           xlim=(-15, 100), xticks=(0, 20, 40, 60, 80, 100))
    ax.tick_params(axis="y", length=0)
    grid(ax, "x")
    ax.set_xlabel("Unused life at the median first order of five seeds (%)")
    ax.text(late_x, -0.95, "late", ha="center", va="bottom", fontsize=9, color=BURGUNDY)
    ax.text(99, min(longest) - 0.42, "longer than any FEMTO test", ha="right", va="top", fontsize=9, color=TEXT2)
    handles = [Line2D([], [], ls="", marker="o", ms=7, color=MUTED, mec="white"),
               Line2D([], [], ls="", marker="o", ms=7, color=BLUE, mec="white"),
               Line2D([], [], ls="", marker="x", ms=6, color=BURGUNDY, mew=1.4)]
    figure.legend(handles, ["Full: operating age in the prediction and the policy", "No age", "Late order"],
                  loc="upper center", ncol=3, bbox_to_anchor=(0.6, 1.0), handletextpad=0.3, columnspacing=1.8)
    figure.subplots_adjust(left=0.22, right=0.985, top=0.9, bottom=0.11)
    save(figure, output, "fig_transfer_age", png)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=PROJECT / "docs" / "figures")
    parser.add_argument("--png", action="store_true", help="also write PNG copies")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    for draw in (degradation_laws, order_margins, new_laws, transfer):
        draw(args.output, args.png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
