"""Native, data-driven scientific figures with captions supplied by the caller."""
from pathlib import Path
import numpy as np


def plot_trajectory(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = result["trace"]
    if not rows:
        raise ValueError("A trajectory plot requires trace=True")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    time = np.asarray([r["calendar_start"] for r in rows])
    observations = np.asarray([r["observation"] for r in rows])
    figure, axes = plt.subplots(1, 3, figsize=(10.0, 2.8), layout="constrained")
    axes[0].plot(time, observations[:, 0], color="#287f8e")
    axes[0].set_ylabel("Observation coordinate 0")
    axes[1].step(time, [r["executed_action"] for r in rows], where="post", color="#3858a0", label="Executed")
    if result["filtered"]:
        axes[1].scatter(time, [r["proposed_action"] for r in rows], s=12, facecolors="none", edgecolors="#cf7344", label="Proposed")
        axes[1].legend(frameon=False)
    axes[1].set_ylabel("Action code")
    axes[2].plot([r["calendar_end"] for r in rows], np.cumsum([r["raw_profit"] for r in rows]), color="#287f8e")
    axes[2].set_ylabel("Cumulative raw profit (CU)")
    for letter, ax in zip("abc", axes):
        ax.set_xlabel("Calendar step")
        ax.text(.02, .97, f"({letter})", transform=ax.transAxes, va="top")
    for extension in ("pdf", "svg", "png"):
        figure.savefig(Path(output) / f"trajectory.{extension}", dpi=300)
    plt.close(figure)
def plot_comparison(result, directory):
    """Equal-scenario means and individual paired units from a core evaluation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    if not result["fixture"]["profile"].startswith("paper-core-"):
        raise ValueError("The timing/life comparison plot is defined for core profiles")
    policies = [row["policy"] for row in result["summary"]]
    fields = [("profit_per_1000_calendar_steps", "Profit / 1,000 calendar steps"),
              ("failures", "Failure per lifecycle"),
              ("normalized_discarded_life", "Unused life fraction"),
              ("first_trigger_true_rul", "Order true RUL (diagnostic)")]
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 4, figsize=(210/25.4, 68/25.4), sharey=True, layout="constrained")
    for letter, ax, (field, label) in zip("abcd", axes, fields):
        for index, policy in enumerate(policies):
            rows = [r for r in result["units"] if r["policy"] == policy and r[field] is not None]
            values = [r[field] for r in rows]
            if values:
                ax.scatter(values, index + np.linspace(-.16, .16, len(values)), s=10, color="#287f8e", alpha=.5)
                ax.scatter([np.mean(values)], [index], marker="|", s=130, color="#172d35", zorder=3)
        ax.set_xlabel(label)
        ax.text(.03, 1.02, f"({letter})", transform=ax.transAxes)
        ax.grid(axis="x", alpha=.15)
    axes[0].set_yticks(range(len(policies)), [p.replace("_", " ") for p in policies])
    axes[0].invert_yaxis()
    for extension in ("pdf", "svg", "png"):
        fig.savefig(Path(directory) / f"comparison.{extension}", dpi=300)
    plt.close(fig)
    (Path(directory) / "comparison_caption.md").write_text(
        "Each dot is one paired synthetic lifecycle; dark ticks are equal-lifecycle means. "
        "True RUL and unused life are evaluator diagnostics, never extra policy inputs. "
        "These fresh software scenarios are pooled; no family-equal aggregation is applied.\n", encoding="utf-8")

