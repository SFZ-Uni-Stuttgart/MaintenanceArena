"""Figure data of the article's synthetic figures and, with the ``plots`` extra, simple figures.

    python experiments/synthetic/figures.py --output outputs/figures --source outputs/source \
        --input-sets outputs/input-sets --preparation outputs/preparation --adaptation outputs/adaptation --plots

Every figure is computed when its inputs are given (the directories are outputs of
the other runners):

* degradation laws, always: ``curves.json``, the health of five lives per law with
  the last safe order (end of life minus 13 steps), drawn as damage d = -log5(h)
  as in Fig. 3 of the article;
* first orders on one life per source law (``--source``): ``order_cases.json``,
  the ten source policies and the Time rule (stored as Rule) on the 800
  validation lives;
* preparation-time policy maps (``--preparation``): ``preparation_maps.csv`` with
  the state count and mean P(ORDER) of every occupied bin, ``preparation_native.csv``
  with the mean at every distinct (health, damage rate) coordinate (the support of
  the 0.5 contour) and ``preparation_maps.json`` with the support;
* adaptation policy maps (``--source`` and ``--adaptation``): the same three files
  (``adaptation_*``) for the source policies, the fine-tuned policies and their
  difference, plus the last safe order of every life and the source and
  adapted Time rule thresholds;
* failed training episodes (``--input-sets`` and/or ``--adaptation``):
  ``training_failures.csv`` per run and bin and ``training_failures.json`` with the
  median of the runs in every bin.

``--plots`` also draws ``fig_*.png`` with matplotlib (``pip install
maintenance-arena[plots]``). These are plain renderings of the data above, not
the article's typeset figures.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np

import common as c
from maintenance_arena import analysis as an
from maintenance_arena.banks import concatenate_banks
from maintenance_arena.metrics import read_outcomes

SOURCE_LAWS = ("linear_reference", "concave_running_in", "convex_accelerating_fatigue", "bearing_late_knee")
NEW_LAWS = ("paris_inspired", "two_knee", "gompertz", "power_law")
NAMES = {"linear_reference": "Linear", "concave_running_in": "Concave", "convex_accelerating_fatigue": "Convex",
         "bearing_late_knee": "Late-knee", "paris_inspired": "Paris-inspired", "two_knee": "Two-knee",
         "gompertz": "Gompertz", "power_law": "Power-law"}
MAP_COLUMNS = ("law", "panel", "health_low", "health_high", "rate_low", "rate_high", "states", "mean_p_order")


# --- Curves and order cases ---

def curves(config):
    spec = config["curves"]
    laws = {}
    for bank_spec in spec["banks"]:
        bank = c.bank_from_spec(c.resolve_bank(config, bank_spec))
        for path in an.health_paths(bank, per_law=spec["lives_per_law"], mobilization_lead=spec["mobilization_lead"]):
            laws.setdefault(path["law"], []).append(path)
    return {"description": spec["description"], "laws": {law: laws[law] for law in (*SOURCE_LAWS, *NEW_LAWS)}}


def order_cases(config, source):
    spec = config["order_cases"]
    source_config = c.load_config(source / "configuration.json")
    lead = source_config["environment"].get("mobilization_lead", 12)
    bank = c.bank_from_spec(c.resolve_bank(source_config, source_config["development_bank"]))
    rows = read_outcomes(source / spec["outcomes"])
    lives = {int(row["unit"]): (row["group"], int(row["last_productive_step"]) + 1)
             for row in rows if row["method"] == "Last-safe reference"}
    cases = []
    for law in SOURCE_LAWS:
        unit = min((abs(eol - spec["end_of_life"]), unit) for unit, (name, eol) in lives.items() if name == law)[1]
        part = bank["pairs"][unit]["active"]
        if part["family_id"] != law or int(bank["pairs"][unit]["pair_index"]) != unit:
            raise ValueError("the outcome table and the development bank disagree")
        health = np.asarray(part["health"], dtype=np.float32)
        last_safe = int(part["eol_operating_age"]) - 1 - lead
        orders = []
        for row in rows:
            if int(row["unit"]) == unit and row["method"] in spec["methods"]:
                step = None if row["order_step"] == "" else int(row["order_step"])
                orders.append({"method": row["method"], "run": row["run"], "order_step": step,
                               "margin": None if step is None else last_safe - step,
                               "health_at_order": None if step is None else float(health[step]),
                               "failed": row["failed"] == "True"})
        margins = Counter(order["margin"] for order in orders if order["method"] == spec["methods"][0])
        cases.append({"law": law, "pair_index": unit, "end_of_life": int(part["eol_operating_age"]),
                      "last_safe_order_age": last_safe, "health": [float(v) for v in health], "orders": orders,
                      "policies_per_margin": {str(k): v for k, v in sorted(margins.items(), key=lambda kv: str(kv[0]))}})
    return {"description": spec["description"], "cases": cases}


# --- Policy maps ---

def map_panel(law, panel, x, y, mean, x_edges, y_edges, where=None):
    """Bin rows, support and arrays (counts, means, native coordinates and means) of one map panel."""
    inside = an.in_view(x, y, x_edges, y_edges, where=where)
    counts, means = an.binned_mean(x, y, mean, x_edges, y_edges, where=where)
    coordinates, native = an.native_means(np.asarray(x, dtype=np.float64)[inside],
                                          np.asarray(y, dtype=np.float64)[inside], mean[inside])
    rows = [{"law": law, "panel": panel, "health_low": float(x_edges[i]), "health_high": float(x_edges[i + 1]),
             "rate_low": float(y_edges[j]), "rate_high": float(y_edges[j + 1]), "states": int(counts[i, j]),
             "mean_p_order": float(means[i, j])} for i, j in zip(*np.nonzero(counts))]
    support = {"states": int(len(x) if where is None else np.sum(where)), "states_in_view": int(inside.sum()),
               "occupied_bins": int((counts > 0).sum()), "unique_native_coordinates": int(len(coordinates))}
    return rows, support, (counts, means, coordinates, native)


def native_table(laws, panels_of, panels):
    """Columns of the native means: law, health, damage rate and the mean of every panel (shared coordinates)."""
    rows = []
    for law in laws:
        coordinates = panels[(law, panels_of[0])][2]
        if any(not np.array_equal(panels[(law, panel)][2], coordinates) for panel in panels_of):
            raise AssertionError("the panels of a law must share their native coordinates")
        means = [panels[(law, panel)][3] for panel in panels_of]
        rows += [{"law": law, "health": float(a), "damage_rate": float(b),
                  **{f"mean_p_order {panel}": float(values[k]) for panel, values in zip(panels_of, means)}}
                 for k, (a, b) in enumerate(coordinates)]
    return rows, ("law", "health", "damage_rate", *[f"mean_p_order {panel}" for panel in panels_of])


def policies(paths):
    from maintenance_arena.learning.collector import policy_agent, single_thread
    with single_thread():
        return [policy_agent(path) for path in paths]


def probabilities(agents, states):
    from maintenance_arena.learning.collector import single_thread
    with single_thread():
        return an.order_probabilities(agents, states)


def preparation_maps(config, preparation):
    spec = config["preparation_maps"]
    roles = c.load_config(preparation / "configuration.json")["roles"]
    first, *others = [c.bank_from_spec(c.resolve_bank(config, entry)) for entry in spec["banks"]]
    bank = first
    for other in others:
        bank = concatenate_banks(bank, other)
    chosen = an.first_lives_per_law(bank, spec["lives_per_law"])
    lives = sorted(index for law in SOURCE_LAWS for index in chosen[law])
    records = c.read_fits(preparation, [f"L{lead}/{role}" for lead in spec["preparation_times"] for role in roles])
    x_edges, y_edges = an.edges(**spec["health"]), an.edges(**spec["damage_rate"])
    rows, support, panels = [], {}, {}
    for lead in spec["preparation_times"]:
        states = an.map_states(bank, mobilization_lead=lead, lives=lives)
        agents = policies([c.checkpoint_path(preparation, records[f"L{lead}/{role}"], spec["milestone"])
                           for role in roles])
        mean = an.ensemble_mean(probabilities(agents, states.states))
        for law in SOURCE_LAWS:
            panel = f"L={lead}"
            r, s, arrays = map_panel(law, panel, states.health, states.damage_rate, mean, x_edges, y_edges,
                                     where=states.valid & (states.law == law))
            rows += r
            support.setdefault(law, {})[panel] = s
            panels[(law, panel)] = arrays
    record = {"description": spec["description"], "lives_per_law": {law: len(chosen[law]) for law in SOURCE_LAWS},
              "lives": {law: chosen[law] for law in SOURCE_LAWS}, "roles": roles, "milestone": spec["milestone"],
              "health_edges": x_edges.tolist(), "damage_rate_edges": y_edges.tolist(), "support": support}
    natives = native_table(SOURCE_LAWS, [f"L={lead}" for lead in spec["preparation_times"]], panels)
    return record, rows, natives, (x_edges, y_edges, panels)


def adaptation_maps(config, source, adaptation):
    spec = config["adaptation_maps"]
    roles = c.load_config(adaptation / "configuration.json")["roles"]
    sources = c.read_fits(source, [f"source/{role}" for role in roles])
    tuned = c.read_fits(adaptation, [f"{law}/{role}/fine-tuned" for law in spec["laws"] for role in roles])
    source_agents = policies([c.checkpoint_path(source, sources[f"source/{role}"]) for role in roles])
    x_edges, y_edges = an.edges(**spec["health"]), an.edges(**spec["damage_rate"])
    rows, support, panels, deadline = [], {}, {}, {}
    for law in spec["laws"]:
        bank = c.bank_from_spec(c.resolve_bank(config, spec["bank"], law=law))
        states = an.map_states(bank, min_age=spec["min_age"])
        x, y = states.health, states.observed_rate
        tuned_agents = policies([c.checkpoint_path(adaptation, tuned[f"{law}/{role}/fine-tuned"],
                                                   spec["fine_tuned_milestone"]) for role in roles])
        means = {"source": an.ensemble_mean(probabilities(source_agents, states.states)),
                 "fine-tuned": an.ensemble_mean(probabilities(tuned_agents, states.states))}
        for panel, mean in means.items():
            r, s, arrays = map_panel(law, panel, x, y, mean, x_edges, y_edges)
            rows += r
            support.setdefault(law, {})[panel] = s
            panels[(law, panel)] = arrays
        counts, tuned_means = panels[(law, "fine-tuned")][:2]
        difference = tuned_means - panels[(law, "source")][1]
        rows += [{"law": law, "panel": "difference", "health_low": float(x_edges[i]), "health_high": float(x_edges[i + 1]),
                  "rate_low": float(y_edges[j]), "rate_high": float(y_edges[j + 1]), "states": int(counts[i, j]),
                  "mean_p_order": float(difference[i, j])} for i, j in zip(*np.nonzero(counts))]
        panels[(law, "difference")] = (counts, difference, None, None)
        # The last safe order of every life (remaining life L + 1), with the rate channel over L in float64.
        last = states.remaining == states.mobilization_lead + 1
        deadline[law] = [[float(a), float(b) / states.mobilization_lead]
                         for a, b in zip(x[last], states.states[last, 1].astype(np.float64))]
    rules = c.load_config(adaptation / "rules.json")
    thresholds = {"source": {key: rules["source_rule"][key] for key in ("family", "threshold")},
                  **{law: {key: rules["recalibrated_rules"][law][key] for key in ("family", "threshold")}
                     for law in spec["laws"]}}
    record = {"description": spec["description"], "roles": roles, "health_edges": x_edges.tolist(),
              "damage_rate_edges": y_edges.tolist(), "support": support, "last_safe_orders": deadline,
              "rule_thresholds": thresholds}
    natives = native_table(spec["laws"], ["source", "fine-tuned"], panels)
    return record, rows, natives, (x_edges, y_edges, panels)


# --- Training failures ---

def training_logs(directory, label):
    return json.loads((Path(directory) / "fits" / label / "training.json").read_text(encoding="utf-8"))["training_log"]


def training_failures(config, input_sets, adaptation):
    spec = config["training_failures"]
    rows, medians = [], {}

    def add(panel, arm, runs, budget, width):
        shares = []
        for run, log in runs:
            grid, share, steps = an.failure_share_curve(log, budget, width)
            shares.append(share)
            rows.extend({"panel": panel, "arm": arm, "run": run, "bin_start": int(grid[i]), "bin_end": int(grid[i + 1]),
                         "failure_pct": float(share[i]), "steps_per_episode": float(steps[i])} for i in range(len(share)))
        medians.setdefault(panel, {})[arm] = {"bin_edges": [int(v) for v in grid],
                                              "median_failure_pct": [float(v) for v in np.median(shares, axis=0)]}
    if input_sets is not None:
        part = spec["input_sets"]
        roles = c.load_config(input_sets / "configuration.json")["roles"]
        for name in part["names"]:
            add("input sets", name, [(role, training_logs(input_sets, f"{name}/{role}")) for role in roles],
                part["budget"], part["width"])
    if adaptation is not None:
        part = spec["adaptation"]
        roles = c.load_config(adaptation / "configuration.json")["roles"]
        for law in NEW_LAWS:
            for arm in part["arms"]:
                add(law, arm, [(role, training_logs(adaptation, f"{law}/{role}/{arm}")) for role in roles],
                    part["budget"], part["width"])
    return {"description": spec["description"], "medians": medians}, rows


# --- Plain figures (matplotlib) ---

def contour_mesh(coordinates, scales, max_span):
    from matplotlib.tri import Triangulation
    if coordinates is None or len(coordinates) < 3 or np.linalg.matrix_rank(coordinates - coordinates[0]) < 2:
        return None
    candidate = Triangulation(*(coordinates / scales).T)
    mask = np.any(np.ptp(coordinates[candidate.triangles] / scales, axis=1) > max_span, axis=1)
    return None if mask.all() else Triangulation(coordinates[:, 0], 1000 * coordinates[:, 1],
                                                 triangles=candidate.triangles, mask=mask)


def rule_line(ax, rule, health, style):
    """A Time rule threshold in the (health, 1000 x damage rate) plane: projected time (1 + ln h / ln 5) / rate."""
    if rule["family"] == "health":
        ax.axvline(rule["threshold"], color="grey", lw=.8, ls=style)
    else:
        ax.plot(health, 1000 * (1 + np.log(health) / np.log(5.)) / rule["threshold"], color="grey", lw=.8, ls=style)


def draw_maps(path, grid, rows_by, columns, config, diverging=(), rules=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x_edges, y_edges, panels = grid
    scales = np.asarray([x_edges[1] - x_edges[0], y_edges[1] - y_edges[0]])
    figure, axes = plt.subplots(len(rows_by), len(columns), figsize=(2.2 * len(columns), 1.9 * len(rows_by)),
                                sharex=True, sharey=True, squeeze=False)
    for i, panel in enumerate(rows_by):
        signed = panel in diverging
        for j, law in enumerate(columns):
            ax = axes[i, j]
            counts, means, coordinates, native = panels[(law, panel)]
            image = ax.pcolormesh(x_edges, 1000 * y_edges, np.ma.masked_invalid(means.T),
                                  cmap="RdBu_r" if signed else "viridis", vmin=-1 if signed else 0, vmax=1)
            mesh = contour_mesh(coordinates, scales, config["contour"]["max_span_bins"])
            if mesh is not None and np.nanmin(native) < config["contour"]["level"] < np.nanmax(native):
                ax.tricontour(mesh, native, levels=[config["contour"]["level"]], colors="black", linewidths=1.)
            if rules is not None:
                health = np.linspace(x_edges[0], x_edges[-1], 200)
                rule_line(ax, rules["source"], health, ":")
                rule_line(ax, rules[law], health, "-")
                ax.set_ylim(1000 * y_edges[0], 1000 * y_edges[-1])
            ax.set_title(f"{NAMES[law]}, {panel}", fontsize=8)
            ax.tick_params(labelsize=7)
        bar = figure.colorbar(image, ax=axes[i, :], shrink=.9)
        bar.set_label("Difference" if signed else "Mean P(ORDER)", fontsize=7)
        bar.ax.tick_params(labelsize=6)
    figure.supxlabel("Health", fontsize=8)
    figure.supylabel("Damage rate (1e-3 per step)", fontsize=8)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def draw(output, results, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    laws = results["curves"]["laws"]
    # The law names describe damage, so the paths are drawn as damage d = -log5(h), failure at d = 1.
    damage = lambda health: -np.log(np.asarray(health, dtype=float)) / np.log(5.)
    figure, axes = plt.subplots(2, 4, figsize=(9, 4.2), sharex=True, sharey=True)
    for ax, law in zip(axes.ravel(), (*SOURCE_LAWS, *NEW_LAWS)):
        for path in laws[law]:
            ax.plot(damage(path["health"]), color="black", lw=.7)
            ax.plot(path["last_safe_order_age"], damage(path["health_at_last_safe_order"]), "o", color="firebrick", ms=3)
        ax.axhline(1., color="grey", ls="--", lw=.6)
        ax.set_title(NAMES[law], fontsize=8)
    figure.supxlabel("Operating age (steps)", fontsize=8)
    figure.supylabel("Damage", fontsize=8)
    figure.savefig(output / "fig_curves.png", dpi=150)
    plt.close(figure)
    if "order_cases" in results:
        figure, axes = plt.subplots(1, 4, figsize=(9, 2.4), sharey=True)
        for ax, case in zip(axes, results["order_cases"]["cases"]):
            ax.plot(case["health"], color="black", lw=.8)
            ax.axvline(case["last_safe_order_age"], color="grey", ls="--", lw=.6)
            for order in case["orders"]:
                if order["order_step"] is not None:
                    ax.plot(order["order_step"], order["health_at_order"], "|" if order["method"] == "PPO" else "x",
                            color="tab:blue" if order["method"] == "PPO" else "tab:orange", ms=8)
            ax.set_title(f"{NAMES[case['law']]} (life {case['pair_index']})", fontsize=8)
            ax.set_xlim(case["last_safe_order_age"] - 40, case["end_of_life"])
        figure.supxlabel("Operating age (steps); ticks: PPO first orders, cross: Time rule", fontsize=8)
        figure.savefig(output / "fig_order_cases.png", dpi=150)
        plt.close(figure)
    if "preparation_maps" in results:
        draw_maps(output / "fig_preparation_maps.png", results["preparation_maps"],
                  [f"L={lead}" for lead in config["preparation_maps"]["preparation_times"]], SOURCE_LAWS, config)
    if "adaptation_maps" in results:
        draw_maps(output / "fig_adaptation_maps.png", results["adaptation_maps"], ["source", "fine-tuned", "difference"],
                  config["adaptation_maps"]["laws"], config, diverging=("difference",),
                  rules=results["adaptation_rules"])
    if "training_failures" in results:
        medians = results["training_failures"]
        figure, axes = plt.subplots(1, len(medians), figsize=(2.2 * len(medians), 2.2), sharey=True, squeeze=False)
        for ax, (panel, arms) in zip(axes[0], medians.items()):
            for arm, values in arms.items():
                edges = np.asarray(values["bin_edges"])
                ax.plot(.5 * (edges[1:] + edges[:-1]), values["median_failure_pct"], lw=1., label=arm)
            ax.set_title(NAMES.get(panel, panel), fontsize=8)
            ax.legend(fontsize=6)
        figure.supxlabel("Training steps", fontsize=8)
        figure.supylabel("Failed episodes (%)", fontsize=8)
        figure.savefig(output / "fig_training_failures.png", dpi=150)
        plt.close(figure)


# --- Command line ---

def main():
    parser = c.parser(__doc__.splitlines()[0])
    for name, text in (("--source", "output of source_learning.py"), ("--input-sets", "output of input_sets.py"),
                       ("--preparation", "output of preparation_time.py"), ("--adaptation", "output of adaptation.py")):
        parser.add_argument(name, type=Path, default=None, help=text)
    parser.add_argument("--plots", action="store_true", help="also draw plain figures (needs matplotlib)")
    arguments = parser.parse_args()
    if arguments.budget is not None or arguments.roles is not None:
        raise SystemExit("--budget and --roles do not apply to the figure data")
    for name in ("source", "input_sets", "preparation", "adaptation"):
        value = getattr(arguments, name)
        if value is not None and not (value / "configuration.json").is_file():
            raise SystemExit(f"{value} is not a runner output")
    if arguments.plots:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            raise SystemExit("--plots needs matplotlib: pip install maintenance-arena[plots]") from None
    output, config, record = c.start(arguments, c.HERE / "figures.json")
    results, drawn = {"curves": curves(config)}, {}
    c.write_json(output / "curves.json", results["curves"])
    if arguments.source is not None:
        results["order_cases"] = order_cases(config, arguments.source)
        c.write_json(output / "order_cases.json", results["order_cases"])
    for name, function, needed in (("preparation_maps", preparation_maps, (arguments.preparation,)),
                                   ("adaptation_maps", adaptation_maps, (arguments.source, arguments.adaptation))):
        if all(value is not None for value in needed):
            summary, rows, (natives, columns), drawn[name] = function(config, *needed)
            if name == "adaptation_maps":
                drawn["adaptation_rules"] = summary["rule_thresholds"]
            stem = name.split("_")[0]
            c.write_json(output / f"{name}.json", summary)
            c.write_csv(output / f"{name}.csv", rows, MAP_COLUMNS)
            c.write_csv(output / f"{stem}_native.csv", natives, columns)
    if arguments.input_sets is not None or arguments.adaptation is not None:
        summary, rows = training_failures(config, arguments.input_sets, arguments.adaptation)
        results["training_failures"] = summary["medians"]
        c.write_json(output / "training_failures.json", summary)
        c.write_csv(output / "training_failures.csv", rows,
                    ("panel", "arm", "run", "bin_start", "bin_end", "failure_pct", "steps_per_episode"))
    if arguments.plots:
        draw(output, {**results, **drawn}, config)
    c.finish(output, record, figures=sorted([*results, *(name for name in drawn if name.endswith("_maps"))]))


if __name__ == "__main__":
    main()
