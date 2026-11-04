"""Plotting functions for Assignment 2."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_feeder_import(
    community,
    results,
    output_dir,
):
    """
    Question 1: Compare hourly feeder imports against the DSO capacity limit.

    Parameters
    ----------
    community:
        CommunityData object containing the DSO capacity limit.

    results:
        Dictionary mapping model names to optimization results.
        Example: {"R": results_R, "I": results_I}

    output_dir:
        Directory where figures are saved.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capacity = np.asarray(
        community.contract.capacity_limit_kW,
        dtype=float,
    )

    hours = np.arange(1, len(capacity) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    for model_name, model_results in results.items():

        if model_name == "C":

            # Community-meter import after internal netting
            feeder_import = np.asarray(
                model_results.community_import_kW,
                dtype=float,
            )

        else:

            # R and I: sum of individual gross imports
            feeder_import = np.sum(
                [
                    r.import_kW
                    for r in model_results.values()
                ],
                axis=0,
            )

        ax.step(
            hours,
            feeder_import,
            where="mid",
            linewidth=2,
            label=f"Model {model_name}",
        )
        
    # DSO capacity limit
    ax.step(
        hours,
        capacity,
        where="mid",
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="DSO capacity limit",
    )

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Feeder import [kW]")

    # Dynamic title
    model_names = ", ".join(f"Model {name}" for name in results)

    ax.set_title(
        f"Hourly feeder import – {model_names}"
    )

    ax.set_xticks(hours)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir / "feeder_import.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir / "feeder_import.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_settlement(result, output_dir):
    """Plot Q1 hourly prices (DKK/kWh) and daily member gains (DKK).

    Each benefit panel contains one slope class and four asset classes. Positive
    bars mean a member gains from C's dual settlement against the named model.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hours = np.array([row["hour"] for row in result.hourly])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    series = [
        ("day_ahead_price_DKK_per_kWh", "Day-ahead", "#777777", ":"),
        ("discounted_import_price_DKK_per_kWh", "Discounted import", "#1f77b4", "--"),
        ("discounted_export_price_DKK_per_kWh", "Discounted export", "#ff7f0e", "--"),
        ("internal_price_DKK_per_kWh", "Internal settlement", "#2ca02c", "-"),
    ]
    for key, label, colour, style in series:
        ax.step(hours, [row[key] for row in result.hourly], where="mid",
                label=label, color=colour, linestyle=style, linewidth=1.8)
    first_binding = True
    for row in result.hourly:
        if row["import_kW"] >= row["cap_kW"] - result.summary["tolerance"]:
            ax.axvspan(row["hour"] - 0.5, row["hour"] + 0.5, color="#2ca02c", alpha=0.08,
                       label="Cap binding or exceeded" if first_binding else None)
            first_binding = False
    ax.set(xlabel="Hour of day", ylabel="Price [DKK/kWh]", xlim=(0.5, len(hours) + 0.5),
           title="Internal settlement price against external prices")
    ax.set_xticks(hours)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"settlement_prices.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    classes = sorted({row["slope_class"] for row in result.members})
    ncols = min(2, len(classes))
    nrows = (len(classes) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3 * nrows), sharey=True, squeeze=False)
    for ax, slope in zip(axes.flat, classes):
        rows = [row for row in result.members if row["slope_class"] == slope]
        x = np.arange(len(rows))
        ax.bar(x - 0.18, [row["gain_vs_R_DKK"] for row in rows], width=0.36,
               label="Gain vs R", color="#1f77b4")
        ax.bar(x + 0.18, [row["gain_vs_I_DKK"] for row in rows], width=0.36,
               label="Gain vs I", color="#ff7f0e")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, [row["consumer_id"] for row in rows], rotation=25, ha="right")
        ax.set_title(f"Disutility class L{slope}")
        ax.set_ylabel("Member gain [DKK/day]")
        ax.grid(axis="y", alpha=0.25)
    for ax in list(axes.flat)[len(classes):]:
        ax.set_visible(False)
    axes[0, 0].legend()
    fig.suptitle("Member gains under dual settlement", fontsize=12)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"settlement_member_gains.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_contract_sensitivity(rows, output_dir, dimension, base_value, member_count, thresholds=None):
    """Plot contract costs/exceedance and settlement outcomes against beta or f."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = "Tariff discount beta" if dimension == "discount" else "Cap-energy factor f"
    colors = {"R": "#1f77b4", "I": "#ff7f0e", "C": "#2ca02c"}
    by_model = {kind: sorted([row for row in rows if row["model"] == kind],
                             key=lambda row: row["parameter_value"]) for kind in colors}
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9), sharex=True)
    for kind, records in by_model.items():
        x = [r["parameter_value"] for r in records]
        axes[0].plot(x, [r["total_objective_DKK"] for r in records], color=colors[kind],
                     label=kind, linewidth=2, linestyle="--" if kind == "R" else "-")
        axes[1].plot(x, [r["aggregate_metered_exceedance_kWh"] for r in records],
                     color=colors[kind], linewidth=2, linestyle="--" if kind == "R" else "-")
        axes[2].plot(x, [r["metered_peak_import_kW"] for r in records], label=kind,
                     color=colors[kind], linewidth=2, linestyle="--" if kind == "R" else "-")
    axes[0].set_ylabel("Community dispatch cost [DKK/day]")
    axes[1].set_ylabel("Aggregate metered exceedance\n[kWh/day]")
    axes[2].set_ylabel("Peak metered import [kW]")
    axes[2].set_xlabel(label)
    axes[2].plot([r["parameter_value"] for r in by_model["C"]],
                 [r["cap_max_kW"] for r in by_model["C"]], color="black", linestyle="-.",
                 label="Max hourly cap", linewidth=1)
    axes[2].legend(ncol=4, fontsize=8)
    axes[0].set_title("Contract terms, community costs and import outcomes")
    if thresholds:
        reference_cost = by_model["R"][0]["total_objective_DKK"]
        for kind in ("I", "C"):
            for bracket in thresholds[kind]["break_even_brackets"]:
                low, high = bracket["lower"], bracket["upper"]
                middle = (low + high) / 2
                axes[0].scatter([middle], [reference_cost], color=colors[kind], s=25, zorder=5)
                axes[0].annotate(f"{kind} break-even\n[{low:.3f}, {high:.3f}]",
                                 xy=(middle, reference_cost), xytext=(100, 50) if kind == "I" else (20, 20),
                                 textcoords="offset points", fontsize=8, color=colors[kind],
                                 bbox=dict(facecolor="white", edgecolor="none", alpha=.85),
                                 arrowprops=dict(arrowstyle="-", color=colors[kind]))
    for ax in axes:
        ax.axvline(base_value, color="#777777", linestyle=":", label="Base contract")
        ax.grid(alpha=.25)
    axes[0].legend(ncol=4)
    fig.text(.5, .01, "R/I: sum of individual-meter imports. C: shared-meter import.\n"
             "Individual contractual exceedances are recorded separately in metrics.csv.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"costs_and_exceedance.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    records = by_model["C"]
    x = [r["parameter_value"] for r in records]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True)
    axes[0].plot(x, [r["manager_surplus_DKK"] for r in records], color=colors["C"], linewidth=2)
    axes[0].axhline(0, color="black", linewidth=.8)
    axes[0].set_ylabel("Manager surplus [DKK/day]")
    for kind in ("R", "I"):
        axes[1].plot(x, [r[f"members_not_worse_off_vs_{kind}"] for r in records],
                     label=f"Compared with {kind}", color=colors[kind], linewidth=1,
                     marker="o", markersize=3, linestyle=":")
    axes[1].set(xlabel=label, ylabel="Members not worse off", ylim=(-.5, member_count + .5))
    axes[1].set_yticks(sorted(set([0, member_count // 4, member_count // 2,
                                 3 * member_count // 4, member_count])))
    axes[1].legend()
    for ax in axes:
        ax.axvline(base_value, color="#777777", linestyle=":")
        ax.grid(alpha=.25)
    axes[0].set_title("Manager surplus and member participation under dual settlement")
    fig.text(.5, .01, "Counts at sampled contracts; dotted joins are guides. No surplus is redistributed.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"settlement_sensitivity.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    plot_report_contract_sensitivity(rows, output_dir, dimension, base_value, thresholds)


def plot_report_contract_sensitivity(rows, output_dir, dimension, base_value, thresholds=None):
    """Save concise Q1 figures: costs (DKK/day), and cap-study peaks (kW).

    The discount figure uses every point. The cap figure focuses on sampled
    factors 0.8--2.5, covering the base case and both default break-even brackets.
    If a custom sweep has no points in that window, its full range is shown.
    Full-range figures and all numerical results are retained by the caller.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"R": "#1f77b4", "I": "#ff7f0e", "C": "#2ca02c"}
    selected = rows
    if dimension == "cap":
        focused = [r for r in rows if 0.8 <= r["parameter_value"] <= 2.5]
        if focused:
            selected = focused
    by_model = {
        kind: sorted([r for r in selected if r["model"] == kind],
                     key=lambda r: r["parameter_value"])
        for kind in colors
    }
    x = [r["parameter_value"] for r in by_model["C"]]
    nrows = 1 if dimension == "discount" else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 3.6 if nrows == 1 else 6),
                             sharex=True, squeeze=False)
    cost_ax = axes[0, 0]
    for kind, records in by_model.items():
        values = [r["parameter_value"] for r in records]
        style = "--" if kind == "R" else "-"
        cost_ax.plot(values, [r["total_objective_DKK"] for r in records],
                     color=colors[kind], linestyle=style, linewidth=2, label=f"Model {kind}")
        if nrows == 2:
            axes[1, 0].plot(values, [r["metered_peak_import_kW"] for r in records],
                            color=colors[kind], linestyle=style, linewidth=2)
    cost_ax.set_ylabel("Community cost [DKK/day]")
    cost_ax.set_title("Tariff discount" if dimension == "discount" else "Cap tightness")
    if thresholds:
        reference_cost = by_model["R"][0]["total_objective_DKK"]
        for kind in ("I", "C"):
            for bracket in thresholds[kind]["break_even_brackets"]:
                low, high = bracket["lower"], bracket["upper"]
                middle = (low + high) / 2
                if min(x) <= middle <= max(x):
                    cost_ax.scatter([middle], [reference_cost], color=colors[kind], s=25, zorder=5)
                    cost_ax.annotate(f"{kind} break-even: {middle:.3f}",
                                     xy=(middle, reference_cost), xytext=(12, 16),
                                     textcoords="offset points", color=colors[kind], fontsize=9,
                                     bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
    for ax in axes.flat:
        if min(x) <= base_value <= max(x):
            ax.axvline(base_value, color="#777777", linestyle=":", label="Base contract")
        ax.grid(alpha=.3)
    cost_ax.legend(ncol=4, fontsize=9)
    axes[-1, 0].set_xlabel("Tariff discount beta" if dimension == "discount" else "Cap-energy factor f")
    bottom_margin = .02
    if nrows == 2:
        peak_ax = axes[1, 0]
        peak_ax.plot(x, [r["cap_max_kW"] for r in by_model["C"]],
                     color="black", linestyle="-.", linewidth=1.3, label="Hourly cap")
        peak_ax.set_ylabel("Peak metered import [kW]")
        peak_ax.legend(loc="upper left", fontsize=9)
        fig.text(.5, .01, "R/I: summed individual-meter imports; C: shared-meter import.",
                 ha="center", fontsize=8)
        bottom_margin = .06
    fig.tight_layout(rect=(0, bottom_margin, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"report_contract_sensitivity.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
