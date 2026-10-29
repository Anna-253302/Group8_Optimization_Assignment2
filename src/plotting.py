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