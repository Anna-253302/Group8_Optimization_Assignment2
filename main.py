"""Entry point of the Assignment 2 starter package: load a community and summarize it.

    Examples:
    python main.py
    python main.py --model R
    python main.py --model I
    python main.py --model both --day d1
    python main.py                                   # base case (16 members), day d1
    python main.py --community my_mix                # your own data/params_community_my_mix.json
    python main.py --day d4                          # another day of the week (and that day's cap)
    python main.py --members                         # also print every member's Assignment 1 summary
    python main.py --demo                            # also run the scenario-generation demonstration (Question 3)

Models R (Reference), I (Individual DSO Contracts) and C (Community DSO Contract) are implemented.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import load_community, list_communities
from src.model import solve_community

from src.analysis import (
    analyse_reference,
    analyse_individual_contracts,
    analyse_centralized_contract,
    validate_reference,
    print_three_model_comparison,
)

from src.plotting import plot_feeder_import

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default="base", help=f"base (shipped) or your own params_community_<name>.json; available: {list_communities()}")
    ap.add_argument("--day", default="d1", help="d1 (base case) ... d7")
    ap.add_argument("--members", action="store_true", help="print the Assignment 1 summary of every member")
    ap.add_argument("--demo", action="store_true", help="run the scenario-generation demonstration")
    ap.add_argument("--model", choices=["R", "I", "C", "both", "all", "none"], default="all", help="Select optimization model: R, I, C, both (R+I), all, or none",)
    ap.add_argument("--verbose-solver", action="store_true", help="Show Gurobi solver output",)
    args = ap.parse_args()

    com = load_community(args.community, day=args.day)
    print(com.summary(), "\n")
    if args.members:
        for m in com.members:
            print(m.summary(), "\n")
    print("Cap profile (kW):", " ".join(f"{x:.1f}" for x in com.contract.capacity_limit_kW))
    ic = com.contract.individual_capacity_limit_kW
    print("Individual caps in model I (kW), hour 0 / hour 11:",
          "  ".join(f"{cid} ({com.member(cid).archetype}) {ic[cid][0]:.2f}/{ic[cid][11]:.2f}" for cid in ("C1", "C1_PVB", "C4", "C4_PVB") if cid in com.member_ids))
    (RESULTS_DIR / args.community).mkdir(parents=True, exist_ok=True)
    
        # ==========================================================
    # QUESTION 1: OPTIMIZATION MODELS R, I AND C
    # ==========================================================

    metrics_R = None
    metrics_I = None
    metrics_C = None
    results_C = None

    plot_results = {}

    # ----------------------------------------------------------
    # MODEL R - Reference
    # ----------------------------------------------------------

    if args.model in ("R", "both", "all"):

        print("\n" + "=" * 60)
        print("MODEL R - REFERENCE")
        print("=" * 60)

        results_R = solve_community(
            com,
            model_type="R",
            verbose=args.verbose_solver,
        )

        metrics_R = analyse_reference(
            com,
            results_R,
        )

        validate_reference(
            com,
            metrics_R,
        )

        print(f"Total objective: {metrics_R.total_objective_DKK:.2f} DKK")
        print(f"Daily gross import: {metrics_R.daily_import_kWh:.2f} kWh")
        print(f"Peak feeder import: {metrics_R.peak_import_kW:.2f} kW")

        plot_results["R"] = results_R

    # ----------------------------------------------------------
    # MODEL I - Individual contracts
    # ----------------------------------------------------------

    if args.model in ("I", "both", "all"):

        print("\n" + "=" * 60)
        print("MODEL I - INDIVIDUAL CONTRACTS")
        print("=" * 60)

        results_I = solve_community(
            com,
            model_type="I",
            verbose=args.verbose_solver,
        )

        metrics_I = analyse_individual_contracts(
            com,
            results_I,
        )

        print(f"Total objective: {metrics_I.feeder.total_objective_DKK:.2f} DKK")
        print(f"Daily gross import: {metrics_I.feeder.daily_import_kWh:.2f} kWh")
        print(f"Peak feeder import: {metrics_I.feeder.peak_import_kW:.2f} kW")
        print(f"Individual exceedance: {metrics_I.individual_exceedance_kWh:.2f} kWh")
        print(f"Individual penalties: {metrics_I.individual_penalty_DKK:.2f} DKK")

        plot_results["I"] = results_I

    # ----------------------------------------------------------
    # MODEL C - Centralized community contract
    # ----------------------------------------------------------

    if args.model in ("C", "all"):

        print("\n" + "=" * 60)
        print("MODEL C - CENTRALIZED COMMUNITY CONTRACT")
        print("=" * 60)

        results_C = solve_community(com,model_type="C",verbose=args.verbose_solver,)
        
        metrics_C = analyse_centralized_contract(com, results_C,)

        community_import = results_C.community_import_kW
        community_export = results_C.community_export_kW
        community_exceedance = results_C.community_exceedance_kW

        print(f"Total objective: {results_C.objective_DKK:.2f} DKK")
        print(f"Daily community import: {sum(community_import):.2f} kWh")
        print(f"Daily community export: {sum(community_export):.2f} kWh")
        print(f"Peak community import: {max(community_import):.2f} kW")
        print(f"Community exceedance: {sum(community_exceedance):.2f} kWh")
        print(f"Congested hours: {sum(community_exceedance > 1e-6)}")

        plot_results["C"] = results_C

        # ----------------------------------------------------------
    # MODEL COMPARISON: R vs I vs C
    # ----------------------------------------------------------

    if (
        metrics_R is not None
        and metrics_I is not None
        and metrics_C is not None
    ):

        print_three_model_comparison(
            metrics_R,
            metrics_I,
            metrics_C,
        )
        
    # ----------------------------------------------------------
    # FIGURE 1 - HOURLY FEEDER IMPORT
    # ----------------------------------------------------------

    if plot_results:

        output_dir = RESULTS_DIR / args.community / args.day

        plot_feeder_import(
            com,
            plot_results,
            output_dir,
        )

        print(f"\nFigure saved to: {output_dir}")

    if args.demo:
        import runpy
        runpy.run_module("src.scenario_generation", run_name="__main__")


if __name__ == "__main__":
    main()
