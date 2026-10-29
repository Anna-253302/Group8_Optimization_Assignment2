"""Post-processing and validation for Assignment 2, Question 1.

Reference R and individual-contract I have *individual meters*: gross feeder
import is the SUM of individual imports, not the positive part of net trade.
The community contract C has a *single meter* and must be analysed separately.

Power is in kW, with hourly intervals (dt=1 h); energy is in kWh.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .data_loader import CommunityData
from .model import ReferenceResult, CommunityResult


@dataclass
class ReferenceMetrics:
    """Aggregated technical/economic results for the independent R models."""

    hours: np.ndarray
    gross_import_kW: np.ndarray
    gross_export_kW: np.ndarray
    net_exchange_kW: np.ndarray
    capacity_limit_kW: np.ndarray
    exceedance_kW: np.ndarray
    daily_import_kWh: float
    daily_export_kWh: float
    daily_net_import_kWh: float
    peak_import_kW: float
    exceedance_kWh: float
    congested_hours: int
    total_objective_DKK: float
    member_objectives_DKK: dict[str, float]


def analyse_reference(
    community: CommunityData,
    results: Mapping[str, ReferenceResult],
    *,
    dt_hours: float = 1.0,
    tolerance: float = 1e-7,
) -> ReferenceMetrics:
    """Aggregate Model R schedules and evaluate the *hypothetical* DSO cap.

    The cap and exceedance do NOT enter the optimization of R; they are used
    only to quantify the congestion R would create relative to the contract cap.
    """
    if dt_hours <= 0:
        raise ValueError('dt_hours must be positive')
    if not community.members:
        raise ValueError('community has no members')

    ids = [member.consumer_id for member in community.members]
    if set(ids) != set(results):
        raise ValueError(
            f'Result IDs do not match community: missing={set(ids)-set(results)}, '
            f'extra={set(results)-set(ids)}'
        )

    n = community.members[0].n_hours
    cap = np.asarray(community.contract.capacity_limit_kW, dtype=float)
    if cap.shape != (n,):
        raise ValueError(f'Capacity limit must have {n} hourly values, got {cap.shape}')

    imports = []
    exports = []
    for cid in ids:
        imp = np.asarray(results[cid].import_kW, dtype=float)
        exp = np.asarray(results[cid].export_kW, dtype=float)
        if imp.shape != (n,) or exp.shape != (n,):
            raise ValueError(f'{cid}: import/export arrays must have {n} values')
        if np.any(imp < -tolerance) or np.any(exp < -tolerance):
            raise ValueError(f'{cid}: negative import/export encountered')
        imports.append(np.maximum(imp, 0.0))
        exports.append(np.maximum(exp, 0.0))

    gross_import = np.sum(imports, axis=0)
    gross_export = np.sum(exports, axis=0)
    net_exchange = gross_import - gross_export
    exceedance = np.maximum(gross_import - cap, 0.0)

    physical_feeder_import = np.maximum(net_exchange, 0.0)

    physical_feeder_peak = float(
        np.max(physical_feeder_import)
    )

    physical_feeder_exceedance = float(
        np.sum(
            np.maximum(
                physical_feeder_import - cap,
                0.0,
            )
        ) * dt_hours
    )

    objectives = {cid: float(results[cid].objective_DKK) for cid in ids}
    return ReferenceMetrics(
        hours=np.arange(1, n + 1),
        gross_import_kW=gross_import,
        gross_export_kW=gross_export,
        net_exchange_kW=net_exchange,
        capacity_limit_kW=cap.copy(),
        exceedance_kW=exceedance,
        daily_import_kWh=float(gross_import.sum() * dt_hours),
        daily_export_kWh=float(gross_export.sum() * dt_hours),
        daily_net_import_kWh=float(net_exchange.sum() * dt_hours),
        peak_import_kW=float(gross_import.max()),
        exceedance_kWh=float(exceedance.sum() * dt_hours),
        congested_hours=int(np.count_nonzero(exceedance > tolerance)),
        total_objective_DKK=float(sum(objectives.values())),
        member_objectives_DKK=objectives,
    )


def validate_reference(
    community: CommunityData,
    metrics: ReferenceMetrics,
    *,
    atol: float = 0.11,
    print_table: bool = True,
) -> dict[str, bool]:
    """Compare R with supplied reference_model_R values for the selected day.

    Reference numbers are rounded (typically one decimal place); default
    tolerance 0.11 permits this rounding but still flags material differences.
    Returns a dictionary of pass/fail checks; does not silently assert success.
    """
    expected = community.contract.reference_model_R
    fields = {
        'daily_import_kWh': metrics.daily_import_kWh,
        'peak_import_kW': metrics.peak_import_kW,
        'cost_DKK': metrics.total_objective_DKK,
        'exceedance_kWh': metrics.exceedance_kWh,
    }
    checks: dict[str, bool] = {}
    if print_table:
        print(f"{'Metric':<24} {'Calculated':>12} {'Reference':>12} {'Difference':>12} {'Check':>8}")
        print('-' * 73)
    for key, actual in fields.items():
        if key not in expected:
            if print_table:
                print(f'{key:<24} {actual:>12.3f} {"N/A":>12} {"N/A":>12} {"SKIP":>8}')
            continue
        target = float(expected[key])
        diff = actual - target
        passed = abs(diff) <= atol
        checks[key] = passed
        if print_table:
            print(f'{key:<24} {actual:>12.3f} {target:>12.3f} {diff:>+12.3f} {"PASS" if passed else "FAIL":>8}')
    if print_table:
        print(f'Congested hours: {metrics.congested_hours}')
        print(f'Daily gross export: {metrics.daily_export_kWh:.2f} kWh')
    return checks

@dataclass
class IndividualContractMetrics:
    """Technical and economic results for Model I."""

    # Aggregated feeder results
    feeder: ReferenceMetrics

    # Individual contract exceedance
    individual_exceedance_kWh: float
    individual_penalty_DKK: float

    # Economic breakdown
    energy_market_DKK: float
    grid_tariffs_DKK: float
    pv_generation_DKK: float
    disutility_DKK: float

    # Results for each consumer
    member_cost_breakdown_DKK: dict[str, dict[str, float]]

@dataclass
class CommunityContractMetrics:
    """Technical and economic metrics for centralized Model C."""

    # Community meter
    community_import_kW: np.ndarray
    community_export_kW: np.ndarray
    community_net_exchange_kW: np.ndarray

    # Congestion
    capacity_limit_kW: np.ndarray
    exceedance_kW: np.ndarray
    peak_import_kW: float
    exceedance_kWh: float
    congested_hours: int

    # Daily energy
    daily_import_kWh: float
    daily_export_kWh: float
    daily_net_import_kWh: float

    # Economic results
    total_objective_DKK: float
    energy_market_DKK: float
    grid_tariffs_DKK: float
    pv_generation_DKK: float
    disutility_DKK: float
    community_penalty_DKK: float

    # Dual variables
    balance_duals: np.ndarray

def analyse_individual_contracts(
    community: CommunityData,
    results: Mapping[str, ReferenceResult],
    *,
    dt_hours: float = 1.0,
) -> IndividualContractMetrics:
    """
    Analyse Model I: Individual DSO contracts.

    Calculates:
    - Feeder congestion
    - Individual contract exceedance
    - Individual congestion penalties
    - Electricity market costs
    - Discounted grid tariffs
    - PV generation costs
    - Load disutility
    """

    # --------------------------------------------------
    # 1. Calculate aggregated feeder metrics
    # --------------------------------------------------

    feeder = analyse_reference(
        community,
        results,
        dt_hours=dt_hours,
    )

    contract = community.contract

    beta = float(contract.tariff_discount_ratio)
    alpha = float(contract.penalty_DKK_per_kWh)

    breakdown = {}

    # --------------------------------------------------
    # 2. Calculate costs for every consumer
    # --------------------------------------------------

    for member in community.members:

        cid = member.consumer_id
        r = results[cid]

        imp = np.asarray(r.import_kW, dtype=float)
        exp = np.asarray(r.export_kW, dtype=float)

        price = np.asarray(member.energy_price, dtype=float)

        # Individual DSO capacity limit
        cap = np.asarray(
            contract.individual_capacity_limit_kW[cid],
            dtype=float,
        )

        # --------------------------------------------------
        # Individual exceedance
        # --------------------------------------------------

        excess = np.maximum(imp - cap, 0.0)

        excess_kWh = float(
            np.sum(excess) * dt_hours
        )

        # --------------------------------------------------
        # Electricity market cost
        # --------------------------------------------------

        energy_market = float(
            np.sum(price * (imp - exp)) * dt_hours
        )

        # --------------------------------------------------
        # Discounted grid tariffs
        # --------------------------------------------------

        grid_tariffs = float(
            (1 - beta)
            * np.sum(
                member.import_tariff * imp
                + member.export_tariff * exp
            )
            * dt_hours
        )

        # --------------------------------------------------
        # PV generation costs
        # --------------------------------------------------

        pv_generation = float(
            member.pv_marginal_cost
            * np.sum(r.pv_kW)
            * dt_hours
        )

        # --------------------------------------------------
        # Load disutility
        # --------------------------------------------------

        disutility = float(
            member.linear_disutility
            * np.sum(r.deviation_kW)
            * dt_hours
        )

        # --------------------------------------------------
        # Individual congestion penalty
        # --------------------------------------------------

        penalty = alpha * excess_kWh

        # --------------------------------------------------
        # Store individual results
        # --------------------------------------------------

        breakdown[cid] = {
            "energy_market": energy_market,
            "grid_tariffs": grid_tariffs,
            "pv_generation": pv_generation,
            "disutility": disutility,
            "penalty": penalty,
            "individual_exceedance_kWh": excess_kWh,
        }

        # Check that our cost breakdown matches Gurobi
        calculated_objective = (
            energy_market
            + grid_tariffs
            + pv_generation
            + disutility
            + penalty
        )

        if not np.isclose(
            calculated_objective,
            r.objective_DKK,
            atol=1e-4,
        ):
            raise ValueError(
                f"Cost breakdown does not match "
                f"objective for {cid}: "
                f"{calculated_objective:.4f} vs "
                f"{r.objective_DKK:.4f}"
            )

    # --------------------------------------------------
    # 3. Aggregate costs across all consumers
    # --------------------------------------------------

    def total(key):
        return sum(
            values[key]
            for values in breakdown.values()
        )

    return IndividualContractMetrics(
        feeder=feeder,
        individual_exceedance_kWh=total(
            "individual_exceedance_kWh"
        ),
        individual_penalty_DKK=total("penalty"),
        energy_market_DKK=total("energy_market"),
        grid_tariffs_DKK=total("grid_tariffs"),
        pv_generation_DKK=total("pv_generation"),
        disutility_DKK=total("disutility"),
        member_cost_breakdown_DKK=breakdown,
    )

def print_three_model_comparison(
    r: ReferenceMetrics,
    i: IndividualContractMetrics,
    c: CommunityContractMetrics,
):
    """
    Compare the economic and technical performance
    of Models R, I and C.
    """

    # --------------------------------------------------
    # Model comparison
    # --------------------------------------------------

    rows = [
        (
            "Total objective [DKK]",
            r.total_objective_DKK,
            i.feeder.total_objective_DKK,
            c.total_objective_DKK,
        ),
        (
            "Daily import [kWh]",
            r.daily_import_kWh,
            i.feeder.daily_import_kWh,
            c.daily_import_kWh,
        ),
        (
            "Daily export [kWh]",
            r.daily_export_kWh,
            i.feeder.daily_export_kWh,
            c.daily_export_kWh,
        ),
        (
            "Peak import [kW]",
            r.peak_import_kW,
            i.feeder.peak_import_kW,
            c.peak_import_kW,
        ),
        (
            "Capacity exceedance [kWh]",
            r.exceedance_kWh,
            i.feeder.exceedance_kWh,
            c.exceedance_kWh,
        ),
        (
            "Congested hours",
            r.congested_hours,
            i.feeder.congested_hours,
            c.congested_hours,
        ),
    ]

    print("\n" + "=" * 75)
    print("MODEL COMPARISON: R vs I vs C")
    print("=" * 75)

    print(
        f"{'Metric':<35}"
        f"{'Model R':>13}"
        f"{'Model I':>13}"
        f"{'Model C':>13}"
    )

    print("-" * 75)

    for label, value_r, value_i, value_c in rows:
        print(
            f"{label:<35}"
            f"{value_r:>13.2f}"
            f"{value_i:>13.2f}"
            f"{value_c:>13.2f}"
        )

    # --------------------------------------------------
    # Economic savings from centralized coordination
    # --------------------------------------------------

    savings_R = r.total_objective_DKK - c.total_objective_DKK
    savings_I = i.feeder.total_objective_DKK - c.total_objective_DKK

    savings_R_pct = (
        savings_R / r.total_objective_DKK * 100
    )

    savings_I_pct = (
        savings_I / i.feeder.total_objective_DKK * 100
    )

    print("\nCENTRALIZED COMMUNITY SAVINGS")

    print(
        f"Compared with Model R: "
        f"{savings_R:.2f} DKK ({savings_R_pct:.1f}%)"
    )

    print(
        f"Compared with Model I: "
        f"{savings_I:.2f} DKK ({savings_I_pct:.1f}%)"
    )

    # --------------------------------------------------
    # Economic breakdown: Model I
    # --------------------------------------------------

    print("\nMODEL I - ECONOMIC BREAKDOWN")

    costs_I = [
        ("Energy market", i.energy_market_DKK),
        ("Grid tariffs", i.grid_tariffs_DKK),
        ("PV generation", i.pv_generation_DKK),
        ("Disutility", i.disutility_DKK),
        ("Individual penalties", i.individual_penalty_DKK),
    ]

    for label, value in costs_I:
        print(f"{label:<30} {value:>12.2f} DKK")

    print("-" * 48)
    print(
        f"{'Total objective':<30}"
        f"{i.feeder.total_objective_DKK:>12.2f} DKK"
    )

    # --------------------------------------------------
    # Economic breakdown: Model C
    # --------------------------------------------------

    print("\nMODEL C - ECONOMIC BREAKDOWN")

    costs_C = [
        ("Energy market", c.energy_market_DKK),
        ("Grid tariffs", c.grid_tariffs_DKK),
        ("PV generation", c.pv_generation_DKK),
        ("Disutility", c.disutility_DKK),
        ("Community penalties", c.community_penalty_DKK),
    ]

    for label, value in costs_C:
        print(f"{label:<30} {value:>12.2f} DKK")

    print("-" * 48)
    print(
        f"{'Total objective':<30}"
        f"{c.total_objective_DKK:>12.2f} DKK"
    )
    
def analyse_centralized_contract(
    community: CommunityData,
    results: CommunityResult,
    *,
    dt_hours: float = 1.0,
    tolerance: float = 1e-7,
) -> CommunityContractMetrics:
    """Analyse the centralized community contract (Model C)."""

    contract = community.contract

    beta = float(contract.tariff_discount_ratio)
    alpha = float(contract.penalty_DKK_per_kWh)

    imp = np.asarray(results.community_import_kW, dtype=float)
    exp = np.asarray(results.community_export_kW, dtype=float)

    cap = np.asarray(contract.capacity_limit_kW, dtype=float)

    net = imp - exp
    excess = np.maximum(imp - cap, 0.0)

    # Community market prices and tariffs
    member = community.members[0]

    price = np.asarray(member.energy_price, dtype=float)

    energy_market = float(
        np.sum(price * net) * dt_hours
    )

    grid_tariffs = float(
        (1 - beta)
        * np.sum(
            member.import_tariff * imp
            + member.export_tariff * exp
        )
        * dt_hours
    )

    pv_generation = float(
        sum(
            m.pv_marginal_cost
            * np.sum(results.member_results[m.consumer_id].pv_kW)
            * dt_hours
            for m in community.members
        )
    )

    disutility = float(
        sum(
            m.linear_disutility
            * np.sum(results.member_results[m.consumer_id].deviation_kW)
            * dt_hours
            for m in community.members
        )
    )

    penalty = float(
        alpha * np.sum(excess) * dt_hours
    )

    calculated_objective = (
        energy_market
        + grid_tariffs
        + pv_generation
        + disutility
        + penalty
    )

    if not np.isclose(
        calculated_objective,
        results.objective_DKK,
        atol=1e-4,
    ):
        raise ValueError(
            f"Model C cost breakdown mismatch: "
            f"{calculated_objective:.4f} vs "
            f"{results.objective_DKK:.4f}"
        )

    return CommunityContractMetrics(
        community_import_kW=imp,
        community_export_kW=exp,
        community_net_exchange_kW=net,
        capacity_limit_kW=cap,
        exceedance_kW=excess,
        peak_import_kW=float(np.max(imp)),
        exceedance_kWh=float(np.sum(excess) * dt_hours),
        congested_hours=int(np.count_nonzero(excess > tolerance)),
        daily_import_kWh=float(np.sum(imp) * dt_hours),
        daily_export_kWh=float(np.sum(exp) * dt_hours),
        daily_net_import_kWh=float(np.sum(net) * dt_hours),
        total_objective_DKK=float(results.objective_DKK),
        energy_market_DKK=energy_market,
        grid_tariffs_DKK=grid_tariffs,
        pv_generation_DKK=pv_generation,
        disutility_DKK=disutility,
        community_penalty_DKK=penalty,
        balance_duals=np.asarray(results.balance_duals),
    )