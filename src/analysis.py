"""Post-processing and validation for Assignment 2, Questions 1 and 2.

Reference R and individual-contract I have *individual meters*: gross feeder
import is the SUM of individual imports, not the positive part of net trade.
The community contract C has a *single meter* and must be analysed separately.

Power is in kW, with hourly intervals (dt=1 h); energy is in kWh.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from gurobipy import GRB

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
    exceedance_range: dict | None = None,
) -> dict[str, bool]:
    """Compare R with supplied reference_model_R values for the selected day.

    Reference numbers are rounded (typically one decimal place); default
    tolerance 0.11 permits this rounding but still flags material differences.
    An optional certified optimal-face diagnostic can establish that differing
    exceedances are both optimal. Its optimum must match this result and the
    primary benchmark checks must pass. No tolerance is widened.
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
    primary_pass = all(abs(fields[key] - float(expected[key])) <= atol
                       for key in fields if key != 'exceedance_kWh' and key in expected)
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
        status = "PASS" if passed else "FAIL"
        if (key == 'exceedance_kWh' and not passed and primary_pass
                and exceedance_range is not None and exceedance_range.get('certified')
                and np.isclose(metrics.total_objective_DKK,
                               exceedance_range['optimal_cost_DKK'], atol=1e-5, rtol=0)):
            lower = exceedance_range['minimum_exceedance_kWh']
            upper = exceedance_range['maximum_exceedance_kWh']
            if lower - 1e-5 <= actual <= upper + 1e-5 and lower - 1e-5 <= target <= upper + 1e-5:
                passed = True
                status = "ALT OPT"
        checks[key] = passed
        if print_table:
            print(f'{key:<24} {actual:>12.3f} {target:>12.3f} {diff:>+12.3f} {status:>8}')
    if print_table:
        if exceedance_range is not None and exceedance_range.get('certified'):
            print('Certified optimal R exceedance range: '
                  f"{exceedance_range['minimum_exceedance_kWh']:.6f} to "
                  f"{exceedance_range['maximum_exceedance_kWh']:.6f} kWh")
            print('ALT OPT means both exceedances belong to the optimal economic solution set.')
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


# ==========================================================
# DUAL SETTLEMENT AND MEMBER PARTICIPATION
# ==========================================================

@dataclass
class SettlementResult:
    """Daily DKK totals, member records and hourly prices in DKK/kWh."""

    members: list[dict]
    hourly: list[dict]
    costs: dict[str, dict[str, float]]
    summary: dict


def analyse_settlement(
    community: CommunityData,
    reference: Mapping[str, ReferenceResult],
    individual: Mapping[str, ReferenceResult],
    centralized: CommunityResult,
    *,
    tolerance: float = 1e-5,
) -> SettlementResult:
    """Settle C using its balance duals and compare every member with R and I.

    The existing model has one-hour intervals. Its balance is
    ``P_imp - P_exp == sum(q_i)``: Gurobi's RHS marginal (Pi) is the
    positive internal purchase price, without a sign flip. Check this using
    marginal external prices, including hours at and above the soft cap.
    Inputs are not mutated; C's diagnostic member objectives remain undefined.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    ids = community.member_ids
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Community must have unique, nonempty member IDs")
    if centralized.model_type != "C":
        raise ValueError("Settlement requires centralized Model C")
    for label, results in (("R", reference), ("I", individual),
                           ("C", centralized.member_results)):
        if set(results) != set(ids):
            raise ValueError(f"{label}: result IDs do not match community members")
        if any(results[cid].consumer_id != cid for cid in ids):
            raise ValueError(f"{label}: result consumer_id does not match its key")
    n = community.members[0].n_hours
    if any(m.n_hours != n for m in community.members):
        raise ValueError("All members must have the same horizon")

    def array(value, label):
        a = np.asarray(value, dtype=float)
        if a.shape != (n,) or not np.all(np.isfinite(a)):
            raise ValueError(f"{label}: expected {n} finite hourly values")
        return a

    def check(actual, expected, label):
        if not np.allclose(actual, expected, atol=tolerance, rtol=0):
            raise ValueError(f"Settlement validation failed: {label}")

    beta = float(community.contract.tariff_discount_ratio)
    alpha = float(community.contract.penalty_DKK_per_kWh)
    price = array(community.energy_price, "energy price")
    internal = array(centralized.balance_duals, "balance duals")
    imp = array(centralized.community_import_kW, "community import")
    exp = array(centralized.community_export_kW, "community export")
    cap = array(community.contract.capacity_limit_kW, "community cap")
    if np.any(imp < -tolerance) or np.any(exp < -tolerance):
        raise ValueError("Community import and export must be nonnegative")
    excess = np.maximum(imp - cap, 0.0)
    check(array(centralized.community_exceedance_kW, "exceedance"), excess,
          "community exceedance")
    buy = price + (1 - beta) * community.import_tariff
    sell = price - (1 - beta) * community.export_tariff
    # At the cap the subgradient is an interval, not a single marginal price.
    if np.any(internal < sell - tolerance) or np.any(internal > buy + alpha + tolerance):
        raise ValueError("Internal prices violate external price bounds; check dual sign")
    slack_import = (imp > tolerance) & (imp < cap - tolerance)
    above_cap = imp > cap + tolerance
    at_cap = (imp > tolerance) & (np.abs(imp - cap) <= tolerance)
    check(internal[slack_import], buy[slack_import], "slack-import price / dual sign")
    check(internal[above_cap], (buy + alpha)[above_cap], "above-cap price")
    check(internal[exp > tolerance], sell[exp > tolerance], "export price")
    if np.any(internal[at_cap] < buy[at_cap] - tolerance):
        raise ValueError("Internal price below import price at the binding cap")
    idle = (imp <= tolerance) & (exp <= tolerance) & (cap > tolerance)
    if np.any(internal[idle] > buy[idle] + tolerance):
        raise ValueError("Internal price above import price at an idle, slack meter")

    energy = price * (imp - exp)
    tariffs = (1 - beta) * (community.import_tariff * imp + community.export_tariff * exp)
    penalty = alpha * excess
    external = energy + tariffs + penalty
    components = ("energy_market_DKK", "grid_tariffs_DKK", "penalty_DKK",
                  "pv_generation_DKK", "disutility_DKK")
    costs = {kind: dict.fromkeys(components, 0.0) for kind in ("R", "I", "C")}
    costs["C"].update(energy_market_DKK=float(energy.sum()),
                      grid_tariffs_DKK=float(tariffs.sum()),
                      penalty_DKK=float(penalty.sum()))
    members = []
    aggregate_net = np.zeros(n)
    hourly_payments = np.zeros(n)
    for member in community.members:
        cid = member.consumer_id
        local = {}
        for kind, result in (("R", reference[cid]), ("I", individual[cid]),
                             ("C", centralized.member_results[cid])):
            load = array(result.load_kW, f"{kind}/{cid} load")
            pv = array(result.pv_kW, f"{kind}/{cid} PV")
            deviation = array(result.deviation_kW, f"{kind}/{cid} deviation")
            check(deviation, np.abs(load - member.reference_load),
                  f"{kind}/{cid} absolute deviation")
            pv_cost = float(member.pv_marginal_cost * pv.sum())
            disutility = float(member.linear_disutility * deviation.sum())
            local[kind] = pv_cost + disutility
            costs[kind]["pv_generation_DKK"] += pv_cost
            costs[kind]["disutility_DKK"] += disutility
            if kind != "C":
                member_imp = array(result.import_kW, f"{kind}/{cid} import")
                member_exp = array(result.export_kW, f"{kind}/{cid} export")
                discount = 0.0 if kind == "R" else beta
                own_energy = float(np.dot(member.energy_price, member_imp - member_exp))
                own_tariffs = float((1 - discount) * (
                    member.import_tariff * member_imp + member.export_tariff * member_exp
                ).sum())
                own_penalty = (0.0 if kind == "R" else float(alpha * np.maximum(
                    member_imp - community.contract.individual_capacity_limit_kW[cid], 0
                ).sum()))
                costs[kind]["energy_market_DKK"] += own_energy
                costs[kind]["grid_tariffs_DKK"] += own_tariffs
                costs[kind]["penalty_DKK"] += own_penalty
                check(result.objective_DKK, own_energy + own_tariffs + own_penalty + local[kind],
                      f"{kind}/{cid} benchmark cost reconstruction")
        result = centralized.member_results[cid]
        net = array(result.net_exchange_kW, f"C/{cid} net exchange")
        charge = np.zeros(n) if result.charge_kW is None else array(result.charge_kW, "charge")
        discharge = (np.zeros(n) if result.discharge_kW is None
                     else array(result.discharge_kW, "discharge"))
        check(net, result.load_kW + charge - result.pv_kW - discharge,
              f"C/{cid} physical exchange")
        payments = internal * net
        payment = float(payments.sum())
        settled = payment + local["C"]
        gain_r = float(reference[cid].objective_DKK - settled)
        gain_i = float(individual[cid].objective_DKK - settled)
        members.append(dict(
            consumer_id=cid, archetype=member.archetype, slope_class=member.slope_class,
            has_pv=member.has_pv, has_battery=member.has_battery,
            reference_cost_DKK=float(reference[cid].objective_DKK),
            individual_cost_DKK=float(individual[cid].objective_DKK),
            net_energy_C_kWh=float(net.sum()), payment_C_DKK=payment,
            local_cost_C_DKK=local["C"], settled_cost_C_DKK=settled,
            gain_vs_R_DKK=gain_r, gain_vs_I_DKK=gain_i,
            individually_rational_vs_R=bool(gain_r >= -tolerance),
            individually_rational_vs_I=bool(gain_i >= -tolerance)))
        aggregate_net += net
        hourly_payments += payments

    check(aggregate_net, imp - exp, "community power balance")
    check(hourly_payments, internal * (imp - exp), "hourly payment aggregation")
    for kind in costs:
        costs[kind]["total_objective_DKK"] = float(sum(costs[kind].values()))
    check(costs["C"]["total_objective_DKK"], centralized.objective_DKK,
          "C objective reconstruction")
    total_payments = float(hourly_payments.sum())
    grid_bill = float(external.sum())
    budget = total_payments - grid_bill
    cap_rent = float(np.dot(np.maximum(internal - buy, 0.0), cap))
    check(budget, cap_rent, "manager surplus equals the cap rent")
    total_settled = float(sum(row["settled_cost_C_DKK"] for row in members))
    identity_residual = total_settled - centralized.objective_DKK - budget
    check(identity_residual, 0.0, "sum(member costs) = C objective + manager surplus")
    summary = dict(
        community=community.community, day=community.day, dt_hours=1.0,
        balance_dual_sign="Pi (P_import - P_export == sum(member net exchange))",
        tolerance=tolerance, total_payments_DKK=total_payments,
        external_energy_DKK=float(energy.sum()), external_tariffs_DKK=float(tariffs.sum()),
        external_penalty_DKK=float(penalty.sum()), external_bill_DKK=grid_bill,
        manager_surplus_DKK=budget, manager_budget_balanced=bool(abs(budget) <= tolerance),
        cap_rent_DKK=cap_rent,
        total_settled_member_cost_DKK=total_settled,
        centralized_objective_DKK=float(centralized.objective_DKK),
        worse_off_vs_R=[r["consumer_id"] for r in members if not r["individually_rational_vs_R"]],
        worse_off_vs_I=[r["consumer_id"] for r in members if not r["individually_rational_vs_I"]],
        accounting_identity_residual_DKK=float(identity_residual), validation_passed=True)
    hourly = [dict(
        hour=t + 1, internal_price_DKK_per_kWh=float(internal[t]),
        day_ahead_price_DKK_per_kWh=float(price[t]),
        discounted_import_price_DKK_per_kWh=float(buy[t]),
        discounted_export_price_DKK_per_kWh=float(sell[t]),
        import_kW=float(imp[t]), export_kW=float(exp[t]), cap_kW=float(cap[t]),
        exceedance_kW=float(excess[t]), payments_collected_DKK=float(hourly_payments[t]),
        external_bill_DKK=float(external[t]),
        manager_surplus_DKK=float(hourly_payments[t] - external[t])) for t in range(n)]
    return SettlementResult(members, hourly, costs, summary)


def print_settlement(result: SettlementResult) -> None:
    """Print daily member costs/benefits and manager receipts/costs in DKK."""
    print("\nQ1 - DUAL SETTLEMENT (DKK; positive gain means the member benefits)")
    print(f"{'Member':<12} {'R cost':>10} {'I cost':>10} {'C settled':>11} "
          f"{'Gain vs R':>11} {'Gain vs I':>11}")
    for row in result.members:
        print(f"{row['consumer_id']:<12} {row['reference_cost_DKK']:>10.2f} "
              f"{row['individual_cost_DKK']:>10.2f} {row['settled_cost_C_DKK']:>11.2f} "
              f"{row['gain_vs_R_DKK']:>11.2f} {row['gain_vs_I_DKK']:>11.2f}")
    s = result.summary
    print(f"Payments collected: {s['total_payments_DKK']:.2f} DKK")
    print(f"External grid bill: {s['external_bill_DKK']:.2f} DKK")
    print(f"Manager surplus:   {s['manager_surplus_DKK']:.2f} DKK")
    print(f"Budget balanced:   {s['manager_budget_balanced']}")
    print("Members worse off vs R:", ", ".join(s["worse_off_vs_R"]) or "none")
    print("Members worse off vs I:", ", ".join(s["worse_off_vs_I"]) or "none")
    print("Settlement validation: PASS")


def _write_csv(path, rows):
    """Write one table; column names retain the quantities and units."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_settlement(result: SettlementResult, output_dir: Path) -> None:
    """Write member/hourly/cost CSVs and a strict JSON summary; amounts are DKK."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = {
        "settlement_members.csv": result.members,
        "settlement_hourly.csv": result.hourly,
        "cost_breakdown.csv": [dict(model=kind, **costs) for kind, costs in result.costs.items()],
    }
    for name, rows in records.items():
        _write_csv(output_dir / name, rows)
    (output_dir / "settlement_summary.json").write_text(
        json.dumps(result.summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")


# ==========================================================
# QUESTION 2: DYNAMIC PRICING, CAP ONLY
# ==========================================================

def analyse_dp(community, reference, individual, centralized, dp, *, tolerance=1e-5):
    """Analyse DP's shared meter, internal transfers and member participation.

    All intervals are one hour. Checks use absolute tolerance (rtol=0), in
    DKK for accounting and kW/kWh for physical quantities. Return the existing
    SettlementResult container, with DP-specific records; no C cap-rent identity
    is imposed. Prices and schedules need not equal C's, even at equal welfare.
    Optimistic LP responses may be nonunique; schedule differences alone do not
    establish a particular consumer's indifference.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be finite and positive')
    ids = community.member_ids
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Community must have unique, nonempty member IDs')
    if centralized.model_type != 'C' or dp.model_type != 'DP':
        raise ValueError('DP analysis requires Model C and Model DP results')
    if dp.individual_rationality and not dp.budget_balance:
        raise ValueError('Individual rationality requires Budget Balance')
    if centralized.solver_status not in (None, GRB.OPTIMAL):
        raise ValueError('The centralized benchmark must be optimal')
    n = community.members[0].n_hours
    if any(member.n_hours != n for member in community.members):
        raise ValueError('All members must have the same horizon')
    for kind, results in (('R', reference), ('I', individual),
                          ('C', centralized.member_results), ('DP', dp.member_results)):
        if set(results) != set(ids) or any(results[cid].consumer_id != cid for cid in ids):
            raise ValueError(f'{kind}: result IDs do not match community members')

    def array(value, label):
        a = np.asarray(value, dtype=float)
        if a.shape != (n,) or not np.all(np.isfinite(a)):
            raise ValueError(f'{label}: expected {n} finite hourly values')
        return a

    def scalar(value, label):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f'{label}: expected a finite value')
        return value

    def check(actual, expected, label):
        if not np.allclose(actual, expected, atol=tolerance, rtol=0):
            raise ValueError(f'DP validation failed: {label}')

    price = array(community.energy_price, 'day-ahead price')
    internal = array(dp.internal_prices, 'DP internal price')
    c_price = array(centralized.balance_duals, 'C balance dual price')  # Pi, no sign flip.
    cap = array(community.contract.capacity_limit_kW, 'DSO capacity limit')
    if np.any(cap < 0):
        raise ValueError('DSO capacity limit must be nonnegative')
    discount = 1 - scalar(community.contract.tariff_discount_ratio, 'tariff discount')
    alpha = scalar(community.contract.penalty_DKK_per_kWh, 'penalty')
    buy = price + discount * community.import_tariff
    sell = price - discount * community.export_tariff
    if np.any(internal < sell - tolerance) or np.any(internal > buy + alpha + tolerance):
        raise ValueError('DP internal prices violate Cap-Only price bounds')
    objectives = {
        'R': sum(scalar(reference[cid].objective_DKK, f'R/{cid} cost') for cid in ids),
        'I': sum(scalar(individual[cid].objective_DKK, f'I/{cid} cost') for cid in ids),
        'C': scalar(centralized.objective_DKK, 'C objective'),
        'DP': scalar(dp.objective_DKK, 'DP objective'),
    }
    efficiency_loss = objectives['DP'] - objectives['C']
    if efficiency_loss < -tolerance:
        raise ValueError('DP validation failed: DP costs less than the optimal C benchmark')

    # Reconstruct both community objectives with the same external tariff rule.
    costs, meters, local_costs = {}, {}, {}
    for kind, result in (('C', centralized), ('DP', dp)):
        imp = array(result.community_import_kW, f'{kind} import')
        exp = array(result.community_export_kW, f'{kind} export')
        if np.any(imp < -tolerance) or np.any(exp < -tolerance):
            raise ValueError(f'{kind}: community import and export must be nonnegative')
        excess = np.maximum(imp - cap, 0.0)
        check(array(result.community_exceedance_kW, f'{kind} exceedance'), excess,
              f'{kind} community exceedance')
        energy = price * (imp - exp)
        tariffs = discount * (community.import_tariff * imp + community.export_tariff * exp)
        penalty = alpha * excess
        external = energy + tariffs + penalty
        meters[kind] = dict(imp=imp, exp=exp, excess=excess, external=external)
        local_costs[kind] = {}
        aggregate_net = np.zeros(n)
        for member in community.members:
            cid = member.consumer_id
            own = result.member_results[cid]
            pv = array(own.pv_kW, f'{kind}/{cid} PV')
            deviation = array(own.deviation_kW, f'{kind}/{cid} deviation')
            load = array(own.load_kW, f'{kind}/{cid} load')
            net = array(own.net_exchange_kW, f'{kind}/{cid} net exchange')
            charge = np.zeros(n) if own.charge_kW is None else array(own.charge_kW, 'charge')
            discharge = (np.zeros(n) if own.discharge_kW is None
                         else array(own.discharge_kW, 'discharge'))
            check(net, load + charge - pv - discharge, f'{kind}/{cid} physical exchange')
            check(deviation, np.abs(load - member.reference_load),
                  f'{kind}/{cid} absolute deviation')
            local_costs[kind][cid] = (
                scalar(member.pv_marginal_cost * pv.sum(), f'{kind}/{cid} PV cost'),
                scalar(member.linear_disutility * deviation.sum(), f'{kind}/{cid} disutility'))
            aggregate_net += net
        check(aggregate_net, imp - exp, f'{kind} community power balance')
        costs[kind] = dict(
            energy_market_DKK=float(energy.sum()), grid_tariffs_DKK=float(tariffs.sum()),
            penalty_DKK=float(penalty.sum()),
            pv_generation_DKK=sum(v[0] for v in local_costs[kind].values()),
            disutility_DKK=sum(v[1] for v in local_costs[kind].values()))
        reconstructed = sum(costs[kind].values())
        check(reconstructed, objectives[kind], f'{kind} objective reconstruction')
        costs[kind]['total_objective_DKK'] = reconstructed

    members = []
    hourly_payments = np.zeros(n)
    for member in community.members:
        cid = member.consumer_id
        own = dp.member_results[cid]
        payments = internal * own.net_exchange_kW
        payment = float(payments.sum())
        pv_cost, disutility = local_costs['DP'][cid]
        settled = payment + pv_cost + disutility
        check(settled, scalar(own.objective_DKK, f'DP/{cid} cost'),
              f'{cid} settled member cost reconstruction')
        gain_r = float(reference[cid].objective_DKK - settled)
        gain_i = float(individual[cid].objective_DKK - settled)
        if dp.individual_rationality and gain_r < -tolerance:
            raise ValueError(f'DP + BB + IR validation failed: {cid} is worse off than Model R')
        members.append(dict(
            consumer_id=cid, archetype=member.archetype, slope_class=member.slope_class,
            has_pv=bool(member.has_pv), has_battery=bool(member.has_battery),
            reference_cost_DKK=float(reference[cid].objective_DKK),
            individual_cost_DKK=float(individual[cid].objective_DKK),
            payment_DP_DKK=payment, pv_generation_DP_DKK=pv_cost,
            disutility_DP_DKK=disutility, settled_cost_DP_DKK=settled,
            gain_vs_R_DKK=gain_r, gain_vs_I_DKK=gain_i,
            individually_rational_vs_R=bool(gain_r >= -tolerance),
            individually_rational_vs_I=bool(gain_i >= -tolerance),
            net_energy_DP_kWh=float(np.sum(own.net_exchange_kW))))
        hourly_payments += payments

    imp, exp = meters['DP']['imp'], meters['DP']['exp']
    excess, external = meters['DP']['excess'], meters['DP']['external']
    aggregate_net = np.sum([dp.member_results[cid].net_exchange_kW for cid in ids], axis=0)
    check(hourly_payments, internal * aggregate_net, 'hourly payment aggregation')
    total_payments = float(hourly_payments.sum())
    check(total_payments, sum(row['payment_DP_DKK'] for row in members), 'daily payment aggregation')
    external_bill = float(external.sum())
    surplus = total_payments - external_bill
    if dp.budget_balance:
        if np.any((imp > tolerance) & (exp > tolerance)):
            raise ValueError('DP + BB validation failed: simultaneous community import and export')
        check(surplus, 0.0, 'hard budget balance from actual prices and exchanges')
    total_settled = sum(row['settled_cost_DP_DKK'] for row in members)
    residual = float(total_settled - objectives['DP'] - surplus)
    check(residual, 0.0, 'sum(member costs) = DP objective + manager surplus')
    status = int(dp.solver_status) if dp.solver_status is not None else None
    status_names = {GRB.OPTIMAL: 'OPTIMAL', GRB.TIME_LIMIT: 'TIME_LIMIT',
                    GRB.SUBOPTIMAL: 'SUBOPTIMAL', GRB.INTERRUPTED: 'INTERRUPTED',
                    GRB.NODE_LIMIT: 'NODE_LIMIT', GRB.SOLUTION_LIMIT: 'SOLUTION_LIMIT'}
    gap = float(dp.mip_gap) if dp.mip_gap is not None else None
    if gap is not None and not np.isfinite(gap):
        gap = None  # Unavailable/infinite solver gaps are represented by JSON null.
    if gap is not None and gap < 0:
        raise ValueError('MIP gap must be nonnegative')
    difference = internal - c_price
    summary = dict(
        community=community.community, day=community.day,
        model_variant=('DP_budget_balance_ir' if dp.individual_rationality else
                       'DP_budget_balance' if dp.budget_balance else 'DP_cap_only'),
        dt_hours=1.0, tolerance=tolerance, relative_tolerance=0.0,
        units=dict(costs='DKK/day', prices='DKK/kWh', power='kW', energy='kWh/day', mip_gap='fraction'),
        objectives_DKK=objectives, efficiency_loss_DKK=efficiency_loss,
        objectives_equal_within_tolerance=bool(abs(efficiency_loss) <= tolerance),
        daily_import_kWh=float(imp.sum()), daily_export_kWh=float(exp.sum()),
        peak_import_kW=float(imp.max()), exceedance_kWh=float(excess.sum()),
        congested_hours=int(np.count_nonzero(excess > tolerance)),
        solver_status=status, solver_status_name=status_names.get(status, 'UNKNOWN' if status is None else f'STATUS_{status}'),
        mip_gap=gap, dp_proven_optimal=bool(status == GRB.OPTIMAL),
        max_absolute_price_difference_DKK_per_kWh=float(np.abs(difference).max()),
        mean_absolute_price_difference_DKK_per_kWh=float(np.abs(difference).mean()),
        total_payments_DKK=total_payments, external_bill_DKK=external_bill,
        external_energy_DKK=costs['DP']['energy_market_DKK'],
        external_tariffs_DKK=costs['DP']['grid_tariffs_DKK'],
        external_penalty_DKK=costs['DP']['penalty_DKK'],
        manager_surplus_DKK=surplus, manager_budget_balanced=bool(abs(surplus) <= tolerance),
        total_settled_member_cost_DKK=total_settled,
        worse_off_vs_R=[row['consumer_id'] for row in members if not row['individually_rational_vs_R']],
        worse_off_vs_I=[row['consumer_id'] for row in members if not row['individually_rational_vs_I']],
        accounting_identity_residual_DKK=residual, validation_passed=True,
        interpretation_note='Optimistic follower responses may be nonunique. Equal community costs do not imply equal prices or schedules. BB and IR are diagnostics only; no surplus is redistributed.')
    if dp.budget_balance:
        summary.update(budget_balance_enforced=True, budget_balance_validation_passed=True,
                       interpretation_note='Optimistic follower responses may be nonunique. Equal community costs do not imply equal prices or schedules. BB is enforced; IR remains diagnostic. No surplus is redistributed.')
    if dp.individual_rationality:
        summary.update(individual_rationality_enforced=True, individual_rationality_validation_passed=True,
                       worse_off_vs_R_count=len(summary['worse_off_vs_R']),
                       interpretation_note='Optimistic follower responses may be nonunique. Equal community costs do not imply equal prices or schedules. BB and IR against Model R are enforced. No surplus is redistributed.')
    hourly = [dict(
        hour=t + 1, day_ahead_price_DKK_per_kWh=float(price[t]),
        discounted_import_price_DKK_per_kWh=float(buy[t]),
        discounted_export_price_DKK_per_kWh=float(sell[t]),
        c_dual_price_DKK_per_kWh=float(c_price[t]),
        dp_internal_price_DKK_per_kWh=float(internal[t]),
        dp_minus_c_price_DKK_per_kWh=float(difference[t]),
        dp_import_kW=float(imp[t]), dp_export_kW=float(exp[t]),
        c_import_kW=float(meters['C']['imp'][t]), c_export_kW=float(meters['C']['exp'][t]),
        cap_kW=float(cap[t]), dp_exceedance_kW=float(excess[t]),
        member_net_exchange_kW=float(aggregate_net[t]),
        payments_collected_DKK=float(hourly_payments[t]), external_bill_DKK=float(external[t]),
        manager_surplus_DKK=float(hourly_payments[t] - external[t])) for t in range(n)]
    return SettlementResult(members, hourly, costs, summary)


def export_dp_analysis(analysis, output_dir):
    """Write all validated DP results to one strict JSON file in the DP directory."""
    payload = dict(summary=dict(analysis.summary, cost_breakdown=analysis.costs),
                   hourly=analysis.hourly, members=analysis.members)
    content = json.dumps(payload, indent=2, allow_nan=False) + '\n'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = analysis.summary['model_variant'].removeprefix('DP_') + '.json'
    output_path = output_dir / filename
    output_path.write_text(content, encoding='utf-8')
    return output_path


# ==========================================================
# CONTRACT SENSITIVITY METRICS AND EXPORTS
# ==========================================================

def analyse_contract_sensitivity(community, reference, individual, centralized, settlement, dimension, value):
    """Return three rows with explicit metering and contract-exceedance definitions."""
    rows = []
    cap = np.asarray(community.contract.capacity_limit_kW)
    r_cost = settlement.costs["R"]["total_objective_DKK"]
    i_cost = settlement.costs["I"]["total_objective_DKK"]
    for kind, results in (("R", reference), ("I", individual),
                           ("C", centralized.member_results)):
        if kind == "C":
            imp = centralized.community_import_kW
            exp = centralized.community_export_kW
        else:
            imp = np.sum([results[cid].import_kW for cid in community.member_ids], axis=0)
            exp = np.sum([results[cid].export_kW for cid in community.member_ids], axis=0)
        net_import = np.maximum(imp - exp, 0)
        aggregate_excess = np.maximum(imp - cap, 0)
        individual_excess = (float(sum(np.maximum(
            results[cid].import_kW - community.contract.individual_capacity_limit_kW[cid], 0
        ).sum() for cid in community.member_ids)) if kind == "I" else None)
        s = settlement.summary
        row = dict(
            dimension=dimension, parameter_value=float(value), model=kind,
            discount=community.contract.tariff_discount_ratio,
            cap_factor=community.contract.capacity_limit_daily_energy_factor,
            cap_min_kW=float(cap.min()), cap_max_kW=float(cap.max()),
            penalty_DKK_per_kWh=community.contract.penalty_DKK_per_kWh,
            **settlement.costs[kind],
            savings_vs_R_DKK=r_cost - settlement.costs[kind]["total_objective_DKK"],
            savings_vs_I_DKK=i_cost - settlement.costs[kind]["total_objective_DKK"],
            metered_daily_import_kWh=float(imp.sum()),
            metered_peak_import_kW=float(imp.max()),
            aggregate_metered_exceedance_kWh=float(aggregate_excess.sum()),
            aggregate_metered_exceedance_hours=int(np.count_nonzero(aggregate_excess > 1e-6)),
            net_feeder_peak_import_kW=float(net_import.max()),
            net_feeder_exceedance_kWh=float(np.maximum(net_import - cap, 0).sum()),
            individual_contract_exceedance_kWh=individual_excess,
            community_contract_exceedance_kWh=float(aggregate_excess.sum()) if kind == "C" else None,
            cap_binding_hours=int(np.count_nonzero(np.abs(imp - cap) <= 1e-6)),
            total_absolute_load_deviation_kWh=float(sum(r.deviation_kW.sum() for r in results.values())),
            manager_surplus_DKK=s["manager_surplus_DKK"] if kind == "C" else None,
            manager_budget_balanced=s["manager_budget_balanced"] if kind == "C" else None,
            members_not_worse_off_vs_R=len(community.members) - len(s["worse_off_vs_R"]) if kind == "C" else None,
            members_not_worse_off_vs_I=len(community.members) - len(s["worse_off_vs_I"]) if kind == "C" else None,
            worst_member_gain_vs_R_DKK=min(r["gain_vs_R_DKK"] for r in settlement.members) if kind == "C" else None,
            worst_member_gain_vs_I_DKK=min(r["gain_vs_I_DKK"] for r in settlement.members) if kind == "C" else None)
        rows.append(row)
    return rows


def export_contract_sensitivity(rows, members, report, output_dir):
    """Save the combined Q1 study tables and parameters under output_dir.

    rows contains R/I/C metrics (DKK/day, kW and kWh/day); members contains
    per-member costs and gains at every sampled value. report records the
    actual contract values, break-even brackets and validation diagnostics.
    No separate copy of each point's settlement files is needed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "metrics.csv", rows)
    _write_csv(output_dir / "member_benefits.csv", members)
    (output_dir / "study.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
