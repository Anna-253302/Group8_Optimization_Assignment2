"""Assignment 2: load inputs, run named experiments, analyse and save results.

Examples:
    python main.py --model all --day d1
    python main.py --model all --day d1 --settlement
    python main.py --model DP --day d1
    python main.py --day d1 --sensitivity both --reference-diagnostics

R, I and C use the shared parametrized model in src/model.py. Experiment
configurations and runners stay here, following Assignment 1. Quantities,
settlement and exports are handled by src/analysis.py; figures by plotting.py.
"""
from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from src.data_loader import load_community, list_communities, with_contract
from src.model import CommunityModel, solve_community, reference_exceedance_range
from src.analysis import (
    analyse_reference,
    analyse_individual_contracts,
    analyse_centralized_contract,
    validate_reference,
    print_three_model_comparison,
    analyse_settlement,
    print_settlement,
    export_settlement,
    analyse_contract_sensitivity,
    export_contract_sensitivity,
    analyse_dp,
    export_dp_analysis,
)
from src.plotting import plot_feeder_import, plot_settlement, plot_contract_sensitivity

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Named experiments: changing a value does not change the provided input files.
CONTRACT_STUDIES = {
    "discount": {
        "values": [0, .1, .2, .3, .4, .5, .6, .7, .75, .8, .9, 1],
        "parameter": "discount",
        "base_attribute": "tariff_discount_ratio",
    },
    "cap": {
        "values": [.8, 1, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2, 2.5, 3, 4, 6, 8],
        "parameter": "cap_factor",
        "base_attribute": "capacity_limit_daily_energy_factor",
    },
}


def run_base_case(
    community, output_dir, *, model="all", settlement=False,
    reference_diagnostics=False, verbose=False,
):
    """Solve the selected base models and save Q1 figures and settlement tables.

    community is the loaded CommunityData; output_dir is a results directory.
    model selects R/I/C, both (R+I), all or none. Settlement needs all three
    benchmarks. Return a dictionary of solved results for reuse; costs are DKK,
    powers kW and daily energies kWh (one-hour intervals).
    """
    selections = {
        "R": ("R",), "I": ("I",), "C": ("C",),
        "both": ("R", "I"), "all": ("R", "I", "C"), "none": (),
    }
    if model not in selections:
        raise ValueError(f"Unknown model selection: {model}")
    if settlement and model != "all":
        raise ValueError("Settlement requires all three member benchmarks")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results, metrics = {}, {}
    titles = {"R": "REFERENCE", "I": "INDIVIDUAL CONTRACTS",
              "C": "CENTRALIZED COMMUNITY CONTRACT"}
    for kind in selections[model]:
        print("\n" + "=" * 60)
        print(f"MODEL {kind} - {titles[kind]}")
        print("=" * 60)
        result = solve_community(community, model_type=kind, verbose=verbose)
        results[kind] = result
        if kind == "R":
            metric = analyse_reference(community, result)
            diagnostic = None
            if reference_diagnostics:
                diagnostic = reference_exceedance_range(community)
                (output_dir / "reference_exceedance_diagnostic.json").write_text(
                    json.dumps(diagnostic, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
            validate_reference(community, metric, exceedance_range=diagnostic)
            print(f"Total objective: {metric.total_objective_DKK:.2f} DKK")
            print(f"Daily gross import: {metric.daily_import_kWh:.2f} kWh")
            print(f"Peak feeder import: {metric.peak_import_kW:.2f} kW")
        elif kind == "I":
            metric = analyse_individual_contracts(community, result)
            print(f"Total objective: {metric.feeder.total_objective_DKK:.2f} DKK")
            print(f"Daily gross import: {metric.feeder.daily_import_kWh:.2f} kWh")
            print(f"Peak feeder import: {metric.feeder.peak_import_kW:.2f} kW")
            print(f"Individual exceedance: {metric.individual_exceedance_kWh:.2f} kWh")
            print(f"Individual penalties: {metric.individual_penalty_DKK:.2f} DKK")
        else:
            metric = analyse_centralized_contract(community, result)
            print(f"Total objective: {result.objective_DKK:.2f} DKK")
            print(f"Daily community import: {sum(result.community_import_kW):.2f} kWh")
            print(f"Daily community export: {sum(result.community_export_kW):.2f} kWh")
            print(f"Peak community import: {max(result.community_import_kW):.2f} kW")
            print(f"Community exceedance: {sum(result.community_exceedance_kW):.2f} kWh")
            print(f"Congested hours: {sum(result.community_exceedance_kW > 1e-6)}")
        metrics[kind] = metric
    if all(kind in metrics for kind in ("R", "I", "C")):
        print_three_model_comparison(metrics["R"], metrics["I"], metrics["C"])
    if settlement:
        payments = analyse_settlement(community, results["R"], results["I"], results["C"])
        print_settlement(payments)
        export_settlement(payments, output_dir)
        plot_settlement(payments, output_dir)
    if results:
        plot_feeder_import(community, results, output_dir)
        print(f"\nFigure saved to: {output_dir}")
    return results


def run_dp_cap_only(community, output_dir, *, verbose=False, budget_balance=False,
                    individual_rationality=False, price_bound_extension: float = 0.0):
    """Solve each of DP/R/I/C once and report the deterministic Q2 experiment.

    output_dir is the DP directory; Q1 runners and their exports
    are separate. IR optionally adds participation constraints against Model R to BB.
    """
    started = time.perf_counter()
    price_bound_extension = float(price_bound_extension)
    if not np.isfinite(price_bound_extension) or price_bound_extension < 0:
        raise ValueError('Price bound extension must be finite and non-negative')
    if individual_rationality and not budget_balance:
        raise ValueError('Individual rationality requires Budget Balance')
    results = {}
    variant = 'Budget Balance' if budget_balance else 'Cap Only'
    if individual_rationality:
        variant += ' + Individual Rationality'
    for kind in ('R', 'I', 'C', 'DP'):
        print(f'\nSolving Model {kind} for DP {variant}...', flush=True)
        options = {'budget_balance': True} if kind == 'DP' and budget_balance else {}
        if kind == 'DP':
            options['price_bound_extension'] = price_bound_extension
        if kind == 'DP' and individual_rationality:
            options.update(individual_rationality=True, reference_costs_DKK={
                cid: result.objective_DKK for cid, result in results['R'].items()})
        results[kind] = solve_community(community, model_type=kind, verbose=verbose, **options)
    analysis = analyse_dp(community, results['R'], results['I'], results['C'], results['DP'])
    output_path = export_dp_analysis(analysis, output_dir)

    s = analysis.summary
    print(f'\nQ2 - DYNAMIC PRICING, {variant.upper()}')
    for kind, cost in s['objectives_DKK'].items():
        print(f'Model {kind:>2} community cost: {cost:.6f} DKK')
    solution = 'optimal within solver tolerance' if s['dp_proven_optimal'] else 'feasible incumbent; optimality not proven'
    gap = 'unavailable' if s['mip_gap'] is None else f"{s['mip_gap']:.6%}"
    print(f"DP solution: {solution}; status={s['solver_status_name']} ({s['solver_status']}), gap={gap}")
    print(f"DP minus C cost: {s['efficiency_loss_DKK']:.9f} DKK; "
          f"equal within {s['tolerance']:g} DKK: {s['objectives_equal_within_tolerance']}")
    print(f"DP import/export: {s['daily_import_kWh']:.3f} / {s['daily_export_kWh']:.3f} kWh; "
          f"peak import: {s['peak_import_kW']:.3f} kW")
    print(f"DP cap exceedance: {s['exceedance_kWh']:.3f} kWh in {s['congested_hours']} hours")
    print(f"DP/C absolute price difference: max {s['max_absolute_price_difference_DKK_per_kWh']:.6f}, "
          f"mean {s['mean_absolute_price_difference_DKK_per_kWh']:.6f} DKK/kWh")
    print(f"Member payments: {s['total_payments_DKK']:.6f} DKK; external grid bill: {s['external_bill_DKK']:.6f} DKK")
    requirement = 'enforced' if budget_balance else 'diagnostic'
    print(f"Manager surplus: {s['manager_surplus_DKK']:.6f} DKK; budget balanced: {s['manager_budget_balanced']} ({requirement})")
    print('Members worse off vs R:', ', '.join(s['worse_off_vs_R']) or 'none')
    if individual_rationality:
        print('Individual rationality against Model R: enforced and validated for all members')
    print('Members worse off vs I:', ', '.join(s['worse_off_vs_I']) or 'none')
    print(f"Accounting residual: {s['accounting_identity_residual_DKK']:.3g} DKK; validation: PASS")
    print(f'DP results saved to {output_path} ({time.perf_counter() - started:.1f} s)')
    return analysis


def diagnose_bb_ir_feasibility(community, reference_results, *, price_bound_extension: float = 0.0):
    """Check the unchanged BB + IR constraints; print results without exporting.

    The objective/search overrides apply only to this disposable diagnostic
    model. Financial checks use actual prices and schedules, not KKT dual bills.
    """
    reference_costs = {cid: result.objective_DKK for cid, result in reference_results.items()}
    optimization = CommunityModel(
        community, model_type='DP', budget_balance=True, individual_rationality=True,
        reference_costs_DKK=reference_costs, verbose=True,
        price_bound_extension=price_bound_extension)
    model = optimization.model
    try:
        optimization.build()
        model.update()
        original_objective = model.getObjective()
        # Diagnostic-only override: the standard community objective is unchanged.
        model.setObjective(0.0, GRB.MINIMIZE)
        model.Params.TimeLimit = 300
        model.Params.MIPFocus = 1
        model.Params.SolutionLimit = 1
        model.Params.OutputFlag = 1
        model.optimize()
        names = {GRB.OPTIMAL: 'OPTIMAL', GRB.INFEASIBLE: 'INFEASIBLE',
                 GRB.TIME_LIMIT: 'TIME_LIMIT', GRB.SOLUTION_LIMIT: 'SOLUTION_LIMIT',
                 GRB.INTERRUPTED: 'INTERRUPTED', GRB.INF_OR_UNBD: 'INF_OR_UNBD'}
        report = dict(solver_status=model.Status,
                      solver_status_name=names.get(model.Status, f'STATUS_{model.Status}'),
                      solution_count=model.SolCount, runtime_seconds=model.Runtime)
        print('\n=== DP + BB + IR FEASIBILITY DIAGNOSTIC ===')
        print(f"Solver status: {report['solver_status_name']} ({model.Status})")
        print(f'Feasible solutions found: {model.SolCount}')
        print(f'Runtime [s]: {model.Runtime:.3f}')
        if model.SolCount > 0:
            tolerance = 1e-5  # Absolute DKK/kW tolerances; one-hour intervals.
            payments, local_cost, violations = 0.0, 0.0, {}
            for cid, block in optimization.blocks.items():
                payment = sum(optimization.internal_prices[t].X * block.net_exchange[t].getValue()
                              for t in optimization.T)
                local = block.physical_cost.getValue() + block.disutility_cost.getValue()
                payments += payment
                local_cost += local
                violations[cid] = max(0.0, payment + local - reference_costs[cid])
            actual_cost = original_objective.getValue()  # Diagnostic ObjVal is zero.
            external = (optimization.objective_components['procurement'].getValue()
                        + optimization.objective_components['penalty'].getValue())
            bb_residual = payments - external
            objective_residual = actual_cost - external - local_cost
            maximum_ir = max(violations.values(), default=0.0)
            violating = [cid for cid, value in violations.items() if value > tolerance]
            passed = bool(np.all(np.isfinite([actual_cost, external, payments, local_cost,
                                             *violations.values()]))
                          and abs(bb_residual) <= tolerance and maximum_ir <= tolerance
                          and abs(objective_residual) <= tolerance)
            report.update(actual_community_cost_DKK=actual_cost, external_bill_DKK=external,
                          total_payments_DKK=payments, bb_residual_DKK=bb_residual,
                          objective_residual_DKK=objective_residual,
                          maximum_ir_violation_DKK=maximum_ir,
                          sum_ir_violations_DKK=sum(violations.values()),
                          violating_member_count=len(violating), violating_members=violating,
                          validation_passed=passed)
            print(f'Actual community cost [DKK]: {actual_cost:.9f}')
            print(f'BB residual [DKK]: {bb_residual:.9g}')
            print(f'Maximum / sum of positive IR violations [DKK]: {maximum_ir:.9g} / {sum(violations.values()):.9g}')
            print(f'IR-violating members: {len(violating)}; IDs: {", ".join(violating) or "none"}')
            print(f'Community objective reconstruction residual [DKK]: {objective_residual:.9g}')
            report['feasibility_conclusion'] = 'validated feasible' if passed else 'candidate failed validation'
        elif model.Status == GRB.INFEASIBLE:
            report['feasibility_conclusion'] = 'proven infeasible within solver tolerances'
        else:
            report['feasibility_conclusion'] = 'unresolved; no incumbent, infeasibility not proven'
        print(f"Feasibility conclusion: {report['feasibility_conclusion']}")
        if model.SolCount > 0 and not report['validation_passed']:
            raise ValueError('Feasibility candidate failed independent BB/IR/cost validation')
        return report
    finally:
        model.dispose()


def diagnose_bb_ir_slacks(community, reference_results, *, price_bound_extension: float = 0.0):
    """Minimize IR violations in an isolated diagnostic with hard BB intact."""
    reference_costs = {cid: float(result.objective_DKK) for cid, result in reference_results.items()}
    if set(reference_costs) != set(community.member_ids):
        raise ValueError('Reference costs must match all community members')
    if not all(np.isfinite(value) for value in reference_costs.values()):
        raise ValueError('Model R reference costs must be finite')
    optimization = CommunityModel(
        community, model_type='DP', budget_balance=True, individual_rationality=False,
        verbose=True, price_bound_extension=price_bound_extension)
    model = optimization.model
    try:
        optimization.build()
        model.update()
        original_objective = model.getObjective()
        member_ids = list(optimization.blocks)
        slacks = model.addVars(member_ids, lb=0.0, name='ir_slack')
        for cid in member_ids:
            model.addConstr(optimization.follower_dual_objectives[cid]
                            <= reference_costs[cid] + slacks[cid], name=f'ir_relaxed_{cid}')
        # Diagnostic only: the ordinary community-cost objective is unchanged.
        model.setObjective(gp.quicksum(slacks[cid] for cid in member_ids), GRB.MINIMIZE)
        model.Params.TimeLimit = 300
        model.Params.MIPFocus = 1
        model.Params.OutputFlag = 1
        model.optimize()  # Keep the default SolutionLimit; improve the slack objective.
        names = {GRB.OPTIMAL: 'OPTIMAL', GRB.INFEASIBLE: 'INFEASIBLE',
                 GRB.TIME_LIMIT: 'TIME_LIMIT', GRB.SOLUTION_LIMIT: 'SOLUTION_LIMIT',
                 GRB.INTERRUPTED: 'INTERRUPTED', GRB.INF_OR_UNBD: 'INF_OR_UNBD',
                 GRB.UNBOUNDED: 'UNBOUNDED'}
        try:
            bound = float(model.ObjBound)
            if not np.isfinite(bound):
                bound = None
        except (gp.GurobiError, AttributeError):
            bound = None
        report = dict(solver_status=model.Status,
                      solver_status_name=names.get(model.Status, f'STATUS_{model.Status}'),
                      solution_count=model.SolCount, runtime_seconds=model.Runtime,
                      best_bound_DKK=bound)
        print('\n=== DP + BB IR-SLACK DIAGNOSTIC ===')
        print(f"Solver status: {report['solver_status_name']} ({model.Status})")
        print(f'Runtime [s]: {model.Runtime:.3f}')
        print(f'SolCount: {model.SolCount}')
        print(f'Best bound [DKK]: {bound if bound is not None else "unavailable"}')
        if model.SolCount > 0:
            tolerance = 1e-5  # Absolute DKK tolerance, with one-hour intervals.
            rows, payments, local_cost = [], 0.0, 0.0
            for cid, block in optimization.blocks.items():
                payment = sum(optimization.internal_prices[t].X * block.net_exchange[t].getValue()
                              for t in optimization.T)
                local = block.physical_cost.getValue() + block.disutility_cost.getValue()
                settled = payment + local
                rows.append(dict(consumer_id=cid, reference_cost_DKK=reference_costs[cid],
                                 settled_cost_DKK=settled,
                                 ir_violation_DKK=max(0.0, settled - reference_costs[cid]),
                                 slack_DKK=float(slacks[cid].X)))
                payments += payment
                local_cost += local
            rows.sort(key=lambda row: row['ir_violation_DKK'], reverse=True)
            best_slack = float(model.ObjVal)  # Total slack, NOT community cost.
            total_slack = sum(row['slack_DKK'] for row in rows)
            actual_violation = sum(row['ir_violation_DKK'] for row in rows)
            maximum_ir = max((row['ir_violation_DKK'] for row in rows), default=0.0)
            external = (optimization.objective_components['procurement'].getValue()
                        + optimization.objective_components['penalty'].getValue())
            actual_cost = original_objective.getValue()
            bb_residual = payments - external
            cost_residual = actual_cost - external - local_cost
            slack_residual = actual_violation - total_slack
            gap = float(model.MIPGap)
            passed = bool(np.all(np.isfinite([best_slack, actual_cost, external, payments, local_cost,
                                             *(value for row in rows for key, value in row.items()
                                               if key != 'consumer_id')]))
                          and all(row['slack_DKK'] >= -tolerance
                                  and row['ir_violation_DKK'] <= row['slack_DKK'] + tolerance
                                  for row in rows)
                          and abs(best_slack - total_slack) <= tolerance
                          and abs(slack_residual) <= tolerance
                          and abs(bb_residual) <= tolerance and abs(cost_residual) <= tolerance)
            report.update(best_total_ir_slack_DKK=best_slack,
                          mip_gap=gap if np.isfinite(gap) else None,
                          actual_community_cost_DKK=actual_cost, external_bill_DKK=external,
                          total_payments_DKK=payments, bb_residual_DKK=bb_residual,
                          maximum_ir_violation_DKK=maximum_ir,
                          actual_total_ir_violation_DKK=actual_violation,
                          slack_reconstruction_residual_DKK=slack_residual,
                          objective_residual_DKK=cost_residual,
                          violating_member_count=sum(row['ir_violation_DKK'] > tolerance for row in rows),
                          members=rows, validation_passed=passed)
            print(f'Best total IR slack [DKK]: {best_slack:.9f}')
            print(f'MIP gap: {gap:.6%}' if np.isfinite(gap) else 'MIP gap: unavailable')
            print(f'Actual community cost [DKK]: {actual_cost:.9f}')
            print(f'BB residual [DKK]: {bb_residual:.9g}')
            print(f'Maximum / total actual IR violation [DKK]: {maximum_ir:.9g} / {actual_violation:.9g}')
            print(f'Actual violation minus total slack [DKK]: {slack_residual:.9g}')
            print(f'Community-cost reconstruction residual [DKK]: {cost_residual:.9g}')
            print(f'{"Member":<16} {"R cost":>12} {"DP cost":>12} {"IR violation":>14} {"IR slack":>12}  [DKK]')
            for row in rows:
                print(f"{row['consumer_id']:<16} {row['reference_cost_DKK']:>12.6f} "
                      f"{row['settled_cost_DKK']:>12.6f} {row['ir_violation_DKK']:>14.6f} "
                      f"{row['slack_DKK']:>12.6f}")
            print(f'Validation: {"PASS" if passed else "FAIL"}')
            if not passed:
                conclusion = 'candidate failed independent validation; no validated feasibility conclusion'
            elif total_slack <= tolerance:
                conclusion = 'BB + IR is feasible within numerical tolerances'
            elif model.Status == GRB.OPTIMAL:
                conclusion = 'positive optimal total IR slack; hard BB + IR is infeasible within solver tolerances'
            elif bound is not None and bound > tolerance:
                conclusion = 'positive lower bound rules out zero total IR slack under the current assumptions'
            else:
                conclusion = 'positive-slack incumbent is only an upper bound; minimum IR relaxation remains unresolved'
        elif model.Status == GRB.INFEASIBLE:
            conclusion = 'relaxed diagnostic proven infeasible; the difficulty is not solely hard IR'
            print('No feasible diagnostic incumbent was found.')
        else:
            conclusion = 'no feasible diagnostic incumbent; minimum IR relaxation remains unresolved'
            print('No feasible diagnostic incumbent was found.')
        report['feasibility_conclusion'] = conclusion
        print(f'Conclusion: {conclusion}')
        if model.SolCount > 0 and not report['validation_passed']:
            raise ValueError('IR-slack candidate failed independent slack/BB/cost validation')
        return report
    finally:
        model.dispose()


def refine_crossings(points, evaluate, tolerance=.005):
    """Bracket zeroes of savings vs R, refining only observed sign changes.

    points maps parameter values to signed savings in DKK. evaluate(value)
    solves/evaluates another point. Returned brackets are sampled thresholds,
    not interpolated exact break-even values or claims outside the study range.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Refinement tolerance must be finite and positive")
    ordered = sorted(points)
    brackets = []
    for low, high in zip(ordered, ordered[1:]):
        gl, gh = float(points[low]), float(points[high])
        if gl * gh >= 0:
            continue
        for _ in range(40):
            if high - low <= tolerance:
                break
            mid = float((low + high) / 2)
            gm = float(evaluate(mid))
            if not np.isfinite(gm):
                raise ValueError("Nonfinite savings during threshold refinement")
            if gm == 0:
                low = high = mid
                gl = gh = 0.0
                break
            if gl * gm < 0:
                high, gh = mid, gm
            else:
                low, gl = mid, gm
        brackets.append(dict(lower=low, upper=high,
                             savings_at_lower_DKK=gl, savings_at_upper_DKK=gh))
    return brackets


def run_contract_sensitivity(
    community, output_dir, *, dimensions=("discount", "cap"), discounts=None,
    cap_factors=None, refine=True, solver=solve_community, reference_diagnostics=False,
):
    """Solve R once, then I and C per contract; export metrics and settlement.

    Discount and cap studies each start from an untouched copy of the base case.
    Values near observed aggregate break-even points are refined to brackets of
    width <= .005 in beta or f. Per-member IR counts describe returned LP duals.
    solver is injectable for independent verification of the same model matrix.
    """

    dimensions = tuple(dimensions)
    if not dimensions or any(d not in ("discount", "cap") for d in dimensions):
        raise ValueError("Study dimensions must be discount and/or cap")
    values_by_dimension = {
        "discount": list(CONTRACT_STUDIES["discount"]["values"] if discounts is None else discounts),
        "cap": list(CONTRACT_STUDIES["cap"]["values"] if cap_factors is None else cap_factors),
    }
    # Validate every requested value before starting any optimization.
    for dimension in dimensions:
        if len(values_by_dimension[dimension]) < 2:
            raise ValueError("Each study needs at least two parameter values")
        for value in values_by_dimension[dimension]:
            with_contract(community, **{CONTRACT_STUDIES[dimension]["parameter"]: value})
    started = time.perf_counter()
    reference = solver(community, model_type="R")
    reference_metrics = analyse_reference(community, reference)
    diagnostic = None
    if reference_diagnostics:
        diagnostic = reference_exceedance_range(community)
    reference_checks = validate_reference(community, reference_metrics, exceedance_range=diagnostic)
    if any(not passed for key, passed in reference_checks.items() if key != "exceedance_kWh"):
        raise ValueError("Primary R benchmark checks failed; investigate before the sensitivity study")

    output_dir = Path(output_dir)
    reports = {}
    for dimension in dimensions:
        folder = output_dir / dimension
        folder.mkdir(parents=True, exist_ok=True)
        study = CONTRACT_STUDIES[dimension]
        base_value = getattr(community.contract, study["base_attribute"])
        coarse_values = sorted(set(float(x) for x in values_by_dimension[dimension]) | {float(base_value)})
        cache = {}

        def evaluate(value):
            value = float(value)
            if value not in cache:
                case = with_contract(community, **{
                    study["parameter"]: value})
                point_start = time.perf_counter()
                individual = solver(case, model_type="I")
                centralized = solver(case, model_type="C")
                settlement = analyse_settlement(case, reference, individual, centralized)
                rows = analyse_contract_sensitivity(case, reference, individual, centralized, settlement, dimension, value)
                if rows[2]["total_objective_DKK"] > rows[1]["total_objective_DKK"] + 1e-5:
                    raise ValueError("C costs more than I under the same contract")
                cache[value] = (rows, settlement, time.perf_counter() - point_start)
                print(f"{dimension}={value:.5f}: I={rows[1]['total_objective_DKK']:.2f}, "
                      f"C={rows[2]['total_objective_DKK']:.2f} DKK; "
                      f"C members not worse off vs R={rows[2]['members_not_worse_off_vs_R']}/{len(case.members)}")
            return cache[value]

        for value in coarse_values:
            evaluate(value)
        thresholds = {}
        for kind, index in (("I", 1), ("C", 2)):
            gains = {value: cache[value][0][index]["savings_vs_R_DKK"] for value in coarse_values}
            brackets = (refine_crossings(gains, lambda value, j=index:
                        evaluate(value)[0][j]["savings_vs_R_DKK"]) if refine else [])
            thresholds[kind] = dict(
                break_even_brackets=brackets,
                sampled_zero_savings=[v for v, g in gains.items() if abs(g) <= 1e-5],
                minimum_coarse_savings_DKK=min(gains.values()),
                maximum_coarse_savings_DKK=max(gains.values()))
        rows = [row for value in sorted(cache) for row in cache[value][0]]
        for index, kind in ((1, "I"), (2, "C")):
            costs = [cache[value][0][index]["total_objective_DKK"] for value in sorted(cache)]
            if np.any(np.diff(costs) > 1e-5):
                raise ValueError(f"{kind} cost increased as {dimension} increased")
        members = [dict(dimension=dimension, parameter_value=value, **member)
                   for value in sorted(cache) for member in cache[value][1].members]
        report = dict(
            community=community.community, day=community.day, dimension=dimension,
            base_value=base_value, coarse_values=coarse_values, evaluated_values=sorted(cache),
            thresholds_against_R=thresholds, reference_checks=reference_checks,
            reference_exceedance_diagnostic=diagnostic,
            fixed_penalty_DKK_per_kWh=community.contract.penalty_DKK_per_kWh,
            fixed_discount=community.contract.tariff_discount_ratio if dimension == "cap" else None,
            fixed_cap_factor=community.contract.capacity_limit_daily_energy_factor if dimension == "discount" else None,
            hypothesis="I/C optimal costs cannot increase with more discount or more cap; R is fixed. C <= I. Individual gains need not be monotone.",
            units="One-hour intervals; costs DKK/day; powers kW; exceedances kWh/day",
            metering_note="R/I aggregate own-meter gross imports; C uses the shared meter. Individual contractual exceedance is separate from aggregate exceedance. Net feeder diagnostics use positive aggregate net exchange.",
            interpretation_note="No surplus redistribution. Counts refer to returned optimal dispatch/duals and do not prove uniqueness or feasibility under all optimal prices. A break-even bracket concerns aggregate dispatch cost, not every member's participation.",
            elapsed_seconds_by_value={str(value): cache[value][2] for value in sorted(cache)},
            validation_passed=True)
        export_contract_sensitivity(rows, members, report, folder)
        plot_contract_sensitivity(rows, folder, dimension, base_value, len(community.members), thresholds)
        reports[dimension] = report
    print(f"Sensitivity results saved to {output_dir} ({time.perf_counter() - started:.1f} s)")
    return reports


def main() -> None:
    """Parse the experiment options and run the selected reproducible command."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default="base",
                    help=f"base or your own community; available: {list_communities()}")
    ap.add_argument("--day", default="d1", help="d1 (base case) ... d7")
    ap.add_argument("--members", action="store_true", help="Print every member's input summary")
    ap.add_argument("--demo", action="store_true", help="Run the Question 3 scenario demonstration")
    ap.add_argument("--model", choices=["R", "I", "C", "DP", "both", "all", "none"], default="all",
                    help="R, I, C, DP (Q2 Cap Only), both (R+I), all (R+I+C), or none")
    ap.add_argument("--verbose-solver", action="store_true", help="Show Gurobi solver output")
    ap.add_argument("--budget-balance", action="store_true", help="Enforce budget balance (requires --model DP)")
    ap.add_argument("--price-bound-extension", type=float, default=0.0,
                    help="Symmetric extension of DP internal price bounds in DKK/kWh")
    ap.add_argument("--individual-rationality", action="store_true",
                    help="Enforce individual rationality for DP + Budget Balance")
    diagnostics = ap.add_mutually_exclusive_group()
    diagnostics.add_argument("--feasibility-check", action="store_true",
                             help="Run feasibility-only diagnosis for DP + BB + IR")
    diagnostics.add_argument("--ir-slack-check", action="store_true",
                             help="Minimize individual rationality violations under hard Budget Balance")
    ap.add_argument("--settlement", action="store_true",
                    help="Calculate member benefits and manager budget (requires --model all)")
    ap.add_argument("--sensitivity", choices=["discount", "cap", "both"],
                    help="Run Q1 contract studies (requires --model all)")
    ap.add_argument("--discount-values", nargs="+", type=float,
                    help="Coarse beta values; base beta is always included")
    ap.add_argument("--cap-factors", nargs="+", type=float,
                    help="Coarse absolute cap-energy factors f; base f is always included")
    ap.add_argument("--no-refine", action="store_true", help="Skip break-even refinement")
    ap.add_argument("--reference-diagnostics", action="store_true",
                    help="Certify R's exceedance range across alternative economic optima")
    args = ap.parse_args()
    if not np.isfinite(args.price_bound_extension) or args.price_bound_extension < 0:
        ap.error('--price-bound-extension must be finite and non-negative')
    if args.price_bound_extension != 0 and args.model != 'DP':
        ap.error('Nonzero --price-bound-extension requires --model DP')
    if args.feasibility_check and (args.model != 'DP' or not args.budget_balance or not args.individual_rationality):
        ap.error('--feasibility-check requires --model DP --budget-balance --individual-rationality')
    if args.ir_slack_check and (args.model != 'DP' or not args.budget_balance or not args.individual_rationality):
        ap.error('--ir-slack-check requires --model DP --budget-balance --individual-rationality')
    if args.budget_balance and args.model != 'DP':
        ap.error('--budget-balance requires --model DP')
    if args.individual_rationality and (args.model != 'DP' or not args.budget_balance):
        ap.error('--individual-rationality requires --model DP --budget-balance')
    if args.model == "DP" and (args.settlement or args.sensitivity or args.reference_diagnostics):
        ap.error("DP has its own analysis; --settlement, --sensitivity and --reference-diagnostics are Q1-only")
    if (args.settlement or args.sensitivity) and args.model != "all":
        ap.error("--settlement/--sensitivity requires --model all to calculate both member benchmarks")
    if (args.discount_values or args.cap_factors or args.no_refine) and not args.sensitivity:
        ap.error("Sweep values and --no-refine require --sensitivity")
    community = load_community(args.community, day=args.day)
    output_dir = RESULTS_DIR / args.community / args.day
    print(community.summary(), "\n")
    if args.sensitivity:
        dimensions = ("discount", "cap") if args.sensitivity == "both" else (args.sensitivity,)
        run_contract_sensitivity(
            community, output_dir / "sensitivity", dimensions=dimensions,
            discounts=args.discount_values, cap_factors=args.cap_factors,
            refine=not args.no_refine,
            solver=partial(solve_community, verbose=args.verbose_solver),
            reference_diagnostics=args.reference_diagnostics)
        return
    if args.members:
        for member in community.members:
            print(member.summary(), "\n")
    if args.feasibility_check or args.ir_slack_check:
        reference = solve_community(community, model_type='R', verbose=args.verbose_solver)
        if args.ir_slack_check:
            diagnose_bb_ir_slacks(community, reference, price_bound_extension=args.price_bound_extension)
        else:
            diagnose_bb_ir_feasibility(community, reference, price_bound_extension=args.price_bound_extension)
        return
    if args.model == "DP":
        run_dp_cap_only(community, output_dir / "DP", verbose=args.verbose_solver,
                        budget_balance=args.budget_balance, individual_rationality=args.individual_rationality,
                        price_bound_extension=args.price_bound_extension)
    else:
        run_base_case(community, output_dir, model=args.model, settlement=args.settlement,
                      reference_diagnostics=args.reference_diagnostics, verbose=args.verbose_solver)
    if args.demo:
        import runpy
        runpy.run_module("src.scenario_generation", run_name="__main__")


if __name__ == "__main__":
    main()
