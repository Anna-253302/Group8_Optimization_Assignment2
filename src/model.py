"""Assignment 2 consumer optimization building blocks and Q1 models R, I and C.

Adapted from Assignment 1 FlexibleConsumerModel: physical balance, PV,
linear absolute-deviation disutility, daily energy and cyclic battery SoC.

R is independent member LPs, with no DSO contract in the objective.
All hourly power values are average kW over one-hour intervals (numerically kWh).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from .data_loader import CommunityData, InputData

# ==========================================================
# MODEL CONFIGURATION
# ==========================================================

@dataclass(frozen=True)
class ModelConfiguration:
    """Contract and metering configuration for Models R, I and C."""

    model_type: Literal["R", "I", "C"]

    tariff_discount: float

    metering: Literal["individual", "community"]

    capacity_contract: Literal["none", "individual", "community"]


def get_model_configuration(
    community: CommunityData,
    model_type: str,
) -> ModelConfiguration:
    """Return the configuration corresponding to the selected model."""

    beta = float(community.contract.tariff_discount_ratio)

    configurations = {
        "R": ModelConfiguration(
            model_type="R",
            tariff_discount=0.0,
            metering="individual",
            capacity_contract="none",
        ),
        "I": ModelConfiguration(
            model_type="I",
            tariff_discount=beta,
            metering="individual",
            capacity_contract="individual",
        ),
        "C": ModelConfiguration(
            model_type="C",
            tariff_discount=beta,
            metering="community",
            capacity_contract="community",
        ),
    }

    if model_type not in configurations:
        raise ValueError(f"Unknown model type: {model_type}")

    return configurations[model_type]


@dataclass
class ConsumerBlock:
    """Variables and objective expressions for one member inside any Gurobi model."""
    member: InputData
    variables: dict[str, Any]
    constraints: dict[str, Any]
    physical_cost: gp.LinExpr
    disutility_cost: gp.LinExpr
    net_exchange: dict[int, gp.LinExpr]


def add_consumer_constraints(
    model: gp.Model,
    member: InputData,
    *,
    terminal_soc: str = "cyclic",
) -> ConsumerBlock:
    """Add one member's Assignment 1 LP constraints to an EXISTING model.

    No external grid purchase variables are created here: net_exchange[t]
    is the member's load + charging - PV - discharging. R/I/C can use
    that expression with different external-meter and payment structures.

    terminal_soc='cyclic' requires SoC[24] == SoC[0].
    terminal_soc='free' leaves final SoC unconstrained (not recommended for R).
    """
    if terminal_soc not in {"cyclic", "free"}:
        raise ValueError("terminal_soc must be 'cyclic' or 'free'")
    if member.reference_load is None or member.linear_disutility is None:
        raise ValueError(f"{member.consumer_id}: linear reference load required")
    if member.min_daily_energy_kWh is None:
        raise ValueError(f"{member.consumer_id}: daily minimum energy required")
    if member.load_lower_kWh is None or member.load_upper_kWh is None:
        raise ValueError(f"{member.consumer_id}: hourly bounds required")
    n = member.n_hours
    T = range(n)
    name = member.consumer_id
    v: dict[str, Any] = {}
    c: dict[str, Any] = {}
    v['load'] = model.addVars(T, lb=0, name=f'{name}_load')
    v['pv'] = model.addVars(T, lb=0, name=f'{name}_pv')
    v['deviation'] = model.addVars(T, lb=0, name=f'{name}_deviation')
    c['load_lower'] = model.addConstrs(
        (v['load'][t] >= float(member.load_lower_kWh[t]) for t in T),
        name=f'{name}_load_lower')
    c['load_upper'] = model.addConstrs(
        (v['load'][t] <= float(member.load_upper_kWh[t]) for t in T),
        name=f'{name}_load_upper')
    c['daily_energy'] = model.addConstr(
        gp.quicksum(v['load'][t] for t in T) >= member.min_daily_energy_kWh,
        name=f'{name}_daily_energy')
    c['pv_available'] = model.addConstrs(
        (v['pv'][t] <= float(member.pv_available[t]) for t in T),
        name=f'{name}_pv_available')
    c['deviation_positive'] = model.addConstrs(
        (v['deviation'][t] >= v['load'][t] - float(member.reference_load[t]) for t in T),
        name=f'{name}_dev_pos')
    c['deviation_negative'] = model.addConstrs(
        (v['deviation'][t] >= float(member.reference_load[t]) - v['load'][t] for t in T),
        name=f'{name}_dev_neg')

    charge = {t: 0.0 for t in T}
    discharge = {t: 0.0 for t in T}
    if member.has_battery:
        v['charge'] = model.addVars(T, lb=0, ub=member.battery_max_charge_kW,
                                    name=f'{name}_charge')
        v['discharge'] = model.addVars(T, lb=0, ub=member.battery_max_discharge_kW,
                                       name=f'{name}_discharge')
        v['soc'] = model.addVars(range(n + 1), lb=0,
                                 ub=member.battery_capacity_kWh, name=f'{name}_soc')
        c['initial_soc'] = model.addConstr(
            v['soc'][0] == member.battery_initial_soc_kWh, name=f'{name}_initial_soc')
        c['soc_balance'] = model.addConstrs((
            v['soc'][t+1] == v['soc'][t]
            + member.battery_charging_efficiency * v['charge'][t]
            - v['discharge'][t] / member.battery_discharging_efficiency
            for t in T), name=f'{name}_soc_balance')
        if terminal_soc == 'cyclic':
            c['final_soc'] = model.addConstr(v['soc'][n] == v['soc'][0],
                                             name=f'{name}_final_soc')
        charge = v['charge']
        discharge = v['discharge']

    net_exchange = {
        t: v['load'][t] + charge[t] - v['pv'][t] - discharge[t]
        for t in T
    }
    physical_cost = gp.quicksum(member.pv_marginal_cost * v['pv'][t] for t in T)
    disutility_cost = gp.quicksum(member.linear_disutility * v['deviation'][t] for t in T)
    return ConsumerBlock(member, v, c, physical_cost, disutility_cost, net_exchange)


@dataclass
class ReferenceResult:
    """Primal solution and objective for one member in R or I; C uses diagnostic flows."""
    consumer_id: str
    objective_DKK: float
    load_kW: np.ndarray
    pv_kW: np.ndarray
    import_kW: np.ndarray
    export_kW: np.ndarray
    net_exchange_kW: np.ndarray
    deviation_kW: np.ndarray
    soc_kWh: np.ndarray | None
    charge_kW: np.ndarray | None
    discharge_kW: np.ndarray | None

@dataclass
class CommunityResult:
    """Centralized dispatch: connection-point flows and community balance duals."""
    model_type: str
    objective_DKK: float
    member_results: dict[str, ReferenceResult]
    community_import_kW: np.ndarray
    community_export_kW: np.ndarray
    community_exceedance_kW: np.ndarray
    balance_duals: np.ndarray


# ==========================================================
# COMMON MODEL INTERFACE
# ==========================================================

def solve_community(
    community: CommunityData,
    *,
    model_type: Literal["R", "I", "C"] = "R",
    terminal_soc: str = "cyclic",
    verbose: bool = False,
):
    """
    Solve the selected community optimization configuration.

    R: Reference
    I: Individual contracts
    C: Centralized community contract
    """

    optimization = CommunityModel(
        community,
        model_type=model_type,
        terminal_soc=terminal_soc,
        verbose=verbose,
    )

    optimization.build()

    return optimization.solve()

def solve_individual_member(
    member: InputData,
    *,
    contract=None,
    individual_cap_kW=None,
    terminal_soc: str = "cyclic",
    verbose: bool = False,
) -> ReferenceResult:
    """
    Solve one consumer under Model R or Model I.

    Model R: contract=None
    Model I: contract provided, with individual capacity limit.
    """

    model = gp.Model(
        f'{"I" if contract is not None else "R"}_{member.consumer_id}'
    )
    model.Params.OutputFlag = int(verbose)

    block = add_consumer_constraints(
        model, member, terminal_soc=terminal_soc
    )

    T = range(member.n_hours)

    # Individual electricity meter
    imp = model.addVars(T, lb=0, name="grid_import")
    exp = model.addVars(T, lb=0, name="grid_export")

    model.addConstrs(
        (
            imp[t] - exp[t] == block.net_exchange[t]
            for t in T
        ),
        name="individual_meter_balance",
    )

    if member.max_import_kW is not None:
        model.addConstrs(
            (imp[t] <= member.max_import_kW for t in T),
            name="import_limit",
        )

    if member.max_export_kW is not None:
        model.addConstrs(
            (exp[t] <= member.max_export_kW for t in T),
            name="export_limit",
        )

    # --------------------------------------------------------
    # Grid tariffs: full tariffs in R, discounted tariffs in I
    # --------------------------------------------------------

    beta = (
        0.0
        if contract is None
        else float(contract.tariff_discount_ratio)
    )

    procurement = gp.quicksum(
        (
            float(member.energy_price[t])
            + (1 - beta) * member.import_tariff
        ) * imp[t]
        -
        (
            float(member.energy_price[t])
            - (1 - beta) * member.export_tariff
        ) * exp[t]
        for t in T
    )

    # --------------------------------------------------------
    # Congestion penalty: only applies to Model I
    # --------------------------------------------------------

    penalty = gp.LinExpr(0.0)

    if contract is not None:

        if individual_cap_kW is None:
            raise ValueError("Model I requires individual_cap_kW")

        cap = np.asarray(individual_cap_kW, dtype=float)

        if cap.shape != (member.n_hours,) or np.any(cap < 0):
            raise ValueError("Invalid individual capacity limit")

        excess = model.addVars(
            T, lb=0, name="individual_exceedance"
        )

        model.addConstrs(
            (
                excess[t] >= imp[t] - float(cap[t])
                for t in T
            ),
            name="individual_cap_exceedance",
        )

        penalty = (
            float(contract.penalty_DKK_per_kWh)
            * gp.quicksum(excess[t] for t in T)
        )

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    model.setObjective(
        procurement
        + block.physical_cost
        + block.disutility_cost
        + penalty,
        GRB.MINIMIZE,
    )

    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(
            f"Consumer {member.consumer_id}: "
            f"Gurobi status {model.Status}"
        )

    # --------------------------------------------------------
    # Extract results
    # --------------------------------------------------------

    v = block.variables

    def arr(x):
        return np.array([x[t].X for t in T], dtype=float)

    soc = (
        np.array(
            [v["soc"][t].X for t in range(member.n_hours + 1)]
        )
        if member.has_battery
        else None
    )

    return ReferenceResult(
        consumer_id=member.consumer_id,
        objective_DKK=model.ObjVal,
        load_kW=arr(v["load"]),
        pv_kW=arr(v["pv"]),
        import_kW=arr(imp),
        export_kW=arr(exp),
        net_exchange_kW=arr(imp) - arr(exp),
        deviation_kW=arr(v["deviation"]),
        soc_kWh=soc,
        charge_kW=arr(v["charge"]) if member.has_battery else None,
        discharge_kW=arr(v["discharge"]) if member.has_battery else None,
    )


def solve_reference_member(
    member: InputData,
    *,
    terminal_soc: str = "cyclic",
    verbose: bool = False,
) -> ReferenceResult:
    """Solve one consumer under Reference Model R."""

    return solve_individual_member(
        member,
        terminal_soc=terminal_soc,
        verbose=verbose,
    )


def solve_reference(community: CommunityData, *, terminal_soc: str = 'cyclic',
                    verbose: bool = False) -> dict[str, ReferenceResult]:
    """Solve all members independently, as required by reference model R."""
    return {m.consumer_id: solve_reference_member(m, terminal_soc=terminal_soc,
                                                  verbose=verbose)
            for m in community.members}

def solve_individual_contracts(
    community: CommunityData,
    *,
    terminal_soc: str = "cyclic",
    verbose: bool = False,
) -> dict[str, ReferenceResult]:
    """
    Model I: Individual DSO contracts.

    Each consumer:
    - Optimizes independently
    - Receives discounted grid tariffs
    - Has an individual capacity limit
    - Pays penalties for exceeding that limit
    """

    contract = community.contract

    # Check that every consumer has an individual capacity limit
    if set(contract.individual_capacity_limit_kW) != set(
        community.member_ids
    ):
        raise ValueError(
            "Individual capacity limits do not match community members"
        )

    # Check that individual limits sum to the community limit
    total_individual_cap = np.sum(
        [
            contract.individual_capacity_limit_kW[cid]
            for cid in community.member_ids
        ],
        axis=0,
    )

    if not np.allclose(
        total_individual_cap,
        contract.capacity_limit_kW,
        atol=1e-4,
    ):
        raise ValueError(
            "Individual caps do not sum to the community cap"
        )

    # Solve each consumer independently
    return {
        member.consumer_id: solve_individual_member(
            member,
            contract=contract,
            individual_cap_kW=(
                contract.individual_capacity_limit_kW[
                    member.consumer_id
                ]
            ),
            terminal_soc=terminal_soc,
            verbose=verbose,
        )
        for member in community.members
    }



# ==========================================================
# PARAMETRIZED COMMUNITY MODEL (R / I / C)
# ==========================================================

class CommunityModel:
    """One LP formulation, with metering and contract switches for R/I/C."""

    def __init__(self, community: CommunityData, model_type: Literal['R', 'I', 'C'] = 'R',
                 terminal_soc: str = 'cyclic', verbose: bool = False):
        self.community = community
        self.config = get_model_configuration(community, model_type)
        self.terminal_soc = terminal_soc
        self.model = gp.Model(f'Community_{model_type}')
        self.model.Params.OutputFlag = int(verbose)
        n = community.members[0].n_hours
        if any(member.n_hours != n for member in community.members):
            raise ValueError('All members must have the same hourly horizon')
        self.T = range(n)
        self.blocks = {}
        self.member_import = {}
        self.member_export = {}
        self.community_import = None
        self.community_export = None
        self.exceedance = None
        self.balance_constraints = None
        self.penalty_cost = gp.LinExpr(0.0)
        self.objective_components = {}

    def build(self):
        self._build_consumers()
        self._build_metering()
        self._build_contract()
        self._build_objective()
        return self

    def _build_consumers(self):
        for member in self.community.members:
            self.blocks[member.consumer_id] = add_consumer_constraints(
                self.model, member, terminal_soc=self.terminal_soc
            )

    def _build_metering(self):
        m, T = self.model, self.T
        if self.config.metering == 'individual':
            for member in self.community.members:
                cid = member.consumer_id
                block = self.blocks[cid]
                imp = m.addVars(T, lb=0, name=f'{cid}_grid_import')
                exp = m.addVars(T, lb=0, name=f'{cid}_grid_export')
                self.member_import[cid] = imp
                self.member_export[cid] = exp
                m.addConstrs((imp[t] - exp[t] == block.net_exchange[t] for t in T),
                             name=f'{cid}_meter_balance')
                if member.max_import_kW is not None:
                    m.addConstrs((imp[t] <= member.max_import_kW for t in T),
                                 name=f'{cid}_import_limit')
                if member.max_export_kW is not None:
                    m.addConstrs((exp[t] <= member.max_export_kW for t in T),
                                 name=f'{cid}_export_limit')
        else:
            # One external meter, with individual physical connection limits retained.
            reference = self.community.members[0]
            for member in self.community.members[1:]:
                if (not np.allclose(member.energy_price, reference.energy_price)
                        or not np.isclose(member.import_tariff, reference.import_tariff)
                        or not np.isclose(member.export_tariff, reference.export_tariff)):
                    raise ValueError('Community meter requires common prices and tariffs')
            self.community_import = m.addVars(T, lb=0, name='community_import')
            self.community_export = m.addVars(T, lb=0, name='community_export')
            self.balance_constraints = m.addConstrs((
                self.community_import[t] - self.community_export[t]
                == gp.quicksum(self.blocks[cid].net_exchange[t]
                               for cid in self.community.member_ids)
                for t in T
            ), name='community_balance')
            for member in self.community.members:
                cid = member.consumer_id
                net = self.blocks[cid].net_exchange
                if member.max_import_kW is not None:
                    m.addConstrs((net[t] <= member.max_import_kW for t in T),
                                 name=f'{cid}_physical_import_limit')
                if member.max_export_kW is not None:
                    m.addConstrs((net[t] >= -member.max_export_kW for t in T),
                                 name=f'{cid}_physical_export_limit')

    def _build_contract(self):
        contract, T, m = self.community.contract, self.T, self.model
        kind = self.config.capacity_contract
        if kind == 'none':
            return
        alpha = float(contract.penalty_DKK_per_kWh)
        if kind == 'individual':
            if set(contract.individual_capacity_limit_kW) != set(self.community.member_ids):
                raise ValueError('Individual caps do not match community members')
            total_cap = np.sum([contract.individual_capacity_limit_kW[cid]
                                for cid in self.community.member_ids], axis=0)
            if not np.allclose(total_cap, contract.capacity_limit_kW, atol=1e-4):
                raise ValueError('Individual caps do not sum to community cap')
            self.exceedance = {}
            for cid in self.community.member_ids:
                cap = np.asarray(contract.individual_capacity_limit_kW[cid], dtype=float)
                if cap.shape != (len(T),) or np.any(cap < 0):
                    raise ValueError(f'Invalid cap for {cid}')
                excess = m.addVars(T, lb=0, name=f'{cid}_exceedance')
                self.exceedance[cid] = excess
                m.addConstrs((excess[t] >= self.member_import[cid][t] - float(cap[t])
                              for t in T), name=f'{cid}_capacity_exceedance')
                self.penalty_cost += alpha * gp.quicksum(excess[t] for t in T)
        else:
            cap = np.asarray(contract.capacity_limit_kW, dtype=float)
            if cap.shape != (len(T),) or np.any(cap < 0):
                raise ValueError('Invalid community capacity profile')
            self.exceedance = m.addVars(T, lb=0, name='community_exceedance')
            m.addConstrs((self.exceedance[t] >= self.community_import[t] - float(cap[t])
                          for t in T), name='community_capacity_exceedance')
            self.penalty_cost = alpha * gp.quicksum(self.exceedance[t] for t in T)

    def _build_objective(self):
        T, beta = self.T, self.config.tariff_discount
        physical = gp.quicksum(b.physical_cost for b in self.blocks.values())
        disutility = gp.quicksum(b.disutility_cost for b in self.blocks.values())
        procurement = gp.LinExpr(0.0)
        if self.config.metering == 'individual':
            for member in self.community.members:
                imp = self.member_import[member.consumer_id]
                exp = self.member_export[member.consumer_id]
                procurement += gp.quicksum(
                    (float(member.energy_price[t]) + (1-beta)*member.import_tariff)*imp[t]
                    - (float(member.energy_price[t]) - (1-beta)*member.export_tariff)*exp[t]
                    for t in T)
        else:
            member = self.community.members[0]
            procurement = gp.quicksum(
                (float(member.energy_price[t]) + (1-beta)*member.import_tariff)
                * self.community_import[t]
                - (float(member.energy_price[t]) - (1-beta)*member.export_tariff)
                * self.community_export[t]
                for t in T)
        self.objective_components = dict(procurement=procurement, physical=physical,
                                         disutility=disutility, penalty=self.penalty_cost)
        self.model.setObjective(procurement + physical + disutility + self.penalty_cost,
                                GRB.MINIMIZE)

    def solve(self):
        self.model.optimize()
        if self.model.Status != GRB.OPTIMAL:
            raise RuntimeError(f'Model {self.config.model_type}: Gurobi status {self.model.Status}')
        return self._extract_results()

    def _extract_results(self):
        T = self.T
        def arr(vars_):
            return np.array([vars_[t].X for t in T], dtype=float)

        results = {}
        beta = self.config.tariff_discount
        for member in self.community.members:
            cid = member.consumer_id
            block = self.blocks[cid]
            v = block.variables
            net = np.array([block.net_exchange[t].getValue() for t in T], dtype=float)
            if self.config.metering == 'individual':
                imp, exp = arr(self.member_import[cid]), arr(self.member_export[cid])
            else:
                # Diagnostic member flows only: these are NOT external DSO-meter flows.
                imp, exp = np.maximum(net, 0.0), np.maximum(-net, 0.0)
            soc = (np.array([v['soc'][t].X for t in range(member.n_hours+1)])
                   if member.has_battery else None)
            # In C, a member objective is not defined until the dual settlement.
            cost = float('nan')
            if self.config.metering == 'individual':
                procurement = sum(
                    (float(member.energy_price[t]) + (1-beta)*member.import_tariff)*imp[t]
                    - (float(member.energy_price[t]) - (1-beta)*member.export_tariff)*exp[t]
                    for t in T)
                physical = sum(member.pv_marginal_cost * v['pv'][t].X for t in T)
                disutility = sum(member.linear_disutility * v['deviation'][t].X for t in T)
                penalty = (float(self.community.contract.penalty_DKK_per_kWh)
                           * sum(self.exceedance[cid][t].X for t in T)
                           if self.config.model_type == 'I' else 0.0)
                cost = float(procurement + physical + disutility + penalty)
            results[cid] = ReferenceResult(
                consumer_id=cid, objective_DKK=cost, load_kW=arr(v['load']),
                pv_kW=arr(v['pv']), import_kW=imp, export_kW=exp,
                net_exchange_kW=net, deviation_kW=arr(v['deviation']), soc_kWh=soc,
                charge_kW=arr(v['charge']) if member.has_battery else None,
                discharge_kW=arr(v['discharge']) if member.has_battery else None)
        if self.config.model_type == 'C':
            return CommunityResult(
                model_type='C', objective_DKK=float(self.model.ObjVal),
                member_results=results, community_import_kW=arr(self.community_import),
                community_export_kW=arr(self.community_export),
                community_exceedance_kW=arr(self.exceedance),
                balance_duals=np.array([self.balance_constraints[t].Pi for t in T], dtype=float))
        total = sum(r.objective_DKK for r in results.values())
        if not np.isclose(total, self.model.ObjVal, atol=1e-5):
            raise RuntimeError(f'Member costs ({total}) differ from model objective ({self.model.ObjVal})')
        return results
