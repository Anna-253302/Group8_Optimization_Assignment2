"""Analytical DP accounting, workflow and small MILP accounting checks."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import gurobipy as gp
from gurobipy import GRB

import main
from src.analysis import analyse_dp, export_dp_analysis
from src.data_loader import load_community
from src.model import CommunityModel, CommunityResult, ReferenceResult, get_model_configuration
from src.plotting import plot_feeder_import


def analytical_case():
    """Two hours: import above the cap, then export; unequal C/DP prices."""
    members = [SimpleNamespace(
        consumer_id=cid, n_hours=2, reference_load=np.array(load),
        pv_marginal_cost=0.1, linear_disutility=0.25, slope_class=1,
        has_pv=has_pv, has_battery=False, archetype='L1+PV' if has_pv else 'L1',
    ) for cid, load, has_pv in (('buyer', [2.0, 0.5], False), ('supplier', [0.0, 0.0], True))]
    community = SimpleNamespace(
        community='test', day='d1', members=members, member_ids=['buyer', 'supplier'],
        energy_price=np.array([1.0, 2.0]), import_tariff=0.4, export_tariff=0.2,
        contract=SimpleNamespace(tariff_discount_ratio=0.5, penalty_DKK_per_kWh=3.0,
                                 capacity_limit_kW=np.ones(2)), summary=lambda: 'Test community')
    dp_prices = np.array([4.0, 2.1])
    own = {}
    for member in members:
        pv = np.array([0.0, 1.5]) if member.has_pv else np.zeros(2)
        net = member.reference_load - pv
        own[member.consumer_id] = ReferenceResult(
            consumer_id=member.consumer_id,
            objective_DKK=float(dp_prices @ net + member.pv_marginal_cost * pv.sum()),
            load_kW=member.reference_load.copy(), pv_kW=pv,
            import_kW=np.maximum(net, 0), export_kW=np.maximum(-net, 0),
            net_exchange_kW=net, deviation_kW=np.zeros(2),
            soc_kWh=None, charge_kW=None, discharge_kW=None)
    dp = CommunityResult(
        model_type='DP', objective_DKK=3.65, member_results=own,
        community_import_kW=np.array([2.0, 0.0]), community_export_kW=np.array([0.0, 1.0]),
        community_exceedance_kW=np.array([1.0, 0.0]), balance_duals=None,
        internal_prices=dp_prices, mip_gap=0.0, solver_status=GRB.OPTIMAL)
    centralized = copy.deepcopy(dp)
    centralized.model_type = 'C'
    centralized.balance_duals = np.array([4.2, 1.9])
    centralized.internal_prices = None
    centralized.mip_gap = None
    for result in centralized.member_results.values():
        result.objective_DKK = float('nan')  # C's unassigned member bills are normal.
    reference, individual = copy.deepcopy(own), copy.deepcopy(own)
    reference['buyer'].objective_DKK, reference['supplier'].objective_DKK = 10.0, -3.0
    individual['buyer'].objective_DKK, individual['supplier'].objective_DKK = 8.0, -3.2
    return community, reference, individual, centralized, dp


class DPAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.case = analytical_case()

    def temporary_output(self):
        parent = Path(__file__).resolve().parents[1] / 'results'
        parent.mkdir(exist_ok=True)
        directory = tempfile.TemporaryDirectory(prefix='dp_test_', dir=parent)
        self.addCleanup(directory.cleanup)
        return Path(directory.name)

    def test_transfers_budget_and_participation(self):
        result = analyse_dp(*self.case)
        s = result.summary
        self.assertAlmostEqual(s['external_bill_DKK'], 3.5)
        self.assertAlmostEqual(s['total_payments_DKK'], 5.9)
        self.assertAlmostEqual(s['manager_surplus_DKK'], 2.4)
        self.assertFalse(s['manager_budget_balanced'])
        self.assertAlmostEqual(s['accounting_identity_residual_DKK'], 0)
        self.assertTrue(s['objectives_equal_within_tolerance'])
        self.assertAlmostEqual(s['max_absolute_price_difference_DKK_per_kWh'], 0.2)
        self.assertEqual(s['worse_off_vs_R'], [])
        self.assertEqual(s['worse_off_vs_I'], ['buyer', 'supplier'])
        self.assertAlmostEqual(result.members[1]['settled_cost_DP_DKK'], -3.0)
        self.assertAlmostEqual(result.costs['DP']['total_objective_DKK'], 3.65)
        # Arbitrary DP prices need not produce C's cap rent (3 DKK here).
        self.assertNotAlmostEqual(s['manager_surplus_DKK'], 3.0)

    def test_nonoptimal_status_and_unavailable_gap(self):
        self.case[-1].solver_status = GRB.TIME_LIMIT
        self.case[-1].mip_gap = float('inf')
        result = analyse_dp(*self.case)
        self.assertFalse(result.summary['dp_proven_optimal'])
        self.assertEqual(result.summary['solver_status_name'], 'TIME_LIMIT')
        self.assertIsNone(result.summary['mip_gap'])
        export_dp_analysis(result, self.temporary_output())

    def test_failed_validation_raises(self):
        mutations = [
            lambda dp: setattr(dp, 'community_import_kW', np.array([-1.0, 0.0])),
            lambda dp: setattr(dp, 'community_import_kW', np.array([2.1, 0.0])),
            lambda dp: setattr(dp, 'community_exceedance_kW', np.zeros(2)),
            lambda dp: setattr(dp, 'objective_DKK', 4.0),
            lambda dp: setattr(dp, 'objective_DKK', 3.0),
            lambda dp: setattr(dp.member_results['buyer'], 'objective_DKK', 100.0),
            lambda dp: setattr(dp, 'internal_prices', np.array([np.nan, 2.1])),
            lambda dp: setattr(dp, 'internal_prices', None),
        ]
        for i, mutate in enumerate(mutations):
            with self.subTest(case=i):
                case = copy.deepcopy(self.case)
                mutate(case[-1])
                with self.assertRaises(ValueError):
                    analyse_dp(*case)

    def test_export_rows_and_strict_json(self):
        result = analyse_dp(*self.case)
        folder = self.temporary_output()
        export_dp_analysis(result, folder)
        payload = json.loads((folder / 'cap_only.json').read_text(),
                             parse_constant=lambda value: self.fail(f'Nonfinite JSON: {value}'))
        self.assertEqual(set(payload), {'summary', 'hourly', 'members'})
        self.assertTrue(payload['summary']['validation_passed'])
        self.assertEqual(payload['summary']['cost_breakdown'], result.costs)
        self.assertEqual(payload['hourly'], result.hourly)
        self.assertEqual(payload['members'], result.members)
        self.assertEqual([path.name for path in folder.iterdir()], ['cap_only.json'])

    def test_shared_meter_plot_does_not_sum_member_imports(self):
        community, reference, _, centralized, dp = self.case
        try:
            with patch('matplotlib.figure.Figure.savefig'), patch('matplotlib.pyplot.close'):
                plot_feeder_import(community, {'R': reference, 'C': centralized, 'DP': dp}, self.temporary_output())
                lines = plt.gcf().axes[0].lines
                np.testing.assert_allclose(lines[0].get_ydata(), [2.0, 0.5])
                np.testing.assert_allclose(lines[1].get_ydata(), [2.0, 0.0])
                np.testing.assert_allclose(lines[2].get_ydata(), [2.0, 0.0])
        finally:
            plt.close('all')

    def test_runner_solves_each_model_once(self):
        community, reference, individual, centralized, dp = self.case
        results = {'R': reference, 'I': individual, 'C': centralized, 'DP': dp}
        folder = self.temporary_output()
        with patch('main.solve_community', side_effect=lambda c, model_type, verbose, **kwargs: results[model_type]) as solver, \
             patch('main.plot_feeder_import') as plot, redirect_stdout(io.StringIO()):
            main.run_dp_cap_only(community, folder, verbose=True)
        self.assertEqual([call.kwargs['model_type'] for call in solver.call_args_list], ['R', 'I', 'C', 'DP'])
        self.assertTrue(all(call.kwargs['verbose'] for call in solver.call_args_list))
        plot.assert_not_called()
        self.assertEqual([path.name for path in folder.iterdir()], ['cap_only.json'])

    def test_cli_keeps_all_as_q1_and_routes_dp(self):
        for kind in ('all', 'DP'):
            with self.subTest(model=kind), patch('sys.argv', ['main.py', '--model', kind]), \
                 patch('main.load_community', return_value=self.case[0]), \
                 patch('main.run_dp_cap_only') as dp_runner, patch('main.run_base_case') as q1_runner, \
                 redirect_stdout(io.StringIO()):
                main.main()
                self.assertEqual(dp_runner.call_count, int(kind == 'DP'))
                self.assertEqual(q1_runner.call_count, int(kind == 'all'))
        for option in ('--settlement', '--sensitivity=both', '--reference-diagnostics'):
            with self.subTest(option=option), patch('sys.argv', ['main.py', '--model', 'DP', option]), \
                 redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main.main()
            self.assertEqual(error.exception.code, 2)

    def test_budget_balance_configuration_and_cli(self):
        community = self.case[0]
        self.assertFalse(get_model_configuration(community, 'DP').budget_balance)
        self.assertTrue(get_model_configuration(community, 'DP', budget_balance=True).budget_balance)
        for kind in ('R', 'I', 'C'):
            with self.subTest(model=kind), self.assertRaisesRegex(ValueError, 'only supported for Model DP'):
                get_model_configuration(community, kind, budget_balance=True)
        with patch('sys.argv', ['main.py', '--model', 'DP', '--budget-balance']), \
             patch('main.load_community', return_value=community), \
             patch('main.run_dp_cap_only') as runner, redirect_stdout(io.StringIO()):
            main.main()
        self.assertTrue(runner.call_args.kwargs['budget_balance'])
        with patch('sys.argv', ['main.py', '--model', 'C', '--budget-balance']), \
             redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main.main()
        self.assertEqual(error.exception.code, 2)

    def test_budget_balance_analysis_and_separate_export(self):
        folder = self.temporary_output()
        export_dp_analysis(analyse_dp(*self.case), folder)
        original = (folder / 'cap_only.json').read_bytes()
        dp = self.case[-1]
        dp.budget_balance = True
        dp.internal_prices = np.array([2.75, 2.0])  # 2*2.75 - 2 = 3.5 DKK external bill.
        dp.solver_status = GRB.TIME_LIMIT
        dp.mip_gap = 0.01
        for member in self.case[0].members:
            own = dp.member_results[member.consumer_id]
            own.objective_DKK = float(dp.internal_prices @ own.net_exchange_kW
                                      + member.pv_marginal_cost * own.pv_kW.sum())
        result = analyse_dp(*self.case)
        self.assertAlmostEqual(result.summary['manager_surplus_DKK'], 0)
        self.assertTrue(result.summary['budget_balance_validation_passed'])
        self.assertFalse(result.summary['dp_proven_optimal'])
        export_dp_analysis(result, folder)
        self.assertEqual((folder / 'cap_only.json').read_bytes(), original)
        self.assertEqual({p.name for p in folder.iterdir()}, {'cap_only.json', 'budget_balance.json'})
        payload = json.loads((folder / 'budget_balance.json').read_text())
        self.assertEqual(payload['summary']['model_variant'], 'DP_budget_balance')

    def test_budget_balance_rejects_surplus_and_simultaneous_trade(self):
        dp = self.case[-1]
        dp.budget_balance = True
        with self.assertRaisesRegex(ValueError, 'hard budget balance'):
            analyse_dp(*self.case)
        # Add simultaneous trade in the export hour; adjust costs and payments
        # consistently so accounting identities alone cannot detect the problem.
        dp.community_import_kW[1] = 0.2
        dp.community_export_kW[1] = 1.2
        dp.objective_DKK = 3.71
        dp.internal_prices = np.array([2.78, 2.0])
        for member in self.case[0].members:
            own = dp.member_results[member.consumer_id]
            own.objective_DKK = float(dp.internal_prices @ own.net_exchange_kW
                                      + member.pv_marginal_cost * own.pv_kW.sum())
        with self.assertRaisesRegex(ValueError, 'simultaneous community import and export'):
            analyse_dp(*self.case)

    def test_exact_bb_metering_and_exceedance(self):
        for imp, exp, excess, feasible in ((2, 1, 1, False), (2, 0, 2, False),
                                            (2, 0, 1, True), (0, 1, 0, True)):
            with self.subTest(imp=imp, exp=exp, excess=excess), gp.Model() as model:
                model.Params.OutputFlag = 0
                opt = object.__new__(CommunityModel)
                opt.model, opt.T = model, range(1)
                opt.community = SimpleNamespace(contract=SimpleNamespace(capacity_limit_kW=[1]))
                opt.community_import = model.addVars(1, lb=imp, ub=imp)
                opt.community_export = model.addVars(1, lb=exp, ub=exp)
                opt.exceedance = model.addVars(1, lb=excess, ub=excess)
                model.addConstr(opt.exceedance[0] >= opt.community_import[0] - 1)
                opt._enforce_exact_bb_accounting()
                model.optimize()
                self.assertEqual(model.Status, GRB.OPTIMAL if feasible else GRB.INFEASIBLE)
                self.assertEqual(model.NumQConstrs, 0)

    def test_budget_balance_solver_status_handling(self):
        opt = object.__new__(CommunityModel)
        opt.config = SimpleNamespace(model_type='DP', budget_balance=True, individual_rationality=False)
        for status, label in ((GRB.INFEASIBLE, 'INFEASIBLE'), (GRB.TIME_LIMIT, 'TIME_LIMIT')):
            opt.model = SimpleNamespace(Status=status, SolCount=0, optimize=lambda: None)
            with self.subTest(status=status), self.assertRaisesRegex(RuntimeError, label):
                opt.solve()
        opt.model = SimpleNamespace(Status=GRB.TIME_LIMIT, SolCount=1,
                                    MIPGap=0.1, optimize=lambda: None)
        with patch.object(opt, '_extract_results', return_value='incumbent') as extract, \
             self.assertWarnsRegex(RuntimeWarning, 'not proven optimal'):
            self.assertEqual(opt.solve(), 'incumbent')
        extract.assert_called_once()

    def test_individual_rationality_configuration_and_cli(self):
        community = self.case[0]
        config = get_model_configuration(community, 'DP', budget_balance=True, individual_rationality=True)
        self.assertTrue(config.individual_rationality)
        self.assertFalse(get_model_configuration(community, 'DP', budget_balance=True).individual_rationality)
        for kind, bb in (('DP', False), ('R', True), ('I', True), ('C', True)):
            with self.subTest(model=kind), self.assertRaises(ValueError):
                get_model_configuration(community, kind, budget_balance=bb, individual_rationality=True)
        with patch('sys.argv', ['main.py', '--model', 'DP', '--budget-balance', '--individual-rationality']), \
             patch('main.load_community', return_value=community), \
             patch('main.run_dp_cap_only') as runner, redirect_stdout(io.StringIO()):
            main.main()
        self.assertTrue(runner.call_args.kwargs['individual_rationality'])
        for args in (['--model', 'DP', '--individual-rationality'],
                     ['--model', 'R', '--budget-balance', '--individual-rationality']):
            with self.subTest(args=args), patch('sys.argv', ['main.py'] + args), \
                 redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main.main()
            self.assertEqual(error.exception.code, 2)

    def test_ir_rows_restrict_only_the_bb_model(self):
        community = load_community('base')
        refs = {cid: float(i + 1) for i, cid in enumerate(community.member_ids)}
        models = []
        try:
            for ir in (False, True):
                opt = CommunityModel(community, 'DP', budget_balance=True,
                                     individual_rationality=ir, reference_costs_DKK=refs).build()
                models.append(opt.model)
                opt.model.update()
                rows = [row for row in opt.model.getConstrs() if row.ConstrName.startswith('individual_rationality_')]
                self.assertEqual(len(rows), 16 if ir else 0)
                for cid in community.member_ids if ir else ():
                    row = opt.model.getConstrByName(f'individual_rationality_{cid}')
                    dual = opt.follower_dual_objectives[cid]
                    self.assertEqual(row.Sense, GRB.LESS_EQUAL)
                    self.assertAlmostEqual(row.RHS, refs[cid] - dual.getConstant())
                    actual = {opt.model.getRow(row).getVar(j).VarName: opt.model.getRow(row).getCoeff(j)
                              for j in range(opt.model.getRow(row).size())}
                    expected = {}
                    for j in range(dual.size()):
                        name = dual.getVar(j).VarName
                        expected[name] = expected.get(name, 0.0) + dual.getCoeff(j)
                    self.assertEqual(actual, {name: value for name, value in expected.items() if value != 0})
                self.assertIsNotNone(opt.model.getConstrByName('budget_balance'))
                self.assertEqual(opt.model.NumQConstrs, 0)
                self.assertFalse(opt.model.IsQP)
            self.assertEqual(models[1].NumConstrs - models[0].NumConstrs, 16)
            self.assertEqual(models[1].NumVars, models[0].NumVars)
            self.assertEqual(models[1].NumSOS, models[0].NumSOS)
            self.assertEqual(models[1].NumGenConstrs, models[0].NumGenConstrs)
        finally:
            for model in models:
                model.dispose()

    def test_ir_requires_complete_finite_reference_costs(self):
        opt = object.__new__(CommunityModel)
        opt.blocks = {'buyer': None, 'supplier': None}
        for refs in (None, {'buyer': 1.0}, {'buyer': 1.0, 'supplier': 2.0, 'extra': 3.0},
                     {'buyer': float('nan'), 'supplier': 1.0},
                     {'buyer': float('inf'), 'supplier': 1.0}):
            opt.reference_costs_DKK = refs
            with self.subTest(refs=refs), self.assertRaises(ValueError):
                opt._build_individual_rationality()

    def test_ir_actual_cost_validation_and_export(self):
        folder = self.temporary_output()
        export_dp_analysis(analyse_dp(*self.case), folder)
        dp = self.case[-1]
        dp.budget_balance = True
        dp.internal_prices = np.array([2.75, 2.0])
        for member in self.case[0].members:
            own = dp.member_results[member.consumer_id]
            own.objective_DKK = float(dp.internal_prices @ own.net_exchange_kW
                                      + member.pv_marginal_cost * own.pv_kW.sum())
        export_dp_analysis(analyse_dp(*self.case), folder)
        previous = {path.name: path.read_bytes() for path in folder.iterdir()}
        dp.individual_rationality = True
        # The existing reference of -3 is below this supplier's -2.85 DP bill.
        with self.assertRaisesRegex(ValueError, 'supplier is worse off than Model R'):
            analyse_dp(*self.case)
        self.case[1]['supplier'].objective_DKK = -2.8
        result = analyse_dp(*self.case)
        self.assertTrue(result.summary['individual_rationality_validation_passed'])
        self.assertEqual(result.summary['worse_off_vs_R_count'], 0)
        export_dp_analysis(result, folder)
        self.assertEqual({path.name for path in folder.iterdir()},
                         {'cap_only.json', 'budget_balance.json', 'budget_balance_ir.json'})
        for name, content in previous.items():
            self.assertEqual((folder / name).read_bytes(), content)
        # Verify the runner passes R costs once, after the three benchmarks.
        community, reference, individual, centralized, dp = self.case
        with patch('main.solve_community', side_effect=[reference, individual, centralized, dp]) as solver, \
             redirect_stdout(io.StringIO()):
            main.run_dp_cap_only(community, folder, budget_balance=True, individual_rationality=True)
        self.assertEqual([call.kwargs['model_type'] for call in solver.call_args_list], ['R', 'I', 'C', 'DP'])
        self.assertEqual(solver.call_args.kwargs['reference_costs_DKK'],
                         {cid: own.objective_DKK for cid, own in reference.items()})
        self.assertTrue(solver.call_args.kwargs['individual_rationality'])

    def test_feasibility_cli_uses_only_one_reference_solve(self):
        args = ['main.py', '--model', 'DP', '--budget-balance', '--individual-rationality', '--feasibility-check']
        with patch('sys.argv', args), patch('main.load_community', return_value=self.case[0]), \
             patch('main.solve_community', return_value=self.case[1]) as solver, \
             patch('main.diagnose_bb_ir_feasibility') as diagnose, \
             patch('main.run_dp_cap_only') as ordinary, patch('main.export_dp_analysis') as export, \
             redirect_stdout(io.StringIO()):
            main.main()
        solver.assert_called_once_with(self.case[0], model_type='R', verbose=False)
        diagnose.assert_called_once_with(self.case[0], self.case[1], price_bound_extension=0.0)
        ordinary.assert_not_called()
        export.assert_not_called()
        for invalid in (['--model', 'DP', '--budget-balance'], ['--model', 'DP', '--individual-rationality'],
                        ['--model', 'C']):
            with self.subTest(args=invalid), patch('sys.argv', ['main.py', '--feasibility-check'] + invalid), \
                 redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main.main()
            self.assertEqual(error.exception.code, 2)

    def test_feasibility_status_without_incumbent(self):
        for status, conclusion in ((GRB.INFEASIBLE, 'proven infeasible'),
                                   (GRB.TIME_LIMIT, 'unresolved'), (GRB.INTERRUPTED, 'unresolved')):
            model = Mock(Status=status, SolCount=0, Runtime=300.0)
            model.Params = SimpleNamespace(TimeLimit=300, MIPFocus=0, SolutionLimit=2000000000)
            optimization = Mock(model=model)
            with self.subTest(status=status), patch('main.CommunityModel', return_value=optimization) as constructor, \
                 redirect_stdout(io.StringIO()):
                result = main.diagnose_bb_ir_feasibility(self.case[0], self.case[1], price_bound_extension=0.5)
            self.assertIn(conclusion, result['feasibility_conclusion'])
            self.assertNotIn('actual_community_cost_DKK', result)
            model.getObjective.return_value.getValue.assert_not_called()
            model.setObjective.assert_called_once_with(0.0, GRB.MINIMIZE)
            self.assertEqual((model.Params.TimeLimit, model.Params.MIPFocus, model.Params.SolutionLimit), (300, 1, 1))
            self.assertEqual(constructor.call_args.kwargs['price_bound_extension'], 0.5)
            self.assertTrue(constructor.call_args.kwargs['individual_rationality'])
            optimization.solve.assert_not_called()
            model.dispose.assert_called_once()

    def test_ir_slack_cli_activation_and_exclusivity(self):
        args = ['main.py', '--model', 'DP', '--budget-balance', '--individual-rationality', '--ir-slack-check']
        with patch('sys.argv', args), patch('main.load_community', return_value=self.case[0]), \
             patch('main.solve_community', return_value=self.case[1]) as solver, \
             patch('main.diagnose_bb_ir_slacks') as diagnose, \
             patch('main.diagnose_bb_ir_feasibility') as feasibility, \
             patch('main.run_dp_cap_only') as ordinary, patch('main.export_dp_analysis') as export, \
             redirect_stdout(io.StringIO()):
            main.main()
        solver.assert_called_once_with(self.case[0], model_type='R', verbose=False)
        diagnose.assert_called_once_with(self.case[0], self.case[1], price_bound_extension=0.0)
        feasibility.assert_not_called()
        ordinary.assert_not_called()
        export.assert_not_called()
        for invalid in (args + ['--feasibility-check'], ['main.py', '--model', 'DP', '--ir-slack-check'],
                        ['main.py', '--model', 'C', '--ir-slack-check']):
            with self.subTest(args=invalid), patch('sys.argv', invalid), \
                 redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main.main()
            self.assertEqual(error.exception.code, 2)

    def test_ir_slack_structure_and_no_incumbent_handling(self):
        community = load_community('base')
        refs = {cid: SimpleNamespace(objective_DKK=1.0) for cid in community.member_ids}
        for invalid in ({}, {**refs, community.member_ids[0]: SimpleNamespace(objective_DKK=float('nan'))}):
            with self.subTest(refs=invalid), patch('main.CommunityModel') as constructor, self.assertRaises(ValueError):
                main.diagnose_bb_ir_slacks(community, invalid)
            constructor.assert_not_called()
        for status, conclusion in ((GRB.TIME_LIMIT, 'unresolved'), (GRB.INFEASIBLE, 'not solely hard IR')):
            model = Mock(Status=status, SolCount=0, Runtime=300.0, ObjBound=0.0)
            model.Params = SimpleNamespace(TimeLimit=300, MIPFocus=0, SolutionLimit=2000000000)
            # Constant linear stand-ins exercise construction without a solver run.
            model.addVars.return_value = {cid: gp.LinExpr(0.0) for cid in refs}
            optimization = Mock(model=model, blocks=dict.fromkeys(refs),
                                follower_dual_objectives={cid: gp.LinExpr(1.0) for cid in refs})
            with self.subTest(status=status), patch('main.CommunityModel', return_value=optimization) as constructor, \
                 redirect_stdout(io.StringIO()):
                report = main.diagnose_bb_ir_slacks(community, refs, price_bound_extension=2.0)
            self.assertTrue(constructor.call_args.kwargs['budget_balance'])
            self.assertFalse(constructor.call_args.kwargs['individual_rationality'])
            self.assertEqual(constructor.call_args.kwargs['price_bound_extension'], 2.0)
            model.addVars.assert_called_once_with(community.member_ids, lb=0.0, name='ir_slack')
            self.assertEqual(model.addConstr.call_count, 16)
            self.assertEqual({call.kwargs['name'] for call in model.addConstr.call_args_list},
                             {f'ir_relaxed_{cid}' for cid in refs})
            self.assertIsInstance(model.setObjective.call_args.args[0], gp.LinExpr)
            self.assertEqual(model.Params.SolutionLimit, 2000000000)
            self.assertEqual((model.Params.TimeLimit, model.Params.MIPFocus), (300, 1))
            model.getObjective.return_value.getValue.assert_not_called()
            self.assertNotIn('actual_community_cost_DKK', report)
            self.assertIn(conclusion, report['feasibility_conclusion'])
            model.dispose.assert_called_once()

    def test_feasibility_solution_limit_checks_actual_cost(self):
        self.case[1]['supplier'].objective_DKK = -2.8
        for first_price in (2.75, 4.0):
            model = Mock(Status=GRB.SOLUTION_LIMIT, SolCount=1, Runtime=12.5, ObjVal=0.0)
            model.Params = SimpleNamespace()
            model.getObjective.return_value.getValue.return_value = 3.65
            blocks = {}
            for member in self.case[0].members:
                own = self.case[-1].member_results[member.consumer_id]
                blocks[member.consumer_id] = SimpleNamespace(
                    physical_cost=Mock(getValue=Mock(return_value=member.pv_marginal_cost * own.pv_kW.sum())),
                    disutility_cost=Mock(getValue=Mock(return_value=0.0)),
                    net_exchange={t: Mock(getValue=Mock(return_value=float(net)))
                                  for t, net in enumerate(own.net_exchange_kW)})
            optimization = Mock(model=model, T=range(2), blocks=blocks,
                                internal_prices={0: SimpleNamespace(X=first_price), 1: SimpleNamespace(X=2.0)},
                                objective_components={'procurement': Mock(getValue=Mock(return_value=0.5)),
                                                      'penalty': Mock(getValue=Mock(return_value=3.0))})
            with self.subTest(price=first_price), patch('main.CommunityModel', return_value=optimization), \
                 redirect_stdout(io.StringIO()):
                if first_price == 4.0:
                    with self.assertRaisesRegex(ValueError, 'failed independent'):
                        main.diagnose_bb_ir_feasibility(self.case[0], self.case[1])
                else:
                    result = main.diagnose_bb_ir_feasibility(self.case[0], self.case[1])
                    self.assertEqual(result['solver_status_name'], 'SOLUTION_LIMIT')
                    self.assertTrue(result['validation_passed'])
                    self.assertAlmostEqual(result['actual_community_cost_DKK'], 3.65)
                    self.assertAlmostEqual(result['bb_residual_DKK'], 0.0)
                    self.assertEqual(result['violating_member_count'], 0)


    def test_price_bound_extension_validation_and_forwarding(self):
        from src.model import solve_community

        community = load_community('base')
        for delta in (-0.1, float('nan'), float('inf'), -float('inf')):
            with self.subTest(delta=delta), patch('src.model.gp.Model') as constructor:
                with self.assertRaises(ValueError):
                    CommunityModel(community, 'DP', price_bound_extension=delta)
                constructor.assert_not_called()
        for kind in ('R', 'I', 'C'):
            with self.subTest(model=kind), patch('src.model.gp.Model') as constructor:
                with self.assertRaises(ValueError):
                    CommunityModel(community, kind, price_bound_extension=0.5)
                constructor.assert_not_called()
                CommunityModel(community, kind)  # Zero remains valid for Q1.
        with patch('src.model.CommunityModel') as constructor:
            solve_community(community, model_type='DP', budget_balance=True,
                            individual_rationality=True, reference_costs_DKK={'buyer': 1},
                            price_bound_extension=2.0)
        self.assertEqual(constructor.call_args.kwargs['price_bound_extension'], 2.0)
        self.assertTrue(constructor.call_args.kwargs['budget_balance'])
        self.assertTrue(constructor.call_args.kwargs['individual_rationality'])
        constructor.return_value.build.assert_called_once()
        constructor.return_value.solve.assert_called_once()

    def test_extension_changes_only_price_bounds(self):
        community = load_community('base')
        refs = {cid: 100.0 for cid in community.member_ids}
        baseline = None
        for delta in (0.0, 0.5, 2.0):
            opt = CommunityModel(community, 'DP', budget_balance=True,
                                 individual_rationality=True, reference_costs_DKK=refs,
                                 price_bound_extension=delta)
            try:
                opt.build()
                model = opt.model
                model.update()
                member = community.members[0]
                discount = 1 - community.contract.tariff_discount_ratio
                lower = member.energy_price - discount * member.export_tariff
                upper = (member.energy_price + discount * member.import_tariff
                         + community.contract.penalty_DKK_per_kWh)
                np.testing.assert_array_equal([opt.internal_prices[t].LB for t in opt.T], lower - delta)
                np.testing.assert_array_equal([opt.internal_prices[t].UB for t in opt.T], upper + delta)
                # All linear rows (including BB, IR and KKT), objective and
                # non-price variable bounds must be identical; no optimization.
                structure = (
                    [(row.ConstrName, row.Sense, row.RHS,
                      [(model.getRow(row).getVar(j).VarName, model.getRow(row).getCoeff(j))
                       for j in range(model.getRow(row).size())]) for row in model.getConstrs()],
                    [(v.VarName, v.VType, v.Obj,
                      None if v.VarName.startswith('internal_price[') else (v.LB, v.UB))
                     for v in model.getVars()],
                    model.NumSOS, model.NumGenConstrs, model.ObjCon, model.ModelSense)
                if baseline is None:
                    baseline = structure
                else:
                    self.assertEqual(structure, baseline)
                self.assertEqual(model.Params.TimeLimit, 300)
                self.assertEqual(model.Params.MIPGap, 1e-6)
            finally:
                opt.model.dispose()

    def test_extended_analysis_and_exports_protect_baselines(self):
        folder = self.temporary_output()
        for variant in ('DP_cap_only', 'DP_budget_balance', 'DP_budget_balance_ir'):
            baseline = analyse_dp(*self.case)
            baseline.summary['model_variant'] = variant
            path = export_dp_analysis(baseline, folder)
            original = path.read_bytes()
            for delta, label in ((0.5, '0p5'), (2.0, '2')):
                case = copy.deepcopy(self.case)
                dp = case[-1]
                dp.price_bound_extension = delta
                dp.best_bound_DKK = 3.6
                # Use the extended upper bound, outside the original interval.
                dp.internal_prices[0] = 4.2 + delta
                for member in case[0].members:
                    own = dp.member_results[member.consumer_id]
                    own.objective_DKK = float(dp.internal_prices @ own.net_exchange_kW
                                             + member.pv_marginal_cost * own.pv_kW.sum())
                result = analyse_dp(*case)
                result.summary['model_variant'] = variant
                output = export_dp_analysis(result, folder)
                self.assertEqual(output, folder / 'price_bound_sensitivity' / f'delta_{label}' / path.name)
                self.assertEqual(path.read_bytes(), original)
                payload = json.loads(output.read_text())
                self.assertEqual(set(payload), {'summary', 'hourly', 'members'})
                self.assertEqual(payload['summary']['price_bound_extension_DKK_per_kWh'], delta)
                self.assertEqual(payload['summary']['best_bound_DKK'], 3.6)
                dp.price_bound_extension = 0.0
                with self.assertRaisesRegex(ValueError, 'configured price bounds'):
                    analyse_dp(*case)

    def test_runner_forwards_extension_only_to_dp(self):
        community, reference, individual, centralized, dp = self.case
        dp.price_bound_extension = 0.5
        folder = self.temporary_output()
        with patch('main.solve_community', side_effect=[reference, individual, centralized, dp]) as solver, \
             redirect_stdout(io.StringIO()):
            main.run_dp_cap_only(community, folder, price_bound_extension=0.5)
        self.assertEqual([call.kwargs['model_type'] for call in solver.call_args_list], ['R', 'I', 'C', 'DP'])
        for call in solver.call_args_list[:3]:
            self.assertNotIn('price_bound_extension', call.kwargs)
        self.assertEqual(solver.call_args.kwargs['price_bound_extension'], 0.5)
        self.assertFalse((folder / 'cap_only.json').exists())
        self.assertTrue((folder / 'price_bound_sensitivity' / 'delta_0p5' / 'cap_only.json').exists())

    def test_extension_cli_routes_and_invalid_values(self):
        for option, target in ((None, 'run_dp_cap_only'),
                               ('--feasibility-check', 'diagnose_bb_ir_feasibility'),
                               ('--ir-slack-check', 'diagnose_bb_ir_slacks')):
            args = ['main.py', '--model', 'DP', '--budget-balance',
                    '--individual-rationality', '--price-bound-extension', '0.5']
            if option:
                args.append(option)
            with self.subTest(option=option), patch('sys.argv', args), \
                 patch('main.load_community', return_value=self.case[0]), \
                 patch('main.solve_community', return_value=self.case[1]), \
                 patch(f'main.{target}') as runner, redirect_stdout(io.StringIO()):
                main.main()
            self.assertEqual(runner.call_args.kwargs['price_bound_extension'], 0.5)
        for kind, value in (('DP', '-1'), ('DP', 'nan'), ('DP', 'inf'), ('DP', '-inf'),
                            ('R', '0.5'), ('I', '0.5'), ('C', '0.5'), ('all', '2')):
            with self.subTest(model=kind, value=value), \
                 patch('sys.argv', ['main.py', '--model', kind, f'--price-bound-extension={value}']), \
                 patch('main.load_community') as loader, redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit) as error:
                main.main()
            self.assertEqual(error.exception.code, 2)
            loader.assert_not_called()

if __name__ == '__main__':
    unittest.main()
