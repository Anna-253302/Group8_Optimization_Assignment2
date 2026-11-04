"""Analytical DP accounting and workflow checks; no optimizer licence required."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import GRB

import main
from src.analysis import analyse_dp, export_dp_analysis
from src.model import CommunityResult, ReferenceResult
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
        with patch('main.solve_community', side_effect=lambda c, model_type, verbose: results[model_type]) as solver, \
             patch('main.plot_feeder_import') as plot, redirect_stdout(io.StringIO()):
            main.run_dp_cap_only(community, folder, verbose=True)
        self.assertEqual([call.kwargs['model_type'] for call in solver.call_args_list], ['DP', 'R', 'I', 'C'])
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


if __name__ == '__main__':
    unittest.main()
