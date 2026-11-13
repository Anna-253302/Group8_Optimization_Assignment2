import json
from pathlib import Path

import numpy as np
import pytest

from main import prepare_q3_scenarios
from src.analysis import evaluate_fixed_prices_under_uncertainty
from src.data_loader import load_community


def _dp_prices():
    result_path = Path("results/base/d1/DP/cap_only.json")
    with result_path.open(encoding="utf-8") as f:
        dp_results = json.load(f)
    hourly = sorted(dp_results["hourly"], key=lambda row: row["hour"])
    return np.array(
        [row["dp_internal_price_DKK_per_kWh"] for row in hourly],
        dtype=float,
    ), dp_results


def test_fixed_price_response_matches_dp():
    community = load_community("base")
    prices, dp_results = _dp_prices()
    scenarios = prepare_q3_scenarios(community, n=1, seed=1)

    evaluation = evaluate_fixed_prices_under_uncertainty(
        community,
        prices,
        scenarios,
    )

    assert len(prices) == 24
    assert len(community.members) == 16
    assert evaluation["scenario_count"] == 1
    result_scenario = evaluation["scenarios"][0]
    dp_members = {
        row["consumer_id"]: row
        for row in dp_results["members"]
    }

    for member in community.members:
        result = result_scenario["member_results"][member.consumer_id]
        expected_cost = dp_members[member.consumer_id]["settled_cost_DP_DKK"]
        np.testing.assert_allclose(
            result.objective_DKK,
            expected_cost,
            atol=1e-5,
            rtol=0,
            err_msg=f"Mismatch for {member.consumer_id}",
        )
        assert (
            result_scenario["members"][member.consumer_id][
                "maximum_constraint_violation"
            ]
            <= 1e-6
        )


def test_q3_scenarios_use_selected_day_and_preserve_community():
    community = load_community("base", day="d3")
    original_profiles = {
        member.consumer_id: member.reference_load.copy()
        for member in community.members
    }
    original_minimums = {
        member.consumer_id: member.min_daily_energy_kWh
        for member in community.members
    }

    for count in (1, 5, 10, 100):
        scenarios = prepare_q3_scenarios(community, n=count, seed=1)
        assert len(scenarios) == count
        assert np.isclose(sum(s["probability"] for s in scenarios), 1.0)
        for scenario in scenarios:
            assert len(scenario["members"]) == 16
            assert set(scenario["theta"]["profile"]) == set(community.member_ids)
            assert all(member.n_hours == 24 for member in scenario["members"])

    baseline = prepare_q3_scenarios(community, n=1, seed=1)[0]
    assert all(value == 1.0 for value in baseline["theta"]["profile"].values())
    for member in baseline["members"]:
        np.testing.assert_allclose(
            member.reference_load,
            original_profiles[member.consumer_id],
            atol=1e-12,
            rtol=0,
        )
        assert member.min_daily_energy_kWh == pytest.approx(
            original_minimums[member.consumer_id]
        )
    for member in community.members:
        np.testing.assert_allclose(
            member.reference_load,
            original_profiles[member.consumer_id],
            atol=0,
            rtol=0,
        )


def test_fixed_price_evaluation_rejects_invalid_inputs():
    community = load_community("base")
    prices, _ = _dp_prices()
    scenarios = prepare_q3_scenarios(community, n=1, seed=1)

    with pytest.raises(ValueError, match="exactly 24"):
        evaluate_fixed_prices_under_uncertainty(community, prices[:-1], scenarios)

    bad_probability = [dict(scenarios[0], probability=0.5)]
    with pytest.raises(ValueError, match="sum to one"):
        evaluate_fixed_prices_under_uncertainty(community, prices, bad_probability)

    with pytest.raises(ValueError, match="At least one scenario"):
        evaluate_fixed_prices_under_uncertainty(community, prices, [])
