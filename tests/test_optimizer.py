"""買い目最適化エンジンのテスト"""

import pytest
from src.betting.optimizer import (
    kelly_criterion,
    compute_expected_value,
    optimize_bets,
    BettingPlan,
)


class TestKellyCriterion:
    def test_positive_ev_returns_positive(self):
        """期待値プラスなら正の比率"""
        f = kelly_criterion(prob=0.3, odds=5.0)
        assert f > 0

    def test_negative_ev_returns_zero(self):
        """期待値マイナスなら0"""
        f = kelly_criterion(prob=0.01, odds=2.0)
        assert f == 0.0

    def test_max_cap(self):
        """30%を超えない"""
        f = kelly_criterion(prob=0.8, odds=10.0)
        assert f <= 0.3

    def test_half_kelly(self):
        """ハーフケリーはフルケリーの半分"""
        full = kelly_criterion(prob=0.3, odds=5.0, fraction=1.0)
        half = kelly_criterion(prob=0.3, odds=5.0, fraction=0.5)
        assert abs(half - full * 0.5) < 0.001 or half == 0.3  # cap考慮


class TestComputeExpectedValue:
    def test_basic(self):
        ev = compute_expected_value(prob=0.1, odds=15.0)
        assert ev == pytest.approx(1.5)

    def test_zero_prob(self):
        ev = compute_expected_value(prob=0.0, odds=100.0)
        assert ev == 0.0


class TestOptimizeBets:
    def _sample_data(self):
        trifecta_probs = [
            ("1-2-3", 0.10),
            ("1-3-2", 0.08),
            ("2-1-3", 0.06),
            ("1-2-4", 0.05),
            ("2-3-1", 0.04),
            ("3-1-2", 0.03),
            ("1-4-2", 0.02),
            ("4-1-2", 0.015),
        ]
        odds_dict = {
            "1-2-3": 8.5,
            "1-3-2": 12.0,
            "2-1-3": 15.0,
            "1-2-4": 20.0,
            "2-3-1": 25.0,
            "3-1-2": 35.0,
            "1-4-2": 50.0,
            "4-1-2": 70.0,
        }
        return trifecta_probs, odds_dict

    def test_budget_not_exceeded(self):
        probs, odds = self._sample_data()
        plan = optimize_bets(probs, odds, budget=3000)

        assert plan.total_bet <= 3000

    def test_min_bet_unit(self):
        probs, odds = self._sample_data()
        plan = optimize_bets(probs, odds, budget=3000)

        for s in plan.suggestions:
            assert s.bet_amount >= 100
            assert s.bet_amount % 100 == 0

    def test_balance_strategy(self):
        probs, odds = self._sample_data()
        plan = optimize_bets(probs, odds, budget=3000, strategy="balance")

        assert plan.strategy == "balance"
        assert len(plan.suggestions) > 0

    def test_roi_strategy_filters_low_ev(self):
        probs, odds = self._sample_data()
        plan = optimize_bets(probs, odds, budget=3000, strategy="roi")

        for s in plan.suggestions:
            assert s.expected_value > 1.0

    def test_hit_rate_strategy_sorted_by_prob(self):
        probs, odds = self._sample_data()
        plan = optimize_bets(probs, odds, budget=5000, strategy="hit_rate")

        probs_list = [s.predicted_prob for s in plan.suggestions]
        assert probs_list == sorted(probs_list, reverse=True)

    def test_no_candidates_returns_empty(self):
        plan = optimize_bets(
            trifecta_probs=[("1-2-3", 0.001)],
            odds_dict={"1-2-3": 1.1},
            budget=3000,
            strategy="roi",
        )
        assert plan.suggestions == []
        assert plan.total_bet == 0

    def test_combined_hit_rate(self):
        probs, odds = self._sample_data()
        plan = optimize_bets(probs, odds, budget=3000)

        expected_hit = sum(s.predicted_prob for s in plan.suggestions)
        assert abs(plan.combined_hit_rate - expected_hit) < 0.001
