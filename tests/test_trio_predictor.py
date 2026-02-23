"""3連複予測のテスト"""

import pytest
from src.scraper.race_data import RaceInfo, Racer, WeatherInfo
from src.model.predictor import predict_trio_probabilities


def _make_race_info() -> RaceInfo:
    racers = [
        Racer(waku=w, name=f"選手{w}", rank="A1" if w <= 2 else "B1",
              win_rate_all=7.5 - w * 0.5, win_rate_2r_all=45.0 - w * 3,
              win_rate_local=7.0 - w * 0.4, win_rate_2r_local=40.0 - w * 2,
              motor_2r=38.0 - w, boat_2r=35.0 - w,
              exhibit_time=6.60 + w * 0.02)
        for w in range(1, 7)
    ]
    weather = WeatherInfo(wind_speed=3, wave_height=2)
    return RaceInfo(race_no=5, date="20260223", racers=racers, weather=weather)


class TestPredictTrioProbabilities:
    def test_returns_correct_count(self):
        race = _make_race_info()
        combos = predict_trio_probabilities(race, top_n=10)
        assert len(combos) == 10

    def test_sorted_by_probability(self):
        race = _make_race_info()
        combos = predict_trio_probabilities(race, top_n=20)
        probs = [p for _, p in combos]
        assert probs == sorted(probs, reverse=True)

    def test_combo_format(self):
        race = _make_race_info()
        combos = predict_trio_probabilities(race, top_n=5)
        for combo, prob in combos:
            parts = combo.split("=")
            assert len(parts) == 3
            assert all(p.isdigit() and 1 <= int(p) <= 6 for p in parts)
            # 3連複はソートされた順序（小=中=大）
            assert int(parts[0]) < int(parts[1]) < int(parts[2])

    def test_probabilities_sum_approx_one(self):
        """全20通りの確率合計がほぼ1.0"""
        race = _make_race_info()
        combos = predict_trio_probabilities(race, top_n=20)
        total = sum(p for _, p in combos)
        assert 0.99 < total < 1.01

    def test_empty_racers(self):
        race = RaceInfo(race_no=1, racers=[], weather=WeatherInfo())
        combos = predict_trio_probabilities(race)
        assert combos == []

    def test_top_combo_contains_waku1(self):
        """1号艇が最強なので、最上位のコンボに1が含まれるはず"""
        race = _make_race_info()
        combos = predict_trio_probabilities(race, top_n=1)
        assert len(combos) == 1
        assert "1" in combos[0][0].split("=")
