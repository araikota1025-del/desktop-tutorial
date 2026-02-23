"""予測モデルのテスト"""

import pytest
from src.scraper.race_data import RaceInfo, Racer, WeatherInfo
from src.model.predictor import predict_win_probabilities, predict_trifecta_probabilities


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


class TestPredictWinProbabilities:
    def test_probabilities_sum_to_one(self):
        race = _make_race_info()
        probs = predict_win_probabilities(race)

        assert len(probs) == 6
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.001

    def test_all_probabilities_positive(self):
        race = _make_race_info()
        probs = predict_win_probabilities(race)

        for waku, p in probs.items():
            assert p > 0, f"枠{waku}の確率が0以下: {p}"

    def test_waku1_has_highest_probability(self):
        """1号艇（高勝率 + インコース）は確率最大のはず"""
        race = _make_race_info()
        probs = predict_win_probabilities(race)

        assert probs[1] == max(probs.values())

    def test_empty_racers_returns_uniform(self):
        race = RaceInfo(race_no=1, racers=[], weather=WeatherInfo())
        probs = predict_win_probabilities(race)

        for waku in range(1, 7):
            assert abs(probs[waku] - 1 / 6) < 0.001


class TestPredictTrifectaProbabilities:
    def test_returns_correct_count(self):
        race = _make_race_info()
        combos = predict_trifecta_probabilities(race, top_n=10)

        assert len(combos) == 10

    def test_sorted_by_probability(self):
        race = _make_race_info()
        combos = predict_trifecta_probabilities(race, top_n=30)

        probs = [p for _, p in combos]
        assert probs == sorted(probs, reverse=True)

    def test_combo_format(self):
        race = _make_race_info()
        combos = predict_trifecta_probabilities(race, top_n=5)

        for combo, prob in combos:
            parts = combo.split("-")
            assert len(parts) == 3
            assert all(p.isdigit() and 1 <= int(p) <= 6 for p in parts)
            # 3連単は全て異なる枠番
            assert len(set(parts)) == 3

    def test_probabilities_are_valid(self):
        race = _make_race_info()
        combos = predict_trifecta_probabilities(race, top_n=120)

        total = sum(p for _, p in combos)
        assert 0.99 < total < 1.01  # 全120通りの合計≒1.0

    def test_empty_racers(self):
        race = RaceInfo(race_no=1, racers=[], weather=WeatherInfo())
        combos = predict_trifecta_probabilities(race)
        assert combos == []
