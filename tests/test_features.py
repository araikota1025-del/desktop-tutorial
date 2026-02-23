"""特徴量パイプラインのテスト"""

import pytest
from src.scraper.race_data import RaceInfo, Racer, WeatherInfo
from src.features.pipeline import (
    build_racer_features,
    build_race_features,
    HEIWAJIMA_COURSE_WIN_RATE,
)


def _make_race_info(racers=None, wind_direction="", wind_speed=0) -> RaceInfo:
    """テスト用のRaceInfoを生成する"""
    if racers is None:
        racers = [
            Racer(waku=w, name=f"選手{w}", rank="A1" if w == 1 else "B1",
                  win_rate_all=7.0 - w * 0.5, win_rate_2r_all=40.0 - w * 3,
                  win_rate_local=6.5 - w * 0.4, win_rate_2r_local=35.0 - w * 2,
                  motor_2r=35.0 + w, boat_2r=33.0 + w,
                  exhibit_time=6.60 + w * 0.03)
            for w in range(1, 7)
        ]
    weather = WeatherInfo(
        wind_direction=wind_direction,
        wind_speed=wind_speed,
        wave_height=3,
    )
    return RaceInfo(race_no=1, date="20260223", racers=racers, weather=weather)


class TestBuildRacerFeatures:
    def test_basic_features(self):
        race = _make_race_info()
        features = build_racer_features(race.racers[0], race)

        assert features["waku"] == 1
        assert features["win_rate_all"] == 6.5
        assert features["rank_score"] == 4  # A1
        assert features["course_base_win_rate"] == HEIWAJIMA_COURSE_WIN_RATE[1]

    def test_all_waku_have_course_win_rate(self):
        race = _make_race_info()
        for racer in race.racers:
            features = build_racer_features(racer, race)
            assert features["course_base_win_rate"] == HEIWAJIMA_COURSE_WIN_RATE[racer.waku]

    def test_headwind_favors_outer(self):
        race = _make_race_info(wind_direction="北")
        f1 = build_racer_features(race.racers[0], race)  # 1号艇
        f6 = build_racer_features(race.racers[5], race)  # 6号艇

        # 向かい風ならアウト（6号艇）にボーナス
        assert f6["wind_course_interaction"] > f1["wind_course_interaction"]

    def test_tailwind_favors_inner(self):
        race = _make_race_info(wind_direction="南")
        f1 = build_racer_features(race.racers[0], race)
        f6 = build_racer_features(race.racers[5], race)

        # 追い風ならイン（1号艇）にボーナス
        assert f1["wind_course_interaction"] > f6["wind_course_interaction"]


class TestBuildRaceFeatures:
    def test_returns_6_features(self):
        race = _make_race_info()
        features = build_race_features(race)
        assert len(features) == 6

    def test_zscore_mean_near_zero(self):
        race = _make_race_info()
        features = build_race_features(race)

        zscores = [f["exhibit_time_zscore"] for f in features]
        assert abs(sum(zscores) / len(zscores)) < 0.01  # 平均≒0

    def test_win_rate_zscore_exists(self):
        race = _make_race_info()
        features = build_race_features(race)
        for f in features:
            assert "win_rate_zscore" in f

    def test_empty_racers(self):
        race = _make_race_info(racers=[])
        features = build_race_features(race)
        assert features == []
