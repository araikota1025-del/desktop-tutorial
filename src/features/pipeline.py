"""選手・レースデータから予測用特徴量を生成するパイプライン"""

import numpy as np
from ..scraper.race_data import RaceInfo, Racer


# 平和島のコース別1着率（過去統計）
HEIWAJIMA_COURSE_WIN_RATE = {
    1: 0.449,
    2: 0.152,
    3: 0.130,
    4: 0.122,
    5: 0.087,
    6: 0.060,
}

# 級別スコア
RANK_SCORE = {"A1": 4, "A2": 3, "B1": 2, "B2": 1, "": 1}


def build_racer_features(racer: Racer, race_info: RaceInfo) -> dict:
    """1選手分の特徴量を生成する"""
    features = {}

    # 基本特徴量
    features["waku"] = racer.waku
    features["win_rate_all"] = racer.win_rate_all
    features["win_rate_2r_all"] = racer.win_rate_2r_all
    features["win_rate_local"] = racer.win_rate_local
    features["win_rate_2r_local"] = racer.win_rate_2r_local
    features["rank_score"] = RANK_SCORE.get(racer.rank, 1)

    # 機材特徴量
    features["motor_2r"] = racer.motor_2r
    features["boat_2r"] = racer.boat_2r

    # 平和島固有特徴量: コース別期待1着率
    features["course_base_win_rate"] = HEIWAJIMA_COURSE_WIN_RATE.get(racer.waku, 0.1)

    # 直前特徴量
    features["exhibit_time"] = racer.exhibit_time

    # 気象影響スコア（平和島固有ロジック）
    weather = race_info.weather
    features["wind_speed"] = weather.wind_speed
    features["wave_height"] = weather.wave_height

    # 風向とコースの相互作用
    # 追い風 → イン有利、向かい風 → アウト有利
    wind_dir = weather.wind_direction
    is_headwind = "向" in wind_dir or "北" in wind_dir
    if is_headwind:
        # 向かい風: アウトコースにボーナス
        features["wind_course_interaction"] = (racer.waku - 3.5) * 0.05
    else:
        # 追い風: インコースにボーナス
        features["wind_course_interaction"] = (3.5 - racer.waku) * 0.05

    return features


def build_race_features(race_info: RaceInfo) -> list[dict]:
    """1レース分の全選手の特徴量を生成する"""
    all_features = []
    for racer in race_info.racers:
        features = build_racer_features(racer, race_info)
        all_features.append(features)

    # レース内での相対特徴量を追加
    if all_features:
        # 展示タイムの偏差値化
        exhibit_times = [f["exhibit_time"] for f in all_features if f["exhibit_time"] > 0]
        if exhibit_times:
            mean_et = np.mean(exhibit_times)
            std_et = np.std(exhibit_times) if len(exhibit_times) > 1 else 1.0
            std_et = max(std_et, 0.01)  # ゼロ除算防止
            for f in all_features:
                if f["exhibit_time"] > 0:
                    f["exhibit_time_zscore"] = (f["exhibit_time"] - mean_et) / std_et
                else:
                    f["exhibit_time_zscore"] = 0.0
        else:
            for f in all_features:
                f["exhibit_time_zscore"] = 0.0

        # 勝率の偏差値化
        win_rates = [f["win_rate_all"] for f in all_features if f["win_rate_all"] > 0]
        if win_rates:
            mean_wr = np.mean(win_rates)
            std_wr = np.std(win_rates) if len(win_rates) > 1 else 1.0
            std_wr = max(std_wr, 0.01)
            for f in all_features:
                f["win_rate_zscore"] = (f["win_rate_all"] - mean_wr) / std_wr
        else:
            for f in all_features:
                f["win_rate_zscore"] = 0.0

    return all_features
