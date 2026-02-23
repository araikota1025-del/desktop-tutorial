"""選手・レースデータから予測用特徴量を生成するパイプライン

特徴量一覧:
- 基本: waku, win_rate_all, win_rate_2r_all, win_rate_local, win_rate_2r_local, rank_score
- 機材: motor_2r, boat_2r
- 平和島固有: course_base_win_rate (実際の進入コースがあればそちらを使用)
- 直前: exhibit_time, exhibit_st (展示ST)
- 気象: wind_speed, wave_height, wind_course_interaction
- 相対: exhibit_time_zscore, win_rate_zscore, motor_2r_zscore
"""

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


def build_racer_features(
    racer: Racer,
    race_info: RaceInfo,
    course_entry: dict[int, int] | None = None,
    exhibit_st: dict[int, float] | None = None,
) -> dict:
    """1選手分の特徴量を生成する

    Args:
        racer: 選手データ
        race_info: レース情報
        course_entry: 進入コース（heiwajima.gr.jpから取得、なければ枠なり想定）
        exhibit_st: 展示スタートタイミング
    """
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

    # ── 進入コースの反映（平和島固有の重要特徴量）──
    # 展示航走で進入コースが変わることが多い（前付け等）
    actual_course = racer.waku  # デフォルトは枠なり
    if course_entry and racer.waku in course_entry:
        actual_course = course_entry[racer.waku]
    features["actual_course"] = actual_course
    features["course_base_win_rate"] = HEIWAJIMA_COURSE_WIN_RATE.get(actual_course, 0.1)

    # 枠なりかどうか（前付けは有利/不利が発生）
    features["is_makunari"] = 1 if actual_course == racer.waku else 0

    # ── 直前特徴量 ──
    features["exhibit_time"] = racer.exhibit_time

    # 展示スタートタイミング（小さいほど良い、マイナスはフライング気味）
    if exhibit_st and racer.waku in exhibit_st:
        features["exhibit_st"] = exhibit_st[racer.waku]
    else:
        features["exhibit_st"] = 0.0

    # ── 気象影響スコア（平和島固有ロジック）──
    weather = race_info.weather
    features["wind_speed"] = weather.wind_speed
    features["wave_height"] = weather.wave_height

    # 風向とコースの相互作用
    # 追い風 → イン有利、向かい風 → アウト有利
    wind_dir = weather.wind_direction
    is_headwind = "向" in wind_dir or "北" in wind_dir
    course_pos = actual_course  # 実際の進入コースを使用
    if is_headwind:
        features["wind_course_interaction"] = (course_pos - 3.5) * 0.05
    else:
        features["wind_course_interaction"] = (3.5 - course_pos) * 0.05

    # 波高×モーター相互作用（荒れた水面ではモーターの差が出やすい）
    if weather.wave_height >= 5:
        features["wave_motor_interaction"] = (racer.motor_2r - 30.0) / 30.0 * 0.3
    else:
        features["wave_motor_interaction"] = 0.0

    return features


def build_race_features(
    race_info: RaceInfo,
    course_entry: dict[int, int] | None = None,
    exhibit_st: dict[int, float] | None = None,
) -> list[dict]:
    """1レース分の全選手の特徴量を生成する"""
    all_features = []
    for racer in race_info.racers:
        features = build_racer_features(racer, race_info, course_entry, exhibit_st)
        all_features.append(features)

    # レース内での相対特徴量を追加
    if all_features:
        # 展示タイムの偏差値化
        exhibit_times = [f["exhibit_time"] for f in all_features if f["exhibit_time"] > 0]
        if exhibit_times:
            mean_et = np.mean(exhibit_times)
            std_et = np.std(exhibit_times) if len(exhibit_times) > 1 else 1.0
            std_et = max(std_et, 0.01)
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

        # モーター2連率の偏差値化
        motor_rates = [f["motor_2r"] for f in all_features if f["motor_2r"] > 0]
        if motor_rates:
            mean_mr = np.mean(motor_rates)
            std_mr = np.std(motor_rates) if len(motor_rates) > 1 else 1.0
            std_mr = max(std_mr, 0.01)
            for f in all_features:
                if f["motor_2r"] > 0:
                    f["motor_2r_zscore"] = (f["motor_2r"] - mean_mr) / std_mr
                else:
                    f["motor_2r_zscore"] = 0.0
        else:
            for f in all_features:
                f["motor_2r_zscore"] = 0.0

        # 展示STの偏差値化
        st_vals = [f["exhibit_st"] for f in all_features if f["exhibit_st"] != 0.0]
        if st_vals:
            mean_st = np.mean(st_vals)
            std_st = np.std(st_vals) if len(st_vals) > 1 else 1.0
            std_st = max(std_st, 0.01)
            for f in all_features:
                if f["exhibit_st"] != 0.0:
                    f["exhibit_st_zscore"] = (f["exhibit_st"] - mean_st) / std_st
                else:
                    f["exhibit_st_zscore"] = 0.0
        else:
            for f in all_features:
                f["exhibit_st_zscore"] = 0.0

    return all_features
