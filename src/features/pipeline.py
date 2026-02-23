"""選手・レースデータから予測用特徴量を生成するパイプライン

特徴量一覧:
- 基本: waku, win_rate_all, win_rate_2r_all, win_rate_local, win_rate_2r_local, rank_score
- 機材: motor_2r, motor_3r, boat_2r, boat_3r
- 平和島固有: course_base_win_rate, actual_course, is_makunari
- 直前: exhibit_time, exhibit_st, avg_start_timing
- 選手リスク: flying_count, late_count, fl_risk_score
- 気象: wind_speed, wave_height, wind_course_interaction, wave_motor_interaction
- 相対(Zスコア): exhibit_time_zscore, win_rate_zscore, motor_2r_zscore, exhibit_st_zscore
- 組合せ: motor_boat_combined, rate_rank_interaction, in_course_advantage
"""

import numpy as np
from ..scraper.race_data import RaceInfo, Racer


# 平和島のコース別1着率（過去統計ベース）
HEIWAJIMA_COURSE_WIN_RATE = {
    1: 0.449,
    2: 0.152,
    3: 0.130,
    4: 0.122,
    5: 0.087,
    6: 0.060,
}

# 平和島のコース別2連率
HEIWAJIMA_COURSE_2R_RATE = {
    1: 0.620,
    2: 0.310,
    3: 0.260,
    4: 0.240,
    5: 0.190,
    6: 0.140,
}

# 平和島のコース別3連率
HEIWAJIMA_COURSE_3R_RATE = {
    1: 0.730,
    2: 0.470,
    3: 0.400,
    4: 0.380,
    5: 0.320,
    6: 0.270,
}

# 級別スコア（細かく設定）
RANK_SCORE = {
    "A1": 4.0,
    "A2": 3.0,
    "B1": 2.0,
    "B2": 1.0,
    "": 1.0,
}


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
        course_entry: 進入コース
        exhibit_st: 展示スタートタイミング
    """
    features = {}

    # ── 基本特徴量 ──
    features["waku"] = racer.waku
    features["win_rate_all"] = racer.win_rate_all
    features["win_rate_2r_all"] = racer.win_rate_2r_all
    features["win_rate_local"] = racer.win_rate_local
    features["win_rate_2r_local"] = racer.win_rate_2r_local
    features["rank_score"] = RANK_SCORE.get(racer.rank, 1.0)

    # ── 機材特徴量 ──
    features["motor_2r"] = racer.motor_2r
    features["motor_3r"] = getattr(racer, "motor_3r", 0.0) or 0.0
    features["boat_2r"] = racer.boat_2r
    features["boat_3r"] = getattr(racer, "boat_3r", 0.0) or 0.0

    # モーター+ボート複合スコア
    features["motor_boat_combined"] = (
        racer.motor_2r * 0.6 + racer.boat_2r * 0.4
    ) if racer.motor_2r > 0 and racer.boat_2r > 0 else 0.0

    # ── 進入コースの反映 ──
    actual_course = racer.waku  # デフォルトは枠なり
    # 1) Racer に直接設定された進入コース
    racer_course = getattr(racer, "course_entry", 0) or 0
    if 1 <= racer_course <= 6:
        actual_course = racer_course
    # 2) 外部から渡された進入コース辞書
    elif course_entry and racer.waku in course_entry:
        actual_course = course_entry[racer.waku]

    features["actual_course"] = actual_course
    features["course_base_win_rate"] = HEIWAJIMA_COURSE_WIN_RATE.get(actual_course, 0.1)
    features["course_2r_rate"] = HEIWAJIMA_COURSE_2R_RATE.get(actual_course, 0.2)
    features["course_3r_rate"] = HEIWAJIMA_COURSE_3R_RATE.get(actual_course, 0.3)

    # 枠なりかどうか
    features["is_makunari"] = 1 if actual_course == racer.waku else 0

    # インコース有利度（1コース=最有利、内枠から外枠に移動した場合のペナルティ）
    if actual_course < racer.waku:
        # 前付け（内側に入った）→ ボーナス
        features["in_course_advantage"] = (racer.waku - actual_course) * 0.1
    elif actual_course > racer.waku:
        # 外に押し出された → ペナルティ
        features["in_course_advantage"] = -(actual_course - racer.waku) * 0.05
    else:
        features["in_course_advantage"] = 0.0

    # ── 直前特徴量 ──
    features["exhibit_time"] = racer.exhibit_time

    # 展示スタートタイミング
    racer_st = getattr(racer, "exhibit_st", 0.0) or 0.0
    if not racer_st and exhibit_st and racer.waku in exhibit_st:
        racer_st = exhibit_st[racer.waku]
    features["exhibit_st"] = racer_st

    # 平均ST
    features["avg_start_timing"] = getattr(racer, "avg_start_timing", 0.0) or 0.0

    # ── 選手リスク特徴量 ──
    f_count = getattr(racer, "flying_count", 0) or 0
    l_count = getattr(racer, "late_count", 0) or 0
    features["flying_count"] = f_count
    features["late_count"] = l_count
    # F/Lリスクスコア（フライング・出遅れが多い選手はスタートが慎重になりやすい）
    features["fl_risk_score"] = f_count * 0.5 + l_count * 0.3

    # ── 勝率×級別相互作用 ──
    features["rate_rank_interaction"] = (
        racer.win_rate_all * features["rank_score"] / 8.0
    )

    # ── 気象影響スコア ──
    weather = race_info.weather
    features["wind_speed"] = weather.wind_speed
    features["wave_height"] = weather.wave_height

    # 風向とコースの相互作用
    wind_dir = weather.wind_direction
    is_headwind = "向" in wind_dir or "北" in wind_dir
    is_tailwind = "追" in wind_dir or "南" in wind_dir

    course_pos = actual_course
    if is_headwind:
        # 向かい風 → アウトコース有利、インコース不利
        features["wind_course_interaction"] = (course_pos - 3.5) * 0.05
    elif is_tailwind:
        # 追い風 → インコース有利
        features["wind_course_interaction"] = (3.5 - course_pos) * 0.05
    else:
        # 横風や弱風 → 影響小
        features["wind_course_interaction"] = (3.5 - course_pos) * 0.02

    # 強風の場合はコース影響を増幅
    if weather.wind_speed >= 5:
        features["wind_course_interaction"] *= 1.5

    # 波高×モーター相互作用
    if weather.wave_height >= 5:
        features["wave_motor_interaction"] = (racer.motor_2r - 30.0) / 30.0 * 0.3
    elif weather.wave_height >= 3:
        features["wave_motor_interaction"] = (racer.motor_2r - 30.0) / 30.0 * 0.15
    else:
        features["wave_motor_interaction"] = 0.0

    return features


def build_race_features(
    race_info: RaceInfo,
    course_entry: dict[int, int] | None = None,
    exhibit_st: dict[int, float] | None = None,
) -> list[dict]:
    """1レース分の全選手の特徴量を生成する

    レース内での相対的な特徴量（Zスコア）も追加する。
    """
    all_features = []
    for racer in race_info.racers:
        features = build_racer_features(racer, race_info, course_entry, exhibit_st)
        all_features.append(features)

    if not all_features:
        return all_features

    # ── レース内 Zスコア（相対特徴量） ──
    _add_zscore(all_features, "exhibit_time", "exhibit_time_zscore",
                filter_fn=lambda v: v > 0)
    _add_zscore(all_features, "win_rate_all", "win_rate_zscore",
                filter_fn=lambda v: v > 0)
    _add_zscore(all_features, "motor_2r", "motor_2r_zscore",
                filter_fn=lambda v: v > 0)
    _add_zscore(all_features, "exhibit_st", "exhibit_st_zscore",
                filter_fn=lambda v: v != 0.0)
    _add_zscore(all_features, "boat_2r", "boat_2r_zscore",
                filter_fn=lambda v: v > 0)
    _add_zscore(all_features, "avg_start_timing", "avg_st_zscore",
                filter_fn=lambda v: v > 0)

    return all_features


def _add_zscore(features_list: list[dict], key: str, zscore_key: str,
                filter_fn=None):
    """レース内のZスコアを計算して追加する"""
    values = []
    for f in features_list:
        v = f.get(key, 0.0)
        if filter_fn is None or filter_fn(v):
            values.append(v)

    if len(values) >= 2:
        mean = np.mean(values)
        std = np.std(values)
        std = max(std, 0.01)
        for f in features_list:
            v = f.get(key, 0.0)
            if filter_fn is None or filter_fn(v):
                f[zscore_key] = (v - mean) / std
            else:
                f[zscore_key] = 0.0
    else:
        for f in features_list:
            f[zscore_key] = 0.0
