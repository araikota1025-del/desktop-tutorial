"""着順確率を予測するモデル

初期版: 統計ベースのヒューリスティックモデル
将来: LightGBMで学習済みモデルに差し替え
"""

import numpy as np
from pathlib import Path
from itertools import permutations

from ..scraper.race_data import RaceInfo
from ..features.pipeline import build_race_features

MODEL_PATH = Path(__file__).parent.parent.parent / "model" / "lgbm_heiwajima.pkl"


def _compute_strength_scores(features_list: list[dict]) -> np.ndarray:
    """各艇の強さスコアを算出する（統計ベース）

    平和島特化の重み付け。将来LightGBMモデルに置き換える箇所。

    スコア構成（合計約 5.0 前後）:
    - コース別基礎勝率: 最大 1.35 (weight=3.0, max=0.449)
    - 全国勝率:         最大 2.00 (weight=2.0)
    - 当地勝率:         最大 1.50 (weight=1.5)
    - 級別:             最大 0.80 (weight=0.2)
    - モーター:         最大 1.00 (weight=1.0)
    - ボート:           最大 0.50 (weight=0.5)
    - 展示タイム:       ±0.30
    - 展示ST:           ±0.40（新規）
    - モーター偏差値:   ±0.20（新規）
    - 風×コース:        ±0.175
    - 波×モーター:      ±0.30（新規）
    """
    # 学習済みモデルが存在する場合はそちらを使用
    if MODEL_PATH.exists():
        return _predict_with_lgbm(features_list)

    scores = []
    for f in features_list:
        score = 0.0

        # コース別基礎勝率（最重要: 平和島はコースの影響大）
        score += f.get("course_base_win_rate", 0.1) * 3.0

        # 全国勝率（選手の実力）
        win_rate = f.get("win_rate_all", 4.0)
        score += (win_rate / 8.0) * 2.0

        # 当地勝率（平和島での実績）
        local_rate = f.get("win_rate_local", 0.0)
        if local_rate > 0:
            score += (local_rate / 8.0) * 1.5
        else:
            score += (win_rate / 8.0) * 0.5

        # 級別スコア
        score += f.get("rank_score", 1) * 0.2

        # モーター2連率
        motor = f.get("motor_2r", 30.0)
        score += (motor / 60.0) * 1.0

        # ボート2連率
        boat = f.get("boat_2r", 30.0)
        score += (boat / 60.0) * 0.5

        # 展示タイム偏差値（低い方が速い → マイナスほど良い）
        et_z = f.get("exhibit_time_zscore", 0.0)
        score -= et_z * 0.3

        # ── 新規特徴量 ──

        # 展示スタートタイミング（STが小さいほどスタートが速い）
        st_z = f.get("exhibit_st_zscore", 0.0)
        score -= st_z * 0.4  # STが速い選手にボーナス

        # モーター偏差値（レース内での相対的なモーター性能）
        motor_z = f.get("motor_2r_zscore", 0.0)
        score += motor_z * 0.2

        # 風×コース相互作用
        score += f.get("wind_course_interaction", 0.0) * 1.0

        # 波×モーター相互作用（荒天ではモーターの差が出る）
        score += f.get("wave_motor_interaction", 0.0)

        scores.append(max(score, 0.01))

    return np.array(scores)


def _predict_with_lgbm(features_list: list[dict]) -> np.ndarray:
    """LightGBMモデルで予測する"""
    try:
        import joblib
        import pandas as pd

        model = joblib.load(MODEL_PATH)
        feature_names = [
            "waku", "win_rate_all", "win_rate_2r_all", "win_rate_local",
            "win_rate_2r_local", "rank_score", "motor_2r", "boat_2r",
            "course_base_win_rate", "exhibit_time", "wind_speed",
            "wave_height", "wind_course_interaction",
            "exhibit_time_zscore", "win_rate_zscore",
        ]
        df = pd.DataFrame(features_list)
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0.0
        predictions = model.predict(df[feature_names])
        return np.array(predictions)
    except Exception:
        # フォールバック
        return np.array([f.get("win_rate_all", 4.0) for f in features_list])


def predict_win_probabilities(
    race_info: RaceInfo,
    course_entry: dict[int, int] | None = None,
    exhibit_st: dict[int, float] | None = None,
) -> dict[int, float]:
    """各艇の1着確率を予測する

    Args:
        race_info: レース情報
        course_entry: 進入コース（heiwajima.gr.jpから取得）
        exhibit_st: 展示スタートタイミング

    Returns:
        {1: 0.35, 2: 0.20, 3: 0.15, 4: 0.13, 5: 0.10, 6: 0.07}
    """
    features_list = build_race_features(race_info, course_entry, exhibit_st)
    if not features_list:
        return {i: 1 / 6 for i in range(1, 7)}

    scores = _compute_strength_scores(features_list)

    # softmax で確率に変換
    exp_scores = np.exp(scores - np.max(scores))
    probabilities = exp_scores / exp_scores.sum()

    return {
        features_list[i]["waku"]: float(probabilities[i])
        for i in range(len(features_list))
    }


def _plackett_luce_scores(
    race_info: RaceInfo,
    course_entry: dict[int, int] | None = None,
    exhibit_st: dict[int, float] | None = None,
):
    """Plackett-Luce モデル用の基礎データを返す"""
    features_list = build_race_features(race_info, course_entry, exhibit_st)
    if not features_list:
        return None, None, None
    scores = _compute_strength_scores(features_list)
    waku_list = [f["waku"] for f in features_list]
    return features_list, scores, waku_list


def predict_trifecta_probabilities(
    race_info: RaceInfo,
    top_n: int = 30,
    course_entry: dict[int, int] | None = None,
    exhibit_st: dict[int, float] | None = None,
) -> list[tuple[str, float]]:
    """3連単の着順確率を予測する（上位N件）

    簡易版: 1着確率をベースに条件付き確率で近似
    """
    features_list, scores, waku_list = _plackett_luce_scores(
        race_info, course_entry, exhibit_st
    )
    if scores is None:
        return []

    # 全120通りの確率を計算（Plackett-Luce モデルの簡易版）
    all_combos = []
    for perm in permutations(range(len(waku_list)), 3):
        i, j, k = perm
        # 1着確率
        remaining_1 = list(range(len(waku_list)))
        p1 = scores[i] / scores[remaining_1].sum()

        # 2着確率（1着を除く）
        remaining_2 = [x for x in remaining_1 if x != i]
        p2 = scores[j] / scores[remaining_2].sum()

        # 3着確率（1着2着を除く）
        remaining_3 = [x for x in remaining_2 if x != j]
        p3 = scores[k] / scores[remaining_3].sum()

        prob = p1 * p2 * p3
        combo_str = f"{waku_list[i]}-{waku_list[j]}-{waku_list[k]}"
        all_combos.append((combo_str, float(prob)))

    # 確率の高い順にソート
    all_combos.sort(key=lambda x: x[1], reverse=True)
    return all_combos[:top_n]


def predict_exacta_probabilities(
    race_info: RaceInfo, top_n: int = 30
) -> list[tuple[str, float]]:
    """2連単の着順確率を予測する（上位N件）"""
    features_list, scores, waku_list = _plackett_luce_scores(race_info)
    if scores is None:
        return []

    all_combos = []
    for perm in permutations(range(len(waku_list)), 2):
        i, j = perm
        remaining_1 = list(range(len(waku_list)))
        p1 = scores[i] / scores[remaining_1].sum()
        remaining_2 = [x for x in remaining_1 if x != i]
        p2 = scores[j] / scores[remaining_2].sum()

        prob = p1 * p2
        combo_str = f"{waku_list[i]}-{waku_list[j]}"
        all_combos.append((combo_str, float(prob)))

    all_combos.sort(key=lambda x: x[1], reverse=True)
    return all_combos[:top_n]


def predict_quinella_probabilities(
    race_info: RaceInfo, top_n: int = 15
) -> list[tuple[str, float]]:
    """2連複の確率を予測する（上位N件）

    2連複は順番不問なので、P(A-B) + P(B-A) を合算する。
    """
    features_list, scores, waku_list = _plackett_luce_scores(race_info)
    if scores is None:
        return []

    from itertools import combinations
    combo_probs: dict[str, float] = {}

    for ci, cj in combinations(range(len(waku_list)), 2):
        # A→B の確率
        remaining = list(range(len(waku_list)))
        p_ab = (scores[ci] / scores[remaining].sum()) * (
            scores[cj] / scores[[x for x in remaining if x != ci]].sum()
        )
        # B→A の確率
        p_ba = (scores[cj] / scores[remaining].sum()) * (
            scores[ci] / scores[[x for x in remaining if x != cj]].sum()
        )
        w_a, w_b = sorted([waku_list[ci], waku_list[cj]])
        combo_str = f"{w_a}={w_b}"
        combo_probs[combo_str] = float(p_ab + p_ba)

    all_combos = sorted(combo_probs.items(), key=lambda x: x[1], reverse=True)
    return all_combos[:top_n]
