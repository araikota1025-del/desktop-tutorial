"""LightGBM ランキング学習モデルの訓練スクリプト

Usage:
    python -m src.model.trainer
    python -m src.model.trainer --data data/raw/heiwajima_20230101_20260101.csv
    python -m src.model.trainer --data data/raw/heiwajima_*.csv --eval-ratio 0.2
"""

import argparse
import glob
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# 平和島のコース別1着率（統計値）
HEIWAJIMA_COURSE_WIN_RATE = {
    1: 0.449, 2: 0.152, 3: 0.130,
    4: 0.122, 5: 0.087, 6: 0.060,
}
RANK_SCORE_MAP = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_race_data(csv_paths: list[str]) -> pd.DataFrame:
    """CSVファイルからレースデータを読み込む"""
    dfs = []
    for path in csv_paths:
        for f in glob.glob(path):
            df = pd.read_csv(f)
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_paths}")

    df = pd.concat(dfs, ignore_index=True)
    print(f"読み込み: {len(df)} レース")
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """横持ちCSVから学習用の縦持ちデータ（1行=1艇）を生成する

    Returns:
        features_df: 特徴量DataFrame
        labels: 着順ラベル（ランキング学習用: 小さいほど良い）
        groups: グループ（レースID）
    """
    rows = []
    labels = []
    groups = []

    for idx, race in df.iterrows():
        race_id = f"{race['date']}_{race['race_no']}"
        finish_1st = race.get("finish_1st", 0)
        finish_2nd = race.get("finish_2nd", 0)
        finish_3rd = race.get("finish_3rd", 0)

        if not finish_1st:
            continue

        # 着順マップ: waku → finish_rank
        finish_map = {}
        if finish_1st:
            finish_map[int(finish_1st)] = 1
        if finish_2nd:
            finish_map[int(finish_2nd)] = 2
        if finish_3rd:
            finish_map[int(finish_3rd)] = 3

        race_features = []
        race_labels = []
        valid = True

        for waku in range(1, 7):
            prefix = f"w{waku}_"

            win_rate_all = race.get(f"{prefix}win_rate_all", 0) or 0
            win_rate_2r_all = race.get(f"{prefix}win_rate_2r_all", 0) or 0
            win_rate_local = race.get(f"{prefix}win_rate_local", 0) or 0
            win_rate_2r_local = race.get(f"{prefix}win_rate_2r_local", 0) or 0
            motor_2r = race.get(f"{prefix}motor_2r", 0) or 0
            motor_3r = race.get(f"{prefix}motor_3r", 0) or 0
            boat_2r = race.get(f"{prefix}boat_2r", 0) or 0
            boat_3r = race.get(f"{prefix}boat_3r", 0) or 0
            exhibit_time = race.get(f"{prefix}exhibit_time", 0) or 0
            exhibit_st = race.get(f"{prefix}exhibit_st", 0) or 0
            avg_st = race.get(f"{prefix}avg_start_timing", 0) or 0
            flying_count = race.get(f"{prefix}flying_count", 0) or 0
            late_count = race.get(f"{prefix}late_count", 0) or 0
            rank_str = str(race.get(f"{prefix}rank", "B2"))
            rank_score = RANK_SCORE_MAP.get(rank_str, 1)

            # コース別基礎勝率
            course_base_wr = HEIWAJIMA_COURSE_WIN_RATE.get(waku, 0.1)

            # 風向×コース交互作用
            wind_speed = race.get("wind_speed", 0) or 0
            wave_height = race.get("wave_height", 0) or 0
            weather_str = str(race.get("weather", ""))
            is_headwind = "向" in weather_str or "北" in weather_str
            wind_course = (waku - 3.5) * 0.05 if is_headwind else (3.5 - waku) * 0.05

            # 複合スコア
            motor_boat = (float(motor_2r) * 0.6 + float(boat_2r) * 0.4
                          if motor_2r and boat_2r else 0.0)
            fl_risk = float(flying_count) * 0.5 + float(late_count) * 0.3
            rate_rank = float(win_rate_all) * rank_score / 8.0

            feature = {
                "waku": waku,
                "win_rate_all": float(win_rate_all),
                "win_rate_2r_all": float(win_rate_2r_all),
                "win_rate_local": float(win_rate_local),
                "win_rate_2r_local": float(win_rate_2r_local),
                "rank_score": rank_score,
                "motor_2r": float(motor_2r),
                "motor_3r": float(motor_3r),
                "boat_2r": float(boat_2r),
                "boat_3r": float(boat_3r),
                "motor_boat_combined": motor_boat,
                "course_base_win_rate": course_base_wr,
                "actual_course": waku,  # CSVでは枠なり前提
                "is_makunari": 1,
                "exhibit_time": float(exhibit_time),
                "exhibit_st": float(exhibit_st),
                "avg_start_timing": float(avg_st),
                "flying_count": int(flying_count),
                "late_count": int(late_count),
                "fl_risk_score": fl_risk,
                "wind_speed": int(wind_speed),
                "wave_height": int(wave_height),
                "wind_course_interaction": wind_course,
                "in_course_advantage": 0.0,
                "rate_rank_interaction": rate_rank,
            }
            race_features.append(feature)

            # ラベル: ランキング学習では「関連度」なので高い方が良い
            finish_rank = finish_map.get(waku, 0)
            if finish_rank == 1:
                relevance = 5
            elif finish_rank == 2:
                relevance = 4
            elif finish_rank == 3:
                relevance = 3
            else:
                relevance = 0
            race_labels.append(relevance)

        if valid and len(race_features) == 6:
            rows.extend(race_features)
            labels.extend(race_labels)
            groups.extend([race_id] * 6)

    features_df = pd.DataFrame(rows)

    # レース内の偏差値化
    _add_zscore_column(features_df, groups, "exhibit_time", "exhibit_time_zscore")
    _add_zscore_column(features_df, groups, "win_rate_all", "win_rate_zscore")
    _add_zscore_column(features_df, groups, "motor_2r", "motor_2r_zscore")
    _add_zscore_column(features_df, groups, "exhibit_st", "exhibit_st_zscore")
    _add_zscore_column(features_df, groups, "boat_2r", "boat_2r_zscore")
    _add_zscore_column(features_df, groups, "avg_start_timing", "avg_st_zscore")

    group_arr = np.array(groups)
    unique_groups = np.unique(group_arr)
    print(f"特徴量生成: {len(features_df)} 行 ({len(unique_groups)} レース × 6艇)")
    return features_df, np.array(labels), group_arr


def _add_zscore_column(df: pd.DataFrame, groups: list, src_col: str, dst_col: str):
    """レース内のZスコアを計算して列を追加する"""
    df[dst_col] = 0.0
    group_arr = np.array(groups)
    for g in np.unique(group_arr):
        mask = group_arr == g
        vals = df.loc[mask, src_col]
        if vals.std() > 0.01:
            df.loc[mask, dst_col] = (vals - vals.mean()) / vals.std()


def train_model(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    eval_ratio: float = 0.2,
) -> lgb.Booster:
    """LightGBM ランキングモデルを学習する"""

    feature_names = [
        "waku", "win_rate_all", "win_rate_2r_all", "win_rate_local",
        "win_rate_2r_local", "rank_score", "motor_2r", "motor_3r",
        "boat_2r", "boat_3r", "motor_boat_combined",
        "course_base_win_rate", "actual_course", "is_makunari",
        "exhibit_time", "exhibit_st", "avg_start_timing",
        "flying_count", "late_count", "fl_risk_score",
        "wind_speed", "wave_height", "wind_course_interaction",
        "in_course_advantage", "rate_rank_interaction",
        "exhibit_time_zscore", "win_rate_zscore",
        "motor_2r_zscore", "exhibit_st_zscore",
        "boat_2r_zscore", "avg_st_zscore",
    ]

    X = features_df[feature_names]
    y = labels

    # レース単位で訓練/検証を分割（同一レースが訓練と検証に分かれないようにする）
    unique_groups = np.unique(groups)
    splitter = GroupShuffleSplit(n_splits=1, test_size=eval_ratio, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups))

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # グループサイズ（各レースの艇数=6）を計算
    train_groups_arr = groups[train_idx]
    val_groups_arr = groups[val_idx]

    train_group_sizes = _compute_group_sizes(train_groups_arr)
    val_group_sizes = _compute_group_sizes(val_groups_arr)

    train_dataset = lgb.Dataset(
        X_train, label=y_train, group=train_group_sizes, feature_name=feature_names,
    )
    val_dataset = lgb.Dataset(
        X_val, label=y_val, group=val_group_sizes,
        feature_name=feature_names, reference=train_dataset,
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "max_depth": 6,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    print("\n学習開始...")
    print(f"  訓練: {len(X_train)} 行 ({len(train_group_sizes)} レース)")
    print(f"  検証: {len(X_val)} 行 ({len(val_group_sizes)} レース)")

    callbacks = [
        lgb.log_evaluation(period=100),
        lgb.early_stopping(stopping_rounds=50),
    ]

    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=1000,
        valid_sets=[val_dataset],
        valid_names=["val"],
        callbacks=callbacks,
    )

    print(f"\n学習完了: best_iteration = {model.best_iteration}")
    return model


def _compute_group_sizes(groups: np.ndarray) -> list[int]:
    """グループ配列からグループサイズのリストを計算する"""
    sizes = []
    current = groups[0]
    count = 0
    for g in groups:
        if g == current:
            count += 1
        else:
            sizes.append(count)
            current = g
            count = 1
    sizes.append(count)
    return sizes


def evaluate_model(model: lgb.Booster, features_df: pd.DataFrame, labels: np.ndarray, groups: np.ndarray):
    """モデルの評価指標を表示する"""
    feature_names = [
        "waku", "win_rate_all", "win_rate_2r_all", "win_rate_local",
        "win_rate_2r_local", "rank_score", "motor_2r", "motor_3r",
        "boat_2r", "boat_3r", "motor_boat_combined",
        "course_base_win_rate", "actual_course", "is_makunari",
        "exhibit_time", "exhibit_st", "avg_start_timing",
        "flying_count", "late_count", "fl_risk_score",
        "wind_speed", "wave_height", "wind_course_interaction",
        "in_course_advantage", "rate_rank_interaction",
        "exhibit_time_zscore", "win_rate_zscore",
        "motor_2r_zscore", "exhibit_st_zscore",
        "boat_2r_zscore", "avg_st_zscore",
    ]

    X = features_df[feature_names]
    predictions = model.predict(X)

    unique_groups = np.unique(groups)
    correct_1st = 0
    correct_top3 = 0
    total = 0

    for g in unique_groups:
        mask = groups == g
        preds = predictions[mask]
        true_labels = labels[mask]

        if len(preds) != 6:
            continue

        # 予測1着（スコア最大）
        pred_1st = np.argmax(preds)
        true_1st = np.argmax(true_labels)

        if pred_1st == true_1st:
            correct_1st += 1

        # 予測上位3艇
        pred_top3 = set(np.argsort(preds)[-3:])
        true_top3 = set(np.argsort(true_labels)[-3:])
        if pred_top3 == true_top3:
            correct_top3 += 1

        total += 1

    print(f"\n=== モデル評価 ===")
    print(f"  レース数: {total}")
    print(f"  1着的中率: {correct_1st / total:.1%} ({correct_1st}/{total})")
    print(f"  上位3艇完全一致率: {correct_top3 / total:.1%} ({correct_top3}/{total})")

    # 特徴量重要度
    importance = model.feature_importance(importance_type="gain")
    fi = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    print(f"\n=== 特徴量重要度 (gain) ===")
    for name, imp in fi:
        bar = "█" * int(imp / max(importance) * 30)
        print(f"  {name:30s} {imp:10.1f} {bar}")


def save_model(model: lgb.Booster, path: Path | None = None):
    """モデルを保存する"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        path = MODEL_DIR / "lgbm_heiwajima.pkl"

    joblib.dump(model, path)
    print(f"\nモデル保存: {path}")
    print(f"ファイルサイズ: {path.stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="LightGBM ランキングモデル学習")
    parser.add_argument(
        "--data", type=str, nargs="+",
        default=[str(DATA_DIR / "heiwajima_*.csv")],
        help="学習データCSVのパス（glob対応）",
    )
    parser.add_argument(
        "--eval-ratio", type=float, default=0.2,
        help="検証データの割合 (デフォルト: 0.2)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="モデル保存先パス",
    )
    args = parser.parse_args()

    # データ読み込み
    df = load_race_data(args.data)

    # 特徴量生成
    features_df, labels, groups = build_features(df)

    if len(features_df) < 60:
        print(f"\nエラー: 学習データが少なすぎます ({len(features_df) // 6} レース)")
        print("最低10レース（60行）以上のデータが必要です。")
        print("先にデータ収集を実行してください:")
        print("  python -m src.scraper.history_collector --months 6")
        return

    # モデル学習
    model = train_model(features_df, labels, groups, args.eval_ratio)

    # 評価
    evaluate_model(model, features_df, labels, groups)

    # 保存
    output_path = Path(args.output) if args.output else None
    save_model(model, output_path)


if __name__ == "__main__":
    main()
