"""バックテスト: 過去データでモデルの収支シミュレーションを行う

Usage:
    python -m src.model.backtester
    python -m src.model.backtester --data data/raw/heiwajima_*.csv --budget 3000
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "model"


def load_data(csv_paths: list[str]) -> pd.DataFrame:
    dfs = []
    for path in csv_paths:
        for f in glob.glob(path):
            dfs.append(pd.read_csv(f))
    if not dfs:
        raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_paths}")
    return pd.concat(dfs, ignore_index=True)


def simulate(
    df: pd.DataFrame,
    budget_per_race: int = 3000,
    strategy: str = "balance",
    use_model: bool = True,
) -> dict:
    """過去データでの収支シミュレーション

    Args:
        df: レースデータ（横持ちCSV）
        budget_per_race: 1レースあたりの掛け金
        strategy: 戦略モード
        use_model: True=LightGBMモデル使用, False=統計ベース

    Returns:
        シミュレーション結果
    """
    from src.scraper.race_data import RaceInfo, Racer, WeatherInfo
    from src.model.predictor import predict_trifecta_probabilities
    from src.betting.optimizer import optimize_bets

    total_investment = 0
    total_return = 0
    races_count = 0
    hits = 0
    results_log = []

    for _, race in df.iterrows():
        finish_1st = race.get("finish_1st", 0)
        finish_2nd = race.get("finish_2nd", 0)
        finish_3rd = race.get("finish_3rd", 0)

        if not finish_1st or not finish_2nd or not finish_3rd:
            continue

        actual_result = f"{int(finish_1st)}-{int(finish_2nd)}-{int(finish_3rd)}"

        # RaceInfo オブジェクトに変換
        racers = []
        for w in range(1, 7):
            prefix = f"w{w}_"
            r = Racer(
                waku=w,
                name=str(race.get(f"{prefix}name", "")),
                rank=str(race.get(f"{prefix}rank", "B2")),
                win_rate_all=float(race.get(f"{prefix}win_rate_all", 0) or 0),
                win_rate_2r_all=float(race.get(f"{prefix}win_rate_2r_all", 0) or 0),
                win_rate_local=float(race.get(f"{prefix}win_rate_local", 0) or 0),
                win_rate_2r_local=float(race.get(f"{prefix}win_rate_2r_local", 0) or 0),
                motor_2r=float(race.get(f"{prefix}motor_2r", 0) or 0),
                boat_2r=float(race.get(f"{prefix}boat_2r", 0) or 0),
                exhibit_time=float(race.get(f"{prefix}exhibit_time", 0) or 0),
            )
            racers.append(r)

        weather = WeatherInfo(
            weather=str(race.get("weather", "")),
            wind_speed=int(race.get("wind_speed", 0) or 0),
            wave_height=int(race.get("wave_height", 0) or 0),
            temperature=float(race.get("temperature", 0) or 0),
            water_temp=float(race.get("water_temp", 0) or 0),
        )

        race_info = RaceInfo(
            race_no=int(race.get("race_no", 0)),
            date=str(race.get("date", "")),
            racers=racers,
            weather=weather,
        )

        # 予測
        trifecta_probs = predict_trifecta_probabilities(race_info, top_n=60)

        # ダミーオッズ（バックテストでは実オッズがCSVにないため概算）
        # 実際にはオッズも収集してCSVに含める改良が可能
        dummy_odds = _estimate_odds(trifecta_probs)

        # 買い目最適化
        plan = optimize_bets(
            trifecta_probs=trifecta_probs,
            odds_dict=dummy_odds,
            budget=budget_per_race,
            strategy=strategy,
        )

        # 的中判定
        race_investment = sum(s.bet_amount for s in plan.suggestions)
        race_return = 0
        hit = False

        for s in plan.suggestions:
            if s.combo == actual_result:
                race_return = int(s.bet_amount * s.odds)
                hit = True
                break

        total_investment += race_investment
        total_return += race_return
        races_count += 1
        if hit:
            hits += 1

        results_log.append({
            "date": race.get("date", ""),
            "race_no": race.get("race_no", 0),
            "actual": actual_result,
            "investment": race_investment,
            "return": race_return,
            "hit": hit,
            "profit": race_return - race_investment,
        })

    recovery_rate = (total_return / total_investment * 100) if total_investment > 0 else 0
    hit_rate = (hits / races_count * 100) if races_count > 0 else 0

    result = {
        "races_count": races_count,
        "total_investment": total_investment,
        "total_return": total_return,
        "profit": total_return - total_investment,
        "recovery_rate": recovery_rate,
        "hit_rate": hit_rate,
        "hits": hits,
        "results_log": results_log,
    }

    return result


def _estimate_odds(trifecta_probs: list[tuple[str, float]]) -> dict[str, float]:
    """的中確率からオッズを概算する（バックテスト用）

    実際の市場オッズを近似。控除率25%を考慮。
    """
    odds = {}
    for combo, prob in trifecta_probs:
        if prob > 0:
            # 理論オッズ = 1/確率 × (1 - 控除率)
            theoretical = (1 / prob) * 0.75
            # ノイズを加えてリアルなオッズに近づける
            odds[combo] = round(max(theoretical, 1.0), 1)
    return odds


def print_report(result: dict):
    """バックテスト結果を表示する"""
    print("\n" + "=" * 60)
    print("  バックテスト結果")
    print("=" * 60)
    print(f"  レース数:     {result['races_count']}")
    print(f"  的中数:       {result['hits']}")
    print(f"  的中率:       {result['hit_rate']:.1f}%")
    print(f"  総投資額:     ¥{result['total_investment']:,}")
    print(f"  総回収額:     ¥{result['total_return']:,}")
    print(f"  収支:         ¥{result['profit']:,}")
    print(f"  回収率:       {result['recovery_rate']:.1f}%")
    print("=" * 60)

    # 月別収支
    if result["results_log"]:
        log_df = pd.DataFrame(result["results_log"])
        log_df["month"] = log_df["date"].astype(str).str[:6]
        monthly = log_df.groupby("month").agg(
            races=("hit", "count"),
            hits=("hit", "sum"),
            investment=("investment", "sum"),
            returns=("return", "sum"),
        )
        monthly["profit"] = monthly["returns"] - monthly["investment"]
        monthly["recovery"] = (monthly["returns"] / monthly["investment"] * 100).round(1)

        print("\n=== 月別収支 ===")
        print(monthly.to_string())


def main():
    parser = argparse.ArgumentParser(description="バックテスト")
    parser.add_argument(
        "--data", type=str, nargs="+",
        default=[str(DATA_DIR / "heiwajima_*.csv")],
    )
    parser.add_argument("--budget", type=int, default=3000)
    parser.add_argument(
        "--strategy", type=str, default="balance",
        choices=["balance", "roi", "hit_rate", "longshot"],
    )
    args = parser.parse_args()

    df = load_data(args.data)
    print(f"レースデータ: {len(df)} 件")
    print(f"予算: ¥{args.budget:,}/レース")
    print(f"戦略: {args.strategy}")

    result = simulate(df, budget_per_race=args.budget, strategy=args.strategy)
    print_report(result)


if __name__ == "__main__":
    main()
