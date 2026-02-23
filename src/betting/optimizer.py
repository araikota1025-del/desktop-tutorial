"""買い目提案エンジン: 期待値算出 + ケリー基準ベースの資金配分"""

from dataclasses import dataclass


@dataclass
class BetSuggestion:
    """1つの買い目提案"""
    combo: str           # "2-1-3"
    odds: float          # オッズ
    predicted_prob: float # 予測的中確率
    expected_value: float # 期待値
    bet_amount: int       # 推奨配分金額（円）
    kelly_fraction: float # ケリー比率


@dataclass
class BettingPlan:
    """レース全体の買い目提案"""
    race_no: int
    total_budget: int
    bet_type: str          # "3連単" / "3連複" / "2連単"
    strategy: str          # "balance" / "roi" / "hit_rate" / "longshot"
    suggestions: list[BetSuggestion]
    total_bet: int
    combined_hit_rate: float
    avg_expected_value: float


def kelly_criterion(prob: float, odds: float, fraction: float = 0.5) -> float:
    """修正ケリー基準で最適配分比率を算出する

    Args:
        prob: 予測的中確率
        odds: オッズ
        fraction: ケリー比率（0.5 = ハーフケリー）

    Returns:
        資金全体に対する配分比率 (0.0 ~ max_ratio)
    """
    b = odds - 1  # 純利益倍率
    q = 1 - prob

    f = (prob * b - q) / b
    f = max(f, 0.0)  # マイナスなら賭けない
    f *= fraction     # ハーフケリー等に調整

    return min(f, 0.3)  # 1つの買い目に30%以上は配分しない


def compute_expected_value(prob: float, odds: float) -> float:
    """期待値を算出する"""
    return prob * odds


def optimize_bets(
    trifecta_probs: list[tuple[str, float]],
    odds_dict: dict[str, float],
    budget: int,
    strategy: str = "balance",
    bet_type: str = "3連単",
    kelly_frac: float = 0.5,
    min_bet: int = 100,
) -> BettingPlan:
    """最適な買い目と資金配分を提案する

    Args:
        trifecta_probs: [(combo, prob), ...] 予測確率
        odds_dict: {combo: odds} オッズ
        budget: 掛け金（円）
        strategy: 戦略モード
        bet_type: 舟券種別
        kelly_frac: ケリー比率
        min_bet: 最小賭け金
    """
    candidates = []

    for combo, prob in trifecta_probs:
        odds = odds_dict.get(combo, 0.0)
        if odds <= 0:
            continue

        ev = compute_expected_value(prob, odds)
        kf = kelly_criterion(prob, odds, kelly_frac)

        candidates.append(BetSuggestion(
            combo=combo,
            odds=odds,
            predicted_prob=prob,
            expected_value=ev,
            bet_amount=0,
            kelly_fraction=kf,
        ))

    # 戦略に応じたフィルタリングとソート
    if strategy == "roi":
        # 回収率重視: 期待値が高い順、EV > 1.0 のみ
        candidates = [c for c in candidates if c.expected_value > 1.0]
        candidates.sort(key=lambda x: x.expected_value, reverse=True)
        candidates = candidates[:8]
    elif strategy == "hit_rate":
        # 的中率重視: 確率が高い順
        candidates.sort(key=lambda x: x.predicted_prob, reverse=True)
        candidates = candidates[:15]
    elif strategy == "longshot":
        # 万舟狙い: オッズ50倍以上で確率がある程度あるもの
        candidates = [c for c in candidates if c.odds >= 50 and c.predicted_prob >= 0.005]
        candidates.sort(key=lambda x: x.expected_value, reverse=True)
        candidates = candidates[:10]
    else:
        # バランス: 期待値 > 0.8 でソート
        candidates = [c for c in candidates if c.expected_value > 0.8]
        candidates.sort(key=lambda x: x.expected_value, reverse=True)
        candidates = candidates[:12]

    if not candidates:
        return BettingPlan(
            race_no=0, total_budget=budget, bet_type=bet_type,
            strategy=strategy, suggestions=[], total_bet=0,
            combined_hit_rate=0.0, avg_expected_value=0.0,
        )

    # ケリー基準で資金配分
    total_kelly = sum(c.kelly_fraction for c in candidates)
    if total_kelly <= 0:
        # ケリーが全部0の場合は均等配分
        total_kelly = len(candidates)
        for c in candidates:
            c.kelly_fraction = 1.0

    for c in candidates:
        ratio = c.kelly_fraction / total_kelly
        raw_amount = budget * ratio
        c.bet_amount = max(int(raw_amount // min_bet) * min_bet, min_bet)

    # 予算オーバーの調整
    total_bet = sum(c.bet_amount for c in candidates)
    while total_bet > budget and candidates:
        # 期待値が最も低い買い目を削除
        candidates.sort(key=lambda x: x.expected_value)
        removed = candidates.pop(0)
        total_bet -= removed.bet_amount
        candidates.sort(key=lambda x: x.expected_value, reverse=True)

    # 予算に余裕がある場合、上位に追加配分
    remaining = budget - sum(c.bet_amount for c in candidates)
    if remaining >= min_bet and candidates:
        candidates[0].bet_amount += (remaining // min_bet) * min_bet

    total_bet = sum(c.bet_amount for c in candidates)
    combined_hit = sum(c.predicted_prob for c in candidates)
    avg_ev = (
        sum(c.expected_value * c.bet_amount for c in candidates) / total_bet
        if total_bet > 0
        else 0.0
    )

    return BettingPlan(
        race_no=0,
        total_budget=budget,
        bet_type=bet_type,
        strategy=strategy,
        suggestions=candidates,
        total_bet=total_bet,
        combined_hit_rate=combined_hit,
        avg_expected_value=avg_ev,
    )
