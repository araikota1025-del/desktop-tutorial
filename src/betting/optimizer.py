"""買い目提案エンジン: 期待値算出 + ケリー基準ベースの資金配分

戦略モード:
- balance:    バランス型。期待値と的中率の両方を考慮
- roi:        回収率重視。期待値 > 1.0 に絞る
- hit_rate:   的中率重視。確率上位を広く買う
- conservative: 安定重視。合成的中率50%以上を目指す（2連複推奨）
- longshot:   万舟狙い。高オッズ + ある程度の確率
"""

from dataclasses import dataclass


@dataclass
class BetSuggestion:
    """1つの買い目提案"""
    combo: str           # "2-1-3" / "1-2" / "1=2"
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
    bet_type: str          # "3連単" / "2連単" / "2連複"
    strategy: str
    suggestions: list[BetSuggestion]
    total_bet: int
    combined_hit_rate: float
    avg_expected_value: float
    risk_label: str = ""   # リスク評価ラベル


def kelly_criterion(prob: float, odds: float, fraction: float = 0.5) -> float:
    """修正ケリー基準で最適配分比率を算出する"""
    b = odds - 1  # 純利益倍率
    if b <= 0:
        return 0.0
    q = 1 - prob

    f = (prob * b - q) / b
    f = max(f, 0.0)
    f *= fraction

    return min(f, 0.3)


def compute_expected_value(prob: float, odds: float) -> float:
    """期待値を算出する"""
    return prob * odds


def _assess_risk(combined_hit: float, avg_ev: float, bet_type: str) -> str:
    """リスク評価ラベルを付与する"""
    if combined_hit >= 0.5 and avg_ev >= 0.9:
        return "低リスク - 安定型"
    elif combined_hit >= 0.35 and avg_ev >= 0.85:
        return "中リスク - バランス型"
    elif combined_hit >= 0.2:
        return "高リスク - 攻撃型"
    else:
        return "超高リスク - ギャンブル型"


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
    if strategy == "conservative":
        # ── 安定重視: 合成的中率50%以上を目標 ──
        # まず確率の高い順にソートし、合成的中率が目標に達するまで追加
        candidates.sort(key=lambda x: x.predicted_prob, reverse=True)
        selected = []
        cumulative_prob = 0.0
        target_hit_rate = 0.50
        for c in candidates:
            if c.expected_value < 0.6:
                continue  # 期待値が極端に低いものは除外
            selected.append(c)
            cumulative_prob += c.predicted_prob
            if cumulative_prob >= target_hit_rate:
                break
        # 目標に届かなくても上位を選択
        if not selected:
            selected = candidates[:10]
        candidates = selected

    elif strategy == "roi":
        # 回収率重視: 期待値が高い順、EV > 1.0 のみ
        candidates = [c for c in candidates if c.expected_value > 1.0]
        candidates.sort(key=lambda x: x.expected_value, reverse=True)
        candidates = candidates[:8]

    elif strategy == "hit_rate":
        # 的中率重視: 確率が高い順、幅広く買う
        candidates.sort(key=lambda x: x.predicted_prob, reverse=True)
        # 合成的中率40%を目標
        selected = []
        cumulative = 0.0
        for c in candidates:
            selected.append(c)
            cumulative += c.predicted_prob
            if cumulative >= 0.40:
                break
        candidates = selected if selected else candidates[:15]

    elif strategy == "longshot":
        # 万舟狙い: オッズ50倍以上で確率がある程度あるもの
        candidates = [c for c in candidates if c.odds >= 50 and c.predicted_prob >= 0.005]
        candidates.sort(key=lambda x: x.expected_value, reverse=True)
        candidates = candidates[:10]

    else:
        # バランス: 期待値 > 0.8 でソート、合成的中率30%目標
        candidates = [c for c in candidates if c.expected_value > 0.8]
        candidates.sort(key=lambda x: x.expected_value, reverse=True)
        candidates = candidates[:12]

    if not candidates:
        return BettingPlan(
            race_no=0, total_budget=budget, bet_type=bet_type,
            strategy=strategy, suggestions=[], total_bet=0,
            combined_hit_rate=0.0, avg_expected_value=0.0,
            risk_label="候補なし",
        )

    # ── ケリー基準で資金配分 ──
    total_kelly = sum(c.kelly_fraction for c in candidates)
    if total_kelly <= 0:
        # ケリーが全部0の場合 → 安定重視なら確率比例配分
        if strategy in ("conservative", "hit_rate"):
            total_prob = sum(c.predicted_prob for c in candidates)
            for c in candidates:
                c.kelly_fraction = c.predicted_prob / total_prob if total_prob > 0 else 1.0
        else:
            for c in candidates:
                c.kelly_fraction = 1.0
        total_kelly = sum(c.kelly_fraction for c in candidates)

    for c in candidates:
        ratio = c.kelly_fraction / total_kelly
        raw_amount = budget * ratio
        c.bet_amount = max(int(raw_amount // min_bet) * min_bet, min_bet)

    # 予算オーバーの調整
    total_bet = sum(c.bet_amount for c in candidates)
    while total_bet > budget and candidates:
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

    risk_label = _assess_risk(combined_hit, avg_ev, bet_type)

    return BettingPlan(
        race_no=0,
        total_budget=budget,
        bet_type=bet_type,
        strategy=strategy,
        suggestions=candidates,
        total_bet=total_bet,
        combined_hit_rate=combined_hit,
        avg_expected_value=avg_ev,
        risk_label=risk_label,
    )
