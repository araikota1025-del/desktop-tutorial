"""平和島ボートレース予想アプリ - Streamlit UI"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import time
from datetime import date
from itertools import permutations as itertools_permutations
import random

from src.scraper.race_data import (
    RaceInfo, Racer, WeatherInfo,
    fetch_race_list, fetch_before_info, fetch_odds_3t,
)
try:
    from src.scraper.race_data import debug_racelist_html
except ImportError:
    debug_racelist_html = None
try:
    from src.scraper.heiwajima import fetch_heiwajima_supplement
except ImportError:
    fetch_heiwajima_supplement = None
from src.model.predictor import predict_win_probabilities, predict_trifecta_probabilities
from src.betting.optimizer import optimize_bets

# ── キャッシュ設定 ──
CACHE_TTL_SEC = 300  # 5分


def _get_cache_key(date_str: str, race_no: int) -> str:
    return f"cache_{date_str}_{race_no}"


def get_cached_data(date_str: str, race_no: int):
    """キャッシュからデータを取得。TTL内ならスクレイピングをスキップする。"""
    cache_key = _get_cache_key(date_str, race_no)
    if cache_key in st.session_state:
        cached = st.session_state[cache_key]
        elapsed = time.time() - cached["timestamp"]
        if elapsed < CACHE_TTL_SEC:
            remaining = int(CACHE_TTL_SEC - elapsed)
            return cached["race_info"], cached["odds"], cached.get("supplement"), True, remaining
    return None, {}, None, False, 0


def set_cache(date_str: str, race_no: int, race_info: RaceInfo, odds: dict, supplement=None):
    """データをキャッシュに保存する"""
    cache_key = _get_cache_key(date_str, race_no)
    st.session_state[cache_key] = {
        "race_info": race_info,
        "odds": odds,
        "supplement": supplement,
        "timestamp": time.time(),
    }


def clear_all_cache():
    """全キャッシュをクリアする"""
    keys_to_remove = [k for k in st.session_state if k.startswith("cache_")]
    for k in keys_to_remove:
        del st.session_state[k]


# ── デモデータ生成関数 ──

def _create_demo_data(race_no: int, date_str: str) -> RaceInfo:
    """開催がない日のデモデータ"""
    rng = random.Random(42 + race_no)

    racers = []
    names = ["山田太郎", "田中一郎", "佐藤健二", "鈴木大介", "高橋裕也", "渡辺光"]
    ranks = ["A1", "A1", "A2", "B1", "A2", "B1"]

    for i in range(6):
        r = Racer(
            waku=i + 1,
            name=names[i],
            register_no=str(3000 + rng.randint(100, 999)),
            rank=ranks[i],
            branch="東京",
            win_rate_all=round(rng.uniform(4.5, 7.5), 2),
            win_rate_2r_all=round(rng.uniform(20, 50), 2),
            win_rate_local=round(rng.uniform(4.0, 8.0), 2),
            win_rate_2r_local=round(rng.uniform(18, 55), 2),
            motor_2r=round(rng.uniform(25, 50), 1),
            boat_2r=round(rng.uniform(25, 45), 1),
            exhibit_time=round(rng.uniform(6.5, 7.0), 2),
        )
        racers.append(r)

    weather = WeatherInfo(
        weather="晴",
        wind_direction="北東",
        wind_speed=3,
        wave_height=5,
        temperature=12.0,
        water_temp=10.0,
    )

    return RaceInfo(
        race_no=race_no,
        race_name=f"一般 第{race_no}R（デモ）",
        date=date_str,
        deadline="14:30",
        racers=racers,
        weather=weather,
    )


def _create_demo_odds() -> dict[str, float]:
    """デモ用の3連単オッズ"""
    rng = random.Random(123)
    odds = {}
    for perm in itertools_permutations(range(1, 7), 3):
        combo = f"{perm[0]}-{perm[1]}-{perm[2]}"
        base = 10.0
        if perm[0] <= 2:
            base *= 0.5
        if perm[0] >= 5:
            base *= 3.0
        odds[combo] = round(base * rng.uniform(0.5, 10.0), 1)
    return odds


# ── ページ設定 ──
st.set_page_config(
    page_title="平和島ボートレース予想",
    page_icon="🚤",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── カスタムCSS（スマホ最適化） ──
st.markdown("""
<style>
    @media (max-width: 768px) {
        .block-container { padding: 1rem 0.5rem; }
        h1 { font-size: 1.4rem !important; }
        h2 { font-size: 1.1rem !important; }
        h3 { font-size: 1.0rem !important; }
        .stDataFrame { font-size: 0.75rem; }
    }
    .app-header {
        background: linear-gradient(135deg, #1E88E5, #1565C0);
        padding: 12px 16px;
        border-radius: 8px;
        color: white;
        margin-bottom: 16px;
    }
    .risk-low { color: #2E7D32; font-weight: bold; }
    .risk-mid { color: #F57F17; font-weight: bold; }
    .risk-high { color: #C62828; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


def get_waku_color(waku: int) -> str:
    colors = {1: "⬜", 2: "⬛", 3: "🟥", 4: "🟦", 5: "🟨", 6: "🟩"}
    return colors.get(waku, "")


def format_date(d: date) -> str:
    return d.strftime("%Y%m%d")


# ── ヘッダー ──
today = date.today()
st.markdown(
    f'<div class="app-header">'
    f'<h1 style="margin:0;color:white;">🚤 平和島ボートレース予想</h1>'
    f'<p style="margin:4px 0 0 0;opacity:0.9;">{today.strftime("%Y年%m月%d日")}</p>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── サイドバー（設定） ──
with st.sidebar:
    st.header("設定")
    target_date = st.date_input("対象日", value=today)
    default_budget = st.number_input(
        "デフォルト掛け金（円）", min_value=100, max_value=100000,
        value=3000, step=100,
    )
    strategy = st.selectbox(
        "戦略モード",
        ["安定重視", "バランス", "的中率重視", "回収率重視", "万舟狙い"],
        help=(
            "安定重視: 合成的中率50%を目標に幅広く買う\n"
            "バランス: 期待値と的中率の両方を考慮\n"
            "的中率重視: 確率上位を広く買う\n"
            "回収率重視: 期待値1.0超のみ\n"
            "万舟狙い: 高配当狙い"
        ),
    )
    strategy_map = {
        "安定重視": "conservative",
        "バランス": "balance",
        "的中率重視": "hit_rate",
        "回収率重視": "roi",
        "万舟狙い": "longshot",
    }

    st.divider()
    use_heiwajima = st.checkbox(
        "平和島公式サイトから追加データ取得",
        value=True,
        help="heiwajima.gr.jp から展示ST・進入コース等を取得",
    )
    debug_mode = st.checkbox("デバッグモード", value=False)
    st.caption("予測は参考情報です。舟券購入は自己責任でお願いします。")


# ── メインコンテンツ ──
date_str = format_date(target_date)

# レース選択
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("レース一覧")
    race_no = st.radio(
        "レースを選択",
        options=list(range(1, 13)),
        format_func=lambda x: f"第{x}R",
        index=0,
        horizontal=False,
    )

with col2:
    btn_col1, btn_col2 = st.columns([3, 1])
    with btn_col1:
        fetch_btn = st.button("🔄 データ取得", use_container_width=True, type="primary")
    with btn_col2:
        clear_btn = st.button("🗑️ キャッシュクリア", use_container_width=True)

    if clear_btn:
        clear_all_cache()
        for key in ["race_info", "odds", "race_no", "supplement"]:
            st.session_state.pop(key, None)
        st.info("キャッシュをクリアしました")

    if fetch_btn:
        # まずキャッシュを確認
        cached_info, cached_odds, cached_supp, cache_hit, remaining = get_cached_data(date_str, race_no)

        if cache_hit:
            st.session_state["race_info"] = cached_info
            st.session_state["odds"] = cached_odds
            st.session_state["supplement"] = cached_supp
            st.session_state["race_no"] = race_no
            st.success(f"第{race_no}R のデータをキャッシュから取得しました（残り {remaining} 秒）")
        else:
            with st.spinner("boatrace.jp からデータ取得中..."):
                race_info = fetch_race_list(date_str, race_no)
                if race_info and race_info.racers:
                    race_info = fetch_before_info(date_str, race_no, race_info)
                    odds = fetch_odds_3t(date_str, race_no)

                    # 平和島公式サイトから追加データ
                    supplement = None
                    if use_heiwajima and fetch_heiwajima_supplement:
                        try:
                            supplement = fetch_heiwajima_supplement(date_str, race_no)
                            if supplement and supplement.success:
                                st.info("平和島公式サイトから追加データを取得しました")
                        except Exception:
                            supplement = None

                    set_cache(date_str, race_no, race_info, odds, supplement)
                    st.session_state["race_info"] = race_info
                    st.session_state["odds"] = odds
                    st.session_state["supplement"] = supplement
                    st.session_state["race_no"] = race_no

                    odds_msg = f"（オッズ {len(odds)} 件取得）" if odds else "（オッズ取得失敗 - デモオッズ使用）"
                    st.success(f"第{race_no}R のデータを取得しました {odds_msg}")
                    if debug_mode and debug_racelist_html:
                        st.session_state["debug_html"] = debug_racelist_html(date_str, race_no)
                else:
                    st.warning(
                        "本日は平和島での開催がないか、データ取得に失敗しました。"
                        "デモデータを表示します。"
                    )
                    st.session_state["race_info"] = _create_demo_data(race_no, date_str)
                    st.session_state["odds"] = _create_demo_odds()
                    st.session_state["supplement"] = None
                    st.session_state["race_no"] = race_no

    # レース情報表示
    if "race_info" in st.session_state:
        race_info_display: RaceInfo = st.session_state["race_info"]
        odds_dict_display: dict = st.session_state.get("odds", {})

        st.subheader(f"第{race_info_display.race_no}R {race_info_display.race_name}")
        if race_info_display.deadline:
            st.caption(f"締切: {race_info_display.deadline}")

        if race_info_display.racers:
            rows = []
            supplement_data = st.session_state.get("supplement")
            for r in race_info_display.racers:
                row_data = {
                    "枠": f"{get_waku_color(r.waku)} {r.waku}",
                    "選手名": r.name or "---",
                    "級": r.rank or "-",
                    "全国勝率": f"{r.win_rate_all:.2f}" if r.win_rate_all else "-",
                    "当地勝率": f"{r.win_rate_local:.2f}" if r.win_rate_local else "-",
                    "モータ2連": f"{r.motor_2r:.1f}%" if r.motor_2r else "-",
                    "ボート2連": f"{r.boat_2r:.1f}%" if r.boat_2r else "-",
                    "展示T": f"{r.exhibit_time:.2f}" if r.exhibit_time else "-",
                }
                # 平和島追加データ
                if supplement_data and supplement_data.success:
                    if supplement_data.exhibit_st and r.waku in supplement_data.exhibit_st:
                        row_data["展示ST"] = f"{supplement_data.exhibit_st[r.waku]:.2f}"
                    if supplement_data.course_entry and r.waku in supplement_data.course_entry:
                        course = supplement_data.course_entry[r.waku]
                        row_data["進入"] = f"{course}コース" if course != r.waku else "枠なり"

                rows.append(row_data)

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # 水面情報
            w = race_info_display.weather
            if w.weather or w.wind_speed or w.wave_height:
                weather_cols = st.columns(4)
                weather_cols[0].metric("天候", w.weather or "-")
                weather_cols[1].metric("風速", f"{w.wind_speed}m" if w.wind_speed else "-")
                weather_cols[2].metric("波高", f"{w.wave_height}cm" if w.wave_height else "-")
                weather_cols[3].metric("気温", f"{w.temperature}℃" if w.temperature else "-")

    # デバッグ表示
    if debug_mode and "debug_html" in st.session_state:
        with st.expander("HTML構造デバッグ", expanded=True):
            for item in st.session_state["debug_html"]:
                is_racer = item.get("is_racer", False)
                marker = "RACER" if is_racer else "SKIP"
                st.markdown(f"**tbody[{item.get('tbody_index')}]** [{marker}] - td数: {item.get('td_count')}, 勝率数: {item.get('rates_count')}, 4桁番号: {item.get('has_4digit')}")
                st.text(f"  テキスト: {item.get('td_texts')}")
                st.text(f"  クラス:   {item.get('td_classes')}")

st.divider()

# ── 予測＆買い目提案セクション ──
st.subheader("💰 3連単 買い目提案")

budget_col, btn_col = st.columns([2, 1])
with budget_col:
    budget = st.number_input(
        "掛け金（円）", min_value=100, max_value=100000,
        value=default_budget, step=100, key="race_budget",
    )
with btn_col:
    st.write("")
    predict_btn = st.button("🎯 予測する", use_container_width=True, type="primary")

if predict_btn and "race_info" in st.session_state:
    race_info_pred: RaceInfo = st.session_state["race_info"]
    odds_dict_pred: dict = st.session_state.get("odds", {})
    supplement_pred = st.session_state.get("supplement")

    # heiwajima 補完データの展開
    course_entry = None
    exhibit_st = None
    if supplement_pred and supplement_pred.success:
        course_entry = supplement_pred.course_entry
        exhibit_st = supplement_pred.exhibit_st

    with st.spinner("予測計算中..."):
        # 1着確率を予測
        win_probs = predict_win_probabilities(race_info_pred, course_entry, exhibit_st)

        st.markdown("**各艇の1着予測確率:**")
        prob_cols = st.columns(6)
        sorted_probs = sorted(win_probs.items(), key=lambda x: x[0])
        for i, (waku, prob) in enumerate(sorted_probs):
            with prob_cols[i]:
                st.metric(
                    f"{get_waku_color(waku)} {waku}号艇",
                    f"{prob:.1%}",
                )

        st.divider()

        # 3連単確率を予測（安定重視なら上位多めに計算）
        top_n = 120 if strategy_map[strategy] == "conservative" else 60
        trifecta_probs = predict_trifecta_probabilities(
            race_info_pred, top_n=top_n,
            course_entry=course_entry, exhibit_st=exhibit_st,
        )

        if not odds_dict_pred:
            st.warning("オッズデータを取得できませんでした。デモオッズで計算します。")
            odds_dict_pred = _create_demo_odds()

        # 買い目最適化
        plan = optimize_bets(
            trifecta_probs=trifecta_probs,
            odds_dict=odds_dict_pred,
            budget=budget,
            strategy=strategy_map[strategy],
            bet_type="3連単",
            kelly_frac=0.5,
        )
        plan.race_no = race_info_pred.race_no

        # 結果表示
        if plan.suggestions:
            st.markdown(f"**戦略: {strategy} / 3連単**")

            # リスク評価の表示
            if "低リスク" in plan.risk_label:
                risk_class = "risk-low"
            elif "中リスク" in plan.risk_label:
                risk_class = "risk-mid"
            else:
                risk_class = "risk-high"
            st.markdown(f'リスク評価: <span class="{risk_class}">{plan.risk_label}</span>', unsafe_allow_html=True)

            rows = []
            for s in plan.suggestions:
                ev_icon = "🟢" if s.expected_value >= 1.0 else "🔴"
                rows.append({
                    "買い目": s.combo,
                    "オッズ": f"{s.odds:.1f}倍",
                    "予測確率": f"{s.predicted_prob:.1%}",
                    "期待値": f"{ev_icon} {s.expected_value:.2f}",
                    "配分金額": f"¥{s.bet_amount:,}",
                })

            df_result = pd.DataFrame(rows)
            st.dataframe(df_result, use_container_width=True, hide_index=True)

            # サマリー
            summary_cols = st.columns(4)
            summary_cols[0].metric("合計投資", f"¥{plan.total_bet:,}")
            summary_cols[1].metric("合成的中率", f"{plan.combined_hit_rate:.1%}")
            summary_cols[2].metric("加重平均期待値", f"{plan.avg_expected_value:.2f}")
            summary_cols[3].metric("買い目数", f"{len(plan.suggestions)}点")

            # 戦略別アドバイス
            if plan.combined_hit_rate >= 0.5:
                st.success("合成的中率50%以上 - 安定的な買い目構成です。長期的にプラス収支が期待できます。")
            elif plan.combined_hit_rate >= 0.3:
                st.info("合成的中率30-50% - バランスの取れた買い目です。")
            else:
                st.warning(
                    "合成的中率30%未満 - リスクが高い構成です。"
                    "「安定重視」戦略への変更を検討してください。"
                )

            # 取得データの詳細（デバッグ用）
            if debug_mode:
                with st.expander("予測詳細"):
                    st.write(f"オッズ取得数: {len(odds_dict_pred)}")
                    st.write(f"3連単確率計算数: {len(trifecta_probs)}")
                    if supplement_pred:
                        st.write(f"平和島追加データ: {'成功' if supplement_pred.success else '失敗'}")
                        if supplement_pred.exhibit_st:
                            st.write(f"展示ST: {supplement_pred.exhibit_st}")
                        if supplement_pred.course_entry:
                            st.write(f"進入コース: {supplement_pred.course_entry}")
        else:
            st.warning("期待値の高い買い目が見つかりませんでした。別の戦略をお試しください。")

elif predict_btn:
    st.warning("先に「データ取得」ボタンでレースデータを取得してください。")
