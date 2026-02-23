"""平和島ボートレース予想アプリ - Streamlit UI

データソース:
  1. BoatraceOpenAPI (GitHub Pages JSON) - プライマリ
  2. heiwajima.gr.jp (平和島公式) - セカンダリ
  3. boatrace.jp (BOAT RACE公式) - フォールバック
"""

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
    unified_fetch_race, unified_fetch_odds,
    fetch_race_list, fetch_before_info, fetch_odds_3t,
)
try:
    from src.scraper.race_data import debug_racelist_html
except ImportError:
    debug_racelist_html = None
from src.model.predictor import (
    predict_win_probabilities,
    predict_trifecta_probabilities,
    predict_exacta_probabilities,
    predict_quinella_probabilities,
    predict_trio_probabilities,
)
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
            return (
                cached["race_info"],
                cached["odds"],
                cached.get("extra_data", {}),
                True,
                remaining,
            )
    return None, {}, {}, False, 0


def set_cache(date_str: str, race_no: int, race_info: RaceInfo,
              odds: dict, extra_data: dict = None):
    """データをキャッシュに保存する"""
    cache_key = _get_cache_key(date_str, race_no)
    st.session_state[cache_key] = {
        "race_info": race_info,
        "odds": odds,
        "extra_data": extra_data or {},
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


def _create_demo_odds(bet_type: str = "3連単") -> dict[str, float]:
    """デモ用オッズ"""
    rng = random.Random(123)
    odds = {}

    if bet_type == "3連単":
        for perm in itertools_permutations(range(1, 7), 3):
            combo = f"{perm[0]}-{perm[1]}-{perm[2]}"
            base = 10.0
            if perm[0] <= 2:
                base *= 0.5
            if perm[0] >= 5:
                base *= 3.0
            odds[combo] = round(base * rng.uniform(0.5, 10.0), 1)
    elif bet_type == "3連複":
        from itertools import combinations
        for c in combinations(range(1, 7), 3):
            combo = f"{c[0]}={c[1]}={c[2]}"
            base = 5.0
            if 1 in c:
                base *= 0.5
            odds[combo] = round(base * rng.uniform(0.5, 8.0), 1)
    elif bet_type == "2連単":
        for perm in itertools_permutations(range(1, 7), 2):
            combo = f"{perm[0]}-{perm[1]}"
            base = 5.0
            if perm[0] <= 2:
                base *= 0.5
            odds[combo] = round(base * rng.uniform(0.5, 5.0), 1)
    elif bet_type == "2連複":
        from itertools import combinations
        for c in combinations(range(1, 7), 2):
            combo = f"{c[0]}={c[1]}"
            base = 3.0
            if 1 in c:
                base *= 0.5
            odds[combo] = round(base * rng.uniform(0.5, 5.0), 1)

    return odds


# ── ページ設定 ──
st.set_page_config(
    page_title="平和島ボートレース予想",
    page_icon="🚤",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── カスタムCSS ──
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
    .source-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .source-heiwajima { background: #4CAF50; color: white; }
    .source-boatrace { background: #2196F3; color: white; }
    .source-demo { background: #FF9800; color: white; }
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

# ── サイドバー ──
with st.sidebar:
    st.header("設定")
    target_date = st.date_input("対象日", value=today)
    default_budget = st.number_input(
        "デフォルト掛け金（円）", min_value=100, max_value=100000,
        value=3000, step=100,
    )

    bet_type_label = st.selectbox(
        "舟券種別",
        ["3連単", "3連複", "2連単", "2連複"],
        help=(
            "3連単: 1-2-3着を順番通り\n"
            "3連複: 1-2-3着を順不同\n"
            "2連単: 1-2着を順番通り\n"
            "2連複: 1-2着を順不同"
        ),
    )
    bet_type_map = {
        "3連単": "3t", "3連複": "3f", "2連単": "2tf", "2連複": "2kt",
    }

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

    # モデル情報
    model_path = ROOT / "model" / "lgbm_heiwajima.pkl"
    if model_path.exists():
        model_size = model_path.stat().st_size / 1024
        st.success(f"LightGBMモデル: 使用中 ({model_size:.0f}KB)")
    else:
        st.info("統計モデル使用中（LightGBM未学習）")

    st.divider()
    debug_mode = st.checkbox("デバッグモード", value=False)
    st.caption("予測は参考情報です。舟券購入は自己責任でお願いします。")


# ── メインコンテンツ ──
date_str = format_date(target_date)

# タブ構成
tab_predict, tab_batch, tab_data = st.tabs(
    ["予測", "全レース一括", "データ管理"]
)

# ══════════════════════════════════════════════
# TAB 1: 個別レース予測
# ══════════════════════════════════════════════
with tab_predict:
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
            fetch_btn = st.button(
                "🔄 データ取得", use_container_width=True, type="primary",
            )
        with btn_col2:
            clear_btn = st.button("🗑️ クリア", use_container_width=True)

        if clear_btn:
            clear_all_cache()
            for key in ["race_info", "odds", "race_no", "extra_data"]:
                st.session_state.pop(key, None)
            st.info("キャッシュをクリアしました")

        if fetch_btn:
            cached_info, cached_odds, cached_extra, cache_hit, remaining = (
                get_cached_data(date_str, race_no)
            )

            if cache_hit:
                st.session_state["race_info"] = cached_info
                st.session_state["odds"] = cached_odds
                st.session_state["extra_data"] = cached_extra
                st.session_state["race_no"] = race_no
                source = cached_extra.get("source", "cache")
                st.success(
                    f"第{race_no}R キャッシュから取得"
                    f"（残り {remaining}秒 / ソース: {source}）"
                )
            else:
                with st.spinner(
                    "データ取得中... (OpenAPI → heiwajima.gr.jp → boatrace.jp)"
                ):
                    race_info, odds, extra_data = unified_fetch_race(
                        date_str, race_no
                    )

                    if race_info and race_info.racers:
                        bet_key = bet_type_map[bet_type_label]
                        if bet_key != "3t" or not odds:
                            additional_odds = unified_fetch_odds(
                                date_str, race_no, bet_key,
                            )
                            if additional_odds:
                                odds = additional_odds

                        set_cache(date_str, race_no, race_info, odds, extra_data)
                        st.session_state["race_info"] = race_info
                        st.session_state["odds"] = odds
                        st.session_state["extra_data"] = extra_data
                        st.session_state["race_no"] = race_no

                        source = extra_data.get("source", "unknown")
                        source_label = {
                            "openapi": "BoatraceOpenAPI",
                            "heiwajima": "平和島公式",
                            "boatrace": "BOAT RACE公式",
                        }.get(source, source)
                        odds_msg = (
                            f"オッズ {len(odds)}件" if odds
                            else "オッズ未取得"
                        )
                        st.success(
                            f"第{race_no}R データ取得完了"
                            f"（{source_label} / {odds_msg}）"
                        )

                        if debug_mode and debug_racelist_html:
                            st.session_state["debug_html"] = (
                                debug_racelist_html(date_str, race_no)
                            )
                    else:
                        st.warning(
                            "本日は平和島での開催がないか、"
                            "データ取得に失敗しました。デモデータを表示します。"
                        )
                        st.session_state["race_info"] = _create_demo_data(
                            race_no, date_str
                        )
                        st.session_state["odds"] = _create_demo_odds(
                            bet_type_label
                        )
                        st.session_state["extra_data"] = {"source": "demo"}
                        st.session_state["race_no"] = race_no

        # レース情報表示
        if "race_info" in st.session_state:
            race_info_display: RaceInfo = st.session_state["race_info"]
            odds_dict_display: dict = st.session_state.get("odds", {})
            extra_data_display: dict = st.session_state.get("extra_data", {})

            source = extra_data_display.get("source", "")
            source_html = ""
            if source == "openapi":
                source_html = (
                    '<span class="source-tag source-heiwajima">OpenAPI</span>'
                )
            elif source == "heiwajima":
                source_html = (
                    '<span class="source-tag source-heiwajima">平和島公式</span>'
                )
            elif source == "boatrace":
                source_html = (
                    '<span class="source-tag source-boatrace">'
                    'BOAT RACE公式</span>'
                )
            elif source == "demo":
                source_html = (
                    '<span class="source-tag source-demo">デモ</span>'
                )

            st.markdown(
                f"### 第{race_info_display.race_no}R "
                f"{race_info_display.race_name} {source_html}",
                unsafe_allow_html=True,
            )
            if race_info_display.deadline:
                st.caption(f"締切: {race_info_display.deadline}")

            if race_info_display.racers:
                rows = []
                course_entry = extra_data_display.get("course_entry", {})
                exhibit_st = extra_data_display.get("exhibit_st", {})

                for r in race_info_display.racers:
                    row_data = {
                        "枠": f"{get_waku_color(r.waku)} {r.waku}",
                        "選手名": r.name or "---",
                        "級": r.rank or "-",
                        "全国勝率": (
                            f"{r.win_rate_all:.2f}" if r.win_rate_all
                            else "-"
                        ),
                        "当地勝率": (
                            f"{r.win_rate_local:.2f}" if r.win_rate_local
                            else "-"
                        ),
                        "モータ2連": (
                            f"{r.motor_2r:.1f}%" if r.motor_2r else "-"
                        ),
                        "ボート2連": (
                            f"{r.boat_2r:.1f}%" if r.boat_2r else "-"
                        ),
                        "展示T": (
                            f"{r.exhibit_time:.2f}" if r.exhibit_time
                            else "-"
                        ),
                    }

                    racer_st = getattr(r, "exhibit_st", 0.0) or 0.0
                    if not racer_st and r.waku in exhibit_st:
                        racer_st = exhibit_st[r.waku]
                    if racer_st:
                        row_data["展示ST"] = f"{racer_st:.2f}"

                    racer_course = getattr(r, "course_entry", 0) or 0
                    if not racer_course and r.waku in course_entry:
                        racer_course = course_entry.get(r.waku, 0)
                    if racer_course and racer_course != r.waku:
                        row_data["進入"] = f"{racer_course}コース"
                    elif racer_course:
                        row_data["進入"] = "枠なり"

                    avg_st = getattr(r, "avg_start_timing", 0.0) or 0.0
                    if avg_st > 0:
                        row_data["平均ST"] = f"{avg_st:.2f}"

                    f_count = getattr(r, "flying_count", 0) or 0
                    l_count = getattr(r, "late_count", 0) or 0
                    if f_count or l_count:
                        row_data["F/L"] = f"F{f_count} L{l_count}"

                    rows.append(row_data)

                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

                w = race_info_display.weather
                if w.weather or w.wind_speed or w.wave_height:
                    weather_cols = st.columns(5)
                    weather_cols[0].metric("天候", w.weather or "-")
                    weather_cols[1].metric(
                        "風速", f"{w.wind_speed}m" if w.wind_speed else "-"
                    )
                    weather_cols[2].metric(
                        "波高",
                        f"{w.wave_height}cm" if w.wave_height else "-",
                    )
                    weather_cols[3].metric(
                        "気温",
                        f"{w.temperature}℃" if w.temperature else "-",
                    )
                    weather_cols[4].metric(
                        "水温",
                        f"{w.water_temp}℃" if w.water_temp else "-",
                    )

        if debug_mode and "debug_html" in st.session_state:
            with st.expander("HTML構造デバッグ", expanded=False):
                for item in st.session_state["debug_html"]:
                    is_racer = item.get("is_racer", False)
                    marker = "RACER" if is_racer else "SKIP"
                    st.markdown(
                        f"**tbody[{item.get('tbody_index')}]** [{marker}] "
                        f"td数: {item.get('td_count')}, "
                        f"勝率数: {item.get('rates_count')}"
                    )

    st.divider()

    # ── 予測＆買い目提案セクション ──
    st.subheader(f"💰 {bet_type_label} 買い目提案")

    budget_col, btn_col = st.columns([2, 1])
    with budget_col:
        budget = st.number_input(
            "掛け金（円）", min_value=100, max_value=100000,
            value=default_budget, step=100, key="race_budget",
        )
    with btn_col:
        st.write("")
        predict_btn = st.button(
            "🎯 予測する", use_container_width=True, type="primary",
        )

    if predict_btn and "race_info" in st.session_state:
        race_info_pred: RaceInfo = st.session_state["race_info"]
        odds_dict_pred: dict = st.session_state.get("odds", {})
        extra_data_pred: dict = st.session_state.get("extra_data", {})

        course_entry = extra_data_pred.get("course_entry") or None
        exhibit_st = extra_data_pred.get("exhibit_st") or None

        with st.spinner("予測計算中..."):
            win_probs = predict_win_probabilities(
                race_info_pred, course_entry, exhibit_st
            )

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

            top_n = 120 if strategy_map[strategy] == "conservative" else 60

            if bet_type_label == "3連単":
                combo_probs = predict_trifecta_probabilities(
                    race_info_pred, top_n=top_n,
                    course_entry=course_entry, exhibit_st=exhibit_st,
                )
            elif bet_type_label == "3連複":
                combo_probs = predict_trio_probabilities(
                    race_info_pred, top_n=20,
                    course_entry=course_entry, exhibit_st=exhibit_st,
                )
            elif bet_type_label == "2連単":
                combo_probs = predict_exacta_probabilities(
                    race_info_pred, top_n=min(top_n, 30),
                    course_entry=course_entry, exhibit_st=exhibit_st,
                )
            elif bet_type_label == "2連複":
                combo_probs = predict_quinella_probabilities(
                    race_info_pred, top_n=15,
                    course_entry=course_entry, exhibit_st=exhibit_st,
                )
            else:
                combo_probs = []

            if not odds_dict_pred:
                bet_key = bet_type_map[bet_type_label]
                odds_dict_pred = unified_fetch_odds(
                    date_str, race_no, bet_key,
                )

            if not odds_dict_pred:
                st.warning(
                    "オッズデータを取得できませんでした。デモオッズで計算します。"
                )
                odds_dict_pred = _create_demo_odds(bet_type_label)

            plan = optimize_bets(
                trifecta_probs=combo_probs,
                odds_dict=odds_dict_pred,
                budget=budget,
                strategy=strategy_map[strategy],
                bet_type=bet_type_label,
                kelly_frac=0.5,
            )
            plan.race_no = race_info_pred.race_no

            if plan.suggestions:
                st.markdown(f"**戦略: {strategy} / {bet_type_label}**")

                if "低リスク" in plan.risk_label:
                    risk_class = "risk-low"
                elif "中リスク" in plan.risk_label:
                    risk_class = "risk-mid"
                else:
                    risk_class = "risk-high"
                st.markdown(
                    f'リスク評価: '
                    f'<span class="{risk_class}">{plan.risk_label}</span>',
                    unsafe_allow_html=True,
                )

                result_rows = []
                for s in plan.suggestions:
                    ev_icon = "🟢" if s.expected_value >= 1.0 else "🔴"
                    result_rows.append({
                        "買い目": s.combo,
                        "オッズ": f"{s.odds:.1f}倍",
                        "予測確率": f"{s.predicted_prob:.1%}",
                        "期待値": f"{ev_icon} {s.expected_value:.2f}",
                        "配分金額": f"¥{s.bet_amount:,}",
                    })

                df_result = pd.DataFrame(result_rows)
                st.dataframe(
                    df_result, use_container_width=True, hide_index=True,
                )

                summary_cols = st.columns(4)
                summary_cols[0].metric("合計投資", f"¥{plan.total_bet:,}")
                summary_cols[1].metric(
                    "合成的中率", f"{plan.combined_hit_rate:.1%}",
                )
                summary_cols[2].metric(
                    "加重平均期待値", f"{plan.avg_expected_value:.2f}",
                )
                summary_cols[3].metric(
                    "買い目数", f"{len(plan.suggestions)}点",
                )

                if plan.combined_hit_rate >= 0.5:
                    st.success(
                        "合成的中率50%以上 - 安定的な買い目構成です。"
                    )
                elif plan.combined_hit_rate >= 0.3:
                    st.info(
                        "合成的中率30-50% - バランスの取れた買い目です。"
                    )
                else:
                    st.warning(
                        "合成的中率30%未満 - リスクが高い構成です。"
                        "「安定重視」戦略への変更を検討してください。"
                    )

                if debug_mode:
                    with st.expander("予測詳細"):
                        st.write(
                            f"データソース: "
                            f"{extra_data_pred.get('source', 'N/A')}"
                        )
                        st.write(f"オッズ取得数: {len(odds_dict_pred)}")
                        st.write(f"確率計算数: {len(combo_probs)}")
                        if course_entry:
                            st.write(f"進入コース: {course_entry}")
                        if exhibit_st:
                            st.write(f"展示ST: {exhibit_st}")
            else:
                st.warning(
                    "期待値の高い買い目が見つかりませんでした。"
                    "別の戦略や舟券種別をお試しください。"
                )

    elif predict_btn:
        st.warning("先に「データ取得」ボタンでレースデータを取得してください。")


# ══════════════════════════════════════════════
# TAB 2: 全レース一括予測
# ══════════════════════════════════════════════
with tab_batch:
    st.subheader("全レース一括予測")
    st.caption("1R〜12Rの全レースを一括でデータ取得・予測します")

    batch_col1, batch_col2 = st.columns([2, 1])
    with batch_col1:
        batch_budget = st.number_input(
            "1レースあたりの掛け金（円）",
            min_value=100, max_value=100000,
            value=default_budget, step=100, key="batch_budget",
        )
    with batch_col2:
        st.write("")
        batch_btn = st.button(
            "🚀 全レース予測", use_container_width=True, type="primary",
        )

    if batch_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()
        all_results = []

        for rno in range(1, 13):
            status_text.text(f"第{rno}R データ取得・予測中...")
            progress_bar.progress(rno / 12)

            cached_info, cached_odds, cached_extra, cache_hit, _ = (
                get_cached_data(date_str, rno)
            )

            if cache_hit:
                ri = cached_info
                od = cached_odds
                ed = cached_extra
            else:
                ri, od, ed = unified_fetch_race(date_str, rno)
                if ri and ri.racers:
                    bet_key = bet_type_map[bet_type_label]
                    if bet_key != "3t" or not od:
                        additional = unified_fetch_odds(
                            date_str, rno, bet_key,
                        )
                        if additional:
                            od = additional
                    set_cache(date_str, rno, ri, od, ed)

            if not ri or not ri.racers:
                continue

            course_entry = (ed or {}).get("course_entry") or None
            exhibit_st_dict = (ed or {}).get("exhibit_st") or None

            win_probs = predict_win_probabilities(
                ri, course_entry, exhibit_st_dict,
            )
            top_waku = max(win_probs, key=win_probs.get)
            top_prob = win_probs[top_waku]

            if bet_type_label == "3連単":
                combo_probs = predict_trifecta_probabilities(
                    ri, top_n=60,
                    course_entry=course_entry, exhibit_st=exhibit_st_dict,
                )
            elif bet_type_label == "3連複":
                combo_probs = predict_trio_probabilities(
                    ri, top_n=20,
                    course_entry=course_entry, exhibit_st=exhibit_st_dict,
                )
            elif bet_type_label == "2連単":
                combo_probs = predict_exacta_probabilities(
                    ri, top_n=30,
                    course_entry=course_entry, exhibit_st=exhibit_st_dict,
                )
            else:
                combo_probs = predict_quinella_probabilities(
                    ri, top_n=15,
                    course_entry=course_entry, exhibit_st=exhibit_st_dict,
                )

            if not od:
                od = _create_demo_odds(bet_type_label)

            plan = optimize_bets(
                trifecta_probs=combo_probs,
                odds_dict=od,
                budget=batch_budget,
                strategy=strategy_map[strategy],
                bet_type=bet_type_label,
            )

            top_bet = (
                plan.suggestions[0].combo if plan.suggestions else "-"
            )
            top_ev = (
                plan.suggestions[0].expected_value if plan.suggestions
                else 0.0
            )

            all_results.append({
                "レース": f"第{rno}R",
                "レース名": ri.race_name or "-",
                "本命": f"{get_waku_color(top_waku)} {top_waku}号艇",
                "1着確率": f"{top_prob:.1%}",
                "推奨買い目": top_bet,
                "最高期待値": f"{top_ev:.2f}",
                "合成的中率": f"{plan.combined_hit_rate:.1%}",
                "投資額": f"¥{plan.total_bet:,}",
                "買い目数": len(plan.suggestions),
                "リスク": plan.risk_label[:4] if plan.risk_label else "-",
            })

        progress_bar.progress(1.0)
        status_text.text("全レース予測完了!")

        if all_results:
            st.dataframe(
                pd.DataFrame(all_results),
                use_container_width=True, hide_index=True,
            )

            total_investment = sum(
                int(r["投資額"].replace("¥", "").replace(",", ""))
                for r in all_results
            )
            st.metric(
                "全レース合計投資額", f"¥{total_investment:,}",
            )
        else:
            st.warning("本日のレースデータが取得できませんでした。")


# ══════════════════════════════════════════════
# TAB 3: データ管理
# ══════════════════════════════════════════════
with tab_data:
    st.subheader("データ管理")

    data_col1, data_col2 = st.columns(2)

    with data_col1:
        st.markdown("#### 過去データ収集")
        st.caption(
            "OpenAPI から過去データを収集してCSVに保存します。"
            "モデル学習に使用します。"
        )

        data_dir = ROOT / "data" / "raw"
        csv_files = sorted(data_dir.glob("heiwajima_*.csv")) if data_dir.exists() else []

        if csv_files:
            st.success(f"CSVファイル数: {len(csv_files)}")
            for csv_file in csv_files[-5:]:
                size_kb = csv_file.stat().st_size / 1024
                st.text(f"  {csv_file.name} ({size_kb:.0f}KB)")
        else:
            st.info("CSVデータなし")

        st.markdown("**収集コマンド:**")
        st.code(
            "python -m src.scraper.history_collector --months 6",
            language="bash",
        )

    with data_col2:
        st.markdown("#### モデル学習")
        st.caption(
            "収集したCSVデータからLightGBMモデルを学習します。"
        )

        model_dir = ROOT / "model"
        model_file = model_dir / "lgbm_heiwajima.pkl"

        if model_file.exists():
            import datetime as dt
            mtime = dt.datetime.fromtimestamp(model_file.stat().st_mtime)
            size_kb = model_file.stat().st_size / 1024
            st.success(
                f"モデルファイル: {size_kb:.0f}KB\n"
                f"更新日時: {mtime.strftime('%Y-%m-%d %H:%M')}"
            )
        else:
            st.info("学習済みモデルなし（統計ベースモデルを使用中）")

        st.markdown("**学習コマンド:**")
        st.code(
            "python -m src.model.trainer",
            language="bash",
        )

    st.divider()

    st.markdown("#### バックテスト")
    st.caption("過去データでモデルの収支シミュレーションを実行します。")
    st.code(
        "python -m src.model.backtester --budget 3000 --strategy balance",
        language="bash",
    )
