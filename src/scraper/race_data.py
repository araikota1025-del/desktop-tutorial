"""boatrace.jp / heiwajima.gr.jp 統合スクレイパー

データ取得優先順位:
1. heiwajima.gr.jp (平和島公式サイト) - プライマリ
2. boatrace.jp (BOAT RACE公式) - フォールバック
"""

import re
from dataclasses import dataclass, field
from bs4 import BeautifulSoup

from .client import fetch_page, CONFIG

VENUE_CODE = CONFIG["app"]["venue_code"]


@dataclass
class Racer:
    """出走選手の情報"""
    waku: int = 0            # 枠番 (1-6)
    name: str = ""           # 選手名
    register_no: str = ""    # 登番
    rank: str = ""           # 級別 (A1/A2/B1/B2)
    branch: str = ""         # 支部
    age: int = 0
    weight: float = 0.0
    win_rate_all: float = 0.0   # 全国勝率
    win_rate_2r_all: float = 0.0  # 全国2連率
    win_rate_local: float = 0.0  # 当地勝率
    win_rate_2r_local: float = 0.0  # 当地2連率
    motor_no: int = 0        # モーター番号
    motor_2r: float = 0.0    # モーター2連率
    motor_3r: float = 0.0    # モーター3連率
    boat_no: int = 0         # ボート番号
    boat_2r: float = 0.0     # ボート2連率
    boat_3r: float = 0.0     # ボート3連率
    # 直前情報
    exhibit_time: float = 0.0   # 展示タイム
    exhibit_st: float = 0.0    # 展示スタートタイミング
    tilt: float = 0.0         # チルト角度
    start_timing: float = 0.0  # スタートタイミング
    # 追加統計
    avg_start_timing: float = 0.0  # 平均ST
    flying_count: int = 0          # フライング回数
    late_count: int = 0            # 出遅れ回数
    course_entry: int = 0          # 実際の進入コース (0=未取得)


@dataclass
class WeatherInfo:
    """水面気象情報"""
    weather: str = ""       # 天候
    wind_direction: str = ""  # 風向
    wind_speed: int = 0     # 風速 (m/s)
    wave_height: int = 0    # 波高 (cm)
    temperature: float = 0.0  # 気温
    water_temp: float = 0.0  # 水温


@dataclass
class RaceInfo:
    """1レースの情報"""
    race_no: int = 0
    race_name: str = ""
    date: str = ""           # YYYYMMDD
    deadline: str = ""       # 締切時刻
    racers: list[Racer] = field(default_factory=list)
    weather: WeatherInfo = field(default_factory=WeatherInfo)


def _safe_float(text: str) -> float:
    try:
        return float(re.sub(r"[^\d.\-]", "", text.strip()))
    except (ValueError, AttributeError):
        return 0.0


def _safe_int(text: str) -> int:
    try:
        return int(re.sub(r"[^\d]", "", text.strip()))
    except (ValueError, AttributeError):
        return 0


def fetch_race_list(date: str, race_no: int) -> RaceInfo | None:
    """出走表を取得する

    Args:
        date: YYYYMMDD形式の日付
        race_no: レース番号 (1-12)
    """
    html = fetch_page(
        "/owpc/pc/race/racelist",
        params={"hd": date, "jcd": VENUE_CODE, "rno": str(race_no)},
    )
    if not html:
        return None

    soup = BeautifulSoup(html, "lxml")
    info = RaceInfo(race_no=race_no, date=date)

    # レース名
    title_el = soup.select_one(".heading2_titleName, .title12__title")
    if title_el:
        info.race_name = title_el.get_text(strip=True)

    # 締切時刻
    deadline_el = soup.select_one(".heading2_titleDetail, .title12__time")
    if deadline_el:
        info.deadline = deadline_el.get_text(strip=True)

    # 各艇の選手情報をパース
    tbody_list = soup.select(".table1 tbody")

    for tbody in tbody_list:
        tds = tbody.select("td")
        if not tds:
            continue

        all_text = [td.get_text(strip=True) for td in tds]
        text_joined = " ".join(all_text)

        # ── レーサーデータかどうかを内容で判定 ──
        # 勝率（X.XX形式）が2つ以上 & 4桁の登番がなければスキップ
        rates = re.findall(r"(\d+\.\d{2})", text_joined)
        has_reg_no = bool(re.search(r"\d{4}", text_joined))
        has_rank = bool(re.search(r"(A1|A2|B1|B2)", text_joined))
        if len(rates) < 2 or not (has_reg_no or has_rank):
            continue

        # ── 枠番を検出 ──
        waku = 0
        # 方法1: is-boatColor クラスから検出
        for td in tds:
            for cls in td.get("class", []):
                color_match = re.search(r"is-boatColor(\d)", cls)
                if color_match:
                    waku = int(color_match.group(1))
                    break
            if waku:
                break
        # 方法2: boatColor がなければ最初のtdから単一数字を検出
        if not waku:
            first_text = tds[0].get_text(strip=True)
            if re.match(r"^[1-6]$", first_text):
                waku = int(first_text)

        if not (1 <= waku <= 6):
            continue
        if any(r.waku == waku for r in info.racers):
            continue

        racer = Racer(waku=waku)

        # ── 選手名を検出 ──
        # 方法1: 専用クラス (.is-fs18, .is-fs14) から取得
        name_el = tbody.select_one(".is-fs18, .is-fs14")
        if name_el:
            racer.name = re.sub(r"\s+", "", name_el.get_text(strip=True))
        else:
            # 方法2: 漢字・ひらがな・カタカナ2文字以上のテキストを探す
            for td in tds:
                td_text = td.get_text(strip=True)
                # 数字のみ、英字のみ、級別はスキップ
                if re.match(r"^[\d.%\-\s]+$", td_text):
                    continue
                if re.match(r"^(A1|A2|B1|B2)$", td_text):
                    continue
                # 日本語名前らしいテキスト（漢字/かな2文字以上）
                name_candidate = re.sub(r"\s+", "", td_text)
                if re.search(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]{2,}", name_candidate):
                    # 締切、天候などのキーワードは除外
                    if not re.search(r"(締切|予定|天候|風|波|気温|水温)", name_candidate):
                        racer.name = name_candidate
                        break

        # 登番を探す
        reg_match = re.search(r"(\d{4})", text_joined)
        if reg_match:
            racer.register_no = reg_match.group(1)

        # 級別を探す
        rank_match = re.search(r"(A1|A2|B1|B2)", text_joined)
        if rank_match:
            racer.rank = rank_match.group(1)

        # 支部を探す
        branch_match = re.search(r"(東京|大阪|愛知|福岡|群馬|埼玉|千葉|静岡|長崎|山口|広島|徳島|香川|岡山|三重|滋賀|福井|佐賀|長崎)", text_joined)
        if branch_match:
            racer.branch = branch_match.group(1)

        # 勝率を代入
        if len(rates) >= 4:
            racer.win_rate_all = _safe_float(rates[0])
            racer.win_rate_2r_all = _safe_float(rates[1])
            racer.win_rate_local = _safe_float(rates[2])
            racer.win_rate_2r_local = _safe_float(rates[3])
        elif len(rates) >= 2:
            racer.win_rate_all = _safe_float(rates[0])
            racer.win_rate_2r_all = _safe_float(rates[1])

        # モーター2連率・ボート2連率
        if len(rates) >= 6:
            racer.motor_2r = _safe_float(rates[4])
            racer.boat_2r = _safe_float(rates[5])

        info.racers.append(racer)

    # 枠番順にソート
    info.racers.sort(key=lambda r: r.waku)

    return info


def fetch_before_info(date: str, race_no: int, race_info: RaceInfo | None = None) -> RaceInfo | None:
    """直前情報（展示タイム・気象）を取得する"""
    html = fetch_page(
        "/owpc/pc/race/beforeinfo",
        params={"hd": date, "jcd": VENUE_CODE, "rno": str(race_no)},
    )
    if not html:
        return race_info

    soup = BeautifulSoup(html, "lxml")

    if race_info is None:
        race_info = RaceInfo(race_no=race_no, date=date)

    # 水面気象情報
    weather = WeatherInfo()

    weather_section = soup.select_one(".weather1, .weatherBody")
    if weather_section:
        items = weather_section.select(".weather1_body, .weatherBody__item, span")
        text = weather_section.get_text()

        # 天候
        w_match = re.search(r"(晴|曇り?|雨|雪|霧)", text)
        if w_match:
            weather.weather = w_match.group(1)

        # 風速
        ws_match = re.search(r"(\d+)\s*m", text)
        if ws_match:
            weather.wind_speed = _safe_int(ws_match.group(1))

        # 波高
        wh_match = re.search(r"(\d+)\s*cm", text)
        if wh_match:
            weather.wave_height = _safe_int(wh_match.group(1))

        # 気温
        temp_match = re.search(r"気温\s*(\d+\.?\d*)", text)
        if temp_match:
            weather.temperature = _safe_float(temp_match.group(1))

        # 水温
        wtemp_match = re.search(r"水温\s*(\d+\.?\d*)", text)
        if wtemp_match:
            weather.water_temp = _safe_float(wtemp_match.group(1))

    race_info.weather = weather

    # 展示タイム
    exhibit_section = soup.select(".table1 tbody tr, table tbody tr")
    time_pattern = re.compile(r"\d+\.\d{2}")
    for row in exhibit_section:
        tds = row.select("td")
        if not tds:
            continue
        text = row.get_text()
        times = time_pattern.findall(text)
        # 枠番を検出
        waku_match = re.search(r"^(\d)", tds[0].get_text(strip=True))
        if waku_match and times:
            waku = int(waku_match.group(1))
            if 1 <= waku <= 6 and waku <= len(race_info.racers):
                race_info.racers[waku - 1].exhibit_time = _safe_float(times[-1])

    return race_info


def _parse_odds_table_positional(
    soup: BeautifulSoup,
    table_selector: str,
    bet_type: str,
) -> dict[str, float]:
    """boatrace.jp のオッズテーブルを位置ベースで解析する。

    boatrace.jp のオッズテーブルは以下の構造をしている:

    【3連単 (odds3t)】
    - contentsFrame1_inner 内の table 要素
    - 6つの tbody ブロック（1着が1号艇〜6号艇に対応）
    - 各 tbody には 5 行の tr（1着以外の残り5艇が2着候補）
    - 各 tr には 4つの td.oddsPoint（2着以外の残り4艇が3着候補）
    - 合計: 6 x 5 x 4 = 120 通り

    【2連単 (odds2tf)】
    - 6つの tbody ブロック（1着が1号艇〜6号艇に対応）
    - 各 tbody には 5 行の tr（1着以外の残り5艇が2着候補）
    - 各 tr には 1つの td.oddsPoint
    - 合計: 6 x 5 = 30 通り

    【2連複 (odds2kt)】
    - 5つの tbody ブロック（小さい方の艇番 1〜5）
    - 各 tbody には (6 - 艇番) 行の tr
    - 合計: 5 + 4 + 3 + 2 + 1 = 15 通り

    Args:
        soup: パース済みの BeautifulSoup オブジェクト
        table_selector: テーブルを特定するCSSセレクタ
        bet_type: "3t" (3連単), "2tf" (2連単), "2kt" (2連複)

    Returns:
        {"1-2-3": 12.5, ...} の形式の辞書
    """
    odds_dict: dict[str, float] = {}
    boats = [1, 2, 3, 4, 5, 6]

    # テーブル要素を取得
    table = soup.select_one(table_selector)
    if not table:
        # フォールバック: 複数のセレクタを試す
        for selector in [
            "div.contentsFrame1_inner table",
            ".table1 table",
            "table.is-w495",
        ]:
            table = soup.select_one(selector)
            if table:
                break
    if not table:
        return odds_dict

    tbody_list = table.select("tbody")

    if bet_type == "3t":
        # ── 3連単: 6 tbody x 5 tr x 4 oddsPoint ──
        for tbody_idx, tbody in enumerate(tbody_list):
            if tbody_idx >= 6:
                break
            first = boats[tbody_idx]  # 1着の艇番

            rows = tbody.select("tr")
            # 2着候補: 1着以外の5艇
            second_candidates = [b for b in boats if b != first]

            for row_idx, row in enumerate(rows):
                if row_idx >= len(second_candidates):
                    break
                second = second_candidates[row_idx]  # 2着の艇番

                odds_cells = row.select("td.oddsPoint")
                # 3着候補: 1着・2着以外の4艇
                third_candidates = [b for b in boats if b != first and b != second]

                for cell_idx, cell in enumerate(odds_cells):
                    if cell_idx >= len(third_candidates):
                        break
                    third = third_candidates[cell_idx]  # 3着の艇番

                    odds_val = _safe_float(
                        cell.get_text(strip=True).replace(",", "")
                    )
                    if odds_val > 0:
                        combo = f"{first}-{second}-{third}"
                        odds_dict[combo] = odds_val

    elif bet_type == "2tf":
        # ── 2連単: 6 tbody x 5 tr x 1 oddsPoint ──
        for tbody_idx, tbody in enumerate(tbody_list):
            if tbody_idx >= 6:
                break
            first = boats[tbody_idx]  # 1着の艇番

            rows = tbody.select("tr")
            second_candidates = [b for b in boats if b != first]

            for row_idx, row in enumerate(rows):
                if row_idx >= len(second_candidates):
                    break
                second = second_candidates[row_idx]  # 2着の艇番

                odds_cells = row.select("td.oddsPoint")
                if odds_cells:
                    odds_val = _safe_float(
                        odds_cells[0].get_text(strip=True).replace(",", "")
                    )
                    if odds_val > 0:
                        combo = f"{first}-{second}"
                        odds_dict[combo] = odds_val

    elif bet_type == "2kt":
        # ── 2連複: 5 tbody（軸艇番 1〜5）──
        for tbody_idx, tbody in enumerate(tbody_list):
            if tbody_idx >= 5:
                break
            boat_a = boats[tbody_idx]  # 小さい方の艇番

            rows = tbody.select("tr")
            # 相手候補: boat_a より大きい艇番
            partner_candidates = [b for b in boats if b > boat_a]

            for row_idx, row in enumerate(rows):
                if row_idx >= len(partner_candidates):
                    break
                boat_b = partner_candidates[row_idx]

                odds_cells = row.select("td.oddsPoint")
                if odds_cells:
                    odds_val = _safe_float(
                        odds_cells[0].get_text(strip=True).replace(",", "")
                    )
                    if odds_val > 0:
                        combo = f"{boat_a}={boat_b}"
                        odds_dict[combo] = odds_val

    return odds_dict


def _parse_odds_fallback_regex(
    soup: BeautifulSoup,
    combo_pattern: str,
) -> dict[str, float]:
    """フォールバック: テキスト内の組番+オッズをregexで抽出する。

    位置ベースの解析が失敗した場合のフォールバック。
    td.oddsPoint に data-id 属性がある場合や、
    組番とオッズが同じセルに入っている場合に対応。

    Args:
        soup: パース済みの BeautifulSoup オブジェクト
        combo_pattern: 組番の正規表現 (例: r"\\d-\\d-\\d")

    Returns:
        {"1-2-3": 12.5, ...} の形式の辞書
    """
    odds_dict: dict[str, float] = {}

    # パターン1: td.oddsPoint に data-id 属性がある場合
    for td in soup.select("td.oddsPoint"):
        data_id = td.get("data-id", "")
        if re.match(combo_pattern, data_id):
            odds_val = _safe_float(td.get_text(strip=True).replace(",", ""))
            if odds_val > 0:
                odds_dict[data_id] = odds_val

    if odds_dict:
        return odds_dict

    # パターン2: テーブル行内で組番テキストとオッズが並んでいる場合
    for row in soup.select("table tr, .table1 tr"):
        tds = row.select("td")
        for i, td in enumerate(tds):
            text = td.get_text(strip=True)
            combo_match = re.search(f"({combo_pattern})", text)
            if combo_match:
                combo = combo_match.group(1)
                # 同じセル内にオッズがある場合
                odds_text = re.search(r"([\d,]+\.\d+)", text)
                if odds_text:
                    odds_val = _safe_float(odds_text.group(1).replace(",", ""))
                    if odds_val > 0:
                        odds_dict[combo] = odds_val
                # 隣のセルにオッズがある場合
                elif i + 1 < len(tds):
                    next_text = tds[i + 1].get_text(strip=True)
                    odds_match = re.search(r"([\d,]+\.\d+)", next_text)
                    if odds_match:
                        odds_val = _safe_float(
                            odds_match.group(1).replace(",", "")
                        )
                        if odds_val > 0:
                            odds_dict[combo] = odds_val

    return odds_dict


def fetch_odds_3t(date: str, race_no: int) -> dict[str, float]:
    """3連単オッズを取得する

    boatrace.jp の odds3t ページの HTML テーブル構造:
    - contentsFrame1_inner 配下の table 要素
    - 6つの tbody（1着=1号艇〜6号艇）
    - 各 tbody に 5行の tr（2着候補: 1着以外の5艇）
    - 各 tr に 4つの td.oddsPoint（3着候補: 1着・2着以外の4艇）
    - 合計 120 通り (6P3 = 6x5x4)

    Returns:
        {"1-2-3": 12.5, "1-2-4": 18.3, ...} の形式
    """
    html = fetch_page(
        "/owpc/pc/race/odds3t",
        params={"hd": date, "jcd": VENUE_CODE, "rno": str(race_no)},
    )
    if not html:
        return {}

    soup = BeautifulSoup(html, "lxml")

    # メイン: 位置ベースの解析（テーブル内の tbody/tr/td.oddsPoint の
    # 並び順から 1着-2着-3着 の組番を決定する）
    table_selector = (
        "div.contentsFrame1_inner table"
    )
    odds_dict = _parse_odds_table_positional(soup, table_selector, "3t")

    # フォールバック: regex ベースの解析
    if not odds_dict:
        odds_dict = _parse_odds_fallback_regex(soup, r"\d-\d-\d")

    return odds_dict


def fetch_odds_2tf(date: str, race_no: int) -> dict[str, float]:
    """2連単オッズを取得する

    boatrace.jp の odds2tf ページの HTML テーブル構造:
    - contentsFrame1_inner 配下の table 要素
    - 6つの tbody（1着=1号艇〜6号艇）
    - 各 tbody に 5行の tr（2着候補: 1着以外の5艇）
    - 各 tr に 1つの td.oddsPoint
    - 合計 30 通り (6P2 = 6x5)

    Returns:
        {"1-2": 3.5, "1-3": 5.2, ...} の形式
    """
    html = fetch_page(
        "/owpc/pc/race/odds2tf",
        params={"hd": date, "jcd": VENUE_CODE, "rno": str(race_no)},
    )
    if not html:
        return {}

    soup = BeautifulSoup(html, "lxml")

    table_selector = "div.contentsFrame1_inner table"
    odds_dict = _parse_odds_table_positional(soup, table_selector, "2tf")

    if not odds_dict:
        odds_dict = _parse_odds_fallback_regex(soup, r"\d-\d")

    return odds_dict


def fetch_odds_3f(date: str, race_no: int) -> dict[str, float]:
    """3連複オッズを取得する

    Returns:
        {"1=2=3": 5.5, "1=2=4": 8.2, ...} の形式
    """
    html = fetch_page(
        "/owpc/pc/race/odds3f",
        params={"hd": date, "jcd": VENUE_CODE, "rno": str(race_no)},
    )
    if not html:
        return {}

    soup = BeautifulSoup(html, "lxml")
    odds_dict = _parse_odds_fallback_regex(soup, r"\d=\d=\d")

    if not odds_dict:
        for td in soup.select("td.oddsPoint"):
            text = td.get_text(strip=True)
            odds_val = _safe_float(text.replace(",", ""))
            if odds_val > 0:
                data_id = td.get("data-id", "")
                if data_id and "=" in data_id:
                    odds_dict[data_id] = odds_val

    return odds_dict


def fetch_odds_2kt(date: str, race_no: int) -> dict[str, float]:
    """2連複オッズを取得する

    boatrace.jp の odds2kt ページの HTML テーブル構造:
    - contentsFrame1_inner 配下の table 要素
    - 5つの tbody（軸艇番 1〜5）
    - 各 tbody に (6-軸艇番) 行の tr（相手: 軸より大きい艇番）
    - 各 tr に 1つの td.oddsPoint
    - 合計 15 通り (6C2 = 15)

    Returns:
        {"1=2": 2.1, "1=3": 4.8, ...} の形式（小さい番号=大きい番号）
    """
    html = fetch_page(
        "/owpc/pc/race/odds2kt",
        params={"hd": date, "jcd": VENUE_CODE, "rno": str(race_no)},
    )
    if not html:
        return {}

    soup = BeautifulSoup(html, "lxml")

    table_selector = "div.contentsFrame1_inner table"
    odds_dict = _parse_odds_table_positional(soup, table_selector, "2kt")

    if not odds_dict:
        odds_dict = _parse_odds_fallback_regex(soup, r"\d=\d")

    return odds_dict


def debug_racelist_html(date: str, race_no: int) -> list[dict]:
    """デバッグ用: 出走表HTMLの各tbodyの構造を返す"""
    html = fetch_page(
        "/owpc/pc/race/racelist",
        params={"hd": date, "jcd": VENUE_CODE, "rno": str(race_no)},
    )
    if not html:
        return [{"error": "HTML取得失敗"}]

    soup = BeautifulSoup(html, "lxml")
    result = []

    tbody_list = soup.select(".table1 tbody")
    for i, tbody in enumerate(tbody_list[:10]):
        tds = tbody.select("td")
        if not tds:
            continue
        td_texts = [td.get_text(strip=True)[:50] for td in tds[:8]]
        td_classes = []
        for td in tds[:8]:
            cls = td.get("class", [])
            td_classes.append(" ".join(cls) if cls else "-")

        text_joined = " ".join(td.get_text(strip=True) for td in tds)
        rates = re.findall(r"(\d+\.\d{2})", text_joined)
        has_reg = bool(re.search(r"\d{4}", text_joined))

        result.append({
            "tbody_index": i,
            "td_count": len(tds),
            "td_texts": td_texts,
            "td_classes": td_classes,
            "rates_count": len(rates),
            "has_4digit": has_reg,
            "is_racer": len(rates) >= 2 and has_reg,
        })

    return result


def fetch_today_race_count(date: str) -> list[int]:
    """今日の平和島の開催レース番号一覧を取得する"""
    html = fetch_page(
        "/owpc/pc/race/index",
        params={"hd": date, "jcd": VENUE_CODE},
    )
    if not html:
        return []

    soup = BeautifulSoup(html, "lxml")
    race_numbers = []

    # レース番号のリンクを探す
    links = soup.select("a[href]")
    for link in links:
        href = link.get("href", "")
        if f"jcd={VENUE_CODE}" in href or "jcd=04" in href:
            rno_match = re.search(r"rno=(\d+)", href)
            if rno_match:
                rno = int(rno_match.group(1))
                if 1 <= rno <= 12 and rno not in race_numbers:
                    race_numbers.append(rno)

    if not race_numbers:
        # デフォルト: 通常開催は12R
        return list(range(1, 13))

    return sorted(race_numbers)


# ── 統合データ取得 API ──
# 優先順位: 1. BoatraceOpenAPI (JSON) → 2. heiwajima.gr.jp → 3. boatrace.jp

def unified_fetch_race(date: str, race_no: int) -> tuple[RaceInfo | None, dict, dict]:
    """レースデータとオッズを統合的に取得する

    3段階のフォールバック:
    1. BoatraceOpenAPI (GitHub Pages JSON) - 最も安定
    2. heiwajima.gr.jp (平和島公式HTML) - 追加データあり
    3. boatrace.jp (BOAT RACE公式HTML) - 最終手段

    Args:
        date: YYYYMMDD形式の日付
        race_no: レース番号 (1-12)

    Returns:
        (race_info, odds_dict, extra_data)
        extra_data: {"source": str, "course_entry": dict, "exhibit_st": dict}
    """
    extra_data = {"source": "none", "course_entry": {}, "exhibit_st": {}}
    race_info = None
    odds_dict = {}

    # ── 1. BoatraceOpenAPI (JSON) ──
    try:
        from .openapi import fetch_openapi_race, openapi_to_race_info

        openapi_data = fetch_openapi_race(date, race_no)
        if openapi_data and len(openapi_data.get("racers", [])) >= 6:
            race_info = openapi_to_race_info(openapi_data)
            extra_data["source"] = "openapi"

            # 展示ST・進入コースを extra_data に反映
            st_dict = {}
            course_dict = {}
            for r in race_info.racers:
                if r.exhibit_st != 0.0:
                    st_dict[r.waku] = r.exhibit_st
                if r.course_entry >= 1:
                    course_dict[r.waku] = r.course_entry
            if st_dict:
                extra_data["exhibit_st"] = st_dict
            if course_dict:
                extra_data["course_entry"] = course_dict
    except Exception:
        pass

    # ── 2. heiwajima.gr.jp (HTMLスクレイピング) ──
    if race_info is None or not race_info.racers:
        try:
            from .heiwajima import (
                fetch_heiwajima_race,
                fetch_heiwajima_odds,
                to_race_info,
            )

            hw_race = fetch_heiwajima_race(date, race_no)
            if hw_race.success and len(hw_race.racers) >= 6:
                race_info = to_race_info(hw_race)
                extra_data["source"] = "heiwajima"

                # 直前情報をRaceInfoのRacerに反映
                for hr in hw_race.racers:
                    for r in race_info.racers:
                        if r.waku == hr.waku:
                            r.exhibit_st = hr.exhibit_st
                            r.avg_start_timing = hr.avg_start_timing
                            r.flying_count = hr.flying_count
                            r.late_count = hr.late_count
                            r.course_entry = hr.course_entry
                            break

                # 進入コース
                if hw_race.course_entries:
                    extra_data["course_entry"] = hw_race.course_entries

                # 展示ST
                st_dict = {}
                for hr in hw_race.racers:
                    if hr.exhibit_st != 0.0:
                        st_dict[hr.waku] = hr.exhibit_st
                if st_dict:
                    extra_data["exhibit_st"] = st_dict

                # オッズ取得
                odds_dict = fetch_heiwajima_odds(date, race_no, "3t")
        except Exception:
            pass

    # ── 3. boatrace.jp (最終フォールバック) ──
    if race_info is None or not race_info.racers:
        try:
            race_info = fetch_race_list(date, race_no)
            if race_info and race_info.racers:
                race_info = fetch_before_info(date, race_no, race_info)
                extra_data["source"] = "boatrace"
        except Exception:
            pass

    if not odds_dict:
        try:
            odds_dict = fetch_odds_3t(date, race_no)
        except Exception:
            pass

    # ── 4. heiwajima で補完 (boatrace.jpデータに追加情報を付ける) ──
    if race_info and race_info.racers and extra_data["source"] in ("boatrace", "openapi"):
        try:
            from .heiwajima import fetch_heiwajima_supplement
            supplement = fetch_heiwajima_supplement(date, race_no)
            if supplement and supplement.success:
                if supplement.course_entry:
                    extra_data["course_entry"] = supplement.course_entry
                if supplement.exhibit_st:
                    extra_data["exhibit_st"] = supplement.exhibit_st
        except Exception:
            pass

    return race_info, odds_dict, extra_data


def unified_fetch_odds(date: str, race_no: int,
                       bet_type: str = "3t",
                       day_no: int = 0) -> dict:
    """オッズを統合的に取得する

    優先順位: heiwajima.gr.jp → boatrace.jp

    Args:
        bet_type: "3t" (3連単), "2tf" (2連単), "2kt" (2連複)
    """
    odds = {}

    # heiwajima.gr.jp から
    try:
        from .heiwajima import fetch_heiwajima_odds
        odds = fetch_heiwajima_odds(date, race_no, bet_type)
    except Exception:
        pass

    # boatrace.jp フォールバック
    if not odds:
        try:
            if bet_type == "3t":
                odds = fetch_odds_3t(date, race_no)
            elif bet_type == "3f":
                odds = fetch_odds_3f(date, race_no)
            elif bet_type == "2tf":
                odds = fetch_odds_2tf(date, race_no)
            elif bet_type == "2kt":
                odds = fetch_odds_2kt(date, race_no)
        except Exception:
            pass

    return odds
