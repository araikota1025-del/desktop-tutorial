"""平和島ボートレース公式サイト (heiwajima.gr.jp/sp) からの全データ取得

プライマリデータソースとして以下を取得:
- 出走表（選手データ、級別、勝率、モーター/ボート情報）
- オッズ（3連単、2連単、2連複）
- 直前情報（展示タイム、展示ST、進入コース、気象）
- レース結果

URL パターン:
  出走表: /asp/kyogi/{venue_code}/sp/syusso{day:02d}{race:02d}.htm
  オッズ: /asp/kyogi/{venue_code}/sp/odds{day:02d}{race:02d}.htm
  直前:   /asp/kyogi/{venue_code}/sp/chokuzen{day:02d}{race:02d}.htm
  結果:   /asp/kyogi/{venue_code}/sp/result{day:02d}{race:02d}.htm
  index:  /asp/heiwajima/sp/kyogi/kyogihtml/index.htm

日数(day)は開催初日=1で、開催シリーズ内の何日目かを表す。
index ページから当日の day 番号を自動検出する。
"""

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from bs4 import BeautifulSoup

from .client import fetch_heiwajima_page

HEIWAJIMA_BASE = "https://www.heiwajima.gr.jp"
VENUE_CODE = "04"  # 平和島の場コード


# ── データクラス ──

@dataclass
class HeiwajimaRacer:
    """平和島公式サイトから取得した選手データ"""
    waku: int = 0
    name: str = ""
    register_no: str = ""
    rank: str = ""       # A1/A2/B1/B2
    branch: str = ""     # 支部
    age: int = 0
    weight: float = 0.0
    win_rate_all: float = 0.0
    win_rate_2r_all: float = 0.0
    win_rate_local: float = 0.0
    win_rate_2r_local: float = 0.0
    motor_no: str = ""
    motor_2r: float = 0.0
    motor_3r: float = 0.0
    boat_no: str = ""
    boat_2r: float = 0.0
    boat_3r: float = 0.0
    # 直前情報
    exhibit_time: float = 0.0
    exhibit_st: float = 0.0
    tilt: float = 0.0
    course_entry: int = 0  # 実際の進入コース (0=未取得)
    # 追加統計
    avg_start_timing: float = 0.0  # 平均ST
    flying_count: int = 0          # フライング回数
    late_count: int = 0            # 出遅れ回数


@dataclass
class HeiwajimaWeather:
    """気象・水面情報"""
    weather: str = ""
    wind_direction: str = ""
    wind_speed: int = 0
    wave_height: int = 0
    temperature: float = 0.0
    water_temp: float = 0.0
    water_condition: str = ""  # 安定板使用等
    is_stable_board: bool = False  # 安定板使用


@dataclass
class HeiwajimaRaceInfo:
    """1レース分の全データ"""
    race_no: int = 0
    race_name: str = ""
    date: str = ""
    deadline: str = ""
    day_no: int = 0  # 開催何日目か
    racers: list = field(default_factory=list)  # list[HeiwajimaRacer]
    weather: HeiwajimaWeather = field(default_factory=HeiwajimaWeather)
    odds_3t: dict = field(default_factory=dict)   # {"1-2-3": 12.5, ...}
    odds_2tf: dict = field(default_factory=dict)   # {"1-2": 3.5, ...}
    odds_2kt: dict = field(default_factory=dict)   # {"1=2": 2.1, ...}
    course_entries: dict = field(default_factory=dict)  # {waku: course}
    result: list = field(default_factory=list)  # 着順 [1st_waku, 2nd_waku, ...]
    success: bool = False


@dataclass
class HeiwajimaSupplement:
    """後方互換: 旧インターフェースとの互換データ"""
    race_no: int = 0
    exhibit_st: Optional[dict] = None
    course_entry: Optional[dict] = None
    water_condition: str = ""
    success: bool = False


# ── URL 生成・検出 ──

def _build_urls(page_type: str, day_no: int, race_no: int) -> list[str]:
    """平和島SPページのURL候補リストを生成する

    Args:
        page_type: "syusso" / "odds" / "chokuzen" / "result"
        day_no: 開催日数 (1-7)
        race_no: レース番号 (1-12)
    """
    d = str(day_no).zfill(2)
    r = str(race_no).zfill(2)

    urls = [
        # パターン1: /asp/kyogi/{venue}/sp/{type}{day}{race}.htm
        f"{HEIWAJIMA_BASE}/asp/kyogi/{d}/sp/{page_type}{d}{r}.htm",
        # パターン2: venue code in path
        f"{HEIWAJIMA_BASE}/asp/kyogi/{VENUE_CODE}/sp/{page_type}{d}{r}.htm",
        # パターン3: index style
        f"{HEIWAJIMA_BASE}/asp/heiwajima/sp/kyogi/{page_type}/{d}{r}.htm",
        # パターン4: alternate path
        f"{HEIWAJIMA_BASE}/asp/kyogi/sp/{page_type}{d}{r}.htm",
    ]
    return urls


def _fetch_with_fallback(urls: list[str]) -> Optional[str]:
    """複数のURL候補を順番に試してHTMLを返す"""
    for url in urls:
        html = fetch_heiwajima_page(url)
        if html and len(html) > 200:
            return html
    return None


def discover_current_day(date_str: str = "") -> int:
    """index ページから当日の開催日数を自動検出する

    Returns:
        day_no: 開催日数 (1-7)。検出失敗時は1-7を全て試す用に0を返す。
    """
    index_urls = [
        f"{HEIWAJIMA_BASE}/asp/heiwajima/sp/kyogi/kyogihtml/index.htm",
        f"{HEIWAJIMA_BASE}/sp/",
        f"{HEIWAJIMA_BASE}/asp/kyogi/sp/index.htm",
    ]

    html = _fetch_with_fallback(index_urls)
    if not html:
        return 0

    soup = BeautifulSoup(html, "lxml")

    # リンクからday番号を検出
    # syusso{DD}{RR}.htm のパターンを探す
    day_numbers = set()
    for a_tag in soup.select("a[href]"):
        href = a_tag.get("href", "")
        m = re.search(r"syusso(\d{2})\d{2}\.htm", href)
        if m:
            day_numbers.add(int(m.group(1)))
        m = re.search(r"odds(\d{2})\d{2}\.htm", href)
        if m:
            day_numbers.add(int(m.group(1)))

    if day_numbers:
        return max(day_numbers)  # 最新の日を返す

    # テキストからも検出を試みる
    text = soup.get_text()
    day_match = re.search(r"(\d+)日目", text)
    if day_match:
        return int(day_match.group(1))

    return 0


def _find_day_no_for_date(date_str: str) -> list[int]:
    """指定日付のday_noを試行するリストを返す"""
    detected = discover_current_day(date_str)
    if detected > 0:
        # 検出成功: その日を最優先に前後も試す
        candidates = [detected]
        for delta in [1, -1, 2, -2]:
            d = detected + delta
            if 1 <= d <= 7:
                candidates.append(d)
        return candidates
    else:
        # 検出失敗: 1-7を全て試す
        return list(range(1, 8))


# ── 出走表パーサー ──

def _safe_float(text: str, default: float = 0.0) -> float:
    """安全にfloat変換"""
    try:
        cleaned = re.sub(r"[^\d.\-]", "", text.strip())
        return float(cleaned) if cleaned else default
    except (ValueError, TypeError):
        return default


def _safe_int(text: str, default: int = 0) -> int:
    """安全にint変換"""
    try:
        cleaned = re.sub(r"[^\d]", "", text.strip())
        return int(cleaned) if cleaned else default
    except (ValueError, TypeError):
        return default


def _detect_waku_from_element(el) -> int:
    """要素から枠番を検出する（CSSクラスやテキストから）"""
    # CSSクラスから
    for cls in el.get("class", []):
        m = re.search(r"(?:boat|waku|color)[\-_]?(\d)", cls, re.IGNORECASE)
        if m and 1 <= int(m.group(1)) <= 6:
            return int(m.group(1))
    # style属性のbackground-colorから
    style = el.get("style", "")
    waku_colors = {
        "#fff": 1, "white": 1,
        "#000": 2, "black": 2,
        "#f00": 3, "red": 3,
        "#00f": 4, "blue": 4,
        "#ff0": 5, "yellow": 5,
        "#0f0": 6, "#0a0": 6, "green": 6,
    }
    for color, waku in waku_colors.items():
        if color in style.lower():
            return waku
    return 0


def _parse_race_entry_table(soup: BeautifulSoup) -> list[HeiwajimaRacer]:
    """出走表HTMLから選手データをパースする"""
    racers = []
    found_wakus = set()

    # テーブルベースのパース
    for table in soup.select("table"):
        rows = table.select("tr")
        for row in rows:
            cells = row.select("td, th")
            if len(cells) < 3:
                continue

            texts = [c.get_text(strip=True) for c in cells]
            joined = " ".join(texts)

            # 枠番の検出
            waku = 0
            # CSSクラスから
            for c in cells:
                w = _detect_waku_from_element(c)
                if w:
                    waku = w
                    break
            # テキストの先頭数字から
            if not waku and texts[0] and re.match(r"^[1-6]$", texts[0]):
                waku = int(texts[0])

            if not (1 <= waku <= 6) or waku in found_wakus:
                continue

            # 選手データっぽいか判定（勝率が含まれるか）
            rates = re.findall(r"\d+\.\d{2}", joined)
            has_reg = bool(re.search(r"\d{4}", joined))
            has_rank = bool(re.search(r"[AB][12]", joined))
            if len(rates) < 2 and not (has_reg and has_rank):
                continue

            racer = HeiwajimaRacer(waku=waku)

            # 登録番号
            reg = re.search(r"(\d{4})", joined)
            if reg:
                racer.register_no = reg.group(1)

            # 級別
            rank = re.search(r"(A1|A2|B1|B2)", joined)
            if rank:
                racer.rank = rank.group(1)

            # 選手名（日本語2文字以上）
            for c in cells:
                name_text = c.get_text(strip=True)
                name_text = re.sub(r"\s+", "", name_text)
                if re.match(r"^[\u3000-\u9fff\uf900-\ufaff]{2,}", name_text):
                    if not re.match(r"^\d", name_text) and name_text != racer.rank:
                        racer.name = name_text
                        break
            # a タグから名前を取得
            if not racer.name:
                a_tag = row.select_one("a")
                if a_tag:
                    name_text = re.sub(r"\s+", "", a_tag.get_text(strip=True))
                    if len(name_text) >= 2:
                        racer.name = name_text

            # 勝率群（数値パターンマッチ）
            if len(rates) >= 2:
                racer.win_rate_all = _safe_float(rates[0])
                racer.win_rate_2r_all = _safe_float(rates[1])
            if len(rates) >= 4:
                racer.win_rate_local = _safe_float(rates[2])
                racer.win_rate_2r_local = _safe_float(rates[3])
            if len(rates) >= 6:
                racer.motor_2r = _safe_float(rates[4])
                racer.boat_2r = _safe_float(rates[5])
            if len(rates) >= 7:
                racer.motor_3r = _safe_float(rates[6])
            if len(rates) >= 8:
                racer.boat_3r = _safe_float(rates[7])

            # 年齢・体重
            age_match = re.search(r"(\d{2})歳", joined)
            if age_match:
                racer.age = int(age_match.group(1))
            weight_match = re.search(r"(\d{2,3}(?:\.\d)?)kg", joined)
            if weight_match:
                racer.weight = _safe_float(weight_match.group(1))

            # 支部
            branch_match = re.search(r"(東京|埼玉|群馬|静岡|愛知|三重|大阪|兵庫|岡山|広島|山口|徳島|香川|愛媛|高知|福岡|佐賀|長崎|熊本|大分|宮崎|鹿児島)", joined)
            if branch_match:
                racer.branch = branch_match.group(1)

            # 平均ST
            st_match = re.search(r"(?:平均ST|ST)\s*[：:]?\s*(\d*\.?\d{2})", joined)
            if st_match:
                racer.avg_start_timing = _safe_float(st_match.group(1))

            # F/L回数
            f_match = re.search(r"F\s*(\d)", joined)
            if f_match:
                racer.flying_count = int(f_match.group(1))
            l_match = re.search(r"L\s*(\d)", joined)
            if l_match:
                racer.late_count = int(l_match.group(1))

            # モーター/ボート番号
            motor_match = re.search(r"モ[ータ]*[：:]?\s*(\d+)", joined)
            if motor_match:
                racer.motor_no = motor_match.group(1)
            boat_match = re.search(r"ボ[ート]*[：:]?\s*(\d+)", joined)
            if boat_match:
                racer.boat_no = boat_match.group(1)

            found_wakus.add(waku)
            racers.append(racer)

    # tbody ベースのパース（boatrace.jp スタイル）
    if len(racers) < 6:
        found_wakus_tbody = set(r.waku for r in racers)
        for tbody in soup.select("tbody"):
            tds = tbody.select("td")
            if not tds:
                continue
            texts = [td.get_text(strip=True) for td in tds]
            joined = " ".join(texts)

            rates = re.findall(r"\d+\.\d{2}", joined)
            has_reg = bool(re.search(r"\d{4}", joined))
            has_rank = bool(re.search(r"[AB][12]", joined))
            if len(rates) < 2 or not (has_reg or has_rank):
                continue

            waku = 0
            for td in tds:
                w = _detect_waku_from_element(td)
                if w:
                    waku = w
                    break
            if not waku:
                first = tds[0].get_text(strip=True)
                if re.match(r"^[1-6]$", first):
                    waku = int(first)

            if not (1 <= waku <= 6) or waku in found_wakus_tbody:
                continue

            racer = HeiwajimaRacer(waku=waku)
            reg = re.search(r"(\d{4})", joined)
            if reg:
                racer.register_no = reg.group(1)
            rank = re.search(r"(A1|A2|B1|B2)", joined)
            if rank:
                racer.rank = rank.group(1)

            name_el = tbody.select_one(".is-fs18, .is-fs14, a")
            if name_el:
                racer.name = re.sub(r"\s+", "", name_el.get_text(strip=True))

            if len(rates) >= 2:
                racer.win_rate_all = _safe_float(rates[0])
                racer.win_rate_2r_all = _safe_float(rates[1])
            if len(rates) >= 4:
                racer.win_rate_local = _safe_float(rates[2])
                racer.win_rate_2r_local = _safe_float(rates[3])
            if len(rates) >= 6:
                racer.motor_2r = _safe_float(rates[4])
                racer.boat_2r = _safe_float(rates[5])

            found_wakus_tbody.add(waku)
            racers.append(racer)

    racers.sort(key=lambda r: r.waku)
    return racers


# ── オッズパーサー ──

def _parse_odds_positional(soup: BeautifulSoup, bet_type: str = "3t") -> dict:
    """オッズテーブルを位置ベースでパースする

    Args:
        bet_type: "3t" (3連単), "2tf" (2連単), "2kt" (2連複)

    Returns:
        {"1-2-3": 12.5, ...} 形式のオッズ辞書
    """
    odds = {}

    # テーブルを探す（複数セレクタ）
    table = None
    for selector in [
        "div.contentsFrame1_inner table",
        "table.oddsTable",
        ".table1 table",
        "table.is-w495",
        "table",
    ]:
        tables = soup.select(selector)
        for t in tables:
            # オッズらしいテーブルか判定
            points = t.select("td.oddsPoint, td.odds-point, td.odds")
            if points:
                table = t
                break
            # テキストにオッズ値が含まれるか
            text = t.get_text()
            if re.search(r"\d+\.\d", text) and len(t.select("tbody")) >= 3:
                table = t
                break
        if table:
            break

    if not table:
        return _parse_odds_from_any_table(soup, bet_type)

    tbodies = table.select("tbody")
    if not tbodies:
        tbodies = [table]

    if bet_type == "3t":
        return _parse_3t_positional(tbodies)
    elif bet_type == "2tf":
        return _parse_2tf_positional(tbodies)
    elif bet_type == "2kt":
        return _parse_2kt_positional(tbodies)

    return odds


def _parse_3t_positional(tbodies: list) -> dict:
    """3連単: 6 tbody x 5 tr x 4 td = 120 通り"""
    odds = {}
    boats = [1, 2, 3, 4, 5, 6]

    for tbody_idx, tbody in enumerate(tbodies[:6]):
        first = boats[tbody_idx] if tbody_idx < 6 else tbody_idx + 1
        rows = tbody.select("tr")
        second_candidates = [b for b in boats if b != first]

        for row_idx, row in enumerate(rows[:5]):
            if row_idx >= len(second_candidates):
                break
            second = second_candidates[row_idx]
            third_candidates = [b for b in boats if b != first and b != second]

            cells = row.select("td.oddsPoint, td.odds-point, td.odds")
            if not cells:
                # 数値を含むtdをオッズとして使う
                cells = [td for td in row.select("td")
                         if re.match(r"[\d,]+\.\d", td.get_text(strip=True))]

            for cell_idx, cell in enumerate(cells[:4]):
                if cell_idx >= len(third_candidates):
                    break
                third = third_candidates[cell_idx]
                val = _parse_odds_value(cell.get_text(strip=True))
                if val > 0:
                    odds[f"{first}-{second}-{third}"] = val

    return odds


def _parse_2tf_positional(tbodies: list) -> dict:
    """2連単: 6 tbody x 5 tr x 1 td = 30 通り"""
    odds = {}
    boats = [1, 2, 3, 4, 5, 6]

    for tbody_idx, tbody in enumerate(tbodies[:6]):
        first = boats[tbody_idx] if tbody_idx < 6 else tbody_idx + 1
        rows = tbody.select("tr")
        second_candidates = [b for b in boats if b != first]

        for row_idx, row in enumerate(rows[:5]):
            if row_idx >= len(second_candidates):
                break
            second = second_candidates[row_idx]

            cells = row.select("td.oddsPoint, td.odds-point, td.odds")
            if not cells:
                cells = [td for td in row.select("td")
                         if re.match(r"[\d,]+\.\d", td.get_text(strip=True))]

            if cells:
                val = _parse_odds_value(cells[-1].get_text(strip=True))
                if val > 0:
                    odds[f"{first}-{second}"] = val

    return odds


def _parse_2kt_positional(tbodies: list) -> dict:
    """2連複: 5 tbody, 行数逓減 = 15 通り"""
    odds = {}
    boats = [1, 2, 3, 4, 5, 6]

    for tbody_idx, tbody in enumerate(tbodies[:5]):
        base = boats[tbody_idx]
        rows = tbody.select("tr")
        partner_candidates = [b for b in boats if b > base]

        for row_idx, row in enumerate(rows):
            if row_idx >= len(partner_candidates):
                break
            partner = partner_candidates[row_idx]

            cells = row.select("td.oddsPoint, td.odds-point, td.odds")
            if not cells:
                cells = [td for td in row.select("td")
                         if re.match(r"[\d,]+\.\d", td.get_text(strip=True))]

            if cells:
                val = _parse_odds_value(cells[-1].get_text(strip=True))
                if val > 0:
                    a, b = sorted([base, partner])
                    odds[f"{a}={b}"] = val

    return odds


def _parse_odds_from_any_table(soup: BeautifulSoup, bet_type: str) -> dict:
    """テーブル構造が不明な場合のフォールバック: 全テーブルからオッズを探す"""
    odds = {}

    for table in soup.select("table"):
        for row in table.select("tr"):
            cells = row.select("td")
            if len(cells) < 2:
                continue

            texts = [c.get_text(strip=True) for c in cells]
            joined = " ".join(texts)

            # "1-2-3" パターンのテキストを探す
            if bet_type == "3t":
                combo_match = re.search(r"([1-6])\s*[-ー]\s*([1-6])\s*[-ー]\s*([1-6])", joined)
                if combo_match:
                    a, b, c = combo_match.groups()
                    combo = f"{a}-{b}-{c}"
                    val_match = re.search(r"([\d,]+\.\d)", joined[combo_match.end():])
                    if val_match:
                        odds[combo] = _parse_odds_value(val_match.group(1))
            elif bet_type == "2tf":
                combo_match = re.search(r"([1-6])\s*[-ー]\s*([1-6])", joined)
                if combo_match:
                    a, b = combo_match.groups()
                    if a != b:
                        combo = f"{a}-{b}"
                        val_match = re.search(r"([\d,]+\.\d)", joined[combo_match.end():])
                        if val_match:
                            odds[combo] = _parse_odds_value(val_match.group(1))
            elif bet_type == "2kt":
                combo_match = re.search(r"([1-6])\s*[=＝]\s*([1-6])", joined)
                if combo_match:
                    a, b = sorted(combo_match.groups())
                    combo = f"{a}={b}"
                    val_match = re.search(r"([\d,]+\.\d)", joined[combo_match.end():])
                    if val_match:
                        odds[combo] = _parse_odds_value(val_match.group(1))

    return odds


def _parse_odds_value(text: str) -> float:
    """オッズ値の文字列をfloatに変換（カンマ対応）"""
    try:
        cleaned = text.replace(",", "").replace("，", "").strip()
        cleaned = re.sub(r"[^\d.]", "", cleaned)
        return float(cleaned) if cleaned else 0.0
    except (ValueError, TypeError):
        return 0.0


# ── 直前情報パーサー ──

def _parse_before_info(soup: BeautifulSoup) -> tuple[dict, dict, HeiwajimaWeather]:
    """直前情報ページから展示データと気象データをパースする

    Returns:
        (exhibit_times, exhibit_sts, weather)
    """
    exhibit_times: dict[int, float] = {}
    exhibit_sts: dict[int, float] = {}
    weather = HeiwajimaWeather()

    text = soup.get_text()

    # ── 気象情報 ──
    w_match = re.search(r"(晴|曇り?|雨|小雨|雪|霧)", text)
    if w_match:
        weather.weather = w_match.group(1)

    wind_dir_match = re.search(r"(北|南|東|西|北東|北西|南東|南西|無風)", text)
    if wind_dir_match:
        weather.wind_direction = wind_dir_match.group(1)

    ws_match = re.search(r"風速?\s*[:：]?\s*(\d+)\s*m", text)
    if not ws_match:
        ws_match = re.search(r"(\d+)\s*m\s*(?:/s)?", text)
    if ws_match:
        weather.wind_speed = int(ws_match.group(1))

    wh_match = re.search(r"波[高]?\s*[:：]?\s*(\d+)\s*cm", text)
    if wh_match:
        weather.wave_height = int(wh_match.group(1))

    temp_match = re.search(r"気温\s*[:：]?\s*(\d+\.?\d*)", text)
    if temp_match:
        weather.temperature = float(temp_match.group(1))

    wt_match = re.search(r"水温\s*[:：]?\s*(\d+\.?\d*)", text)
    if wt_match:
        weather.water_temp = float(wt_match.group(1))

    # 安定板
    if "安定板" in text:
        weather.is_stable_board = True
        weather.water_condition = "安定板使用"

    # ── 展示タイム ──
    time_pattern = re.compile(r"\d+\.\d{2}")
    for table in soup.select("table"):
        table_text = table.get_text()
        if "展示" not in table_text and "タイム" not in table_text:
            continue

        rows = table.select("tr")
        for row in rows:
            tds = row.select("td, th")
            if len(tds) < 2:
                continue
            row_text = row.get_text()

            # 枠番検出
            waku = 0
            for td in tds:
                w = _detect_waku_from_element(td)
                if w:
                    waku = w
                    break
            if not waku:
                first = tds[0].get_text(strip=True)
                waku_match = re.match(r"^([1-6])$", first)
                if waku_match:
                    waku = int(waku_match.group(1))

            if not (1 <= waku <= 6):
                continue

            # 展示タイム
            times = time_pattern.findall(row_text)
            if times:
                # 最後の数値が展示タイムのことが多い
                et = float(times[-1])
                if 6.0 <= et <= 8.0:
                    exhibit_times[waku] = et

    # ── 展示ST ──
    for table in soup.select("table"):
        table_text = table.get_text()
        if "ST" not in table_text and "スタート" not in table_text:
            continue

        rows = table.select("tr")
        for row in rows:
            tds = row.select("td, th")
            if len(tds) < 2:
                continue
            row_text = row.get_text()

            waku = 0
            for td in tds:
                w = _detect_waku_from_element(td)
                if w:
                    waku = w
                    break
            if not waku:
                waku_match = re.search(r"([1-6])\s*号?艇?", row_text)
                if waku_match:
                    waku = int(waku_match.group(1))

            if not (1 <= waku <= 6):
                continue

            # STタイミング
            st_match = re.search(r"[F.]?\s*(\d*\.?\d{2})", row_text)
            if st_match:
                timing = float(st_match.group(1)) if st_match.group(1) else 0.0
                if "F" in row_text[:row_text.find(st_match.group(0)) + 1]:
                    timing = -timing
                if abs(timing) < 1.0:  # 妥当なST値のみ
                    exhibit_sts[waku] = timing

    return exhibit_times, exhibit_sts, weather


def _parse_course_entry(soup: BeautifulSoup) -> dict[int, int]:
    """進入コースをパースする"""
    course_dict: dict[int, int] = {}

    for table in soup.select("table"):
        text = table.get_text()
        if "進入" not in text and "コース" not in text:
            continue

        rows = table.select("tr")
        for row in rows:
            tds = row.select("td")
            if len(tds) < 6:
                continue
            entries = []
            for td in tds[:6]:
                num = re.search(r"([1-6])", td.get_text(strip=True))
                if num:
                    entries.append(int(num.group(1)))
            if len(entries) == 6 and len(set(entries)) == 6:
                for course, waku in enumerate(entries, 1):
                    course_dict[waku] = course
                break

    return course_dict


# ── 公開API ──

def fetch_heiwajima_race(date_str: str, race_no: int,
                          day_no: int = 0) -> HeiwajimaRaceInfo:
    """平和島公式サイトから1レースの全データを取得する

    Args:
        date_str: "YYYYMMDD" 形式の日付
        race_no: レース番号 (1-12)
        day_no: 開催日数 (0=自動検出)

    Returns:
        HeiwajimaRaceInfo
    """
    result = HeiwajimaRaceInfo(race_no=race_no, date=date_str)

    # day_no の決定
    if day_no <= 0:
        day_candidates = _find_day_no_for_date(date_str)
    else:
        day_candidates = [day_no]

    # 出走表を取得
    html = None
    used_day = 0
    for d in day_candidates:
        urls = _build_urls("syusso", d, race_no)
        html = _fetch_with_fallback(urls)
        if html:
            used_day = d
            result.day_no = d
            break

    if html:
        soup = BeautifulSoup(html, "lxml")
        result.racers = _parse_race_entry_table(soup)

        # レース名
        title_el = soup.select_one("h1, h2, .race-name, .raceTitle, title")
        if title_el:
            result.race_name = title_el.get_text(strip=True)

        # 締切時刻
        deadline_match = re.search(r"(\d{1,2}:\d{2})", soup.get_text())
        if deadline_match:
            result.deadline = deadline_match.group(1)

    # 直前情報を取得
    if used_day > 0:
        before_urls = _build_urls("chokuzen", used_day, race_no)
        before_html = _fetch_with_fallback(before_urls)
        if before_html:
            before_soup = BeautifulSoup(before_html, "lxml")
            exhibit_times, exhibit_sts, weather = _parse_before_info(before_soup)
            result.weather = weather

            # 選手データに直前情報をマージ
            for racer in result.racers:
                if racer.waku in exhibit_times:
                    racer.exhibit_time = exhibit_times[racer.waku]
                if racer.waku in exhibit_sts:
                    racer.exhibit_st = exhibit_sts[racer.waku]

            # 進入コース
            course_dict = _parse_course_entry(before_soup)
            if course_dict:
                result.course_entries = course_dict
                for racer in result.racers:
                    if racer.waku in course_dict:
                        racer.course_entry = course_dict[racer.waku]

    result.success = len(result.racers) >= 6
    return result


def fetch_heiwajima_odds(date_str: str, race_no: int,
                          bet_type: str = "3t",
                          day_no: int = 0) -> dict:
    """平和島公式サイトからオッズを取得する

    Args:
        date_str: "YYYYMMDD" 形式の日付
        race_no: レース番号 (1-12)
        bet_type: "3t" / "2tf" / "2kt"
        day_no: 開催日数 (0=自動検出)

    Returns:
        {"1-2-3": 12.5, ...} 形式のオッズ辞書
    """
    if day_no <= 0:
        day_candidates = _find_day_no_for_date(date_str)
    else:
        day_candidates = [day_no]

    for d in day_candidates:
        urls = _build_urls("odds", d, race_no)
        html = _fetch_with_fallback(urls)
        if html:
            soup = BeautifulSoup(html, "lxml")
            odds = _parse_odds_positional(soup, bet_type)
            if odds:
                return odds

    return {}


def fetch_heiwajima_result(date_str: str, race_no: int,
                            day_no: int = 0) -> list[int]:
    """平和島公式サイトからレース結果を取得する

    Returns:
        着順の枠番リスト [1着枠, 2着枠, 3着枠, ...]
    """
    if day_no <= 0:
        day_candidates = _find_day_no_for_date(date_str)
    else:
        day_candidates = [day_no]

    for d in day_candidates:
        urls = _build_urls("result", d, race_no)
        html = _fetch_with_fallback(urls)
        if html:
            soup = BeautifulSoup(html, "lxml")
            result = _parse_result(soup)
            if result:
                return result

    return []


def _parse_result(soup: BeautifulSoup) -> list[int]:
    """レース結果をパースする"""
    finish_order = []

    for table in soup.select("table"):
        rows = table.select("tr")
        for row in rows:
            tds = row.select("td")
            if len(tds) < 2:
                continue
            texts = [td.get_text(strip=True) for td in tds]
            try:
                rank = int(re.sub(r"[^\d]", "", texts[0]))
                waku = int(re.sub(r"[^\d]", "", texts[1]))
                if 1 <= rank <= 6 and 1 <= waku <= 6:
                    finish_order.append((rank, waku))
            except (ValueError, IndexError):
                continue

    if finish_order:
        finish_order.sort(key=lambda x: x[0])
        return [w for _, w in finish_order]

    return []


# ── 後方互換インターフェース ──

def fetch_heiwajima_supplement(date_str: str, race_no: int) -> HeiwajimaSupplement:
    """旧インターフェースとの互換性を保つ補完データ取得

    内部的に fetch_heiwajima_race を使用する。
    """
    result = HeiwajimaSupplement(race_no=race_no)

    try:
        race_info = fetch_heiwajima_race(date_str, race_no)
        if race_info.success:
            # 展示ST
            st_dict = {}
            for r in race_info.racers:
                if r.exhibit_st != 0.0:
                    st_dict[r.waku] = r.exhibit_st
            if st_dict:
                result.exhibit_st = st_dict

            # 進入コース
            if race_info.course_entries:
                result.course_entry = race_info.course_entries

            # 水面状態
            result.water_condition = race_info.weather.water_condition
            result.success = True
    except Exception:
        pass

    return result


# ── RaceInfo 変換 ──

def to_race_info(hw_race: HeiwajimaRaceInfo):
    """HeiwajimaRaceInfo を既存の RaceInfo に変換する"""
    from .race_data import RaceInfo, Racer, WeatherInfo

    racers = []
    for hr in hw_race.racers:
        r = Racer(
            waku=hr.waku,
            name=hr.name,
            register_no=hr.register_no,
            rank=hr.rank,
            branch=hr.branch,
            win_rate_all=hr.win_rate_all,
            win_rate_2r_all=hr.win_rate_2r_all,
            win_rate_local=hr.win_rate_local,
            win_rate_2r_local=hr.win_rate_2r_local,
            motor_2r=hr.motor_2r,
            boat_2r=hr.boat_2r,
            exhibit_time=hr.exhibit_time,
            tilt=hr.tilt,
        )
        racers.append(r)

    weather = WeatherInfo(
        weather=hw_race.weather.weather,
        wind_direction=hw_race.weather.wind_direction,
        wind_speed=hw_race.weather.wind_speed,
        wave_height=hw_race.weather.wave_height,
        temperature=hw_race.weather.temperature,
        water_temp=hw_race.weather.water_temp,
    )

    return RaceInfo(
        race_no=hw_race.race_no,
        race_name=hw_race.race_name,
        date=hw_race.date,
        deadline=hw_race.deadline,
        racers=racers,
        weather=weather,
    )
