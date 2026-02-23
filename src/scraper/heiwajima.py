"""平和島ボートレース公式サイト (heiwajima.gr.jp) からの追加データ取得

boatrace.jp では取得しにくい以下のデータを補完する:
- スタート展示タイミング (ST)
- 進入コース（展示航走での枠番変更）
- 選手コメント
- 平和島固有のコンディション情報

URL パターン:
  出走表: /asp/heiwajima/sp/kyogi/kyogihtml/index.htm
  ※ iframe 内に日付・レース番号ごとの子ページが埋め込まれている想定
"""

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup
from .client import fetch_heiwajima_page

HEIWAJIMA_BASE = "https://www.heiwajima.gr.jp"


@dataclass
class HeiwajimaSupplement:
    """boatrace.jp のデータを補完する平和島固有情報"""
    race_no: int = 0
    # 各艇の展示ST {waku: timing_sec}（マイナスならフライング気味）
    exhibit_st: dict[int, float] | None = None
    # 各艇の進入コース {waku: course}（1〜6、枠なり=waku==course）
    course_entry: dict[int, int] | None = None
    # 追加の気象・水面情報
    water_condition: str = ""
    # データ取得成功フラグ
    success: bool = False


def _try_urls(date_str: str, race_no: int) -> str | None:
    """平和島公式サイトの複数のURLパターンを試行する"""
    # 日付フォーマットのバリエーション
    # YYYYMMDD -> YYMMDD, MM/DD, etc.
    ymd = date_str  # 20260223
    ymd_short = date_str[2:]  # 260223
    y = date_str[:4]
    m = date_str[4:6]
    d = date_str[6:8]
    rno = str(race_no).zfill(2)

    url_patterns = [
        # パターン1: index.htm にパラメータ
        f"{HEIWAJIMA_BASE}/asp/heiwajima/sp/kyogi/kyogihtml/index.htm",
        # パターン2: 日付ディレクトリ + レース番号
        f"{HEIWAJIMA_BASE}/asp/heiwajima/sp/kyogi/kyogihtml/{ymd}/{rno}.htm",
        f"{HEIWAJIMA_BASE}/asp/heiwajima/sp/kyogi/kyogihtml/{ymd_short}/{rno}.htm",
        # パターン3: 番組表直リンク
        f"{HEIWAJIMA_BASE}/asp/heiwajima/sp/kyogi/bangumi/{ymd}_{rno}.htm",
        # パターン4: 直前情報
        f"{HEIWAJIMA_BASE}/asp/heiwajima/sp/kyogi/chokuzen/{ymd}_{rno}.htm",
        # パターン5: よくあるCGIパターン
        f"{HEIWAJIMA_BASE}/asp/heiwajima/sp/kyogi/kyogihtml/race.asp?day={ymd}&race={race_no}",
    ]

    for url in url_patterns:
        html = fetch_heiwajima_page(url)
        if html and len(html) > 500:
            return html
    return None


def _parse_exhibit_st(soup: BeautifulSoup) -> dict[int, float] | None:
    """展示スタートタイミングをパースする"""
    st_dict: dict[int, float] = {}

    # "ST" や "スタート" を含むテーブルセクションを探す
    for table in soup.select("table"):
        text = table.get_text()
        if "ST" not in text and "スタート" not in text:
            continue

        rows = table.select("tr")
        for row in rows:
            tds = row.select("td, th")
            if len(tds) < 2:
                continue
            row_text = row.get_text()
            # 枠番を検出
            waku_match = re.search(r"([1-6])\s*号?艇?", row_text)
            # タイミングを検出（0.12 や .15 や F.12 形式）
            st_match = re.search(r"[F.]?\s*(\d*\.?\d{2})", row_text)
            if waku_match and st_match:
                waku = int(waku_match.group(1))
                timing = float(st_match.group(1)) if st_match.group(1) else 0.0
                # F表記はフライング（マイナス値）
                if "F" in row_text[:row_text.find(st_match.group(0))]:
                    timing = -timing
                st_dict[waku] = timing

    return st_dict if st_dict else None


def _parse_course_entry(soup: BeautifulSoup) -> dict[int, int] | None:
    """進入コース（展示航走）をパースする"""
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
            # 6つのtdにそれぞれ艇番が入っている場合
            entries = []
            for td in tds[:6]:
                num = re.search(r"([1-6])", td.get_text(strip=True))
                if num:
                    entries.append(int(num.group(1)))
            if len(entries) == 6:
                for course, waku in enumerate(entries, 1):
                    course_dict[waku] = course
                break

    return course_dict if len(course_dict) == 6 else None


def fetch_heiwajima_supplement(date_str: str, race_no: int) -> HeiwajimaSupplement:
    """平和島公式サイトから補完データを取得する

    boatrace.jp のデータだけでは不足するST展示や進入コース等を取得する。
    取得に失敗しても空の HeiwajimaSupplement を返す（フォールバック安全）。
    """
    result = HeiwajimaSupplement(race_no=race_no)

    html = _try_urls(date_str, race_no)
    if not html:
        return result

    soup = BeautifulSoup(html, "lxml")

    result.exhibit_st = _parse_exhibit_st(soup)
    result.course_entry = _parse_course_entry(soup)

    # 水面コンディション
    for el in soup.select("div, p, span, td"):
        text = el.get_text(strip=True)
        if any(kw in text for kw in ["水面", "うねり", "安定", "荒れ"]):
            if 5 < len(text) < 100:
                result.water_condition = text
                break

    result.success = bool(result.exhibit_st or result.course_entry)
    return result
