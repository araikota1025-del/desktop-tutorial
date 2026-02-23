"""過去のレースデータを一括収集して CSV に保存するスクリプト

Usage:
    python -m src.scraper.history_collector --months 6
    python -m src.scraper.history_collector --start 20240101 --end 20241231
"""

import argparse
import csv
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

from .client import fetch_page, CONFIG

VENUE_CODE = CONFIG["app"]["venue_code"]
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"


def fetch_race_result(date: str, race_no: int) -> dict | None:
    """1レースの結果ページから着順・払戻を取得する"""
    html = fetch_page(
        "/owpc/pc/race/raceresult",
        params={"hd": date, "jcd": VENUE_CODE, "rno": str(race_no)},
    )
    if not html:
        return None

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    result = {"date": date, "race_no": race_no}

    # 着順テーブルを取得
    tbody_list = soup.select("table.is-w495 tbody, .table1 tbody")
    finish_order = []
    for tbody in tbody_list:
        rows = tbody.select("tr")
        for row in rows:
            tds = row.select("td")
            if len(tds) < 2:
                continue
            text = " ".join(td.get_text(strip=True) for td in tds)

            # 着順と枠番を検出
            rank_match = re.search(r"^(\d)", tds[0].get_text(strip=True))
            waku_match = re.search(r"(\d)\s", text)
            if rank_match:
                finish_order.append({
                    "rank": int(rank_match.group(1)),
                    "text": text,
                })

    if finish_order:
        result["finish_order_raw"] = str(finish_order[:6])

    return result


def fetch_racelist_for_training(date: str, race_no: int) -> dict | None:
    """出走表からモデル学習用のデータを取得する"""
    html = fetch_page(
        "/owpc/pc/race/racelist",
        params={"hd": date, "jcd": VENUE_CODE, "rno": str(race_no)},
    )
    if not html:
        return None

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    race_data = {"date": date, "race_no": race_no}

    # 各艇のデータを取得（is-w748 テーブルを優先）
    racer_table = soup.select_one("table.is-w748")
    if racer_table:
        tbody_list = racer_table.select("tbody")
    else:
        tbody_list = soup.select(".table1 tbody")
    racers = []

    for tbody in tbody_list:
        tds = tbody.select("td")
        if not tds:
            continue

        # 枠番を is-boatColor クラスから検出
        waku = 0
        for td in tds:
            for cls in td.get("class", []):
                color_match = re.search(r"is-boatColor(\d)", cls)
                if color_match:
                    waku = int(color_match.group(1))
                    break
            if waku:
                break

        # フォールバック: 最初のtdが1-6の単一数字か確認
        if not waku:
            first_text = tds[0].get_text(strip=True)
            if re.match(r"^[1-6]$", first_text):
                waku = int(first_text)

        if not (1 <= waku <= 6):
            continue
        if any(r.get("waku") == waku for r in racers):
            continue

        racer = {"waku": waku}

        all_text = [td.get_text(strip=True) for td in tds]
        text_joined = " ".join(all_text)

        # 登番
        reg_match = re.search(r"(\d{4})", text_joined)
        if reg_match:
            racer["register_no"] = reg_match.group(1)

        # 級別
        rank_match = re.search(r"(A1|A2|B1|B2)", text_joined)
        if rank_match:
            racer["rank"] = rank_match.group(1)

        # 選手名
        name_el = tbody.select_one(".is-fs18, .is-fs14, a")
        if name_el:
            racer["name"] = re.sub(r"\s+", "", name_el.get_text(strip=True))

        # 勝率（小数点含む数値）
        rates = re.findall(r"(\d+\.\d{2})", text_joined)
        if len(rates) >= 2:
            racer["win_rate_all"] = float(rates[0])
            racer["win_rate_2r_all"] = float(rates[1])
        if len(rates) >= 4:
            racer["win_rate_local"] = float(rates[2])
            racer["win_rate_2r_local"] = float(rates[3])
        if len(rates) >= 6:
            racer["motor_2r"] = float(rates[4])
            racer["boat_2r"] = float(rates[5])

        racers.append(racer)

    racers.sort(key=lambda r: r["waku"])
    race_data["racers"] = racers
    return race_data


def fetch_beforeinfo_for_training(date: str, race_no: int) -> dict | None:
    """直前情報から展示タイム・気象データを取得する"""
    html = fetch_page(
        "/owpc/pc/race/beforeinfo",
        params={"hd": date, "jcd": VENUE_CODE, "rno": str(race_no)},
    )
    if not html:
        return None

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    info = {"date": date, "race_no": race_no}

    # 気象情報
    weather_section = soup.select_one(".weather1, .weatherBody")
    if weather_section:
        text = weather_section.get_text()

        w_match = re.search(r"(晴|曇り?|雨|雪|霧)", text)
        if w_match:
            info["weather"] = w_match.group(1)

        ws_match = re.search(r"(\d+)\s*m", text)
        if ws_match:
            info["wind_speed"] = int(ws_match.group(1))

        wh_match = re.search(r"(\d+)\s*cm", text)
        if wh_match:
            info["wave_height"] = int(wh_match.group(1))

        temp_match = re.search(r"気温\s*(\d+\.?\d*)", text)
        if temp_match:
            info["temperature"] = float(temp_match.group(1))

        wtemp_match = re.search(r"水温\s*(\d+\.?\d*)", text)
        if wtemp_match:
            info["water_temp"] = float(wtemp_match.group(1))

    # 展示タイム
    exhibit_times = {}
    time_pattern = re.compile(r"\d+\.\d{2}")
    rows = soup.select(".table1 tbody tr, table tbody tr")
    for row in rows:
        tds = row.select("td")
        if not tds:
            continue
        first_text = tds[0].get_text(strip=True)
        waku_match = re.search(r"^(\d)", first_text)
        times = time_pattern.findall(row.get_text())
        if waku_match and times:
            waku = int(waku_match.group(1))
            if 1 <= waku <= 6:
                exhibit_times[waku] = float(times[-1])

    info["exhibit_times"] = exhibit_times
    return info


def fetch_result_for_training(date: str, race_no: int) -> dict | None:
    """レース結果から着順を取得する"""
    html = fetch_page(
        "/owpc/pc/race/raceresult",
        params={"hd": date, "jcd": VENUE_CODE, "rno": str(race_no)},
    )
    if not html:
        return None

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    result = {"date": date, "race_no": race_no, "finish": {}}

    # 着順テーブル
    rows = soup.select("table.is-w495 tbody tr, .table1 tbody tr")
    for row in rows:
        tds = row.select("td")
        if len(tds) < 3:
            continue
        texts = [td.get_text(strip=True) for td in tds]

        # 着順(1列目) と 枠番(2列目)
        try:
            rank = int(re.sub(r"[^\d]", "", texts[0]))
            waku = int(re.sub(r"[^\d]", "", texts[1]))
            if 1 <= rank <= 6 and 1 <= waku <= 6:
                result["finish"][waku] = rank
        except (ValueError, IndexError):
            continue

    return result if result["finish"] else None


def collect_date_range(start_date: str, end_date: str, output_path: Path | None = None):
    """指定期間の平和島レースデータを収集してCSVに保存する

    Args:
        start_date: 開始日 YYYYMMDD
        end_date: 終了日 YYYYMMDD
        output_path: 出力CSVパス
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = DATA_DIR / f"heiwajima_{start_date}_{end_date}.csv"

    fieldnames = [
        "date", "race_no",
        # 着順結果
        "finish_1st", "finish_2nd", "finish_3rd",
        # 気象
        "weather", "wind_speed", "wave_height", "temperature", "water_temp",
        # 各艇データ (1-6号艇)
    ]
    for w in range(1, 7):
        fieldnames.extend([
            f"w{w}_name", f"w{w}_register_no", f"w{w}_rank",
            f"w{w}_win_rate_all", f"w{w}_win_rate_2r_all",
            f"w{w}_win_rate_local", f"w{w}_win_rate_2r_local",
            f"w{w}_motor_2r", f"w{w}_boat_2r",
            f"w{w}_exhibit_time",
        ])

    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    current = start

    rows_collected = 0

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        while current <= end:
            date_str = current.strftime("%Y%m%d")
            print(f"[{date_str}] 収集中...")

            for race_no in range(1, 13):
                row = {"date": date_str, "race_no": race_no}

                # 出走表
                racelist = fetch_racelist_for_training(date_str, race_no)
                if not racelist or not racelist.get("racers"):
                    continue  # この日は開催なし

                for racer in racelist["racers"]:
                    w = racer["waku"]
                    row[f"w{w}_name"] = racer.get("name", "")
                    row[f"w{w}_register_no"] = racer.get("register_no", "")
                    row[f"w{w}_rank"] = racer.get("rank", "")
                    row[f"w{w}_win_rate_all"] = racer.get("win_rate_all", 0)
                    row[f"w{w}_win_rate_2r_all"] = racer.get("win_rate_2r_all", 0)
                    row[f"w{w}_win_rate_local"] = racer.get("win_rate_local", 0)
                    row[f"w{w}_win_rate_2r_local"] = racer.get("win_rate_2r_local", 0)
                    row[f"w{w}_motor_2r"] = racer.get("motor_2r", 0)
                    row[f"w{w}_boat_2r"] = racer.get("boat_2r", 0)

                # 直前情報
                before = fetch_beforeinfo_for_training(date_str, race_no)
                if before:
                    row["weather"] = before.get("weather", "")
                    row["wind_speed"] = before.get("wind_speed", 0)
                    row["wave_height"] = before.get("wave_height", 0)
                    row["temperature"] = before.get("temperature", 0)
                    row["water_temp"] = before.get("water_temp", 0)
                    for waku, et in before.get("exhibit_times", {}).items():
                        row[f"w{waku}_exhibit_time"] = et

                # レース結果
                result = fetch_result_for_training(date_str, race_no)
                if result and result.get("finish"):
                    finish = result["finish"]
                    # 着順 → 枠番の逆引き
                    rank_to_waku = {v: k for k, v in finish.items()}
                    row["finish_1st"] = rank_to_waku.get(1, 0)
                    row["finish_2nd"] = rank_to_waku.get(2, 0)
                    row["finish_3rd"] = rank_to_waku.get(3, 0)
                else:
                    continue  # 結果がなければスキップ

                writer.writerow(row)
                rows_collected += 1

            current += timedelta(days=1)

    print(f"\n収集完了: {rows_collected} レース → {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="平和島レースデータ収集")
    parser.add_argument("--start", type=str, help="開始日 YYYYMMDD")
    parser.add_argument("--end", type=str, help="終了日 YYYYMMDD")
    parser.add_argument("--months", type=int, default=6, help="直近N ヶ月分を収集")
    args = parser.parse_args()

    if args.start and args.end:
        start_date = args.start
        end_date = args.end
    else:
        end = datetime.now()
        start = end - timedelta(days=30 * args.months)
        start_date = start.strftime("%Y%m%d")
        end_date = end.strftime("%Y%m%d")

    print(f"平和島レースデータ収集: {start_date} ～ {end_date}")
    print(f"注意: boatrace.jp への負荷を抑えるため、1リクエストごとに1.5秒の間隔を空けます")
    print(f"      12R × 3ページ × 1.5秒 = 1日あたり約54秒かかります")
    print()

    collect_date_range(start_date, end_date)


if __name__ == "__main__":
    main()
