"""過去のレースデータを一括収集して CSV に保存するスクリプト

データソース:
1. BoatraceOpenAPI (GitHub Pages JSON) - 安定・高速（推奨）
2. boatrace.jp (HTMLスクレイピング) - フォールバック

Usage:
    python -m src.scraper.history_collector --months 6
    python -m src.scraper.history_collector --start 20240101 --end 20241231
    python -m src.scraper.history_collector --source openapi --months 3
"""

import argparse
import csv
import re
from datetime import datetime, timedelta
from pathlib import Path

from .client import fetch_page, CONFIG

VENUE_CODE = CONFIG["app"]["venue_code"]
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"

# CSV フィールド名
FIELDNAMES = [
    "date", "race_no",
    "finish_1st", "finish_2nd", "finish_3rd",
    "finish_4th", "finish_5th", "finish_6th",
    "weather", "wind_direction", "wind_speed", "wave_height",
    "temperature", "water_temp",
]
for _w in range(1, 7):
    FIELDNAMES.extend([
        f"w{_w}_name", f"w{_w}_register_no", f"w{_w}_rank",
        f"w{_w}_win_rate_all", f"w{_w}_win_rate_2r_all",
        f"w{_w}_win_rate_local", f"w{_w}_win_rate_2r_local",
        f"w{_w}_motor_no", f"w{_w}_motor_2r", f"w{_w}_motor_3r",
        f"w{_w}_boat_no", f"w{_w}_boat_2r", f"w{_w}_boat_3r",
        f"w{_w}_exhibit_time", f"w{_w}_exhibit_st",
        f"w{_w}_avg_start_timing",
        f"w{_w}_flying_count", f"w{_w}_late_count",
        f"w{_w}_tilt",
        f"w{_w}_course_entry", f"w{_w}_start_timing",
        f"w{_w}_finish_place",
    ])


# ── OpenAPI ベースの収集（高速・安定）──

def collect_via_openapi(start_date: str, end_date: str,
                        output_path: Path | None = None) -> Path:
    """OpenAPI から過去データを収集する（推奨）

    1日あたり3リクエスト（programs + previews + results）で済むため、
    boatrace.jp スクレイピングより圧倒的に高速。
    """
    from .openapi import (
        fetch_programs, fetch_previews, fetch_results,
        _filter_heiwajima, CLASS_MAP, WEATHER_MAP, WIND_DIR_MAP,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = DATA_DIR / f"heiwajima_{start_date}_{end_date}.csv"

    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    current = start
    rows_collected = 0

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()

        while current <= end:
            date_str = current.strftime("%Y%m%d")
            programs = fetch_programs(date_str)
            previews = fetch_previews(date_str)
            results_data = fetch_results(date_str)

            if not programs and not results_data:
                current += timedelta(days=1)
                continue

            hw_programs = _filter_heiwajima(programs or [], 0) if programs else []
            hw_previews = _filter_heiwajima(previews or [], 0) if previews else []
            hw_results = _filter_heiwajima(
                results_data or [], 0, is_results=True
            ) if results_data else []

            if not hw_programs and not hw_results:
                current += timedelta(days=1)
                continue

            preview_map = {p.get("number", 0): p for p in hw_previews}
            result_map = {r.get("race_number", 0): r for r in hw_results}

            race_count = 0
            for prog in hw_programs:
                rno = prog.get("number", 0)
                if not rno:
                    continue
                row = {"date": date_str, "race_no": rno}

                res = result_map.get(rno)
                if not res:
                    continue

                place_labels = ["1st", "2nd", "3rd", "4th", "5th", "6th"]
                for rb in res.get("boats", []):
                    waku = int(rb.get("racer_boat_number", 0))
                    place = int(rb.get("racer_place_number", 0))
                    if 1 <= waku <= 6 and 1 <= place <= 6:
                        row[f"w{waku}_finish_place"] = place
                        row[f"w{waku}_course_entry"] = int(
                            rb.get("racer_course_number", 0)
                        )
                        row[f"w{waku}_start_timing"] = float(
                            rb.get("racer_start_timing", 0) or 0
                        )
                        row[f"finish_{place_labels[place - 1]}"] = waku

                if not row.get("finish_1st"):
                    continue

                weather_num = res.get("race_weather_number", 0)
                wind_dir_num = res.get("race_wind_direction_number", 0)
                row["weather"] = WEATHER_MAP.get(weather_num, "")
                row["wind_direction"] = WIND_DIR_MAP.get(wind_dir_num, "")
                row["wind_speed"] = int(res.get("race_wind", 0) or 0)
                row["wave_height"] = int(res.get("race_wave", 0) or 0)
                row["temperature"] = float(res.get("race_temperature", 0) or 0)
                row["water_temp"] = float(
                    res.get("race_water_temperature", 0) or 0
                )

                for boat in prog.get("boats", []):
                    waku = int(boat.get("racer_boat_number", 0))
                    if not (1 <= waku <= 6):
                        continue
                    p = f"w{waku}_"
                    row[f"{p}name"] = str(boat.get("racer_name", ""))
                    row[f"{p}register_no"] = str(boat.get("racer_number", ""))
                    class_num = boat.get("racer_class_number", 0)
                    row[f"{p}rank"] = CLASS_MAP.get(class_num, "")
                    row[f"{p}win_rate_all"] = float(
                        boat.get("racer_national_top_1_percent", 0) or 0
                    )
                    row[f"{p}win_rate_2r_all"] = float(
                        boat.get("racer_national_top_2_percent", 0) or 0
                    )
                    row[f"{p}win_rate_local"] = float(
                        boat.get("racer_local_top_1_percent", 0) or 0
                    )
                    row[f"{p}win_rate_2r_local"] = float(
                        boat.get("racer_local_top_2_percent", 0) or 0
                    )
                    row[f"{p}motor_no"] = str(
                        boat.get("racer_assigned_motor_number", "")
                    )
                    row[f"{p}motor_2r"] = float(
                        boat.get("racer_assigned_motor_top_2_percent", 0) or 0
                    )
                    row[f"{p}motor_3r"] = float(
                        boat.get("racer_assigned_motor_top_3_percent", 0) or 0
                    )
                    row[f"{p}boat_no"] = str(
                        boat.get("racer_assigned_boat_number", "")
                    )
                    row[f"{p}boat_2r"] = float(
                        boat.get("racer_assigned_boat_top_2_percent", 0) or 0
                    )
                    row[f"{p}boat_3r"] = float(
                        boat.get("racer_assigned_boat_top_3_percent", 0) or 0
                    )
                    row[f"{p}avg_start_timing"] = float(
                        boat.get("racer_average_start_timing", 0) or 0
                    )
                    row[f"{p}flying_count"] = int(
                        boat.get("racer_flying_count", 0) or 0
                    )
                    row[f"{p}late_count"] = int(
                        boat.get("racer_late_count", 0) or 0
                    )

                prev = preview_map.get(rno)
                if prev:
                    for pb in prev.get("boats", []):
                        waku = int(pb.get("racer_boat_number", 0))
                        if not (1 <= waku <= 6):
                            continue
                        p = f"w{waku}_"
                        et = float(pb.get("racer_exhibition_time", 0) or 0)
                        if et > 0:
                            row[f"{p}exhibit_time"] = et
                        st = float(pb.get("racer_start_timing", 0) or 0)
                        if st != 0:
                            row[f"{p}exhibit_st"] = st
                        row[f"{p}tilt"] = float(
                            pb.get("racer_tilt_adjustment", 0) or 0
                        )

                writer.writerow(row)
                rows_collected += 1
                race_count += 1

            if race_count:
                print(f"[{date_str}] {race_count} レース収集")
            current += timedelta(days=1)

    print(f"\n収集完了: {rows_collected} レース → {output_path}")
    return output_path


# ── boatrace.jp スクレイピング ベースの収集（フォールバック）──

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
    tbody_list = soup.select(".table1 tbody")
    racers = []
    for tbody in tbody_list:
        tds = tbody.select("td")
        if not tds:
            continue
        all_text = [td.get_text(strip=True) for td in tds]
        text_joined = " ".join(all_text)
        rates = re.findall(r"(\d+\.\d{2})", text_joined)
        has_reg_no = bool(re.search(r"\d{4}", text_joined))
        has_rank = bool(re.search(r"(A1|A2|B1|B2)", text_joined))
        if len(rates) < 2 or not (has_reg_no or has_rank):
            continue
        waku = 0
        for td in tds:
            for cls in td.get("class", []):
                color_match = re.search(r"is-boatColor(\d)", cls)
                if color_match:
                    waku = int(color_match.group(1))
                    break
            if waku:
                break
        if not waku:
            first_text = tds[0].get_text(strip=True)
            if re.match(r"^[1-6]$", first_text):
                waku = int(first_text)
        if not (1 <= waku <= 6) or any(r.get("waku") == waku for r in racers):
            continue
        racer = {"waku": waku}
        reg_match = re.search(r"(\d{4})", text_joined)
        if reg_match:
            racer["register_no"] = reg_match.group(1)
        rank_match = re.search(r"(A1|A2|B1|B2)", text_joined)
        if rank_match:
            racer["rank"] = rank_match.group(1)
        name_el = tbody.select_one(".is-fs18, .is-fs14, a")
        if name_el:
            racer["name"] = re.sub(r"\s+", "", name_el.get_text(strip=True))
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
    rows = soup.select("table.is-w495 tbody tr, .table1 tbody tr")
    for row in rows:
        tds = row.select("td")
        if len(tds) < 3:
            continue
        texts = [td.get_text(strip=True) for td in tds]
        try:
            rank = int(re.sub(r"[^\d]", "", texts[0]))
            waku = int(re.sub(r"[^\d]", "", texts[1]))
            if 1 <= rank <= 6 and 1 <= waku <= 6:
                result["finish"][waku] = rank
        except (ValueError, IndexError):
            continue
    return result if result["finish"] else None


def collect_via_scraping(start_date: str, end_date: str,
                         output_path: Path | None = None) -> Path:
    """boatrace.jp スクレイピングで過去データを収集する（フォールバック）"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = DATA_DIR / f"heiwajima_{start_date}_{end_date}.csv"
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    current = start
    rows_collected = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        while current <= end:
            date_str = current.strftime("%Y%m%d")
            print(f"[{date_str}] 収集中...")
            for race_no in range(1, 13):
                row = {"date": date_str, "race_no": race_no}
                racelist = fetch_racelist_for_training(date_str, race_no)
                if not racelist or not racelist.get("racers"):
                    continue
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
                before = fetch_beforeinfo_for_training(date_str, race_no)
                if before:
                    row["weather"] = before.get("weather", "")
                    row["wind_speed"] = before.get("wind_speed", 0)
                    row["wave_height"] = before.get("wave_height", 0)
                    row["temperature"] = before.get("temperature", 0)
                    row["water_temp"] = before.get("water_temp", 0)
                    for waku, et in before.get("exhibit_times", {}).items():
                        row[f"w{waku}_exhibit_time"] = et
                result = fetch_result_for_training(date_str, race_no)
                if result and result.get("finish"):
                    finish = result["finish"]
                    rank_to_waku = {v: k for k, v in finish.items()}
                    row["finish_1st"] = rank_to_waku.get(1, 0)
                    row["finish_2nd"] = rank_to_waku.get(2, 0)
                    row["finish_3rd"] = rank_to_waku.get(3, 0)
                else:
                    continue
                writer.writerow(row)
                rows_collected += 1
            current += timedelta(days=1)
    print(f"\n収集完了: {rows_collected} レース → {output_path}")
    return output_path


def collect_date_range(start_date: str, end_date: str,
                       output_path: Path | None = None,
                       source: str = "openapi") -> Path:
    """指定期間の平和島レースデータを収集してCSVに保存する"""
    if source == "openapi":
        return collect_via_openapi(start_date, end_date, output_path)
    return collect_via_scraping(start_date, end_date, output_path)


def main():
    parser = argparse.ArgumentParser(description="平和島レースデータ収集")
    parser.add_argument("--start", type=str, help="開始日 YYYYMMDD")
    parser.add_argument("--end", type=str, help="終了日 YYYYMMDD")
    parser.add_argument("--months", type=int, default=6, help="直近Nヶ月分を収集")
    parser.add_argument(
        "--source", type=str, default="openapi",
        choices=["openapi", "scraping"],
        help="データソース (openapi=推奨, scraping=フォールバック)",
    )
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
    print(f"データソース: {args.source}")
    if args.source == "openapi":
        print("OpenAPI: 1日あたり3リクエスト（高速）")
    else:
        print("注意: boatrace.jp への負荷を抑えるため、"
              "1リクエストごとに1.5秒の間隔を空けます")
    print()
    collect_date_range(start_date, end_date, source=args.source)


if __name__ == "__main__":
    main()
