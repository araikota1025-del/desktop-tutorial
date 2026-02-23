"""BoatraceOpenAPI からの JSON データ取得

GitHub Pages で公開されている無料 JSON API:
- programs (出走表): https://boatraceopenapi.github.io/programs/v2/YYYY/YYYYMMDD.json
- previews (直前情報): https://boatraceopenapi.github.io/previews/v2/YYYY/YYYYMMDD.json
- results (結果): https://boatraceopenapi.github.io/results/v2/YYYY/YYYYMMDD.json

約30分ごとに GitHub Actions で更新される。
全24場のデータが含まれるため、venue_code (jyo) = "04" でフィルタする。

利点:
- JSON 形式で構造化されたデータ
- HTML パースが不要
- boatrace.jp へのスクレイピング負荷なし
- 安定した URL パターン
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

OPENAPI_BASE = "https://boatraceopenapi.github.io"
HEIWAJIMA_JYO = "04"

# レートリミット
_last_request_time = 0.0


def _fetch_json(url: str) -> Optional[dict]:
    """OpenAPI から JSON を取得する"""
    global _last_request_time

    # レートリミット（1秒間隔）
    elapsed = time.time() - _last_request_time
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)

    headers = {
        "User-Agent": "BoatRaceHeiwajima-Predictor/1.0",
        "Accept": "application/json",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        _last_request_time = time.time()
        if resp.status_code == 200:
            return resp.json()
    except (requests.RequestException, json.JSONDecodeError):
        pass

    return None


def fetch_programs(date_str: str) -> Optional[dict]:
    """出走表データを取得する

    Args:
        date_str: "YYYYMMDD" 形式の日付

    Returns:
        全場の出走表データ (JSON dict) or None
    """
    year = date_str[:4]
    url = f"{OPENAPI_BASE}/programs/v2/{year}/{date_str}.json"
    return _fetch_json(url)


def fetch_previews(date_str: str) -> Optional[dict]:
    """直前情報を取得する"""
    year = date_str[:4]
    url = f"{OPENAPI_BASE}/previews/v2/{year}/{date_str}.json"
    return _fetch_json(url)


def fetch_results(date_str: str) -> Optional[dict]:
    """レース結果を取得する"""
    year = date_str[:4]
    url = f"{OPENAPI_BASE}/results/v2/{year}/{date_str}.json"
    return _fetch_json(url)


def _filter_heiwajima(data: dict, race_no: int = 0) -> list[dict]:
    """JSON データから平和島のデータをフィルタする

    OpenAPI のデータ構造は複数のパターンがありうるため、
    柔軟に対応する。
    """
    races = []

    # パターン1: トップレベルがリスト
    if isinstance(data, list):
        for item in data:
            jyo = str(item.get("jyo", item.get("jcd", item.get("venue", ""))))
            if jyo == HEIWAJIMA_JYO or jyo == "平和島" or jyo == "heiwajima":
                rno = int(item.get("raceNumber", item.get("rno", item.get("race_no", 0))))
                if race_no == 0 or rno == race_no:
                    races.append(item)

    # パターン2: 場コードがキー
    elif isinstance(data, dict):
        # {"04": [...]} or {"04": {"races": [...]}}
        venue_data = data.get(HEIWAJIMA_JYO) or data.get("04") or data.get("heiwajima")
        if venue_data:
            if isinstance(venue_data, list):
                for item in venue_data:
                    rno = int(item.get("raceNumber", item.get("rno", item.get("race_no", 0))))
                    if race_no == 0 or rno == race_no:
                        races.append(item)
            elif isinstance(venue_data, dict):
                race_list = venue_data.get("races", venue_data.get("raceList", []))
                if isinstance(race_list, list):
                    for item in race_list:
                        rno = int(item.get("raceNumber", item.get("rno", 0)))
                        if race_no == 0 or rno == race_no:
                            races.append(item)
                elif race_no == 0:
                    races.append(venue_data)

        # パターン3: "races" キーがトップレベル
        if not races and "races" in data:
            for item in data["races"]:
                jyo = str(item.get("jyo", item.get("jcd", "")))
                if jyo == HEIWAJIMA_JYO:
                    rno = int(item.get("raceNumber", item.get("rno", 0)))
                    if race_no == 0 or rno == race_no:
                        races.append(item)

    return races


def _extract_racer_data(racer_entry: dict) -> dict:
    """レーサーデータを正規化して返す"""
    # OpenAPI のフィールド名は複数パターンがありうる
    waku = int(
        racer_entry.get("waku", 0) or
        racer_entry.get("lane", 0) or
        racer_entry.get("number", 0) or 0
    )

    return {
        "waku": waku,
        "name": str(racer_entry.get("name", racer_entry.get("racer_name", ""))),
        "register_no": str(racer_entry.get("id", racer_entry.get("toban", racer_entry.get("register_no", "")))),
        "rank": str(racer_entry.get("class", racer_entry.get("rank", racer_entry.get("grade", "")))),
        "branch": str(racer_entry.get("branch", racer_entry.get("area", ""))),
        "age": int(racer_entry.get("age", 0) or 0),
        "weight": float(racer_entry.get("weight", 0) or 0),
        "win_rate_all": float(racer_entry.get("national_win_rate", racer_entry.get("win_rate", racer_entry.get("allWinRate", 0))) or 0),
        "win_rate_2r_all": float(racer_entry.get("national_2r_rate", racer_entry.get("win2_rate", racer_entry.get("allRenritsu", 0))) or 0),
        "win_rate_local": float(racer_entry.get("local_win_rate", racer_entry.get("here_win_rate", racer_entry.get("localWinRate", 0))) or 0),
        "win_rate_2r_local": float(racer_entry.get("local_2r_rate", racer_entry.get("here_win2_rate", racer_entry.get("localRenritsu", 0))) or 0),
        "motor_no": str(racer_entry.get("motor_no", racer_entry.get("motorNo", ""))),
        "motor_2r": float(racer_entry.get("motor_2r_rate", racer_entry.get("motor_win2", racer_entry.get("motorRenritsu", 0))) or 0),
        "motor_3r": float(racer_entry.get("motor_3r_rate", racer_entry.get("motor_win3", 0)) or 0),
        "boat_no": str(racer_entry.get("boat_no", racer_entry.get("boatNo", ""))),
        "boat_2r": float(racer_entry.get("boat_2r_rate", racer_entry.get("boat_win2", racer_entry.get("boatRenritsu", 0))) or 0),
        "boat_3r": float(racer_entry.get("boat_3r_rate", racer_entry.get("boat_win3", 0)) or 0),
        "exhibit_time": float(racer_entry.get("exhibit_time", racer_entry.get("exhibitionTime", 0)) or 0),
        "exhibit_st": float(racer_entry.get("exhibit_st", racer_entry.get("startExhibition", 0)) or 0),
        "tilt": float(racer_entry.get("tilt", 0) or 0),
        "flying_count": int(racer_entry.get("flying", racer_entry.get("f_count", 0)) or 0),
        "late_count": int(racer_entry.get("late", racer_entry.get("l_count", 0)) or 0),
        "avg_start_timing": float(racer_entry.get("avg_st", racer_entry.get("averageST", 0)) or 0),
    }


def fetch_openapi_race(date_str: str, race_no: int) -> Optional[dict]:
    """OpenAPI から1レースの出走表 + 直前情報を取得する

    Returns:
        {
            "race_no": 1,
            "race_name": "...",
            "deadline": "14:30",
            "racers": [{"waku": 1, "name": "...", ...}, ...],
            "weather": {"weather": "晴", "wind_speed": 3, ...},
            "source": "openapi",
        }
        or None
    """
    # 出走表を取得
    programs = fetch_programs(date_str)
    if not programs:
        return None

    races = _filter_heiwajima(programs, race_no)
    if not races:
        return None

    race_data = races[0]
    result = {
        "race_no": race_no,
        "race_name": str(race_data.get("raceTitle", race_data.get("race_name", race_data.get("title", "")))),
        "deadline": str(race_data.get("deadline", race_data.get("closingTime", ""))),
        "racers": [],
        "weather": {},
        "source": "openapi",
    }

    # レーサーデータ
    racers_data = (
        race_data.get("racers", []) or
        race_data.get("players", []) or
        race_data.get("entries", []) or
        race_data.get("racer_list", []) or []
    )

    for entry in racers_data:
        racer = _extract_racer_data(entry)
        if racer["waku"] >= 1:
            result["racers"].append(racer)

    # 直前情報を取得してマージ
    previews = fetch_previews(date_str)
    if previews:
        preview_races = _filter_heiwajima(previews, race_no)
        if preview_races:
            preview_data = preview_races[0]

            # 気象情報
            weather = preview_data.get("weather", preview_data.get("condition", {}))
            if isinstance(weather, dict):
                result["weather"] = {
                    "weather": str(weather.get("weather", weather.get("type", ""))),
                    "wind_direction": str(weather.get("wind_direction", weather.get("windDirection", ""))),
                    "wind_speed": int(weather.get("wind_speed", weather.get("windSpeed", 0)) or 0),
                    "wave_height": int(weather.get("wave_height", weather.get("waveHeight", 0)) or 0),
                    "temperature": float(weather.get("temperature", weather.get("temp", 0)) or 0),
                    "water_temp": float(weather.get("water_temp", weather.get("waterTemp", 0)) or 0),
                }
            elif isinstance(weather, str):
                result["weather"] = {"weather": weather}

            # 展示データを各レーサーにマージ
            preview_racers = (
                preview_data.get("racers", []) or
                preview_data.get("players", []) or
                preview_data.get("entries", []) or []
            )
            for pr in preview_racers:
                pr_waku = int(pr.get("waku", pr.get("lane", 0)) or 0)
                for racer in result["racers"]:
                    if racer["waku"] == pr_waku:
                        # 展示タイム
                        et = float(pr.get("exhibit_time", pr.get("exhibitionTime", 0)) or 0)
                        if et > 0:
                            racer["exhibit_time"] = et
                        # 展示ST
                        st = float(pr.get("exhibit_st", pr.get("startExhibition", 0)) or 0)
                        if st != 0:
                            racer["exhibit_st"] = st
                        break

    # レーサーが6人いるか確認
    if len(result["racers"]) >= 6:
        result["racers"].sort(key=lambda r: r["waku"])
        return result

    return None


def openapi_to_race_info(openapi_data: dict):
    """OpenAPI データを既存の RaceInfo に変換する"""
    from .race_data import RaceInfo, Racer, WeatherInfo

    racers = []
    for rd in openapi_data.get("racers", []):
        r = Racer(
            waku=rd["waku"],
            name=rd.get("name", ""),
            register_no=rd.get("register_no", ""),
            rank=rd.get("rank", ""),
            branch=rd.get("branch", ""),
            age=rd.get("age", 0),
            weight=rd.get("weight", 0.0),
            win_rate_all=rd.get("win_rate_all", 0.0),
            win_rate_2r_all=rd.get("win_rate_2r_all", 0.0),
            win_rate_local=rd.get("win_rate_local", 0.0),
            win_rate_2r_local=rd.get("win_rate_2r_local", 0.0),
            motor_2r=rd.get("motor_2r", 0.0),
            motor_3r=rd.get("motor_3r", 0.0),
            boat_2r=rd.get("boat_2r", 0.0),
            boat_3r=rd.get("boat_3r", 0.0),
            exhibit_time=rd.get("exhibit_time", 0.0),
            exhibit_st=rd.get("exhibit_st", 0.0),
            tilt=rd.get("tilt", 0.0),
            avg_start_timing=rd.get("avg_start_timing", 0.0),
            flying_count=rd.get("flying_count", 0),
            late_count=rd.get("late_count", 0),
        )
        racers.append(r)

    wd = openapi_data.get("weather", {})
    weather = WeatherInfo(
        weather=wd.get("weather", ""),
        wind_direction=wd.get("wind_direction", ""),
        wind_speed=wd.get("wind_speed", 0),
        wave_height=wd.get("wave_height", 0),
        temperature=wd.get("temperature", 0.0),
        water_temp=wd.get("water_temp", 0.0),
    )

    return RaceInfo(
        race_no=openapi_data.get("race_no", 0),
        race_name=openapi_data.get("race_name", ""),
        date=openapi_data.get("date", ""),
        deadline=openapi_data.get("deadline", ""),
        racers=racers,
        weather=weather,
    )
