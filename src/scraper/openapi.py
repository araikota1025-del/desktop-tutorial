"""BoatraceOpenAPI からの JSON データ取得

GitHub Pages で公開されている無料 JSON API:
- programs (出走表): https://boatraceopenapi.github.io/programs/v2/YYYY/YYYYMMDD.json
- previews (直前情報): https://boatraceopenapi.github.io/previews/v2/YYYY/YYYYMMDD.json
- results (結果): https://boatraceopenapi.github.io/results/v2/YYYY/YYYYMMDD.json

約30分ごとに GitHub Actions で更新される。
全24場のデータが配列で含まれるため、stadium_number == 4 でフィルタする。

確認済みスキーマ (2026年2月):
  Programs: [{stadium_number, number, boats: [{racer_boat_number, racer_name, ...}]}]
  Previews: [{stadium_number, number, wind_speed, boats: [{racer_boat_number, racer_exhibition_time, ...}]}]
  Results:  [{race_stadium_number, race_number, boats: [{racer_boat_number, racer_place_number, ...}], payouts: {...}}]

注意: オッズデータはこのAPIに含まれない。オッズは heiwajima.gr.jp から取得する。
"""

import json
import time
from typing import Optional

import requests

OPENAPI_BASE = "https://boatraceopenapi.github.io"
HEIWAJIMA_STADIUM = 4  # 平和島の場番号

# 級別番号 → 級別文字列
CLASS_MAP = {1: "A1", 2: "A2", 3: "B1", 4: "B2"}

# 天候番号 → 天候文字列
WEATHER_MAP = {1: "晴", 2: "曇り", 3: "雨", 4: "雪", 5: "霧"}

# 風向番号 → 風向文字列 (16方位の代表的なマッピング)
WIND_DIR_MAP = {
    1: "北", 2: "北北東", 3: "北東", 4: "東北東",
    5: "東", 6: "東南東", 7: "南東", 8: "南南東",
    9: "南", 10: "南南西", 11: "南西", 12: "西南西",
    13: "西", 14: "西北西", 15: "北西", 16: "北北西",
}

# レートリミット
_last_request_time = 0.0


def _fetch_json(url: str) -> Optional[list | dict]:
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


def fetch_programs(date_str: str) -> Optional[list]:
    """出走表データを取得する

    Args:
        date_str: "YYYYMMDD" 形式の日付

    Returns:
        全場の出走表データ (JSON array) or None
    """
    year = date_str[:4]
    url = f"{OPENAPI_BASE}/programs/v2/{year}/{date_str}.json"
    return _fetch_json(url)


def fetch_previews(date_str: str) -> Optional[list]:
    """直前情報を取得する"""
    year = date_str[:4]
    url = f"{OPENAPI_BASE}/previews/v2/{year}/{date_str}.json"
    return _fetch_json(url)


def fetch_results(date_str: str) -> Optional[list]:
    """レース結果を取得する"""
    year = date_str[:4]
    url = f"{OPENAPI_BASE}/results/v2/{year}/{date_str}.json"
    return _fetch_json(url)


def _filter_heiwajima(data: list | dict, race_no: int = 0,
                       is_results: bool = False) -> list[dict]:
    """JSON データから平和島のデータをフィルタする

    Args:
        data: API レスポンス (通常はリスト)
        race_no: レース番号 (0=全レース)
        is_results: Results API はフィールド名が異なる
    """
    races = []

    if not isinstance(data, list):
        return races

    # フィールド名: results は race_ プレフィックス付き
    stadium_key = "race_stadium_number" if is_results else "stadium_number"
    number_key = "race_number" if is_results else "number"

    for item in data:
        stadium = item.get(stadium_key, 0)
        if stadium == HEIWAJIMA_STADIUM:
            rno = item.get(number_key, 0)
            if race_no == 0 or rno == race_no:
                races.append(item)

    return races


def _extract_racer_from_program(boat: dict) -> dict:
    """programs の boats 要素からレーサーデータを正規化して返す

    確認済みフィールド:
        racer_boat_number, racer_name, racer_number, racer_class_number,
        racer_branch_number, racer_age, racer_weight,
        racer_flying_count, racer_late_count, racer_average_start_timing,
        racer_national_top_1_percent, racer_national_top_2_percent, racer_national_top_3_percent,
        racer_local_top_1_percent, racer_local_top_2_percent, racer_local_top_3_percent,
        racer_assigned_motor_number, racer_assigned_motor_top_2_percent, racer_assigned_motor_top_3_percent,
        racer_assigned_boat_number, racer_assigned_boat_top_2_percent, racer_assigned_boat_top_3_percent
    """
    class_num = boat.get("racer_class_number", 0)
    rank_str = CLASS_MAP.get(class_num, "")

    # 勝率: OpenAPI は「1着率(%)」を提供、既存モデルは「勝率(X.XX)」を期待
    # national_top_1_percent は百分率 (例: 45.2%) だが、
    # 勝率 (win_rate) はボートレースの成績ポイント (例: 7.50) なので異なる
    # OpenAPI にはポイント勝率がないため、top_1_percent をそのまま使う
    # (予測モデル側で正規化するので問題なし)
    return {
        "waku": int(boat.get("racer_boat_number", 0)),
        "name": str(boat.get("racer_name", "")),
        "register_no": str(boat.get("racer_number", "")),
        "rank": rank_str,
        "branch": str(boat.get("racer_branch_number", "")),
        "age": int(boat.get("racer_age", 0) or 0),
        "weight": float(boat.get("racer_weight", 0) or 0),
        # 勝率系: top_1_percent をポイント勝率の代わりに使用
        "win_rate_all": float(boat.get("racer_national_top_1_percent", 0) or 0),
        "win_rate_2r_all": float(boat.get("racer_national_top_2_percent", 0) or 0),
        "win_rate_local": float(boat.get("racer_local_top_1_percent", 0) or 0),
        "win_rate_2r_local": float(boat.get("racer_local_top_2_percent", 0) or 0),
        # 3連率 (追加データ)
        "win_rate_3r_all": float(boat.get("racer_national_top_3_percent", 0) or 0),
        "win_rate_3r_local": float(boat.get("racer_local_top_3_percent", 0) or 0),
        # モーター・ボート
        "motor_no": str(boat.get("racer_assigned_motor_number", "")),
        "motor_2r": float(boat.get("racer_assigned_motor_top_2_percent", 0) or 0),
        "motor_3r": float(boat.get("racer_assigned_motor_top_3_percent", 0) or 0),
        "boat_no": str(boat.get("racer_assigned_boat_number", "")),
        "boat_2r": float(boat.get("racer_assigned_boat_top_2_percent", 0) or 0),
        "boat_3r": float(boat.get("racer_assigned_boat_top_3_percent", 0) or 0),
        # ST関連
        "avg_start_timing": float(boat.get("racer_average_start_timing", 0) or 0),
        "flying_count": int(boat.get("racer_flying_count", 0) or 0),
        "late_count": int(boat.get("racer_late_count", 0) or 0),
        # 展示データ (programs には含まれない、previews でマージ)
        "exhibit_time": 0.0,
        "exhibit_st": 0.0,
        "tilt": 0.0,
        "course_entry": 0,
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
        "date": date_str,
        "race_name": str(race_data.get("title", race_data.get("subtitle", ""))),
        "deadline": str(race_data.get("closed_at", "")),
        "racers": [],
        "weather": {},
        "source": "openapi",
    }

    # レーサーデータ (boats 配列)
    boats_data = race_data.get("boats", [])
    for boat in boats_data:
        racer = _extract_racer_from_program(boat)
        if racer["waku"] >= 1:
            result["racers"].append(racer)

    # 直前情報を取得してマージ
    previews = fetch_previews(date_str)
    if previews:
        preview_races = _filter_heiwajima(previews, race_no)
        if preview_races:
            preview_data = preview_races[0]

            # 気象情報 (previews のレースレベルフィールド)
            weather_num = preview_data.get("weather_number", 0)
            wind_dir_num = preview_data.get("wind_direction_number", 0)
            result["weather"] = {
                "weather": WEATHER_MAP.get(weather_num, ""),
                "wind_direction": WIND_DIR_MAP.get(wind_dir_num, ""),
                "wind_speed": int(preview_data.get("wind_speed", 0) or 0),
                "wave_height": int(preview_data.get("wave_height", 0) or 0),
                "temperature": float(preview_data.get("air_temperature", 0) or 0),
                "water_temp": float(preview_data.get("water_temperature", 0) or 0),
            }

            # 展示データを各レーサーにマージ
            preview_boats = preview_data.get("boats", [])
            for pb in preview_boats:
                pb_waku = int(pb.get("racer_boat_number", 0))
                for racer in result["racers"]:
                    if racer["waku"] == pb_waku:
                        # 展示タイム
                        et = float(pb.get("racer_exhibition_time", 0) or 0)
                        if et > 0:
                            racer["exhibit_time"] = et
                        # 展示ST
                        st = float(pb.get("racer_start_timing", 0) or 0)
                        if st != 0:
                            racer["exhibit_st"] = st
                        # チルト
                        tilt = float(pb.get("racer_tilt_adjustment", 0) or 0)
                        if tilt != 0:
                            racer["tilt"] = tilt
                        # 進入コース
                        course = int(pb.get("racer_course_number", 0) or 0)
                        if course >= 1:
                            racer["course_entry"] = course
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
            course_entry=rd.get("course_entry", 0),
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


def fetch_openapi_results(date_str: str, race_no: int) -> Optional[dict]:
    """OpenAPI からレース結果を取得する

    Returns:
        {
            "race_no": 1,
            "boats": [{"waku": 1, "course": 1, "place": 1, "st": 0.15}, ...],
            "payouts": {"trifecta": [{"combination": "1-2-3", "payout": 1230}], ...},
            "weather": {...},
        }
        or None
    """
    results = fetch_results(date_str)
    if not results:
        return None

    races = _filter_heiwajima(results, race_no, is_results=True)
    if not races:
        return None

    race_data = races[0]
    result = {
        "race_no": race_no,
        "boats": [],
        "payouts": race_data.get("payouts", {}),
        "weather": {},
    }

    # 気象情報
    weather_num = race_data.get("race_weather_number", 0)
    wind_dir_num = race_data.get("race_wind_direction_number", 0)
    result["weather"] = {
        "weather": WEATHER_MAP.get(weather_num, ""),
        "wind_direction": WIND_DIR_MAP.get(wind_dir_num, ""),
        "wind_speed": int(race_data.get("race_wind", 0) or 0),
        "wave_height": int(race_data.get("race_wave", 0) or 0),
        "temperature": float(race_data.get("race_temperature", 0) or 0),
        "water_temp": float(race_data.get("race_water_temperature", 0) or 0),
    }

    # ボートデータ
    for boat in race_data.get("boats", []):
        result["boats"].append({
            "waku": int(boat.get("racer_boat_number", 0)),
            "course": int(boat.get("racer_course_number", 0)),
            "place": int(boat.get("racer_place_number", 0)),
            "st": float(boat.get("racer_start_timing", 0) or 0),
            "register_no": str(boat.get("racer_number", "")),
            "name": str(boat.get("racer_name", "")),
        })

    return result
