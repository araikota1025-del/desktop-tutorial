"""boatrace.jp / heiwajima.gr.jp へのHTTPリクエストを管理する共通クライアント"""

import time
import requests
import yaml
from pathlib import Path


def load_config():
    config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "app": {"base_url": "https://www.boatrace.jp", "venue_code": "04"},
        "scraper": {
            "request_interval_sec": 1.5,
            "timeout_sec": 15,
            "user_agent": "Mozilla/5.0",
            "max_retries": 3,
        },
    }


CONFIG = load_config()
_last_request_time = 0.0
_last_heiwajima_request_time = 0.0

# ブラウザに近いヘッダー（heiwajima.gr.jp 用）
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Cache-Control": "max-age=0",
}

# heiwajima.gr.jp 用セッション（Cookie を保持）
_heiwajima_session: requests.Session | None = None


def fetch_page(path: str, params: dict | None = None) -> str | None:
    """boatrace.jp からHTMLを取得する（レートリミット付き）"""
    global _last_request_time

    cfg = CONFIG["scraper"]
    url = CONFIG["app"]["base_url"] + path

    # レートリミット
    elapsed = time.time() - _last_request_time
    wait = cfg["request_interval_sec"] - elapsed
    if wait > 0:
        time.sleep(wait)

    headers = {"User-Agent": cfg["user_agent"]}

    for attempt in range(cfg["max_retries"]):
        try:
            resp = requests.get(
                url, params=params, headers=headers, timeout=cfg["timeout_sec"]
            )
            _last_request_time = time.time()
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding
            return resp.text
        except requests.RequestException:
            if attempt < cfg["max_retries"] - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                return None
    return None


def _get_heiwajima_session() -> requests.Session:
    """heiwajima.gr.jp 用の Session を取得する（Cookie 永続化）"""
    global _heiwajima_session
    if _heiwajima_session is None:
        _heiwajima_session = requests.Session()
        _heiwajima_session.headers.update(_BROWSER_HEADERS)
    return _heiwajima_session


def fetch_heiwajima_page(url: str) -> str | None:
    """heiwajima.gr.jp からHTMLを取得する

    ブラウザライクなヘッダーとセッションCookieを使い、
    403を回避する。
    """
    global _last_heiwajima_request_time

    cfg = CONFIG["scraper"]

    # レートリミット（2秒間隔）
    elapsed = time.time() - _last_heiwajima_request_time
    wait = 2.0 - elapsed
    if wait > 0:
        time.sleep(wait)

    session = _get_heiwajima_session()

    for attempt in range(cfg["max_retries"]):
        try:
            resp = session.get(url, timeout=cfg["timeout_sec"])
            _last_heiwajima_request_time = time.time()
            resp.raise_for_status()
            # Shift_JIS の場合も考慮
            if resp.encoding and "shift" in resp.encoding.lower():
                resp.encoding = "shift_jis"
            elif resp.apparent_encoding:
                resp.encoding = resp.apparent_encoding
            return resp.text
        except requests.RequestException:
            if attempt < cfg["max_retries"] - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                return None
    return None
