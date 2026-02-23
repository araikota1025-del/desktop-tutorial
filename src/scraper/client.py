"""boatrace.jp へのHTTPリクエストを管理する共通クライアント"""

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
