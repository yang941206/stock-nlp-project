"""用 Alpha Vantage NEWS_SENTIMENT API 回溯抓取歷史新聞（免費方案：25 次/天、5 次/分鐘）。

設計重點:
    1. 進度追蹤: 每支股票有一個 state 檔（data/news/{ticker}_av_state.json），
       記錄目前已經回溯到的最早時間點（earliest_covered）。每次執行會從上次
       停下的地方繼續往回抓，不會重複抓同一段時間。
    2. 額度追蹤: data/news/.av_quota.json 記錄「今天」已經呼叫幾次 API
       （以 UTC 日期為界，Alpha Vantage 官方沒有明確公布重置時區，這是保守近似）。
       額度用完，或 Alpha Vantage 回傳額度限制訊息時，會印出明確訊息並乾淨地停止，
       不會跑到一半噴例外。
    3. 輸出格式: 欄位（ticker, id, title, summary, pub_date, provider, url）
       與 fetch_news.py（yfinance 版）相容，寫入同一個
       data/news/{ticker}_news.csv，並依 url 去重、依 pub_date 排序，
       這樣 merge_price_news.py 不需要修改就能直接吃到兩個來源的新聞。

用法:
    python fetch_news_alphavantage.py --ticker AAPL
    python fetch_news_alphavantage.py --ticker AAPL --max-requests 5 --window-days 14
"""

import argparse
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEWS_DIR = PROJECT_ROOT / "data" / "news"
QUOTA_FILE = NEWS_DIR / ".av_quota.json"

AV_URL = "https://www.alphavantage.co/query"
AV_QUERY_TIME_FMT = "%Y%m%dT%H%M"
AV_ITEM_TIME_FMT = "%Y%m%dT%H%M%S"
REQUEST_INTERVAL_SEC = 13  # 免費方案 5 次/分鐘 -> 保守間隔 13 秒


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill historical news via Alpha Vantage NEWS_SENTIMENT API.")
    parser.add_argument("--ticker", default="AAPL", help="股票代號（預設：AAPL）")
    parser.add_argument("--window-days", type=int, default=7, help="每次 API 呼叫回溯的天數區間（預設：7）")
    parser.add_argument(
        "--max-requests", type=int, default=None,
        help="這次執行最多呼叫幾次 API（預設：用完當天剩餘額度）"
    )
    parser.add_argument("--daily-limit", type=int, default=25, help="帳號每日額度上限（免費方案：25）")
    return parser.parse_args()


def load_json(path: Path, default: dict) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def load_quota() -> dict:
    quota = load_json(QUOTA_FILE, {"date": today_str(), "requests_used": 0})
    if quota["date"] != today_str():
        quota = {"date": today_str(), "requests_used": 0}
    return quota


def state_path(ticker: str) -> Path:
    return NEWS_DIR / f"{ticker}_av_state.json"


def load_state(ticker: str) -> dict:
    return load_json(state_path(ticker), {"earliest_covered": None, "latest_covered": None})


def make_id(url: str) -> str:
    return "av_" + hashlib.md5(url.encode("utf-8")).hexdigest()[:16]


def parse_av_time(ts: str) -> str:
    dt = datetime.strptime(ts, AV_ITEM_TIME_FMT)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_window(ticker: str, time_from: datetime, time_to: datetime, api_key: str) -> dict:
    resp = requests.get(
        AV_URL,
        params={
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "time_from": time_from.strftime(AV_QUERY_TIME_FMT),
            "time_to": time_to.strftime(AV_QUERY_TIME_FMT),
            "limit": 1000,
            "sort": "EARLIEST",
            "apikey": api_key,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def merge_into_news_csv(ticker: str, new_rows: list) -> int:
    output_path = NEWS_DIR / f"{ticker}_news.csv"
    new_df = pd.DataFrame(new_rows)

    if output_path.exists():
        existing = pd.read_csv(output_path)
        existing_count = len(existing)
        if "source_api" not in existing.columns:
            existing["source_api"] = "yahoo"
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        existing_count = 0
        combined = new_df

    combined = combined.drop_duplicates(subset=["url"], keep="first")
    # format="ISO8601"：既有資料存回 CSV 後會變成 "YYYY-MM-DD HH:MM:SS+00:00"，這次新抓到的
    # 是 "YYYY-MM-DDTHH:MM:SSZ"，同一欄兩種格式混在一起，不指定 format 會直接 crash（已實測重現）。
    combined["pub_date"] = pd.to_datetime(combined["pub_date"], utc=True, format="ISO8601")
    combined = combined.sort_values("pub_date").reset_index(drop=True)
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")

    net_new = len(combined) - existing_count
    print(f"已合併去重（淨增加 {net_new} 則），存到: {output_path}")
    print(f"目前共有 {len(combined)} 則新聞，涵蓋 {combined['pub_date'].min()} ~ {combined['pub_date'].max()}")
    return net_new


def main() -> None:
    args = parse_args()

    load_dotenv()
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise RuntimeError("ALPHA_VANTAGE_API_KEY 未設定，請先在 .env 填入你的 Alpha Vantage API key。")

    quota = load_quota()
    remaining_today = args.daily_limit - quota["requests_used"]
    if remaining_today <= 0:
        print(f"今天的 Alpha Vantage 額度（{args.daily_limit} 次）已經用完，明天再執行。")
        return

    run_budget = remaining_today if args.max_requests is None else min(args.max_requests, remaining_today)
    print(f"今天已用 {quota['requests_used']}/{args.daily_limit} 次，這次執行最多呼叫 {run_budget} 次 API。")

    state = load_state(args.ticker)
    now = datetime.now(timezone.utc)
    if state["earliest_covered"]:
        window_end = datetime.strptime(state["earliest_covered"], AV_QUERY_TIME_FMT).replace(tzinfo=timezone.utc)
    else:
        window_end = now
    if state["latest_covered"] is None:
        state["latest_covered"] = now.strftime(AV_QUERY_TIME_FMT)

    collected_rows = []
    requests_made = 0
    consecutive_empty = 0

    while requests_made < run_budget:
        window_start = window_end - timedelta(days=args.window_days)
        print(f"抓取 {args.ticker} 新聞: {window_start.date()} ~ {window_end.date()} ...")

        try:
            data = fetch_window(args.ticker, window_start, window_end, api_key)
        except requests.RequestException as e:
            print(f"請求失敗，停止本次執行: {e}")
            break

        requests_made += 1
        quota["requests_used"] += 1
        save_json(QUOTA_FILE, quota)

        if "Information" in data or "Note" in data:
            msg = data.get("Information") or data.get("Note")
            print(f"Alpha Vantage 回報額度/速率限制，停止本次執行: {msg}")
            quota["requests_used"] = args.daily_limit
            save_json(QUOTA_FILE, quota)
            break

        feed = data.get("feed", [])
        print(f"  取得 {len(feed)} 則新聞")

        if not feed:
            consecutive_empty += 1
            if consecutive_empty >= 3:
                print("連續 3 個時間窗都沒有新聞，可能已經到達 Alpha Vantage 歷史涵蓋的最早範圍，停止回溯。")
                window_end = window_start
                state["earliest_covered"] = window_end.strftime(AV_QUERY_TIME_FMT)
                save_json(state_path(args.ticker), state)
                break
        else:
            consecutive_empty = 0
            for item in feed:
                collected_rows.append(
                    {
                        "ticker": args.ticker,
                        "id": make_id(item["url"]),
                        "title": item.get("title", ""),
                        "summary": item.get("summary", ""),
                        "pub_date": parse_av_time(item["time_published"]),
                        "provider": item.get("source", ""),
                        "url": item.get("url", ""),
                        "source_api": "alpha_vantage",
                    }
                )

        window_end = window_start
        state["earliest_covered"] = window_end.strftime(AV_QUERY_TIME_FMT)
        save_json(state_path(args.ticker), state)

        if requests_made < run_budget:
            time.sleep(REQUEST_INTERVAL_SEC)

    print(f"\n本次共呼叫 API {requests_made} 次，新抓到 {len(collected_rows)} 則新聞。")

    net_new_items = 0
    if collected_rows:
        net_new_items = merge_into_news_csv(args.ticker, collected_rows)
    else:
        print("這次沒有新增任何新聞。")

    print(f"目前回溯到: {state['earliest_covered']}（下次執行會從這裡繼續往回抓）")
    print(f"今天已用 {quota['requests_used']}/{args.daily_limit} 次額度。")

    save_json(
        NEWS_DIR / f"{args.ticker}_av_last_run.json",
        {
            "run_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "requests_made": requests_made,
            "net_new_items": net_new_items,
        },
    )


if __name__ == "__main__":
    main()
