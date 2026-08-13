"""用 GDELT DOC 2.0 API 回溯抓取歷史新聞（免費、不需要 API key，速率限制比 Alpha Vantage 寬鬆很多）。

跟 fetch_news_alphavantage.py 的差異:
    1. 不用 API key，額度也不是「25 次/天」這種硬性上限，而是「1 次/5 秒」的軟性速率限制。
    2. 單次查詢最長可回溯 1 年（timespan=1y），但單次查詢結果上限 250 筆，所以用月份區間切
       window 分批抓，確保密度不會被 250 筆上限卡住。
    3. GDELT 沒有股票代號欄位，是全文關鍵字檢索，見 DEFAULT_QUERY_TERMS。因為是「全網新聞」不是
       財經新聞專用 feed，噪音明顯比 Alpha Vantage 多，預設限定一批財經媒體 domain（DEFAULT_DOMAINS）
       降噪，可用 --domains 覆寫或關閉。
    4. GDELT ArtList 回傳沒有 summary 欄位，merge_into_news_csv 會把 summary 留空（不影響情緒分類，
       batch_predict.py 本來就是對 title 打分）。

輸出格式跟 fetch_news_alphavantage.py 一致：欄位（ticker, id, title, summary, pub_date, provider, url,
source_api），寫進同一個 data/news/{ticker}_news.csv，依 url 去重、依 pub_date 排序，下游
merge_price_news.py / merge_cross_market_news.py 不用改。

用法:
    python fetch_news_gdelt.py --ticker AAPL
    python fetch_news_gdelt.py --ticker AAPL --lookback-days 365 --window-days 30
    python fetch_news_gdelt.py --ticker AAPL --query "\"Apple Inc\"" --domains cnbc.com reuters.com
"""

import argparse
import hashlib
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEWS_DIR = PROJECT_ROOT / "data" / "news"

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_QUERY_TIME_FMT = "%Y%m%d%H%M%S"       # startdatetime/enddatetime 查詢參數用的格式
GDELT_ITEM_TIME_FMT = "%Y%m%dT%H%M%SZ"      # 回傳的 seendate 欄位用的格式（例如 20260714T111500Z）
REQUEST_INTERVAL_SEC = 6  # GDELT 官方要求 1 次/5 秒，保守用 6 秒
MAX_RECORDS_PER_QUERY = 250  # GDELT DOC API 單次查詢上限

# 股票代號 -> GDELT 查詢用的公司名稱（GDELT 沒有股票代號欄位，只能全文關鍵字檢索）
DEFAULT_QUERY_TERMS = {
    "AAPL": '"Apple Inc" OR "Apple stock"',
    "NVDA": '"Nvidia"',
    "AMD": '"Advanced Micro Devices" OR "AMD stock"',
    "TSM": '"Taiwan Semiconductor" OR "TSMC"',
    "QCOM": '"Qualcomm"',
}

# 限定財經媒體 domain，降低 GDELT 全網新聞的噪音（--domains 傳空清單可關閉這個限制）。
# 故意只留少數幾個：實測發現 domain OR 子句疊太多會被 GDELT 判定「query was too short or
# too long」拒絕（官方沒有明確公布確切上限，經驗上 3-4 個 domain 是安全的，10 個會失敗）。
DEFAULT_DOMAINS = ["reuters.com", "cnbc.com", "marketwatch.com"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill historical news via the GDELT DOC 2.0 API.")
    parser.add_argument("--ticker", default="AAPL", help="股票代號（預設：AAPL）")
    parser.add_argument(
        "--query", default=None,
        help="自訂 GDELT 查詢字串（預設查 DEFAULT_QUERY_TERMS 對應表，找不到就報錯提醒手動指定）"
    )
    parser.add_argument(
        "--domains", nargs="*", default=None,
        help="限定的媒體 domain 清單（預設用 DEFAULT_DOMAINS；傳 --domains 不接任何值可關閉限制）"
    )
    parser.add_argument("--window-days", type=int, default=30, help="每次 API 呼叫回溯的天數區間（預設：30）")
    parser.add_argument("--lookback-days", type=int, default=365, help="總共要回溯多久（預設：365，GDELT 上限）")
    parser.add_argument("--max-requests", type=int, default=None, help="這次執行最多呼叫幾次 API（預設：不限）")
    return parser.parse_args()


def build_query(args: argparse.Namespace) -> str:
    query = args.query
    if query is None:
        query = DEFAULT_QUERY_TERMS.get(args.ticker)
        if query is None:
            raise ValueError(
                f"DEFAULT_QUERY_TERMS 裡沒有 {args.ticker} 的查詢字串，請用 --query 手動指定"
                f"（例如 --query '\"Company Name\"'）。"
            )
    domains = DEFAULT_DOMAINS if args.domains is None else args.domains
    if domains:
        domain_clause = " OR ".join(f"domain:{d}" for d in domains)
        query = f"({query}) ({domain_clause})"
    return query


def make_id(url: str) -> str:
    return "gdelt_" + hashlib.md5(url.encode("utf-8")).hexdigest()[:16]


def fetch_window(query: str, time_from: datetime, time_to: datetime) -> list:
    resp = requests.get(
        GDELT_URL,
        params={
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "sort": "DateDesc",
            "maxrecords": MAX_RECORDS_PER_QUERY,
            "startdatetime": time_from.strftime(GDELT_QUERY_TIME_FMT),
            "enddatetime": time_to.strftime(GDELT_QUERY_TIME_FMT),
        },
        headers={"User-Agent": "stock-nlp-project (research use)"},
        timeout=30,
    )
    if resp.status_code == 429:
        raise requests.RequestException(f"GDELT rate limit hit: {resp.text.strip()}")
    resp.raise_for_status()
    try:
        data = resp.json()
    except ValueError:
        # GDELT 偶爾會在查詢語法有問題時回傳 HTML 錯誤頁而不是 JSON
        raise requests.RequestException(f"GDELT returned non-JSON response: {resp.text[:300]}")
    return data.get("articles", [])


def parse_gdelt_time(seendate: str) -> str:
    dt = datetime.strptime(seendate, GDELT_ITEM_TIME_FMT).replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


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
    query = build_query(args)
    print(f"GDELT 查詢字串: {query}")

    now = datetime.now(timezone.utc)
    lookback_start = now - timedelta(days=args.lookback_days)

    collected_rows = []
    requests_made = 0
    window_end = now

    while window_end > lookback_start:
        if args.max_requests is not None and requests_made >= args.max_requests:
            print(f"已達 --max-requests 上限（{args.max_requests}），停止本次執行。")
            break

        window_start = max(window_end - timedelta(days=args.window_days), lookback_start)
        print(f"抓取 {args.ticker} 新聞: {window_start.date()} ~ {window_end.date()} ...")

        try:
            articles = fetch_window(query, window_start, window_end)
        except requests.RequestException as e:
            print(f"請求失敗，停止本次執行: {e}")
            break

        requests_made += 1
        print(f"  取得 {len(articles)} 則新聞" + ("（觸及 250 筆上限，這個 window 可能還有漏抓，建議縮小 --window-days）" if len(articles) >= MAX_RECORDS_PER_QUERY else ""))

        for item in articles:
            url = item.get("url", "")
            seendate = item.get("seendate")
            if not url or not seendate:
                continue
            collected_rows.append(
                {
                    "ticker": args.ticker,
                    "id": make_id(url),
                    "title": item.get("title", ""),
                    "summary": "",
                    "pub_date": parse_gdelt_time(seendate),
                    "provider": item.get("domain", ""),
                    "url": url,
                    "source_api": "gdelt",
                }
            )

        window_end = window_start
        if window_end > lookback_start:
            time.sleep(REQUEST_INTERVAL_SEC)

    print(f"\n本次共呼叫 API {requests_made} 次，抓到 {len(collected_rows)} 則新聞（去重前）。")

    net_new_items = 0
    if collected_rows:
        net_new_items = merge_into_news_csv(args.ticker, collected_rows)
    else:
        print("這次沒有抓到任何新聞。")

    print(f"淨新增 {net_new_items} 則。")


if __name__ == "__main__":
    main()
