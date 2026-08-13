"""抓取特定股票相關新聞（透過 yfinance / Yahoo Finance）並存成 CSV。

用法:
    python fetch_news.py --ticker AAPL
    python fetch_news.py --ticker 2330.TW --count 20
"""

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "news"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch ticker-related news via yfinance.")
    parser.add_argument("--ticker", default="AAPL", help="股票代號，例如 AAPL 或 2330.TW（預設：AAPL）")
    parser.add_argument("--count", type=int, default=10, help="最多抓取的新聞則數（預設：10）")
    return parser.parse_args()


def fetch_news(ticker: str, count: int) -> pd.DataFrame:
    raw_items = yf.Ticker(ticker).news
    if not raw_items:
        raise RuntimeError(f"No news returned for {ticker}. Check network connection or ticker symbol.")

    rows = []
    for item in raw_items[:count]:
        content = item.get("content", {})
        provider = content.get("provider") or {}
        link = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        rows.append(
            {
                "ticker": ticker,
                "id": item.get("id"),
                "title": content.get("title"),
                "summary": content.get("summary"),
                "pub_date": content.get("pubDate"),
                "provider": provider.get("displayName"),
                "url": link.get("url"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    print(f"Fetching news for {args.ticker} (count={args.count}) ...")
    data = fetch_news(args.ticker, args.count)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{args.ticker}_news.csv"
    data.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Fetched {len(data)} news items.")
    print(f"Saved to: {output_path}")
    print(data[["pub_date", "title"]].head())


if __name__ == "__main__":
    main()
