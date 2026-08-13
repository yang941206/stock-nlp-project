"""將股價 CSV 與新聞 CSV 依日期整合成一份資料。

每一則新聞會對應到「新聞發布當天，或往前最近一個有交易的日期」的股價資料
（用 merge_asof backward 對齊，因為新聞可能發生在假日或收盤後）。

用法:
    python merge_price_news.py --ticker AAPL
"""

import argparse
import glob
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
NEWS_DIR = PROJECT_ROOT / "data" / "news"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge price CSV and news CSV by date.")
    parser.add_argument("--ticker", default="AAPL", help="股票代號（預設：AAPL）")
    return parser.parse_args()


def find_price_file(ticker: str) -> Path:
    matches = glob.glob(str(RAW_DIR / f"{ticker}_*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No price CSV found for {ticker} in {RAW_DIR}. Run fetch_stock_price.py first."
        )
    return Path(max(matches, key=lambda p: Path(p).stat().st_mtime))


def load_price(ticker: str) -> pd.DataFrame:
    price_path = find_price_file(ticker)
    price = pd.read_csv(price_path)
    price["Date"] = pd.to_datetime(price["Date"], utc=True).dt.tz_localize(None)
    price = price.sort_values("Date").reset_index(drop=True)
    return price


def load_news(ticker: str) -> pd.DataFrame:
    news_path = NEWS_DIR / f"{ticker}_news.csv"
    if not news_path.exists():
        raise FileNotFoundError(f"No news CSV found at {news_path}. Run fetch_news.py first.")
    news = pd.read_csv(news_path)
    news["pub_date"] = pd.to_datetime(news["pub_date"], utc=True).dt.tz_localize(None)
    news = news.sort_values("pub_date").reset_index(drop=True)
    return news


def merge_price_news(ticker: str) -> pd.DataFrame:
    price = load_price(ticker)
    news = load_news(ticker)

    merged = pd.merge_asof(
        news,
        price,
        left_on="pub_date",
        right_on="Date",
        direction="backward",
    )
    merged = merged.rename(columns={"Date": "trading_date"})
    return merged


def main() -> None:
    args = parse_args()

    print(f"Merging price and news data for {args.ticker} ...")
    merged = merge_price_news(args.ticker)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{args.ticker}_merged.csv"
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    unmatched = merged["trading_date"].isna().sum()
    print(f"Merged {len(merged)} news items with price data.")
    if unmatched:
        print(f"Warning: {unmatched} news items had no matching trading date (before earliest price data).")
    print(f"Saved to: {output_path}")
    print(merged[["pub_date", "trading_date", "Close", "title"]].head())


if __name__ == "__main__":
    main()
