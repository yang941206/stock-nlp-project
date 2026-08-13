"""把跨市場合併結果轉成 build_feature_table.py / signals.py 可以直接吃的格式。

這兩支「正式」腳本是設計給同一市場的 {ticker}_indicators.csv / {ticker}_merged_sentiment.csv
用的。這支腳本用一個合成的 ticker 標籤（預設 2330.TW_x_US，代表「2330.TW 股價 + 美股新聞情緒」）
複製/改名檔案，讓後面兩支腳本可以原封不動直接重複使用，不用為了跨市場這個實驗改它們的程式碼。

用法:
    python prepare_crossmarket_adapter.py --price-ticker 2330.TW --news-tickers AAPL NVDA
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare adapter files for build_feature_table.py / signals.py.")
    parser.add_argument("--price-ticker", default="2330.TW")
    parser.add_argument("--news-tickers", nargs="+", default=["AAPL", "NVDA"])
    parser.add_argument("--label", default="2330.TW_x_US", help="合成 ticker 標籤（預設：2330.TW_x_US）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    src_indicators = PROCESSED_DIR / f"{args.price_ticker}_indicators.csv"
    if not src_indicators.exists():
        raise FileNotFoundError(f"{src_indicators} not found. Run indicators.py first.")
    dst_indicators = PROCESSED_DIR / f"{args.label}_indicators.csv"
    shutil.copy(src_indicators, dst_indicators)

    news_tag = "_".join(args.news_tickers)
    src_sentiment = PROCESSED_DIR / f"{args.price_ticker}_news_from_{news_tag}.csv"
    if not src_sentiment.exists():
        raise FileNotFoundError(f"{src_sentiment} not found. Run merge_cross_market_news.py first.")

    df = pd.read_csv(src_sentiment)
    df = df.rename(columns={"aligned_trading_date": "trading_date"})
    dst_sentiment = PROCESSED_DIR / f"{args.label}_merged_sentiment.csv"
    df.to_csv(dst_sentiment, index=False, encoding="utf-8-sig")

    print(f"Adapter files ready for ticker label '{args.label}':")
    print(f"  {dst_indicators}")
    print(f"  {dst_sentiment}")


if __name__ == "__main__":
    main()
