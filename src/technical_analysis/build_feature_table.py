"""把技術指標（indicators.py）與新聞情緒分數（batch_predict.py）依日期合併成一份特徵表。

每個交易日一列：技術指標 + 當天新聞的情緒統計（則數、平均看多/看空/中立機率、net_sentiment）。

用法:
    python build_feature_table.py --ticker AAPL
"""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge technical indicators with daily sentiment aggregates.")
    parser.add_argument("--ticker", default="AAPL", help="股票代號（預設：AAPL）")
    return parser.parse_args()


def build_feature_table(ticker: str) -> pd.DataFrame:
    indicators_path = PROCESSED_DIR / f"{ticker}_indicators.csv"
    sentiment_path = PROCESSED_DIR / f"{ticker}_merged_sentiment.csv"

    if not indicators_path.exists():
        raise FileNotFoundError(f"{indicators_path} not found. Run indicators.py first.")
    if not sentiment_path.exists():
        raise FileNotFoundError(f"{sentiment_path} not found. Run batch_predict.py first.")

    indicators = pd.read_csv(indicators_path)
    indicators["Date"] = pd.to_datetime(indicators["Date"]).dt.normalize()

    sentiment = pd.read_csv(sentiment_path)
    sentiment["trading_date"] = pd.to_datetime(sentiment["trading_date"]).dt.normalize()

    daily_sentiment = (
        sentiment.groupby("trading_date")
        .agg(
            news_count=("sentiment_label", "count"),
            avg_bearish=("sentiment_bearish", "mean"),
            avg_bullish=("sentiment_bullish", "mean"),
            avg_neutral=("sentiment_neutral", "mean"),
        )
        .reset_index()
        .rename(columns={"trading_date": "Date"})
    )
    daily_sentiment["net_sentiment"] = daily_sentiment["avg_bullish"] - daily_sentiment["avg_bearish"]

    merged = indicators.merge(daily_sentiment, on="Date", how="left")
    merged["news_count"] = merged["news_count"].fillna(0).astype(int)

    return merged


def main() -> None:
    args = parse_args()

    print(f"Building feature table for {args.ticker} ...")
    result = build_feature_table(args.ticker)

    output_path = PROCESSED_DIR / f"{args.ticker}_features.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    days_with_news = (result["news_count"] > 0).sum()
    print(f"Saved {len(result)} rows to: {output_path} ({days_with_news} days have news coverage)")
    print(result[["Date", "Close", "rsi_14", "macd", "news_count", "net_sentiment"]].tail(10))


if __name__ == "__main__":
    main()
