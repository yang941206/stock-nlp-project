"""用技術指標 + 新聞情緒的合併特徵表，產生簡單的買賣訊號，並計算訊號後 N 天的報酬做為粗略回測。

策略（示範用，非投資建議）:
    buy  : net_sentiment > sentiment_threshold 且 RSI < rsi_oversold（偏多情緒 + 超賣）
    sell : net_sentiment < -sentiment_threshold 且 RSI > rsi_overbought（偏空情緒 + 超買）
    hold : 其他情況

用法:
    python signals.py --ticker AAPL
"""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate simple sentiment+RSI signals and forward-return check.")
    parser.add_argument("--ticker", default="AAPL", help="股票代號（預設：AAPL）")
    parser.add_argument("--sentiment-threshold", type=float, default=0.1)
    parser.add_argument("--rsi-oversold", type=float, default=35)
    parser.add_argument("--rsi-overbought", type=float, default=65)
    parser.add_argument("--forward-days", type=int, default=5, help="訊號後幾天的報酬做為評估")
    return parser.parse_args()


def generate_signals(df: pd.DataFrame, sentiment_threshold: float, rsi_oversold: float, rsi_overbought: float) -> pd.DataFrame:
    df = df.copy()

    buy = (df["net_sentiment"] > sentiment_threshold) & (df["rsi_14"] < rsi_oversold)
    sell = (df["net_sentiment"] < -sentiment_threshold) & (df["rsi_14"] > rsi_overbought)

    df["signal"] = "hold"
    df.loc[buy, "signal"] = "buy"
    df.loc[sell, "signal"] = "sell"
    return df


def add_forward_return(df: pd.DataFrame, forward_days: int) -> pd.DataFrame:
    df = df.copy()
    df[f"forward_return_{forward_days}d"] = df["Close"].shift(-forward_days) / df["Close"] - 1
    return df


def main() -> None:
    args = parse_args()

    features_path = PROCESSED_DIR / f"{args.ticker}_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"{features_path} not found. Run build_feature_table.py first.")

    df = pd.read_csv(features_path)
    df["Date"] = pd.to_datetime(df["Date"])

    df = generate_signals(df, args.sentiment_threshold, args.rsi_oversold, args.rsi_overbought)
    df = add_forward_return(df, args.forward_days)

    output_path = PROCESSED_DIR / f"{args.ticker}_signals.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved to: {output_path}")
    print(df["signal"].value_counts())

    ret_col = f"forward_return_{args.forward_days}d"
    for label in ["buy", "sell"]:
        subset = df[(df["signal"] == label) & df[ret_col].notna()]
        if len(subset) == 0:
            print(f"{label}: no evaluable signals yet (need more days with both news + {args.forward_days}-day forward price data)")
        else:
            print(f"{label}: n={len(subset)}, avg {args.forward_days}d forward return = {subset[ret_col].mean():.4f}")

    days_with_news = (df["news_count"] > 0).sum()
    print(
        f"\nNote: only {days_with_news} of {len(df)} trading days have news coverage "
        "(Yahoo Finance news source only returns the ~10 most recent items, no historical backfill). "
        "Signal counts/backtest above are not statistically meaningful yet — need a news source with "
        "historical coverage to properly evaluate this strategy."
    )


if __name__ == "__main__":
    main()
