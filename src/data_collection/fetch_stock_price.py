"""抓取股價資料並存成 CSV。

用法:
    python fetch_stock_price.py --ticker AAPL --period 1y
    python fetch_stock_price.py --ticker 2330.TW --period 6mo
"""

import argparse
from pathlib import Path

import yfinance as yf

VALID_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch stock price history via yfinance.")
    parser.add_argument("--ticker", default="AAPL", help="股票代號，例如 AAPL 或 2330.TW（預設：AAPL）")
    parser.add_argument(
        "--period",
        default="1y",
        choices=sorted(VALID_PERIODS),
        help="資料期間（預設：1y）",
    )
    return parser.parse_args()


def fetch_stock_price(ticker: str, period: str) -> "pd.DataFrame":
    data = yf.Ticker(ticker).history(period=period)
    if data.empty:
        raise RuntimeError(f"No data returned for {ticker}. Check network connection or ticker symbol.")
    return data


def main() -> None:
    args = parse_args()

    print(f"Fetching {args.ticker} price data (period={args.period}) ...")
    data = fetch_stock_price(args.ticker, args.period)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{args.ticker}_{args.period}.csv"
    data.to_csv(output_path)

    print(f"Fetched {len(data)} rows.")
    print(f"Saved to: {output_path}")
    print(data.head())


if __name__ == "__main__":
    main()
