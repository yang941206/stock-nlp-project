"""計算技術指標（RSI、MACD、布林通道、移動平均），輸出到 data/processed。

用法:
    python indicators.py --ticker AAPL
"""

import argparse
import glob
from pathlib import Path

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute technical indicators for a ticker's price history.")
    parser.add_argument("--ticker", default="AAPL", help="股票代號（預設：AAPL）")
    return parser.parse_args()


def find_price_file(ticker: str) -> Path:
    matches = glob.glob(str(RAW_DIR / f"{ticker}_*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No price CSV found for {ticker} in {RAW_DIR}. Run fetch_stock_price.py first."
        )
    return Path(max(matches, key=lambda p: Path(p).stat().st_mtime))


def compute_indicators(ticker: str) -> pd.DataFrame:
    price = pd.read_csv(find_price_file(ticker))
    price["Date"] = pd.to_datetime(price["Date"], utc=True).dt.tz_localize(None)
    price = price.sort_values("Date").reset_index(drop=True)

    missing_close = price["Close"].isna().sum()
    if missing_close:
        print(f"  Dropping {missing_close} row(s) with missing Close price (likely incomplete/unsettled data from yfinance).")
        price = price.dropna(subset=["Close"]).reset_index(drop=True)

    close = price["Close"]

    price["sma_20"] = SMAIndicator(close, window=20).sma_indicator()
    price["ema_20"] = EMAIndicator(close, window=20).ema_indicator()

    rsi = RSIIndicator(close, window=14)
    price["rsi_14"] = rsi.rsi()

    macd = MACD(close)
    price["macd"] = macd.macd()
    price["macd_signal"] = macd.macd_signal()
    price["macd_diff"] = macd.macd_diff()

    bb = BollingerBands(close, window=20, window_dev=2)
    price["bb_high"] = bb.bollinger_hband()
    price["bb_low"] = bb.bollinger_lband()
    price["bb_mid"] = bb.bollinger_mavg()

    return price


def main() -> None:
    args = parse_args()

    print(f"Computing technical indicators for {args.ticker} ...")
    result = compute_indicators(args.ticker)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{args.ticker}_indicators.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved {len(result)} rows to: {output_path}")
    print(result[["Date", "Close", "rsi_14", "macd", "bb_high", "bb_low"]].tail())


if __name__ == "__main__":
    main()
