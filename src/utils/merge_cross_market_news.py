"""實驗性腳本：把「美股新聞情緒」對齊到「新聞發布後下一個台股交易日」的股價。

跟 merge_price_news.py 的差異：
    - merge_price_news.py 是同一支股票的新聞對齊「當天或最近一個交易日」（backward，因為新聞跟
      股價是同一個市場，同一個時區的收盤價已經反映了當天的新聞）。
    - 這支腳本是跨市場對齊：新聞來自美股公司（例如 AAPL、NVDA），股價是台股（例如 2330.TW）。
      因為美股新聞多半在台灣時間的深夜/凌晨發布，用「當天」對齊沒有意義，所以改成 forward + 不允許
      精準匹配（allow_exact_matches=False），對齊到「新聞發布時間之後，第一個台股交易日」的收盤價。

用法:
    python merge_cross_market_news.py --price-ticker 2330.TW --news-tickers AAPL NVDA
"""

import argparse
import glob
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentiment_model.batch_predict import batch_predict, DEFAULT_MODEL_DIR  # noqa: E402
from utils.news_filters import is_routine_institutional_filing  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
NEWS_DIR = PROJECT_ROOT / "data" / "news"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align foreign-market news sentiment to the next local trading day's price."
    )
    parser.add_argument("--price-ticker", default="2330.TW", help="股價股票代號（預設：2330.TW）")
    parser.add_argument(
        "--news-tickers", nargs="+", default=["AAPL", "NVDA"], help="新聞來源股票代號（預設：AAPL NVDA）"
    )
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="fine-tune 後的情緒模型資料夾")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def find_price_file(ticker: str) -> Path:
    matches = glob.glob(str(RAW_DIR / f"{ticker}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No price CSV found for {ticker} in {RAW_DIR}. Run fetch_stock_price.py first.")
    return Path(max(matches, key=lambda p: Path(p).stat().st_mtime))


def load_price(ticker: str) -> pd.DataFrame:
    price = pd.read_csv(find_price_file(ticker))
    price["Date"] = pd.to_datetime(price["Date"], utc=True).dt.tz_localize(None)
    return price.sort_values("Date").reset_index(drop=True)


def load_news(tickers: list) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        news_path = NEWS_DIR / f"{ticker}_news.csv"
        if not news_path.exists():
            raise FileNotFoundError(f"No news CSV found at {news_path}. Run a fetch_news* script first.")
        df = pd.read_csv(news_path)
        frames.append(df)
    news = pd.concat(frames, ignore_index=True)
    news["pub_date"] = pd.to_datetime(news["pub_date"], utc=True).dt.tz_localize(None)

    routine_mask = news["title"].apply(is_routine_institutional_filing)
    if routine_mask.any():
        print(
            f"  過濾掉 {routine_mask.sum()} / {len(news)} 則例行機構持股異動公告"
            f"（{routine_mask.mean() * 100:.1f}%）"
        )
    news = news[~routine_mask]

    return news.sort_values("pub_date").reset_index(drop=True)


def main() -> None:
    args = parse_args()

    print(f"Loading price data for {args.price_ticker} ...")
    price = load_price(args.price_ticker)

    print(f"Loading news for {args.news_tickers} ...")
    news = load_news(args.news_tickers)
    print(f"  {len(news)} news items total")

    print("Aligning each news item to the next TW trading day strictly after its pub_date ...")
    merged = pd.merge_asof(
        news,
        price,
        left_on="pub_date",
        right_on="Date",
        direction="forward",
        allow_exact_matches=False,
    )
    merged = merged.rename(columns={"ticker": "news_ticker", "Date": "aligned_trading_date"})
    merged["price_ticker"] = args.price_ticker

    unmatched = merged["aligned_trading_date"].isna().sum()
    if unmatched:
        print(f"  Warning: {unmatched} news items have no next trading day yet (news is after the latest price data).")
    merged = merged.dropna(subset=["aligned_trading_date"]).reset_index(drop=True)

    print(f"Scoring sentiment for {len(merged)} headlines with {args.model_dir} ...")
    scores = batch_predict(merged["title"].fillna("").tolist(), args.model_dir, args.batch_size)
    scores = scores.rename(columns={c: f"sentiment_{c}" for c in scores.columns if c != "sentiment_label"})
    result = pd.concat([merged.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)

    news_tag = "_".join(args.news_tickers)
    output_path = OUTPUT_DIR / f"{args.price_ticker}_news_from_{news_tag}.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved to: {output_path}")
    print(f"Rows: {len(result)}")
    print(result["news_ticker"].value_counts())
    print()
    preview_cols = ["news_ticker", "pub_date", "aligned_trading_date", "title", "sentiment_label", "Close"]
    print(result[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
