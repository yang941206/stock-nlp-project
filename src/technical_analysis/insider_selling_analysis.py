"""內部人/高管申報賣股新聞 vs. 股價後續表現：純統計觀察，不是正式訊號策略。

跟主流程/跨市場流程的差異:
    news_filters.is_insider_selling 抓的是「公司高管/董事賣自家股票」類標題（跟
    is_routine_institutional_filing 抓的法人 13F 持股異動是不同類別，目前故意不過濾）。
    這類新聞提到的公司通常不是查詢用的 ticker 本身（AAPL/NVDA/AMD/TSM/QCOM），而是
    Alpha Vantage NEWS_SENTIMENT 回傳的廣泛市場新聞裡剛好提到的其他公司，所以這裡不是
    「該公司內部人賣股 → 該公司股價後續表現」的因果分析，而是把「內部人賣股新聞」當成
    一種市場情緒的雜訊指標，觀察它跟 AAPL 股價後續表現有沒有可觀察的關聯——之所以用 AAPL
    而不是台股，是因為 AAPL 有完整一年的股價歷史（vs. 台股跨市場對齊目前只有 ~2 週新聞覆蓋），
    樣本數比較不會太小。

用法:
    python insider_selling_analysis.py
    python insider_selling_analysis.py --news-tickers AAPL NVDA AMD TSM QCOM --price-ticker AAPL
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentiment_model.batch_predict import batch_predict, DEFAULT_MODEL_DIR  # noqa: E402
from utils.news_filters import is_insider_selling  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
NEWS_DIR = PROJECT_ROOT / "data" / "news"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

FORWARD_DAYS = [1, 3, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Observational analysis: insider-selling news vs. forward returns.")
    parser.add_argument("--news-tickers", nargs="+", default=["AAPL", "NVDA", "AMD", "TSM", "QCOM"])
    parser.add_argument("--price-ticker", default="AAPL", help="用哪支股票的價格資料（預設 AAPL，歷史最長）")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    return parser.parse_args()


def load_price(ticker: str) -> pd.DataFrame:
    matches = glob.glob(str(RAW_DIR / f"{ticker}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No price CSV found for {ticker}. Run fetch_stock_price.py first.")
    price_path = Path(max(matches, key=lambda p: Path(p).stat().st_mtime))
    price = pd.read_csv(price_path)
    price["Date"] = pd.to_datetime(price["Date"], utc=True).dt.tz_localize(None)
    price = price.sort_values("Date").reset_index(drop=True)
    for n in FORWARD_DAYS:
        price[f"forward_return_{n}d"] = price["Close"].shift(-n) / price["Close"] - 1
    return price


def load_insider_selling_news(tickers: list) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        path = NEWS_DIR / f"{ticker}_news.csv"
        if not path.exists():
            print(f"  (skip {ticker}: no news CSV found)")
            continue
        df = pd.read_csv(path)
        if "title" not in df.columns or len(df) == 0:
            continue
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No news CSVs found for any of the given tickers.")
    news = pd.concat(frames, ignore_index=True)
    news = news.drop_duplicates(subset=["url"]) if "url" in news.columns else news.drop_duplicates(subset=["title"])
    news["pub_date"] = pd.to_datetime(news["pub_date"], utc=True).dt.tz_localize(None)

    mask = news["title"].apply(is_insider_selling)
    print(f"Found {mask.sum()} insider-selling headlines out of {len(news)} total pooled news items.")
    return news.loc[mask].sort_values("pub_date").reset_index(drop=True)


def main() -> None:
    args = parse_args()

    print(f"Loading price data for {args.price_ticker} ...")
    price = load_price(args.price_ticker)
    print(f"  {len(price)} trading days: {price['Date'].min().date()} ~ {price['Date'].max().date()}")

    print(f"\nLoading + filtering insider-selling news from {args.news_tickers} ...")
    insider_news = load_insider_selling_news(args.news_tickers)
    if len(insider_news) == 0:
        print("No insider-selling headlines found. Nothing to analyze.")
        return

    print(f"\nScoring sentiment for {len(insider_news)} headlines with {args.model_dir} ...")
    scores = batch_predict(insider_news["title"].fillna("").tolist(), args.model_dir, batch_size=32)
    insider_news = pd.concat([insider_news.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)

    # 對齊到「新聞發布當天或往前最近一個交易日」，跟 merge_price_news.py 同市場對齊邏輯一致
    # （這些是美股新聞，用 AAPL 自己的交易日曆對齊是合理的「同市場」假設）
    aligned = pd.merge_asof(
        insider_news.sort_values("pub_date"),
        price[["Date"] + [f"forward_return_{n}d" for n in FORWARD_DAYS] + ["Close"]],
        left_on="pub_date",
        right_on="Date",
        direction="backward",
    )
    unmatched = aligned["Date"].isna().sum()
    if unmatched:
        print(f"Warning: {unmatched} insider-selling news items had no matching trading date (before earliest price data), dropped.")
        aligned = aligned.dropna(subset=["Date"])

    output_path = OUTPUT_DIR / f"insider_selling_news_{args.price_ticker}.csv"
    aligned.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved aligned insider-selling news + forward returns to: {output_path}")

    # 依交易日聚合：當天有幾則內部人賣股新聞、平均情緒
    daily = (
        aligned.groupby("Date")
        .agg(
            insider_selling_count=("title", "count"),
            avg_bearish=("bearish", "mean"),
            avg_bullish=("bullish", "mean"),
        )
        .reset_index()
    )
    daily["net_sentiment"] = daily["avg_bullish"] - daily["avg_bearish"]
    daily = daily.merge(price[["Date"] + [f"forward_return_{n}d" for n in FORWARD_DAYS]], on="Date", how="left")

    print(f"\n{len(daily)} distinct trading days had at least one insider-selling headline "
          f"(out of {len(price)} total trading days, {len(daily) / len(price) * 100:.1f}%).")

    print("\n=== 純統計觀察（樣本數偏小，僅供參考，不是正式訊號）===")
    for n in FORWARD_DAYS:
        col = f"forward_return_{n}d"
        event_days = daily.dropna(subset=[col])
        baseline_days = price.dropna(subset=[col])
        if len(event_days) < 3:
            print(f"\n[{n}日後報酬率] 樣本數 {len(event_days)} 筆，太少無法算相關係數。")
            continue

        event_mean = event_days[col].mean()
        baseline_mean = baseline_days[col].mean()
        r, p = stats.pearsonr(event_days["insider_selling_count"], event_days[col])
        sr, sp = None, None
        sent_valid = event_days.dropna(subset=["net_sentiment"])
        if len(sent_valid) >= 3:
            sr, sp = stats.pearsonr(sent_valid["net_sentiment"], sent_valid[col])

        print(f"\n[{n}日後報酬率]")
        print(f"  有內部人賣股新聞當天（n={len(event_days)}）：平均 {event_mean * 100:+.2f}%")
        print(f"  全部交易日基準（n={len(baseline_days)}）：平均 {baseline_mean * 100:+.2f}%")
        print(f"  差異：{(event_mean - baseline_mean) * 100:+.2f} 個百分點")
        print(f"  新聞則數 vs. 報酬率 Pearson r={r:.3f} (p={p:.3f})")
        if sr is not None:
            print(f"  net_sentiment vs. 報酬率 Pearson r={sr:.3f} (p={sp:.3f}, n={len(sent_valid)})")
        if p >= 0.05:
            print("  ⚠️ p-value >= 0.05，統計上不顯著，這個差異很可能只是雜訊。")

    daily_output_path = OUTPUT_DIR / f"insider_selling_daily_{args.price_ticker}.csv"
    daily.to_csv(daily_output_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved daily aggregation to: {daily_output_path}")
    print(
        "\n提醒：這是純觀察性分析，樣本數(幾十筆新聞、十幾個交易日)遠不足以做成正式訊號策略，"
        "而且新聞提到的公司多半不是股價本身（AAPL），此分析把它當成廣義市場情緒雜訊指標，"
        "不是「該公司內部人賣股 -> 該公司股價」的因果分析。"
    )


if __name__ == "__main__":
    main()
