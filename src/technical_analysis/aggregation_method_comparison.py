"""探索性分析：比較不同的「單日新聞情緒聚合方式」跟報酬率的相關性，純觀察，不寫進正式的
{ticker}_correlation_history.csv（原因同 multi_horizon_correlation.py：避免跟排程每天固定
用 baseline 方式追蹤的正式記錄混在一起）。

動機：build_feature_table.py 目前用「當天所有新聞 bullish 機率平均 - bearish 機率平均」算
net_sentiment。實測發現正式新聞裡有 66% 被模型信心十足地判成 neutral（不是模稜兩可，是真的
判定成中性，機率值接近 0/1 兩極化，不是卡在中間），這些 neutral 新聞在簡單平均裡會把有方向性
新聞的訊號稀釋掉。這裡測試三種替代聚合方式，看能不能讓相關性浮現：
    - exclude_neutral：只用非 neutral 標籤的新聞算平均
    - conviction_weighted：用 (1 - neutral 機率) 當權重做加權平均
    - pct_bullish_minus_bearish：不用機率平均，改用「當天 bullish 標籤則數% - bearish 標籤則數%」

用法:
    python aggregation_method_comparison.py
    python aggregation_method_comparison.py --tickers 2330.TW --forward-days 1 5 10
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

METHODS = ["baseline_mean_all", "exclude_neutral", "conviction_weighted", "pct_bullish_minus_bearish"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare sentiment aggregation methods vs. forward returns (observational only).")
    parser.add_argument("--tickers", nargs="+", default=["2330.TW", "0050.TW", "2317.TW", "2454.TW"])
    parser.add_argument("--forward-days", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    return parser.parse_args()


def load_daily_price(ticker: str, forward_days: list) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / f"{ticker}_x_US_features.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    for fd in forward_days:
        df[f"fwd_{fd}"] = df["Close"].shift(-fd) / df["Close"] - 1
    return df[["Date"] + [f"fwd_{fd}" for fd in forward_days]]


def aggregate(sentiment_df: pd.DataFrame, method: str) -> pd.DataFrame:
    sentiment_df = sentiment_df.copy()
    sentiment_df["trading_date"] = pd.to_datetime(sentiment_df["trading_date"]).dt.normalize()

    if method == "baseline_mean_all":
        g = sentiment_df.groupby("trading_date")
        out = g["sentiment_bullish"].mean() - g["sentiment_bearish"].mean()
    elif method == "exclude_neutral":
        non_neutral = sentiment_df[sentiment_df["sentiment_label"] != "neutral"]
        g = non_neutral.groupby("trading_date")
        out = g["sentiment_bullish"].mean() - g["sentiment_bearish"].mean()
    elif method == "conviction_weighted":
        w = 1 - sentiment_df["sentiment_neutral"]
        net = sentiment_df["sentiment_bullish"] - sentiment_df["sentiment_bearish"]
        sentiment_df["_wnet"] = net * w
        g = sentiment_df.groupby("trading_date")
        out = g["_wnet"].sum() / g.apply(lambda d: (1 - d["sentiment_neutral"]).sum())
    elif method == "pct_bullish_minus_bearish":
        g = sentiment_df.groupby("trading_date")["sentiment_label"]
        out = g.apply(lambda s: (s == "bullish").mean() - (s == "bearish").mean())
    else:
        raise ValueError(f"Unknown method: {method}")

    return out.rename("net_sentiment_alt").reset_index().rename(columns={"trading_date": "Date"})


def main() -> None:
    args = parse_args()
    rows = []

    for ticker in args.tickers:
        sentiment_path = PROCESSED_DIR / f"{ticker}_x_US_merged_sentiment.csv"
        if not sentiment_path.exists():
            print(f"  (skip {ticker}: {sentiment_path.name} not found)")
            continue
        sentiment_df = pd.read_csv(sentiment_path)
        price_df = load_daily_price(ticker, args.forward_days)

        for method in METHODS:
            agg = aggregate(sentiment_df, method)
            merged = agg.merge(price_df, on="Date", how="inner")
            for fd in args.forward_days:
                valid = merged.dropna(subset=["net_sentiment_alt", f"fwd_{fd}"])
                n = len(valid)
                if n < 10:
                    continue
                r, p = stats.pearsonr(valid["net_sentiment_alt"], valid[f"fwd_{fd}"])
                rows.append({"ticker": ticker, "method": method, "forward_days": fd, "n": n, "pearson_r": round(r, 4), "pearson_p": round(p, 4)})

    result = pd.DataFrame(rows)
    pd.set_option("display.width", 140)
    print(result.to_string(index=False))

    print("\n=== 依方法彙總（p < 0.05 的組合數） ===")
    for method in METHODS:
        sub = result[result["method"] == method]
        sig = sub[sub["pearson_p"] < 0.05]
        print(f"  {method}: {len(sig)} / {len(sub)} 組合達到 p<0.05，|r| 最大值 = {sub['pearson_r'].abs().max():.4f}")

    sig_all = result[result["pearson_p"] < 0.05]
    if len(sig_all):
        print("\n達到 p<0.05 的組合明細：")
        print(sig_all.to_string(index=False))
    print(
        "\n提醒：這是探索性分析，同時測試多種方法 x horizon 組合，一樣有 multiple comparisons 的問題，"
        "結果不寫進正式的 correlation_history.csv，純觀察用。"
    )


if __name__ == "__main__":
    main()
