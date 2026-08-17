"""探索性分析：net_sentiment vs. 不同天數的累積報酬率，純觀察，不寫進正式的
{ticker}_correlation_history.csv（那份是排程每天固定用 forward_days=1 在追蹤的正式記錄，
這裡是一次性測試多個 horizon，混在一起會讓 dashboard 的歷史圖表失真，所以獨立輸出）。

動機：README「已知限制」記錄了樣本量衝到 129 天之後，net_sentiment vs. 隔天報酬率仍然沒有
統計上顯著的關聯（4 支台股 p-value 全部 > 0.05）。這裡測試「會不會是效果要累積幾天才會顯現」，
拉長 forward_days 到 3/5/10/20 天看看。

用法:
    python multi_horizon_correlation.py
    python multi_horizon_correlation.py --tickers 2330.TW_x_US --forward-days 1 3 5 10 20
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore net_sentiment vs. multi-horizon forward returns (observational only).")
    parser.add_argument(
        "--tickers", nargs="+",
        default=["2330.TW_x_US", "0050.TW_x_US", "2317.TW_x_US", "2454.TW_x_US"],
    )
    parser.add_argument("--forward-days", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []

    for ticker in args.tickers:
        features_path = PROCESSED_DIR / f"{ticker}_features.csv"
        if not features_path.exists():
            print(f"  (skip {ticker}: {features_path.name} not found)")
            continue

        df = pd.read_csv(features_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        for fd in args.forward_days:
            d = df.copy()
            d["forward_return"] = d["Close"].shift(-fd) / d["Close"] - 1
            valid = d[d["news_count"] > 0].dropna(subset=["net_sentiment", "forward_return"])
            n = len(valid)
            if n < 3:
                rows.append({"ticker": ticker, "forward_days": fd, "n": n, "pearson_r": None, "pearson_p": None, "spearman_rho": None, "spearman_p": None})
                continue
            r, p = stats.pearsonr(valid["net_sentiment"], valid["forward_return"])
            rho, p_rho = stats.spearmanr(valid["net_sentiment"], valid["forward_return"])
            rows.append({
                "ticker": ticker, "forward_days": fd, "n": n,
                "pearson_r": round(r, 4), "pearson_p": round(p, 4),
                "spearman_rho": round(rho, 4), "spearman_p": round(p_rho, 4),
            })

    result = pd.DataFrame(rows)
    pd.set_option("display.width", 120)
    print(result.to_string(index=False))

    sig = result[(result["pearson_p"].notna()) & (result["pearson_p"] < 0.05)]
    print(f"\n達到 p < 0.05 的組合數: {len(sig)} / {len(result)}")
    if len(sig):
        print(sig.to_string(index=False))
    print(
        "\n提醒：這是探索性分析，同時測試多個 ticker x forward_days 組合，就算隨機雜訊也有機會"
        "偶然出現 p < 0.05（multiple comparisons，這裡共測了 "
        f"{len(result)} 個組合，純用 0.05 門檻，預期光靠運氣就會有 ~{len(result) * 0.05:.1f} 個"
        "「看起來顯著」的假陽性）。這份結果不寫進正式的 correlation_history.csv，純觀察用。"
    )


if __name__ == "__main__":
    main()
