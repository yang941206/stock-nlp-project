"""算 net_sentiment 跟未來 N 天報酬率的相關係數（純觀察用，不是策略），
並把結果 append 進歷史記錄檔，方便長期追蹤：樣本數有沒有增加、相關係數有沒有穩定下來。

用法:
    python correlation_analysis.py --ticker 2330.TW_x_US
    python correlation_analysis.py --ticker 2330.TW_x_US --forward-days 3
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correlation between news sentiment and forward returns (observational only).")
    parser.add_argument("--ticker", default="2330.TW_x_US", help="feature table 的 ticker 標籤")
    parser.add_argument("--forward-days", type=int, default=1, help="用幾天後的報酬率算相關係數（預設：1，隔天）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    features_path = PROCESSED_DIR / f"{args.ticker}_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"{features_path} not found. Run build_feature_table.py first.")

    df = pd.read_csv(features_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["forward_return"] = df["Close"].shift(-args.forward_days) / df["Close"] - 1

    valid = df[df["news_count"] > 0].dropna(subset=["net_sentiment", "forward_return"])
    n = len(valid)

    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if n < 3:
        print(f"樣本數只有 {n} 天，不足以算相關係數（至少要 3 筆），本次先跳過，只記錄樣本數。")
        row = {
            "run_timestamp": run_timestamp,
            "forward_days": args.forward_days,
            "n_samples": n,
            "pearson_r": None,
            "pearson_p": None,
            "spearman_rho": None,
            "spearman_p": None,
            "date_range_start": valid["Date"].min().strftime("%Y-%m-%d") if n else None,
            "date_range_end": valid["Date"].max().strftime("%Y-%m-%d") if n else None,
        }
    else:
        r, p = stats.pearsonr(valid["net_sentiment"], valid["forward_return"])
        rho, p_rho = stats.spearmanr(valid["net_sentiment"], valid["forward_return"])
        print(f"n={n}, Pearson r={r:.4f} (p={p:.4f}), Spearman rho={rho:.4f} (p={p_rho:.4f})")
        row = {
            "run_timestamp": run_timestamp,
            "forward_days": args.forward_days,
            "n_samples": n,
            "pearson_r": round(r, 4),
            "pearson_p": round(p, 4),
            "spearman_rho": round(rho, 4),
            "spearman_p": round(p_rho, 4),
            "date_range_start": valid["Date"].min().strftime("%Y-%m-%d"),
            "date_range_end": valid["Date"].max().strftime("%Y-%m-%d"),
        }

    history_path = PROCESSED_DIR / f"{args.ticker}_correlation_history.csv"
    history_df = pd.DataFrame([row])
    if history_path.exists():
        existing = pd.read_csv(history_path)
        history_df = pd.concat([existing, history_df], ignore_index=True)
    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")

    print(f"已寫入歷史記錄: {history_path}（累計 {len(history_df)} 筆執行記錄）")
    print(
        "\n提醒: 這是純觀察用的相關係數，不是策略依據。p-value 沒有明顯小於 0.05 之前，"
        "任何看起來的相關性都可能只是樣本數太少造成的巧合。"
    )


if __name__ == "__main__":
    main()
