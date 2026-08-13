"""排程執行完後，把當日摘要發到 Telegram。

內容包含: 今天新增新聞則數與額度用量（所有股票共用同一份新聞，只列一次）、
每支股票各自的累積樣本天數、最新 net_sentiment/RSI、有沒有觸發買賣訊號。

用法:
    python telegram_notify.py --tickers 2330.TW 0050.TW 2317.TW 2454.TW --news-tickers AAPL NVDA
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
NEWS_DIR = PROJECT_ROOT / "data" / "news"
STATUS_PATH = PROJECT_ROOT / "logs" / "last_run_status.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a daily summary to Telegram.")
    parser.add_argument(
        "--tickers", nargs="+", default=["2330.TW", "0050.TW", "2317.TW", "2454.TW"],
        help="要回報的股票（會自動接上 _x_US 找對應的 feature table/signals 檔）"
    )
    parser.add_argument("--news-tickers", nargs="+", default=["AAPL", "NVDA"])
    parser.add_argument("--daily-limit", type=int, default=25, help="Alpha Vantage 每日額度上限")
    parser.add_argument("--sample-threshold", type=int, default=60, help="樣本天數可信門檻下限（預設：60）")
    return parser.parse_args()


def load_json(path: Path, default: dict) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def build_shared_header(args: argparse.Namespace) -> list:
    """今天新增新聞則數 + 額度用量，所有股票共用同一份新聞，只需要算一次。"""
    lines = []

    quota = load_json(NEWS_DIR / ".av_quota.json", {})
    requests_used = quota.get("requests_used", "?")

    per_ticker_new = {}
    for t in args.news_tickers:
        run_info = load_json(NEWS_DIR / f"{t}_av_last_run.json", {})
        per_ticker_new[t] = run_info.get("net_new_items", 0)
    total_new = sum(per_ticker_new.values())
    breakdown = "、".join(f"{k} {v} 則" for k, v in per_ticker_new.items())

    lines.append(f"📰 今日新增新聞: {total_new} 則（{breakdown}，AAPL/NVDA 新聞為所有台股共用）")
    lines.append(f"🔑 Alpha Vantage 今日額度: {requests_used}/{args.daily_limit}")
    return lines


def build_ticker_block(ticker: str, args: argparse.Namespace) -> list:
    label = f"{ticker}_x_US"
    lines = [f"\n— {ticker} —"]

    features_path = PROCESSED_DIR / f"{label}_features.csv"
    if not features_path.exists():
        lines.append("⚠️ 找不到 feature table，可能是今天的 pipeline 還沒跑到 build_feature_table.py 這一步。")
        return lines

    features = pd.read_csv(features_path)
    features["Date"] = pd.to_datetime(features["Date"])
    features = features.sort_values("Date").reset_index(drop=True)

    n_days = int((features["news_count"] > 0).sum())
    remaining = max(0, args.sample_threshold - n_days)
    if remaining > 0:
        lines.append(f"📊 樣本天數: {n_days} 天（距 {args.sample_threshold} 天門檻還差 {remaining} 天）")
    else:
        lines.append(f"📊 樣本天數: {n_days} 天（已達 {args.sample_threshold} 天門檻）")

    latest = features.iloc[-1]
    lines.append(f"📈 最新交易日（{latest['Date'].date()}）RSI: {latest['rsi_14']:.1f}, Close: {latest['Close']:.1f}")

    news_days = features[features["news_count"] > 0]
    if len(news_days):
        sentiment_row = news_days.iloc[-1]
        lines.append(
            f"💬 最新有新聞的交易日（{sentiment_row['Date'].date()}）"
            f"net_sentiment: {sentiment_row['net_sentiment']:+.3f}"
        )
    else:
        lines.append("💬 目前還沒有任何交易日有新聞覆蓋。")

    signals_path = PROCESSED_DIR / f"{label}_signals.csv"
    if signals_path.exists():
        signals = pd.read_csv(signals_path)
        signals["Date"] = pd.to_datetime(signals["Date"])
        signals = signals.sort_values("Date").reset_index(drop=True)
        latest_signal = signals.iloc[-1]["signal"]
        if latest_signal != "hold":
            lines.append(f"🚨 最新交易日觸發訊號: {latest_signal.upper()}")
        else:
            lines.append("⚪ 最新交易日: 無訊號（hold）")
    else:
        lines.append("⚠️ 找不到 signals 檔案。")

    return lines


def build_health_warning() -> str:
    """讀 run_daily_pipeline.ps1 寫的健檢狀態檔，有失敗步驟/資料沒更新就組一段警告文字。"""
    status = load_json(STATUS_PATH, {})
    failed_steps = status.get("failed_steps", [])
    stale_tickers = status.get("stale_tickers", [])
    if not failed_steps and not stale_tickers:
        return ""

    lines = ["🔴 今天執行有異常，以下資料可能不完整/不是最新："]
    if failed_steps:
        lines.append(f"　失敗步驟（{len(failed_steps)}）: " + "、".join(failed_steps))
    if stale_tickers:
        lines.append(f"　資料未更新（{len(stale_tickers)}）: " + "、".join(stale_tickers))
    lines.append("　詳見 logs\\daily_pipeline.log\n")
    return "\n".join(lines)


def build_message(args: argparse.Namespace) -> str:
    lines = build_shared_header(args)
    for ticker in args.tickers:
        lines.extend(build_ticker_block(ticker, args))
    lines.append("\n（相關係數/訊號樣本數還太少，僅供觀察，不是操作建議）")
    return "\n".join(lines)


def send_telegram_message(text: str, token: str, chat_id: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    if not result.get("ok"):
        raise RuntimeError(f"Telegram API error: {result}")


def main() -> None:
    args = parse_args()

    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or token == "your_telegram_bot_token_here":
        raise RuntimeError("TELEGRAM_BOT_TOKEN 未設定，請先在 .env 填入。")
    if not chat_id or chat_id == "your_telegram_chat_id_here":
        raise RuntimeError("TELEGRAM_CHAT_ID 未設定，請先在 .env 填入。")

    header = f"📅 stock-nlp-project 每日報告 — {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    message = header + build_health_warning() + build_message(args)

    send_telegram_message(message, token, chat_id)
    print("Telegram 訊息已送出:")
    print(message)


if __name__ == "__main__":
    main()
