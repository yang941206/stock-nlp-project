"""本機儀表板，用來看股價/新聞情緒/技術指標整合後的樣子。

用法:
    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65
SAMPLE_THRESHOLD = 60

TICKERS = {
    "AAPL": {
        "features": "AAPL_features.csv",
        "signals": "AAPL_signals.csv",
        "sentiment": "AAPL_merged_sentiment.csv",
        "correlation_history": None,
        "note": "新聞來源：AAPL 自己的新聞（Yahoo + Alpha Vantage）",
    },
    "2330.TW": {
        "features": "2330.TW_x_US_features.csv",
        "signals": "2330.TW_x_US_signals.csv",
        "sentiment": "2330.TW_x_US_merged_sentiment.csv",
        "correlation_history": "2330.TW_x_US_correlation_history.csv",
        "note": "跨市場實驗：新聞來自 AAPL/NVDA，對齊到新聞發布後下一個台股交易日",
    },
    "0050.TW": {
        "features": "0050.TW_x_US_features.csv",
        "signals": "0050.TW_x_US_signals.csv",
        "sentiment": "0050.TW_x_US_merged_sentiment.csv",
        "correlation_history": "0050.TW_x_US_correlation_history.csv",
        "note": "跨市場實驗：台灣50 ETF，成分股橫跨多產業，跟半導體新聞的相關性天生較弱",
    },
    "2317.TW": {
        "features": "2317.TW_x_US_features.csv",
        "signals": "2317.TW_x_US_signals.csv",
        "sentiment": "2317.TW_x_US_merged_sentiment.csv",
        "correlation_history": "2317.TW_x_US_correlation_history.csv",
        "note": "跨市場實驗：鴻海，Apple 供應鏈概念股，跟 AAPL 新聞主題相關性較高",
    },
    "2454.TW": {
        "features": "2454.TW_x_US_features.csv",
        "signals": "2454.TW_x_US_signals.csv",
        "sentiment": "2454.TW_x_US_merged_sentiment.csv",
        "correlation_history": "2454.TW_x_US_correlation_history.csv",
        "note": "跨市場實驗：聯發科，半導體概念股，跟 NVDA 新聞主題相關性較高",
    },
}

st.set_page_config(page_title="Stock NLP Dashboard", layout="wide")
st.title("📊 Stock NLP Dashboard")

# 1. 選擇股票
ticker = st.sidebar.selectbox("選擇股票", list(TICKERS.keys()))
config = TICKERS[ticker]
st.sidebar.caption(config["note"])

features_path = PROCESSED_DIR / config["features"]
if not features_path.exists():
    st.error(f"找不到 {features_path.name}，請先執行對應的 pipeline（build_feature_table.py）。")
    st.stop()

features = pd.read_csv(features_path)
features["Date"] = pd.to_datetime(features["Date"])
features = features.sort_values("Date").reset_index(drop=True)

# 2. 股價走勢 + net_sentiment 疊圖
st.subheader("股價走勢 + 新聞情緒")
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=features["Date"], y=features["Close"], name="Close", line=dict(color="steelblue")))
sentiment_days = features.dropna(subset=["net_sentiment"])
fig_price.add_trace(
    go.Scatter(
        x=sentiment_days["Date"],
        y=sentiment_days["net_sentiment"],
        name="net_sentiment",
        mode="markers+lines",
        yaxis="y2",
        line=dict(color="orange"),
    )
)
fig_price.update_layout(
    yaxis=dict(title="Close"),
    yaxis2=dict(title="net_sentiment", overlaying="y", side="right"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=40),
)
st.plotly_chart(fig_price, use_container_width=True)

# 3. RSI 走勢圖
st.subheader("RSI 走勢")
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=features["Date"], y=features["rsi_14"], name="RSI(14)", line=dict(color="purple")))
fig_rsi.add_hline(y=RSI_OVERSOLD, line_dash="dash", line_color="green", annotation_text=f"超賣 {RSI_OVERSOLD}")
fig_rsi.add_hline(y=RSI_OVERBOUGHT, line_dash="dash", line_color="red", annotation_text=f"超買 {RSI_OVERBOUGHT}")
fig_rsi.update_layout(yaxis=dict(title="RSI", range=[0, 100]), margin=dict(t=20))
st.plotly_chart(fig_rsi, use_container_width=True)

latest_rsi = features.iloc[-1]["rsi_14"]
latest_date = features.iloc[-1]["Date"].date()
if latest_rsi >= RSI_OVERBOUGHT:
    st.warning(f"最新交易日（{latest_date}）RSI = {latest_rsi:.1f}，已觸及超買門檻（{RSI_OVERBOUGHT}）")
elif latest_rsi <= RSI_OVERSOLD:
    st.warning(f"最新交易日（{latest_date}）RSI = {latest_rsi:.1f}，已觸及超賣門檻（{RSI_OVERSOLD}）")
else:
    st.info(f"最新交易日（{latest_date}）RSI = {latest_rsi:.1f}，位於中性區間（{RSI_OVERSOLD} ~ {RSI_OVERBOUGHT}）")

# 4. 已對齊新聞的交易日列表
st.subheader("已對齊新聞的交易日")
sentiment_path = PROCESSED_DIR / config["sentiment"]
if sentiment_path.exists():
    sentiment_df = pd.read_csv(sentiment_path)
    date_col = "trading_date" if "trading_date" in sentiment_df.columns else "aligned_trading_date"
    sentiment_df[date_col] = pd.to_datetime(sentiment_df[date_col])

    news_days = sorted(sentiment_df[date_col].dt.date.unique(), reverse=True)
    st.caption(f"共 {len(news_days)} 個交易日有新聞覆蓋")

    display_cols = [
        c
        for c in ["news_ticker", "title", "sentiment_label", "sentiment_bullish", "sentiment_bearish", "sentiment_neutral"]
        if c in sentiment_df.columns
    ]

    for d in news_days:
        day_news = sentiment_df[sentiment_df[date_col].dt.date == d]
        with st.expander(f"{d}（{len(day_news)} 則新聞）"):
            st.dataframe(day_news[display_cols], use_container_width=True, hide_index=True)
else:
    st.info(f"找不到 {sentiment_path.name}，請先執行 batch_predict.py / merge_cross_market_news.py。")

# 5. 相關係數統計
st.subheader("相關係數統計（net_sentiment vs. 隔天報酬率，純觀察用）")

n_days = int((features["news_count"] > 0).sum())
col1, col2, col3 = st.columns(3)
col1.metric("累積樣本天數", n_days)
col2.metric(f"距離 {SAMPLE_THRESHOLD} 天可信門檻", max(0, SAMPLE_THRESHOLD - n_days))

valid = features[features["news_count"] > 0].copy()
valid["forward_return"] = features["Close"].shift(-1) / features["Close"] - 1
valid = valid.dropna(subset=["net_sentiment", "forward_return"])

if len(valid) >= 3:
    r, p = stats.pearsonr(valid["net_sentiment"], valid["forward_return"])
    col3.metric("Pearson r", f"{r:.3f}", help=f"p-value = {p:.3f}（n={len(valid)}）")
else:
    col3.metric("Pearson r", "樣本不足")

if len(valid) < 3:
    st.caption("樣本數不足 3 筆，無法算相關係數。")
elif p >= 0.05:
    st.caption(f"⚠️ p-value = {p:.3f} ≥ 0.05，統計上不顯著，相關係數僅供觀察，不代表真的有關聯。")

if config["correlation_history"]:
    hist_path = PROCESSED_DIR / config["correlation_history"]
    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        hist["run_timestamp"] = pd.to_datetime(hist["run_timestamp"])
        hist = hist.sort_values("run_timestamp")

        st.caption("樣本數隨時間累積")
        st.line_chart(hist.set_index("run_timestamp")[["n_samples"]])

        corr_hist = hist.dropna(subset=["pearson_r"])
        if len(corr_hist):
            st.caption("相關係數隨時間變化")
            st.line_chart(corr_hist.set_index("run_timestamp")[["pearson_r", "spearman_rho"]])
    else:
        st.info("尚未累積相關係數歷史記錄（排程跑過一次後才會有資料）。")
else:
    st.info(f"{ticker} 目前沒有排程在累積相關係數歷史記錄，只顯示即時算出來的數字。")
