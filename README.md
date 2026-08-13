# stock-nlp-project

股價資料 + 新聞情緒分析 + 技術分析整合專案。用 [yfinance](https://github.com/ranaroussi/yfinance) 抓股價、
用 [Alpha Vantage](https://www.alphavantage.co/) / Yahoo Finance 抓新聞，fine-tune 一個 FinBERT-based
的新聞情緒分類模型，再把情緒分數跟 RSI/MACD/布林通道等技術指標合併成一份特徵表，最後產生簡單的買賣訊號。

> 目前是**個人研究/原型驗證用專案**，不是可以直接拿去用真錢交易的系統。詳見下方「已知限制」。

## 這個專案在做什麼

1. **抓股價**（`yfinance`）：歷史 OHLCV 資料。
2. **抓新聞**：Yahoo Finance（即時，但只有最新 ~10 則、無歷史回溯）+ Alpha Vantage
   （`NEWS_SENTIMENT` API，可回溯歷史，但免費方案每天限 25 次請求）。
3. **情緒分類模型**：以 `ProsusAI/finbert` 為起點，用 HuggingFace `transformers` 在
   `zeroshot/twitter-financial-news-sentiment` 資料集上 fine-tune 一個 3 類（bearish / bullish /
   neutral）情緒分類器，用來對新聞標題打分。
4. **技術分析**：用 `ta` 套件算 RSI、MACD、布林通道、SMA/EMA。
5. **整合**：把每日的新聞情緒統計（平均看多/看空機率、net_sentiment）跟當天的技術指標合併成一份
   特徵表，再用簡單規則（情緒 + RSI 極端值同時出現）產生示範用買賣訊號，並計算訊號後 N 天的報酬率
   做為粗略回測。
6. **通知**：排程跑完後自動發一則摘要到 Telegram（`telegram_notify.py`）。
7. **視覺化**：本機 Streamlit 儀表板（`app.py`），可以選股票看股價/情緒疊圖、RSI、新聞列表、相關係數。

## 資料流程圖

```
fetch_stock_price.py                    fetch_news.py (Yahoo)
        │                                        │
        │  data/raw/{ticker}_*.csv               │  fetch_news_alphavantage.py (Alpha Vantage,
        │                                        │  可回溯歷史，會跟 Yahoo 版合併去重)
        │                                        ▼
        │                              data/news/{ticker}_news.csv
        │                                        │
        └───────────────┬────────────────────────┘
                         ▼
              merge_price_news.py
              （新聞依 pub_date 對齊到當天/最近交易日的股價，merge_asof backward）
                         │
                         ▼
          data/processed/{ticker}_merged.csv
                         │
                         ▼
              batch_predict.py
              （用 fine-tune 好的情緒模型，對每則新聞標題打分）
                         │
                         ▼
      data/processed/{ticker}_merged_sentiment.csv
                         │
                         │         indicators.py
                         │         （RSI/MACD/布林通道/SMA/EMA）
                         │                 │
                         │                 ▼
                         │   data/processed/{ticker}_indicators.csv
                         │                 │
                         └────────┬────────┘
                                  ▼
                     build_feature_table.py
                     （技術指標 + 每日新聞情緒統計，依日期合併）
                                  │
                                  ▼
                data/processed/{ticker}_features.csv
                                  │
                                  ▼
                          signals.py
              （情緒 + RSI 極端值 → buy/sell/hold，並計算 forward return）
                                  │
                                  ▼
                data/processed/{ticker}_signals.csv
```

另外獨立的一條線是**情緒模型的訓練**：

```
train.py（在 zeroshot/twitter-financial-news-sentiment 上 fine-tune ProsusAI/finbert）
   │
   ▼
models/finbert-sentiment/final/
   │
   ├─→ predict.py        單句情緒推論（CLI 手動測試用）
   └─→ batch_predict.py  批次情緒推論（接在資料流程圖裡）
```

還有一條**實驗性的跨市場流程**（已排進每日排程，見下方「排程狀態」）：把美股新聞情緒對齊到台股
下一個交易日，觀察有沒有領先/落後的關聯性。跟上面主流程的差異是新聞股票（AAPL/NVDA）跟股價股票
不是同一支。新聞只抓一次，**多支台股共用同一份 AAPL/NVDA 新聞**（目前是 2330.TW、0050.TW、
2317.TW、2454.TW 這 4 支，見下方「排程狀態」）：

```
                                         fetch_news_alphavantage.py (AAPL / NVDA)
                                                         │  （只抓一次，4 支台股共用）
                                                         ▼
                                         data/news/{AAPL,NVDA}_news.csv
                                                         │
for each 台股 ticker (2330.TW / 0050.TW / 2317.TW / 2454.TW):
                    │
   fetch_stock_price.py (該台股)              merge_cross_market_news.py
                    │                          （先用 news_filters.py 濾掉機構持股異動公告，
                    ▼                           再用 merge_asof forward + allow_exact_matches=False
        data/raw/{ticker}_*.csv                 對齊到「新聞發布後下一個台股交易日」，並跑情緒分類）
                    │                                        │
                    ▼                                        ▼
               indicators.py              data/processed/{ticker}_news_from_AAPL_NVDA.csv
                    │                                        │
                    ▼                                        │
data/processed/{ticker}_indicators.csv                       │
                    │                                        │
                    └──────────┬─────────────────────────────┘
                                ▼
                   prepare_crossmarket_adapter.py
                   （複製/改名成 build_feature_table.py / signals.py 看得懂的格式，
                    用合成標籤 {ticker}_x_US 代表「該台股股價 + 美股新聞情緒」）
                                ▼
                   build_feature_table.py → signals.py（沿用主流程的既有腳本，不用改程式碼）
                                ▼
                   correlation_analysis.py
                   （算 net_sentiment vs. 隔天報酬率的相關係數，純觀察用，append 進歷史記錄）
                                ▼
                data/processed/{ticker}_x_US_correlation_history.csv

（所有台股跑完之後）
                                ▼
                   telegram_notify.py
                   （每支台股各自的摘要 + 共用的新聞/額度資訊，發一則訊息到 Telegram）
```

## 資料夾/檔案說明

```
stock-nlp-project/
├── venv/                          Python 虛擬環境（獨立安裝的 Python 3.13，不是 conda）
├── data/
│   ├── raw/                       yfinance 抓下來的原始股價 CSV（{ticker}_{period}.csv）
│   ├── news/                      新聞 CSV + 抓取進度/額度追蹤檔
│   │   ├── {ticker}_news.csv          Yahoo + Alpha Vantage 合併去重後的新聞（有 source_api 欄位區分來源）
│   │   ├── {ticker}_av_state.json     Alpha Vantage 回溯進度（記錄目前回溯到的最早時間點）
│   │   └── .av_quota.json             Alpha Vantage 當日已用請求次數（只追蹤透過 fetch_news_alphavantage.py 發出的請求）
│   └── processed/                 各階段處理後的中間/最終資料（見上方流程圖）
│       ├── {ticker}_merged.csv
│       ├── {ticker}_merged_sentiment.csv
│       ├── {ticker}_indicators.csv
│       ├── {ticker}_features.csv
│       ├── {ticker}_signals.csv
│       ├── {ticker}_news_from_AAPL_NVDA.csv       跨市場：美股新聞對齊到下一個台股交易日 + 情緒分類（{ticker} = 2330.TW/0050.TW/2317.TW/2454.TW）
│       ├── {ticker}_x_US_*.csv                    跨市場實驗用合成標籤（indicators/merged_sentiment/features/signals，內容跟同名的 {ticker}_* 檔案意義不同，見上方流程圖說明）
│       └── {ticker}_x_US_correlation_history.csv  每支台股各自的相關係數歷史記錄，每次排程執行都會 append 一筆
├── models/
│   └── finbert-sentiment/final/   fine-tune 完成的情緒分類模型（HuggingFace 格式，含 tokenizer）
├── src/
│   ├── data_collection/
│   │   ├── fetch_stock_price.py       抓股價 → data/raw
│   │   ├── fetch_news.py              抓最新新聞（Yahoo Finance，經 yfinance）→ data/news
│   │   └── fetch_news_alphavantage.py 回溯歷史新聞（Alpha Vantage），自動追蹤進度與額度
│   ├── sentiment_model/
│   │   ├── train.py                   fine-tune 情緒分類模型 → models/finbert-sentiment/final
│   │   ├── predict.py                 單句情緒推論（手動測試用）
│   │   └── batch_predict.py           批次情緒推論，接在資料流程裡
│   ├── technical_analysis/
│   │   ├── indicators.py              計算 RSI/MACD/布林通道/均線
│   │   ├── build_feature_table.py     技術指標 + 每日情緒統計合併
│   │   ├── signals.py                 產生買賣訊號 + forward return 粗略回測
│   │   └── correlation_analysis.py    情緒 vs. 隔天報酬率的相關係數（觀察用），append 進歷史記錄
│   └── utils/
│       ├── merge_price_news.py         新聞依日期對齊股價（同一支股票，merge_asof backward）
│       ├── merge_cross_market_news.py  跨市場：美股新聞對齊到下一個台股交易日 + 情緒分類（實驗性）
│       ├── news_filters.py             過濾例行機構持股異動公告的關鍵字規則
│       ├── prepare_crossmarket_adapter.py  把跨市場合併結果轉成 build_feature_table.py/signals.py 看得懂的格式
│       └── telegram_notify.py          組每支股票的摘要，發一則訊息到 Telegram
├── scripts/
│   └── run_daily_pipeline.ps1     Windows 工作排程器每日執行用的包裝腳本（見下方「排程狀態」）
├── app.py                          Streamlit 本機儀表板：`streamlit run app.py`
├── logs/
│   ├── daily_pipeline.log             現行排程的執行紀錄（累加寫入）
│   └── fetch_news_alphavantage.log    舊版排程（只抓 AAPL）留下的歷史紀錄，現行排程不再寫入
├── requirements.txt                完整套件清單（含 ML fine-tune 用的 transformers/torch/datasets/accelerate）
├── requirements-base.txt           基本套件（資料抓取/處理/儀表板，不含 ML 套件）
├── .env / .env.example             API key / Bot Token 等敏感設定（.env 不會進 git，.env.example 是範本）
└── .gitignore
```

## 目前已知的限制

1. **新聞歷史資料量還不夠多**：即使已經串接 Alpha Vantage 回溯歷史，目前累積的也只有約 2 週、
   11 個交易日有新聞覆蓋（vs. 全年 252 個交易日）。`signals.py` 算出來的訊號數量/報酬率**目前完全
   不具統計意義**，純粹是驗證程式邏輯正確（見程式輸出的提醒訊息）。免費方案每天限 25 次請求，
   要累積出真正夠長的歷史需要持續跑排程一段時間，或改用付費方案。

2. **情緒分類模型是小樣本驗證版本**：目前的 fine-tune 是在
   `zeroshot/twitter-financial-news-sentiment`（推文語氣）上訓練的，訓練集用了 9,543 筆、驗證集
   2,388 筆，4 epoch（`eval_accuracy` ≈ 0.89）。這足以證明訓練流程正確、模型可用，但：
   - 訓練資料是「推文」語氣，面對「正式新聞標題」（陳述事實、較少表態）時，模型會傾向判斷為
     neutral，鑑別度較低。
   - 沒有針對特定產業/股票做領域微調。
   - 沒有做嚴謹的 train/val/test 三方切分或交叉驗證，`eval_accuracy` 只能當參考，不是嚴謹的
     模型評估結果。

3. **進場時機用「收盤價」，不是即時價格**：`indicators.py` 算的 RSI/MACD 等都是用當天**收盤後**的
   資料，`signals.py` 的 `forward_return` 也是用「訊號當天收盤價 → N 天後收盤價」計算報酬率。這代表
   策略隱含假設「收盤時就能用收盤價成交」，實務上做不到（無法卡在收盤瞬間進場），也沒有考慮滑價、
   手續費、稅金等交易成本。另外 `merge_price_news.py` 用 `merge_asof(direction="backward")` 把新聞
   對應到「當天或最近一個交易日」，如果新聞是當天盤中發布的，對應的當天收盤價其實已經反映了市場
   當天的反應，不是新聞發布當下的價格。

## 排程狀態

Windows 工作排程器已設定每天自動執行**完整的跨市場實驗流程**（4 支台股 × 共用新聞 → 各自整合報表
跟相關係數 → Telegram 通知）：

| 項目 | 內容 |
|---|---|
| 任務名稱 | `StockNLP_DailyPipeline` |
| 執行時間 | 每天 08:30（本機時間） |
| 執行內容 | `scripts\run_daily_pipeline.ps1`，依序執行： |
| | 1. `fetch_news_alphavantage.py --ticker AAPL --max-requests 10`（只抓一次，4 支台股共用） |
| | 2. `fetch_news_alphavantage.py --ticker NVDA --max-requests 10` |
| | 3. 對 `2330.TW`、`0050.TW`、`2317.TW`、`2454.TW` 各自依序跑： |
| | &nbsp;&nbsp;a. `fetch_stock_price.py --ticker {該台股} --period 1y` |
| | &nbsp;&nbsp;b. `merge_cross_market_news.py --price-ticker {該台股} --news-tickers AAPL NVDA` |
| | &nbsp;&nbsp;c. `indicators.py --ticker {該台股}` |
| | &nbsp;&nbsp;d. `prepare_crossmarket_adapter.py`（合成標籤 `{該台股}_x_US`） |
| | &nbsp;&nbsp;e. `build_feature_table.py` → `signals.py` → `correlation_analysis.py`（append 進各自的歷史記錄） |
| | 4. `telegram_notify.py --tickers 2330.TW 0050.TW 2317.TW 2454.TW`（每支台股一個小段落，發一則訊息） |
| 執行紀錄 | `logs\daily_pipeline.log`（累加寫入，UTF-8） |
| 歷史記錄 | 每支台股各自的 `data\processed\{ticker}_x_US_correlation_history.csv`——每次執行 append 一筆
（run_timestamp、樣本數、Pearson/Spearman 相關係數與 p-value），可以定期回來看樣本數有沒有增加、
相關係數有沒有穩定下來 |
| 額度分配 | AAPL / NVDA 各自最多 10 次請求（總共最多 20 次，留 5 次緩衝給手動測試）。這個額度是
「抓新聞」這一步專屬的，4 支台股**共用同一份**抓回來的 AAPL/NVDA 新聞，不會因為台股變多而分食
更多額度——加台股基本上不影響額度用量，只會多花一點點本地運算時間（跑情緒推論 + 指標計算）。
Alpha Vantage 免費方案每天 25 次上限是**帳號共用**的 |
| 通知 | Telegram Bot `@Stock_observer_bot`，Token/Chat ID 存在 `.env`（`TELEGRAM_BOT_TOKEN` /
`TELEGRAM_CHAT_ID`） |
| 限制 | 只有使用者登入時才會執行（沒有存密碼設定「無人登入也執行」）；用 `-StartWhenAvailable`，
錯過時間會在下次登入後盡快補跑，但不保證準時；額度計數器以 **UTC 日期**為界重置，跟台灣時間
（UTC+8）不完全對齊，08:30 本機時間有時可能還沒重置到新的一天額度 |

管理指令：

```powershell
# 查看任務狀態
Get-ScheduledTask -TaskName "StockNLP_DailyPipeline"

# 查看上次/下次執行時間
Get-ScheduledTaskInfo -TaskName "StockNLP_DailyPipeline"

# 移除排程
Unregister-ScheduledTask -TaskName "StockNLP_DailyPipeline" -Confirm:$false
```

> 舊的排程 `StockNLP_FetchNewsAlphaVantage`（只抓 AAPL）已經移除、換成上面這個。

## 之後要擴充的話，大概要改哪些地方

**多支股票：**
- 目前排程已經是多支台股（`2330.TW`/`0050.TW`/`2317.TW`/`2454.TW`）共用同一份 AAPL/NVDA 新聞的設計，
  `scripts\run_daily_pipeline.ps1` 裡的 `$TwTickers` 陣列直接加代號就能擴充，不用動迴圈邏輯本身。
  因為新聞是共用的，加台股幾乎不影響 Alpha Vantage 額度用量。
- 如果要加的是**美股**（自己的新聞、自己的股價，像 AAPL 現在這樣，不是跨市場對齊）：
  `fetch_stock_price.py` / `fetch_news.py` / `fetch_news_alphavantage.py` / `indicators.py` /
  `merge_price_news.py` / `batch_predict.py` / `build_feature_table.py` / `signals.py` 全部都已經是
  `--ticker` 參數化的，照順序把每支腳本跑一次即可。但這種美股會有**自己獨立的** Alpha Vantage 額度
  需求（不像台股是共用 AAPL/NVDA 新聞），加一支美股就要重新分配 25 次/天的額度預算。
- 情緒分類模型（`models/finbert-sentiment/final`）是通用的英文金融情緒模型，不綁定特定股票，
  多支股票可以共用同一個模型，不需要重新訓練。
- `telegram_notify.py --tickers` 跟 `app.py` 的 `TICKERS` dict 都要記得同步加新股票，不然新股票
  的資料算出來了，但看不到通知/儀表板。

**更多新聞來源：**
- 照 `fetch_news_alphavantage.py` 的模式寫新的抓取腳本：只要輸出欄位維持
  `ticker, id, title, summary, pub_date, provider, url`（`source_api` 欄位用來標示來源，非必要但建議加），
  寫進同一個 `data/news/{ticker}_news.csv` 並依 `url` 去重，下游的 `merge_price_news.py` 完全不用改。
- 新來源如果也有請求額度限制，建議照 `fetch_news_alphavantage.py` 的做法做「進度 state 檔 +
  每日額度 tracker」，避免浪費額度重複抓、也避免額度用完時腳本噴例外中斷。

**其他可能的延伸方向：**
- `signals.py` 目前的門檻（`sentiment_threshold` / `rsi_oversold` / `rsi_overbought`）是寫死的預設值，
  之後如果要做參數優化/掃描，建議另外寫一支腳本跑網格搜尋，不要直接改這支腳本的預設值（保持它是
  「目前正式設定」的單一事實來源）。
- `train.py` 目前用固定的 `--max-train-samples` / `--max-eval-samples` 控制訓練資料量，是為了在
  CPU/GPU 上快速跑通流程；要提升模型品質可以拿掉這個上限（用滿整個資料集）、增加 epoch，或改用
  更貼近正式新聞語氣的訓練資料集。
