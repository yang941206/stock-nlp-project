# Wrapper script for Windows Task Scheduler.
# 每天自動跑完整條流程: 抓股價 -> 抓新聞(AAPL/NVDA，只抓一次，4 支台股共用)
# -> 對每支台股: 跨市場對齊+情緒分類 -> 技術指標 -> 合併成 feature table -> 訊號 -> 相關係數分析
# -> 發送 Telegram 摘要

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$ProjectRoot = "C:\Users\WWW\stock-nlp-project"
$LogFile = Join-Path $ProjectRoot "logs\daily_pipeline.log"
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$Src = Join-Path $ProjectRoot "src"

$TwTickers = @("2330.TW", "0050.TW", "2317.TW", "2454.TW")
$NewsTickers = @("AAPL", "NVDA")

Set-Location $ProjectRoot

function Write-Log {
    param([string]$Message)
    $Message | Out-File -FilePath $LogFile -Append -Encoding utf8
}

function Run-Step {
    param([string]$Label, [string]$ScriptPath, [string[]]$ScriptArgs)
    Write-Log "--- $Label ---"
    $quotedArgs = ($ScriptArgs | ForEach-Object { "`"$_`"" }) -join ' '
    cmd /c "`"$Python`" `"$ScriptPath`" $quotedArgs >> `"$LogFile`" 2>&1"
}

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Log "`n===== Run at $timestamp ====="

# 新聞只抓一次，AAPL/NVDA，4 支台股共用同一份新聞
Run-Step "fetch_news_alphavantage.py (AAPL)" "$Src\data_collection\fetch_news_alphavantage.py" @("--ticker", "AAPL", "--max-requests", "10")
Run-Step "fetch_news_alphavantage.py (NVDA)" "$Src\data_collection\fetch_news_alphavantage.py" @("--ticker", "NVDA", "--max-requests", "10")

foreach ($tw in $TwTickers) {
    $label = "${tw}_x_US"
    Write-Log "===== $tw ====="
    Run-Step "fetch_stock_price.py ($tw)" "$Src\data_collection\fetch_stock_price.py" @("--ticker", $tw, "--period", "1y")
    Run-Step "merge_cross_market_news.py ($tw)" "$Src\utils\merge_cross_market_news.py" @("--price-ticker", $tw, "--news-tickers", "AAPL", "NVDA")
    Run-Step "indicators.py ($tw)" "$Src\technical_analysis\indicators.py" @("--ticker", $tw)
    Run-Step "prepare_crossmarket_adapter.py ($tw)" "$Src\utils\prepare_crossmarket_adapter.py" @("--price-ticker", $tw, "--news-tickers", "AAPL", "NVDA", "--label", $label)
    Run-Step "build_feature_table.py ($tw)" "$Src\technical_analysis\build_feature_table.py" @("--ticker", $label)
    Run-Step "signals.py ($tw)" "$Src\technical_analysis\signals.py" @("--ticker", $label)
    Run-Step "correlation_analysis.py ($tw)" "$Src\technical_analysis\correlation_analysis.py" @("--ticker", $label)
}

Run-Step "telegram_notify.py" "$Src\utils\telegram_notify.py" @("--tickers", $TwTickers, "--news-tickers", "AAPL", "NVDA")

Write-Log "===== Finished at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="
