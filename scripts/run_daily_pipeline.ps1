# Wrapper script for Windows Task Scheduler.
# 每天自動跑完整條流程: 抓股價 -> 抓新聞(AAPL/NVDA/AMD/TSM/QCOM，只抓一次，4 支台股共用)
# -> 對每支台股: 跨市場對齊+情緒分類 -> 技術指標 -> 合併成 feature table -> 訊號 -> 相關係數分析
# -> 發送 Telegram 摘要

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
# 排程用 cmd /c 執行 python 時，子行程的 stdout/stderr 預設用系統 ANSI codepage（cp950），
# print() 裡的 emoji（📅📰🔑...）會直接讓 python 崩潰、噴 UnicodeEncodeError（exit code 非 0）。
# 統一在這裡設一次 PYTHONIOENCODING，所有透過 Run-Step 呼叫的 python 子行程都會繼承到，
# 不用每支腳本各自修。這也是為什麼之前 telegram_notify.py 訊息明明送出去了、log 裡卻有
# 一段 traceback的原因——訊息在 print() 之前就送出了，只是印確認訊息時才崩潰。
$env:PYTHONIOENCODING = "utf-8"

$ProjectRoot = "C:\Users\WWW\stock-nlp-project"
$LogFile = Join-Path $ProjectRoot "logs\daily_pipeline.log"
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$Src = Join-Path $ProjectRoot "src"

$TwTickers = @("2330.TW", "0050.TW", "2317.TW", "2454.TW")
# 半導體供應鏈相關新聞來源，4 支台股共用。5 支 x 4 次 = 20 次/天，在 Alpha Vantage
# 免費方案 25 次/天上限內留 5 次緩衝給手動測試（原本 AAPL/NVDA 各 10 次，因為加了
# 3 支新來源所以調降，AAPL/NVDA 的歷史回溯速度會因此變慢）。
$NewsTickers = @("AAPL", "NVDA", "AMD", "TSM", "QCOM")
$NewsMaxRequestsPerTicker = 4

Set-Location $ProjectRoot

function Write-Log {
    param([string]$Message)
    $Message | Out-File -FilePath $LogFile -Append -Encoding utf8
}

# 健檢用：累積這次執行裡失敗的步驟，跑完後寫進 logs/last_run_status.json，
# telegram_notify.py 會讀這個檔案，有失敗就在通知開頭加警告，不會再悄悄跑完卻沒人知道。
$script:FailedSteps = @()

function Run-Step {
    param([string]$Label, [string]$ScriptPath, [string[]]$ScriptArgs)
    Write-Log "--- $Label ---"
    $quotedArgs = ($ScriptArgs | ForEach-Object { "`"$_`"" }) -join ' '
    cmd /c "`"$Python`" `"$ScriptPath`" $quotedArgs >> `"$LogFile`" 2>&1"
    if ($LASTEXITCODE -ne 0) {
        Write-Log "!!! FAILED: $Label (exit code $LASTEXITCODE) !!!"
        $script:FailedSteps += $Label
    }
}

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Log "`n===== Run at $timestamp ====="

# 新聞只抓一次，4 支台股共用同一份新聞
foreach ($newsTicker in $NewsTickers) {
    Run-Step "fetch_news_alphavantage.py ($newsTicker)" "$Src\data_collection\fetch_news_alphavantage.py" @("--ticker", $newsTicker, "--max-requests", $NewsMaxRequestsPerTicker)
}

foreach ($tw in $TwTickers) {
    $label = "${tw}_x_US"
    Write-Log "===== $tw ====="
    Run-Step "fetch_stock_price.py ($tw)" "$Src\data_collection\fetch_stock_price.py" @("--ticker", $tw, "--period", "1y")
    Run-Step "merge_cross_market_news.py ($tw)" "$Src\utils\merge_cross_market_news.py" @("--price-ticker", $tw, "--news-tickers", $NewsTickers)
    Run-Step "indicators.py ($tw)" "$Src\technical_analysis\indicators.py" @("--ticker", $tw)
    Run-Step "prepare_crossmarket_adapter.py ($tw)" "$Src\utils\prepare_crossmarket_adapter.py" @("--price-ticker", $tw, "--news-tickers", $NewsTickers, "--label", $label)
    Run-Step "build_feature_table.py ($tw)" "$Src\technical_analysis\build_feature_table.py" @("--ticker", $label)
    Run-Step "signals.py ($tw)" "$Src\technical_analysis\signals.py" @("--ticker", $label)
    Run-Step "correlation_analysis.py ($tw)" "$Src\technical_analysis\correlation_analysis.py" @("--ticker", $label)
}

# 健檢：就算所有步驟都「沒噴例外」，也可能是某一步靜默用了舊檔案（例如今天的 merge 沒有
# 真的重新產生 signals.csv，下游步驟直接讀到昨天留下的舊檔，全部正常跑完但資料是舊的）。
# 所以額外檢查每支台股的 signals.csv 是不是今天才被寫入。
$staleTickers = @()
$today = Get-Date -Format "yyyy-MM-dd"
foreach ($tw in $TwTickers) {
    $label = "${tw}_x_US"
    $signalsPath = Join-Path $ProjectRoot "data\processed\${label}_signals.csv"
    if (-not (Test-Path $signalsPath)) {
        $staleTickers += "$tw (檔案不存在)"
    } elseif ((Get-Item $signalsPath).LastWriteTime.ToString('yyyy-MM-dd') -ne $today) {
        $staleTickers += "$tw (今天沒有更新，可能還是舊資料)"
    }
}

if ($script:FailedSteps.Count -gt 0) {
    Write-Log "!!! HEALTH CHECK: 這次執行有 $($script:FailedSteps.Count) 個步驟失敗 !!!"
}
if ($staleTickers.Count -gt 0) {
    Write-Log "!!! HEALTH CHECK: $($staleTickers.Count) 支台股的資料今天沒有真的更新: $($staleTickers -join '; ') !!!"
}

$statusPath = Join-Path $ProjectRoot "logs\last_run_status.json"
@{
    run_timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
    failed_steps  = $script:FailedSteps
    stale_tickers = $staleTickers
} | ConvertTo-Json | Out-File -FilePath $statusPath -Encoding utf8

Run-Step "telegram_notify.py" "$Src\utils\telegram_notify.py" @("--tickers", $TwTickers, "--news-tickers", $NewsTickers)

# 把最新的 dashboard 資料 (data/processed 底下沒被 .gitignore 排除的檔案) commit + push 回
# GitHub，讓 Streamlit Cloud 上的版本隔天能抓到當天資料。只 add data/processed，不會動到
# 其他還沒 commit 的程式碼變更。用 .env 裡的 GITHUB_PAT（GitHub Personal Access Token）非
# 互動推送，不依賴 Windows 登入 session 或瀏覽器 OAuth，排程無人值守也能跑。PAT 只在這次
# push 指令的 URL 參數裡用一下，不會寫進 .git/config（git remote -v 看到的還是乾淨的網址）。
Write-Log "--- git commit + push (data/processed) ---"
$Git = "C:\Program Files\Git\cmd\git.exe"
$env:GIT_TERMINAL_PROMPT = "0"

& $Git add data/processed *>> $LogFile
& $Git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    $commitMsg = "Daily data update $(Get-Date -Format 'yyyy-MM-dd')"
    & $Git commit -m $commitMsg *>> $LogFile

    $envLine = Get-Content (Join-Path $ProjectRoot ".env") | Where-Object { $_ -match '^GITHUB_PAT=' }
    $githubPat = ($envLine -replace '^GITHUB_PAT=', '').Trim()
    if (-not $githubPat) {
        Write-Log "WARNING: .env 裡沒有 GITHUB_PAT，無法自動 push，請手動 push 或補上 token。"
    } else {
        $pushUrl = "https://$githubPat@github.com/yang941206/stock-nlp-project.git"
        & $Git push $pushUrl main *>> $LogFile
        if ($LASTEXITCODE -ne 0) {
            Write-Log "WARNING: git push 失敗 (exit code $LASTEXITCODE)，GitHub 上的資料未更新，Streamlit Cloud 會顯示舊資料，請手動檢查（可能是 PAT 過期/被撤銷）。"
        } else {
            # push 用明確網址而不是具名 origin，不會自動更新本機的 origin/main 追蹤參照，
            # 手動同步一下，避免 git status 一直誤報「ahead of origin」
            & $Git update-ref refs/remotes/origin/main (& $Git rev-parse HEAD)
        }
    }
} else {
    Write-Log "No data changes to commit."
}

Write-Log "===== Finished at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="
