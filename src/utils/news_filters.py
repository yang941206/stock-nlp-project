"""過濾掉例行的機構持股異動公告（SEC 13F/13G 申報用語產生的自動化標題）。

這類標題資訊量很低（單純申報揭露，跟公司基本面無關），但情緒模型對表面詞彙
（Sells → bearish, Buys/Purchased/Lifted → bullish）有明顯偏誤，見
data/processed/2330.TW_news_from_AAPL_NVDA.csv 的人工檢查結果。

之後如果要正式重訓情緒模型，這類標題應該收進訓練資料並標註為 neutral，讓模型
自己學會忽略，而不是永遠用規則過濾。
"""

import re

ROUTINE_INSTITUTIONAL_FILING_PATTERN = re.compile(
    r"\b(Sells|Buys|Acquires|Purchases)\s+[\d,]+\s+Shares\s+of\b"
    r"|\bShares\s+(Purchased|Sold|Acquired|Bought)\s+by\b"
    r"|\b(Sells|Buys|Acquires|Purchases|Takes|Establishes|Grows|Cuts|Trims|Boosts|Raises|Lowers|Reduces"
    r"|Decreases|Increases|Lifts|Builds|Has)\s+(its\s+|a\s+|new\s+|\$[\d.,]+\s*(million|billion)?\s+)?"
    r"(Stock\s+)?(Position|Holdings?|Stake)\s+in\b"
    r"|\b(Stock\s+)?(Position|Holdings?|Stake)\s+(Lifted|Raised|Increased|Decreased|Reduced|Trimmed"
    r"|Boosted|Grown|Cut|Lessened|Lowered)\s+by\b",
    re.IGNORECASE,
)


def is_routine_institutional_filing(title: str) -> bool:
    if not isinstance(title, str):
        return False
    return bool(ROUTINE_INSTITUTIONAL_FILING_PATTERN.search(title))


# 內部人/高管申報賣股（Form 4 衍生的新聞標題），跟上面的機構（法人）持股異動是不同類別：
# 這裡是公司自己的高管/董事賣「自家公司」股票，訊號意義不同，故意不過濾掉、單獨拉出來看
# （見 insider_selling_analysis.py）。只抓「有明確賣出動作」的標題，不含方向不明的
# 「Form 4 XXX For: ... By Investing.com」原始申報通知（無法判斷買賣方向）。
INSIDER_TITLE_WORDS = (
    r"CEO|CFO|COO|CTO|President|Chairman|Chairwoman|Director|SVP|EVP|VP"
    r"|Officer|Insider|Founder|Owner"
)
INSIDER_SELLING_PATTERN = re.compile(
    rf"\bInsider\s+(?:Sale|Sell)(?:s|ing)?\b"
    rf"|\b(?:{INSIDER_TITLE_WORDS})\b[^?!]{{0,80}}\b(?:Sale|Sell)(?:s|ing)?\b[^?!]{{0,60}}\b(?:Shares?|Stock)\b",
    re.IGNORECASE,
)


def is_insider_selling(title: str) -> bool:
    if not isinstance(title, str):
        return False
    return bool(INSIDER_SELLING_PATTERN.search(title))
