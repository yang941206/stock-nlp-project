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
