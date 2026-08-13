"""對 merge_price_news.py 產出的新聞+股價整合檔，批次跑情緒分類，
新增 sentiment_label / sentiment_bearish / sentiment_bullish / sentiment_neutral 欄位。

用法:
    python batch_predict.py --ticker AAPL
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "finbert-sentiment" / "final"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch sentiment scoring over a merged news+price CSV.")
    parser.add_argument("--ticker", default="AAPL", help="股票代號（預設：AAPL）")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="fine-tune 後的模型資料夾")
    parser.add_argument("--text-column", default="title", help="要跑情緒分類的欄位名稱")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def batch_predict(texts: list, model_dir: str, batch_size: int) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    id2label = model.config.id2label
    rows = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, max_length=128, padding=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().tolist()

        for p in probs:
            scores = {id2label[i]: round(v, 4) for i, v in enumerate(p)}
            rows.append({"sentiment_label": max(scores, key=scores.get), **scores})

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    merged_path = PROCESSED_DIR / f"{args.ticker}_merged.csv"
    if not merged_path.exists():
        raise FileNotFoundError(f"{merged_path} not found. Run merge_price_news.py first.")
    if not Path(args.model_dir).exists():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}. Run train.py first.")

    df = pd.read_csv(merged_path)
    print(f"Scoring sentiment for {len(df)} rows from {merged_path.name} ...")

    scores = batch_predict(df[args.text_column].fillna("").tolist(), args.model_dir, args.batch_size)
    scores = scores.rename(columns={c: f"sentiment_{c}" for c in scores.columns if c != "sentiment_label"})
    result = pd.concat([df.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)

    output_path = PROCESSED_DIR / f"{args.ticker}_merged_sentiment.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved to: {output_path}")
    print(result["sentiment_label"].value_counts())
    print(result[[args.text_column, "sentiment_label"]].head())


if __name__ == "__main__":
    main()
