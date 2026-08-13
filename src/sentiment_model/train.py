"""金融新聞情緒分類模型 fine-tune 腳本（正式訓練版本）。

Base model:
    ProsusAI/finbert（BERT 在金融語料上 pretrain 並曾用於情緒分類的知名 checkpoint）
    註: yiyanghkust/finbert-pretrain 這個「未接分類頭」版本因 repo 內缺少
    tokenizer_config.json / tokenizer.json，在目前安裝的 transformers 版本下
    無法自動建立 tokenizer，故改用 ProsusAI/finbert 作為起點繼續 fine-tune。

訓練資料:
    zeroshot/twitter-financial-news-sentiment（HuggingFace Hub 上公開、
    無需申請 key 的英文金融新聞/推文情緒資料集），預設用完整資料集（train 9,543 / validation 2,388）。
    標籤: 0 = Bearish（看空）, 1 = Bullish（看多）, 2 = Neutral（中立）

    額外加碼: 把 data/news/*_news.csv 裡符合 news_filters.is_routine_institutional_filing 的
    「例行機構持股異動公告」標題（Sells/Buys X Shares of、Position Lifted/Raised/... 等），
    全部標註為 neutral 加進訓練集（只加進 train，不動 validation，維持驗證指標可比較），
    讓模型學會辨認這類低資訊量標題而不是被表面詞彙（sells→bearish, buys→bullish）誤導。

訓練完會印出驗證集的混淆矩陣，特別標出 bearish/bullish 互判的次數與比例（這是最嚴重的一種錯誤，
比誤判成 neutral 更值得注意）。

用法:
    python train.py                                    # 正式訓練：完整資料集 + neutral 加碼 + 6 epoch
    python train.py --max-train-samples 300 --max-eval-samples 60 --epochs 1  # 快速跑通流程用
    python train.py --no-institutional-neutral         # 不加機構持股異動標題
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.news_filters import is_routine_institutional_filing  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "models"
NEWS_DIR = PROJECT_ROOT / "data" / "news"

ID2LABEL = {0: "bearish", 1: "bullish", 2: "neutral"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a FinBERT-based sentiment classifier.")
    parser.add_argument("--model-name", default="ProsusAI/finbert", help="HuggingFace base model")
    parser.add_argument(
        "--dataset-name", default="zeroshot/twitter-financial-news-sentiment", help="HuggingFace dataset"
    )
    parser.add_argument(
        "--max-train-samples", type=int, default=None, help="訓練用樣本數上限（預設：不限，用完整資料集）"
    )
    parser.add_argument(
        "--max-eval-samples", type=int, default=None, help="驗證用樣本數上限（預設：不限，用完整驗證集）"
    )
    parser.add_argument(
        "--no-institutional-neutral", action="store_true",
        help="不要把 data/news/*_news.csv 裡的機構持股異動標題加進訓練集"
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128, help="tokenizer 最大長度")
    parser.add_argument("--output-dir", default=str(MODEL_ROOT / "finbert-sentiment"), help="模型輸出資料夾")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_institutional_neutral_texts() -> list:
    titles = set()
    for path in NEWS_DIR.glob("*_news.csv"):
        df = pd.read_csv(path)
        if "title" not in df.columns:
            continue
        mask = df["title"].apply(is_routine_institutional_filing)
        titles.update(df.loc[mask, "title"].dropna().tolist())
    return sorted(titles)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def main() -> None:
    args = parse_args()

    print(f"Loading dataset: {args.dataset_name} ...")
    raw = load_dataset(args.dataset_name)

    n_train = len(raw["train"]) if args.max_train_samples is None else min(args.max_train_samples, len(raw["train"]))
    n_eval = (
        len(raw["validation"]) if args.max_eval_samples is None else min(args.max_eval_samples, len(raw["validation"]))
    )
    train_ds = raw["train"].shuffle(seed=args.seed).select(range(n_train))
    eval_ds = raw["validation"].shuffle(seed=args.seed).select(range(n_eval))
    print(f"Using {len(train_ds)} train / {len(eval_ds)} eval examples from {args.dataset_name}.")

    if not args.no_institutional_neutral:
        neutral_texts = load_institutional_neutral_texts()
        print(f"Adding {len(neutral_texts)} institutional-filing headlines as labeled neutral examples (train only) ...")
        if neutral_texts:
            neutral_ds = Dataset.from_dict(
                {"text": neutral_texts, "label": [LABEL2ID["neutral"]] * len(neutral_texts)}
            )
            train_ds = concatenate_datasets([train_ds, neutral_ds]).shuffle(seed=args.seed)
    print(f"Final train size: {len(train_ds)} examples.")

    print(f"Loading tokenizer/model: {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(tokenize, batched=True)
    eval_ds = eval_ds.map(tokenize, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(Path(args.output_dir) / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        logging_steps=20,
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training ...")
    trainer.train()

    print("Final evaluation:")
    metrics = trainer.evaluate()
    print(metrics)

    print("\nRunning predictions on validation set for confusion matrix ...")
    predictions = trainer.predict(eval_ds)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids
    label_names = [ID2LABEL[i] for i in range(len(ID2LABEL))]

    cm = confusion_matrix(labels, preds, labels=list(range(len(ID2LABEL))))
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print("            " + "".join(f"{name:>10s}" for name in label_names))
    for i, row in enumerate(cm):
        print(f"{label_names[i]:>12s}" + "".join(f"{v:>10d}" for v in row))

    print("\nClassification report:")
    print(classification_report(labels, preds, target_names=label_names, digits=3))

    bearish_idx, bullish_idx = LABEL2ID["bearish"], LABEL2ID["bullish"]
    bearish_as_bullish = int(cm[bearish_idx, bullish_idx])
    bullish_as_bearish = int(cm[bullish_idx, bearish_idx])
    cross_confusion = bearish_as_bullish + bullish_as_bearish
    bearish_bullish_total = int(cm[bearish_idx].sum() + cm[bullish_idx].sum())
    cross_rate = cross_confusion / bearish_bullish_total if bearish_bullish_total else 0.0
    print(
        f"\nBearish/Bullish 互判: {cross_confusion} 次"
        f"（實際 bearish 卻判成 bullish: {bearish_as_bullish} 次；"
        f"實際 bullish 卻判成 bearish: {bullish_as_bearish} 次），"
        f"佔所有實際為 bearish/bullish 樣本（{bearish_bullish_total} 筆）的 {cross_rate * 100:.2f}%"
    )

    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nModel saved to: {final_dir}")


if __name__ == "__main__":
    main()
