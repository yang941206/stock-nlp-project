"""用 fine-tune 好的情緒分類模型，輸入一則新聞標題，輸出情緒分數。

用法:
    python predict.py --text "Apple stock surges after strong earnings report"
    python predict.py --text "Company shares plunge amid fraud allegations" --model-dir "../../models/finbert-sentiment/final"
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "finbert-sentiment" / "final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sentiment inference on a single headline.")
    parser.add_argument("--text", required=True, help="新聞標題或句子")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="fine-tune 後的模型資料夾")
    return parser.parse_args()


def predict_sentiment(text: str, model_dir: str) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().tolist()

    id2label = model.config.id2label
    scores = {id2label[i]: round(p, 4) for i, p in enumerate(probs)}
    predicted_label = max(scores, key=scores.get)

    return {"text": text, "predicted_label": predicted_label, "scores": scores}


def main() -> None:
    args = parse_args()

    if not Path(args.model_dir).exists():
        raise FileNotFoundError(
            f"Model directory not found: {args.model_dir}. Run train.py first to produce a fine-tuned model."
        )

    result = predict_sentiment(args.text, args.model_dir)
    print(f"Text:  {result['text']}")
    print(f"Label: {result['predicted_label']}")
    print(f"Scores: {result['scores']}")


if __name__ == "__main__":
    main()
