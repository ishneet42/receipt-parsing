import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

from train_layoutlmv3 import (
    OUTSIDE_LABEL,
    TARGET_MODES,
    get_device_name,
    normalize_bbox,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LayoutLMv3 inference on a single receipt image.")
    parser.add_argument("--model-path", type=Path, default=Path("./layoutlmv3-receipt-model"))
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--target-mode", choices=["full", "total_only"], default="full")
    parser.add_argument(
        "--ocr-source",
        choices=["tesseract"],
        default="tesseract",
        help="OCR engine that supplies words and boxes for inference.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--tesseract-lang", default="eng")
    return parser.parse_args()


def run_tesseract(image: Image.Image, lang: str) -> Tuple[List[str], List[List[int]]]:
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError(
            "pytesseract is required for --ocr-source tesseract. "
            "Install with: pip install pytesseract (and the Tesseract binary, e.g. brew install tesseract)."
        ) from exc

    data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
    words: List[str] = []
    boxes: List[List[int]] = []
    width, height = image.size

    for i, text in enumerate(data["text"]):
        word = (text or "").strip()
        if not word:
            continue
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        words.append(word)
        boxes.append(normalize_bbox([x, y, x + w, y + h], width, height))

    return words, boxes


def group_predictions_by_field(
    words: List[str],
    word_label_ids: List[int],
    word_boxes: List[List[int]],
    id2label: Dict[int, str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Stitch consecutive same-field tokens into spans, ignoring O."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    current_field: Optional[str] = None
    current_tokens: List[str] = []
    current_boxes: List[List[int]] = []

    def flush() -> None:
        nonlocal current_field, current_tokens, current_boxes
        if current_field is not None and current_tokens:
            grouped[current_field].append(
                {
                    "text": " ".join(current_tokens),
                    "boxes": current_boxes,
                }
            )
        current_field = None
        current_tokens = []
        current_boxes = []

    for word, label_id, box in zip(words, word_label_ids, word_boxes):
        label = id2label.get(int(label_id), OUTSIDE_LABEL)
        if label == OUTSIDE_LABEL:
            flush()
            continue

        prefix, _, field = label.partition("-")
        if not field:
            flush()
            continue

        if prefix == "B" or field != current_field:
            flush()
            current_field = field

        current_tokens.append(word)
        current_boxes.append(box)

    flush()
    return grouped


def main() -> None:
    args = parse_args()
    device_name = get_device_name()
    device = torch.device("mps" if device_name == "mps" else device_name)
    selected_fields = TARGET_MODES[args.target_mode]

    processor = LayoutLMv3Processor.from_pretrained(str(args.model_path), apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(str(args.model_path))
    model.to(device)
    model.eval()
    id2label: Dict[int, str] = {int(k): v for k, v in model.config.id2label.items()}

    image = Image.open(args.image_path).convert("RGB")

    if args.ocr_source == "tesseract":
        words, boxes = run_tesseract(image, args.tesseract_lang)
    else:
        raise ValueError(f"Unsupported OCR source: {args.ocr_source}")

    if not words:
        print("No words detected by OCR — nothing to classify.")
        print(json.dumps({field: [] for field in selected_fields}, indent=2))
        return

    encoding = processor(
        image,
        words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        max_length=args.max_length,
        return_tensors="pt",
    )
    # Capture word_ids from this single call before we move tensors to device.
    word_ids = encoding.word_ids(batch_index=0)
    encoding = {key: value.to(device) for key, value in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)

    predictions = outputs.logits.argmax(dim=-1)[0].tolist()

    word_label_ids: List[int] = [-1] * len(words)
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx >= len(words):
            continue
        if word_label_ids[word_idx] == -1:
            word_label_ids[word_idx] = predictions[token_idx]

    # Any word the tokenizer dropped from the window stays as O.
    outside_id = next((idx for idx, lbl in id2label.items() if lbl == OUTSIDE_LABEL), 0)
    word_label_ids = [outside_id if value == -1 else value for value in word_label_ids]

    grouped = group_predictions_by_field(words, word_label_ids, boxes, id2label)
    structured = {field: [span["text"] for span in grouped.get(field, [])] for field in selected_fields}

    print("Per-word predictions:")
    for word, label_id in zip(words, word_label_ids):
        print(f"  {id2label.get(int(label_id), OUTSIDE_LABEL):>30s}  {word}")
    print()
    print("Parsed JSON:")
    print(json.dumps(structured, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
