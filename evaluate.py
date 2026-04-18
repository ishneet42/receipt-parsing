"""
Evaluate LayoutLMv3 and Donut on the CORD v2 test set.

For each model we compute, per receipt:
  - Canonical {field: [values]} predictions for the five item-level fields
  - Entity-level precision/recall/F1 per field (exact + normalized matching)
  - Micro-F1 over all fields
  - Line-item matching accuracy (name + price jointly correct)

Running both models takes ~30-45 min on a MacBook Air. Intermediate predictions
are cached to JSON so you can re-run metrics cheaply.

Usage:
    python3 evaluate.py --models layoutlmv3 donut
    python3 evaluate.py --models donut --limit 20      # quick sanity run
    python3 evaluate.py --models layoutlmv3 --cached   # reuse saved predictions
"""

import argparse
import io
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from datasets import load_dataset
from PIL import Image
from transformers import (
    DonutProcessor,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    VisionEncoderDecoderModel,
)

from data_prep import CORD_DATASET_NAME, build_cord_target_fields, parse_ground_truth
from train_layoutlmv3 import OUTSIDE_LABEL, normalize_bbox, quad_to_bbox


DONUT_MODEL = "naver-clova-ix/donut-base-finetuned-cord-v2"
DONUT_TASK_TOKEN = "<s_cord-v2>"

TARGET_FIELDS = [
    "menu.nm",
    "menu.price",
    "menu.cnt",
    "sub_total.subtotal_price",
    "total.total_price",
]


# ────────────────────────────────────────────────────────────────────────────
#  Data
# ────────────────────────────────────────────────────────────────────────────
def get_device() -> Tuple[torch.device, str]:
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def load_image(example: Dict[str, Any]) -> Image.Image:
    image = example.get("image")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict):
        if image.get("bytes"):
            return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
        if image.get("path"):
            return Image.open(image["path"]).convert("RGB")
    raise ValueError("Unsupported image payload.")


def extract_cord_words_and_boxes(
    example: Dict[str, Any], image: Image.Image
) -> Tuple[List[str], List[List[int]]]:
    """Pull CORD's gold word-level OCR from the ground truth and normalize boxes."""
    gt = parse_ground_truth(example["ground_truth"])
    width, height = image.size
    words: List[str] = []
    boxes: List[List[int]] = []
    for line in gt.get("valid_line", []):
        for word in line.get("words", []):
            text = (word.get("text") or "").strip()
            if not text:
                continue
            words.append(text)
            boxes.append(normalize_bbox(quad_to_bbox(word["quad"]), width, height))
    return words, boxes


def build_ground_truth(example: Dict[str, Any]) -> Dict[str, List[str]]:
    gt = parse_ground_truth(example["ground_truth"])
    return build_cord_target_fields(gt.get("gt_parse", {}))


# ────────────────────────────────────────────────────────────────────────────
#  LayoutLMv3
# ────────────────────────────────────────────────────────────────────────────
def load_layoutlmv3(model_path: str, device: torch.device):
    processor = LayoutLMv3Processor.from_pretrained(model_path, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    return model, processor, id2label


def predict_layoutlmv3(
    image: Image.Image,
    words: List[str],
    boxes: List[List[int]],
    model,
    processor,
    id2label: Dict[int, str],
    device: torch.device,
    max_length: int = 512,
) -> Dict[str, List[str]]:
    empty = {f: [] for f in TARGET_FIELDS}
    if not words:
        return empty

    encoding = processor(
        image,
        words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    word_ids = encoding.word_ids(batch_index=0)
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
    predictions = outputs.logits.argmax(dim=-1)[0].tolist()

    # First sub-token per word carries the label.
    outside_id = next((i for i, l in id2label.items() if l == OUTSIDE_LABEL), 0)
    word_label_ids: List[int] = [-1] * len(words)
    for tok_idx, w_idx in enumerate(word_ids):
        if w_idx is None or w_idx >= len(words):
            continue
        if word_label_ids[w_idx] == -1:
            word_label_ids[w_idx] = predictions[tok_idx]
    word_label_ids = [outside_id if v == -1 else v for v in word_label_ids]

    # Stitch B/I spans into field strings.
    grouped: Dict[str, List[str]] = defaultdict(list)
    current_field = None
    current_tokens: List[str] = []

    def flush():
        nonlocal current_field, current_tokens
        if current_field and current_tokens:
            grouped[current_field].append(" ".join(current_tokens))
        current_field, current_tokens = None, []

    for word, lid in zip(words, word_label_ids):
        label = id2label.get(int(lid), OUTSIDE_LABEL)
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
    flush()

    return {f: grouped.get(f, []) for f in TARGET_FIELDS}


# ────────────────────────────────────────────────────────────────────────────
#  Donut
# ────────────────────────────────────────────────────────────────────────────
def load_donut(device: torch.device):
    processor = DonutProcessor.from_pretrained(DONUT_MODEL)
    model = VisionEncoderDecoderModel.from_pretrained(DONUT_MODEL)
    model.to(device)
    model.eval()
    return model, processor


def donut_sequence_to_json(seq: str, processor) -> Dict[str, Any]:
    seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    seq = re.sub(r"<.*?>", "", seq, count=1).strip()
    try:
        return processor.token2json(seq)
    except Exception:
        return {}


def flatten_donut(parsed: Dict[str, Any]) -> Dict[str, List[str]]:
    out = {f: [] for f in TARGET_FIELDS}
    menu = parsed.get("menu")
    if isinstance(menu, dict):
        menu = [menu]
    if isinstance(menu, list):
        for entry in menu:
            if not isinstance(entry, dict):
                continue
            for src, dst in (("nm", "menu.nm"), ("price", "menu.price"), ("cnt", "menu.cnt")):
                v = entry.get(src)
                if v is not None and str(v).strip():
                    out[dst].append(str(v))
    sub = parsed.get("sub_total")
    if isinstance(sub, dict) and sub.get("subtotal_price"):
        out["sub_total.subtotal_price"].append(str(sub["subtotal_price"]))
    tot = parsed.get("total")
    if isinstance(tot, dict) and tot.get("total_price"):
        out["total.total_price"].append(str(tot["total_price"]))
    return out


def predict_donut(image: Image.Image, model, processor, device: torch.device) -> Dict[str, List[str]]:
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    decoder_input_ids = processor.tokenizer(
        DONUT_TASK_TOKEN, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    with torch.no_grad():
        out = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

    raw = processor.batch_decode(out.sequences)[0]
    return flatten_donut(donut_sequence_to_json(raw, processor))


# ────────────────────────────────────────────────────────────────────────────
#  Metrics
# ────────────────────────────────────────────────────────────────────────────
def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[\s,]", "", s)
    s = re.sub(r"[^\w.-]", "", s)
    return s


def multiset_f1(preds: List[str], golds: List[str], strict: bool) -> Tuple[int, int, int]:
    """Return (tp, pred_count, gold_count) using bag-style matching."""
    p_bag = Counter(preds if strict else [normalize(x) for x in preds])
    g_bag = Counter(golds if strict else [normalize(x) for x in golds])
    tp = sum((p_bag & g_bag).values())
    return tp, sum(p_bag.values()), sum(g_bag.values())


def prf1(tp: int, pred_total: int, gold_total: int) -> Tuple[float, float, float]:
    p = tp / pred_total if pred_total else 0.0
    r = tp / gold_total if gold_total else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def line_item_accuracy(
    all_preds: List[Dict[str, List[str]]],
    all_golds: List[Dict[str, List[str]]],
    strict: bool,
) -> float:
    """Fraction of (menu.nm, menu.price) ground-truth pairs correctly extracted jointly."""
    tp = 0
    total = 0
    for pred, gold in zip(all_preds, all_golds):
        gold_names = gold.get("menu.nm", [])
        gold_prices = gold.get("menu.price", [])
        pred_names = pred.get("menu.nm", [])
        pred_prices = pred.get("menu.price", [])

        gold_pairs = Counter(
            zip(gold_names if strict else [normalize(x) for x in gold_names],
                gold_prices if strict else [normalize(x) for x in gold_prices])
        )
        pred_pairs = Counter(
            zip(pred_names if strict else [normalize(x) for x in pred_names],
                pred_prices if strict else [normalize(x) for x in pred_prices])
        )
        tp += sum((gold_pairs & pred_pairs).values())
        total += sum(gold_pairs.values())
    return tp / total if total else 0.0


def compute_metrics(
    all_preds: List[Dict[str, List[str]]],
    all_golds: List[Dict[str, List[str]]],
) -> Dict[str, Any]:
    out = {"per_field": {}, "micro": {}, "line_item": {}}
    for strict_name, strict in (("exact", True), ("norm", False)):
        micro_tp = micro_pred = micro_gold = 0
        per_field = {}
        for field in TARGET_FIELDS:
            f_tp = f_pred = f_gold = 0
            for pred, gold in zip(all_preds, all_golds):
                tp, p, g = multiset_f1(pred.get(field, []), gold.get(field, []), strict)
                f_tp += tp
                f_pred += p
                f_gold += g
            p, r, f = prf1(f_tp, f_pred, f_gold)
            per_field[field] = {"precision": p, "recall": r, "f1": f,
                                 "tp": f_tp, "pred": f_pred, "gold": f_gold}
            micro_tp += f_tp
            micro_pred += f_pred
            micro_gold += f_gold
        p, r, f = prf1(micro_tp, micro_pred, micro_gold)
        out["per_field"][strict_name] = per_field
        out["micro"][strict_name] = {"precision": p, "recall": r, "f1": f}
        out["line_item"][strict_name] = line_item_accuracy(all_preds, all_golds, strict)
    return out


# ────────────────────────────────────────────────────────────────────────────
#  Main loop
# ────────────────────────────────────────────────────────────────────────────
def evaluate_model(
    model_type: str,
    examples: List[Dict[str, Any]],
    layoutlmv3_path: str,
    device: torch.device,
    cache_path: Path,
    use_cached: bool,
) -> Tuple[List[Dict[str, List[str]]], List[Dict[str, List[str]]]]:
    golds = [build_ground_truth(ex) for ex in examples]

    if use_cached and cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if len(cached) == len(examples):
            print(f"[{model_type}] Using cached predictions from {cache_path}")
            return cached, golds

    predictions: List[Dict[str, List[str]]] = []
    start = time.time()

    if model_type == "layoutlmv3":
        model, processor, id2label = load_layoutlmv3(layoutlmv3_path, device)
        for i, ex in enumerate(examples):
            image = load_image(ex)
            words, boxes = extract_cord_words_and_boxes(ex, image)
            pred = predict_layoutlmv3(image, words, boxes, model, processor, id2label, device)
            predictions.append(pred)
            if (i + 1) % 10 == 0 or i == len(examples) - 1:
                elapsed = time.time() - start
                print(f"  [{model_type}] {i+1}/{len(examples)} ({elapsed:.1f}s, "
                      f"{elapsed/(i+1):.2f}s/sample)")

    elif model_type == "donut":
        model, processor = load_donut(device)
        for i, ex in enumerate(examples):
            image = load_image(ex)
            try:
                pred = predict_donut(image, model, processor, device)
            except Exception as e:
                print(f"  [donut] example {i} failed: {e}")
                pred = {f: [] for f in TARGET_FIELDS}
            predictions.append(pred)
            if (i + 1) % 5 == 0 or i == len(examples) - 1:
                elapsed = time.time() - start
                print(f"  [{model_type}] {i+1}/{len(examples)} ({elapsed:.1f}s, "
                      f"{elapsed/(i+1):.2f}s/sample, "
                      f"ETA {(len(examples)-i-1)*elapsed/(i+1):.0f}s)")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    cache_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2))
    print(f"[{model_type}] Saved predictions to {cache_path}")
    return predictions, golds


def format_table(results: Dict[str, Dict[str, Any]]) -> str:
    labels = list(results.keys())
    lines = []
    lines.append("\n" + "=" * 78)
    lines.append("CORD TEST SET — ENTITY-LEVEL F1 (NORMALIZED MATCHING)")
    lines.append("=" * 78)
    header = f"{'Field':32s}" + "".join(f"{l:>22s}" for l in labels)
    lines.append(header)
    lines.append("-" * 78)
    for field in TARGET_FIELDS:
        row = f"{field:32s}"
        for l in labels:
            f1 = results[l]["per_field"]["norm"][field]["f1"]
            row += f"{f1:>22.3f}"
        lines.append(row)
    lines.append("-" * 78)
    row = f"{'MICRO F1':32s}"
    for l in labels:
        row += f"{results[l]['micro']['norm']['f1']:>22.3f}"
    lines.append(row)
    row = f"{'Line-item matching accuracy':32s}"
    for l in labels:
        row += f"{results[l]['line_item']['norm']:>22.3f}"
    lines.append(row)
    lines.append("=" * 78)

    lines.append("\n" + "=" * 78)
    lines.append("CORD TEST SET — ENTITY-LEVEL F1 (EXACT MATCHING)")
    lines.append("=" * 78)
    lines.append(header)
    lines.append("-" * 78)
    for field in TARGET_FIELDS:
        row = f"{field:32s}"
        for l in labels:
            f1 = results[l]["per_field"]["exact"][field]["f1"]
            row += f"{f1:>22.3f}"
        lines.append(row)
    lines.append("-" * 78)
    row = f"{'MICRO F1':32s}"
    for l in labels:
        row += f"{results[l]['micro']['exact']['f1']:>22.3f}"
    lines.append(row)
    row = f"{'Line-item matching accuracy':32s}"
    for l in labels:
        row += f"{results[l]['line_item']['exact']:>22.3f}"
    lines.append(row)
    lines.append("=" * 78)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate both models on CORD test set.")
    parser.add_argument("--models", nargs="+", default=["layoutlmv3", "donut"],
                        choices=["layoutlmv3", "donut"])
    parser.add_argument("--layoutlmv3-path", default="./layoutlmv3-overnight")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0, help="Limit # examples (0 = all).")
    parser.add_argument("--cache-dir", type=Path, default=Path("./eval-cache"))
    parser.add_argument("--cached", action="store_true",
                        help="Reuse cached predictions if present.")
    parser.add_argument("--results-json", type=Path, default=Path("./eval-results.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    device, device_name = get_device()
    print(f"Device: {device_name}")

    print(f"Loading CORD {args.split} split...")
    ds = load_dataset(CORD_DATASET_NAME, split=args.split)
    examples = list(ds) if args.limit <= 0 else list(ds.select(range(min(args.limit, len(ds)))))
    print(f"Evaluating on {len(examples)} examples.")

    all_results: Dict[str, Dict[str, Any]] = {}
    for model_type in args.models:
        print(f"\n─── Running {model_type} ───")
        cache = args.cache_dir / f"{model_type}-{args.split}-{len(examples)}.json"
        preds, golds = evaluate_model(
            model_type, examples, args.layoutlmv3_path, device, cache, args.cached
        )
        all_results[model_type] = compute_metrics(preds, golds)

    print(format_table(all_results))
    args.results_json.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {args.results_json}")


if __name__ == "__main__":
    main()
