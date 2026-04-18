"""
Run both models on a single receipt image, compose the hybrid, and report
cross-model agreement + the arithmetic checksum.

Field routing for the hybrid:
  - menu.nm / menu.price / menu.cnt      <-- LayoutLMv3 (Tesseract OCR)
  - sub_total.subtotal_price / total.total_price  <-- Donut (OCR-free)

Example:
    python3 predict_hybrid.py --image-path ~/Desktop/bill2.png
    python3 predict_hybrid.py --image-path ~/Desktop/bill.png \
        --layoutlmv3-path ./layoutlmv3-fixed
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from transformers import (
    DonutProcessor,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    VisionEncoderDecoderModel,
)

from train_layoutlmv3 import OUTSIDE_LABEL, normalize_bbox


DONUT_MODEL = "naver-clova-ix/donut-base-finetuned-cord-v2"
DONUT_TASK_TOKEN = "<s_cord-v2>"

TARGET_FIELDS = [
    "menu.nm",
    "menu.price",
    "menu.cnt",
    "sub_total.subtotal_price",
    "total.total_price",
]

HYBRID_SOURCE = {
    "menu.nm": "layoutlmv3",
    "menu.price": "layoutlmv3",
    "menu.cnt": "layoutlmv3",
    "sub_total.subtotal_price": "donut",
    "total.total_price": "donut",
}


def get_device() -> Tuple[torch.device, str]:
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


# ---------- LayoutLMv3 (Tesseract -> token classifier) ----------
def run_tesseract(image: Image.Image, lang: str) -> Tuple[List[str], List[List[int]]]:
    import pytesseract

    data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
    words, boxes = [], []
    width, height = image.size
    for i, text in enumerate(data["text"]):
        w = (text or "").strip()
        if not w:
            continue
        x = int(data["left"][i])
        y = int(data["top"][i])
        ww = int(data["width"][i])
        hh = int(data["height"][i])
        words.append(w)
        boxes.append(normalize_bbox([x, y, x + ww, y + hh], width, height))
    return words, boxes


def predict_layoutlmv3(
    image: Image.Image, model_path: str, device: torch.device, max_length: int = 512, lang: str = "eng"
) -> Dict[str, List[str]]:
    processor = LayoutLMv3Processor.from_pretrained(model_path, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_path).to(device)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    words, boxes = run_tesseract(image, lang)
    empty = {f: [] for f in TARGET_FIELDS}
    if not words:
        return empty

    encoding = processor(
        image, words, boxes=boxes,
        truncation=True, padding="max_length",
        max_length=max_length, return_tensors="pt",
    )
    word_ids = encoding.word_ids(batch_index=0)
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
    preds = outputs.logits.argmax(dim=-1)[0].tolist()

    outside_id = next((i for i, l in id2label.items() if l == OUTSIDE_LABEL), 0)
    word_label_ids = [-1] * len(words)
    for tok, w_idx in enumerate(word_ids):
        if w_idx is None or w_idx >= len(words):
            continue
        if word_label_ids[w_idx] == -1:
            word_label_ids[w_idx] = preds[tok]
    word_label_ids = [outside_id if v == -1 else v for v in word_label_ids]

    grouped = defaultdict(list)
    cur_field, cur_tokens = None, []

    def flush():
        nonlocal cur_field, cur_tokens
        if cur_field and cur_tokens:
            grouped[cur_field].append(" ".join(cur_tokens))
        cur_field, cur_tokens = None, []

    for word, lid in zip(words, word_label_ids):
        label = id2label.get(int(lid), OUTSIDE_LABEL)
        if label == OUTSIDE_LABEL:
            flush()
            continue
        prefix, _, field = label.partition("-")
        if not field:
            flush()
            continue
        if prefix == "B" or field != cur_field:
            flush()
            cur_field = field
        cur_tokens.append(word)
    flush()

    return {f: grouped.get(f, []) for f in TARGET_FIELDS}


# ---------- Donut (pixels -> JSON) ----------
def predict_donut(image: Image.Image, device: torch.device) -> Dict[str, List[str]]:
    processor = DonutProcessor.from_pretrained(DONUT_MODEL)
    model = VisionEncoderDecoderModel.from_pretrained(DONUT_MODEL).to(device)
    model.eval()

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
            use_cache=True, num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
    raw = processor.batch_decode(out.sequences)[0]
    raw = raw.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    raw = re.sub(r"<.*?>", "", raw, count=1).strip()
    try:
        parsed = processor.token2json(raw)
    except Exception:
        parsed = {}

    out_fields = {f: [] for f in TARGET_FIELDS}
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
                    out_fields[dst].append(str(v))
    sub = parsed.get("sub_total")
    if isinstance(sub, dict) and sub.get("subtotal_price"):
        out_fields["sub_total.subtotal_price"].append(str(sub["subtotal_price"]))
    tot = parsed.get("total")
    if isinstance(tot, dict) and tot.get("total_price"):
        out_fields["total.total_price"].append(str(tot["total_price"]))
    return out_fields


# ---------- Hybrid + verification ----------
def build_hybrid(lv3: Dict[str, List[str]], dn: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {
        f: (lv3 if HYBRID_SOURCE[f] == "layoutlmv3" else dn).get(f, [])
        for f in TARGET_FIELDS
    }


def parse_money(text: str) -> float:
    m = re.search(r"[-+]?\d[\d,]*\.?\d*", text or "")
    if not m:
        return 0.0
    raw = m.group(0).replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return 0.0


def verify(pred: Dict[str, List[str]], lv3: Dict[str, List[str]], dn: Dict[str, List[str]]) -> Dict[str, Any]:
    prices = [parse_money(p) for p in pred.get("menu.price", [])]
    subtotal = parse_money(pred["sub_total.subtotal_price"][0]) if pred.get("sub_total.subtotal_price") else None
    total = parse_money(pred["total.total_price"][0]) if pred.get("total.total_price") else None
    items_sum = sum(prices) if prices else None

    agree_sub = lv3.get("sub_total.subtotal_price") == dn.get("sub_total.subtotal_price") \
        and bool(lv3.get("sub_total.subtotal_price"))
    agree_tot = lv3.get("total.total_price") == dn.get("total.total_price") \
        and bool(lv3.get("total.total_price"))

    checks = {
        "items_count": len(prices),
        "items_sum": items_sum,
        "subtotal": subtotal,
        "total": total,
        "items_sum_matches_subtotal": None,
        "items_sum_matches_total": None,
        "implied_tax": None,
        "implied_tax_pct": None,
        "layoutlmv3_donut_agree_subtotal": agree_sub,
        "layoutlmv3_donut_agree_total": agree_tot,
    }

    if items_sum is not None and subtotal is not None and subtotal > 0:
        checks["items_sum_matches_subtotal"] = abs(items_sum - subtotal) <= 0.01
    if items_sum is not None and total is not None and total > 0:
        checks["items_sum_matches_total"] = abs(items_sum - total) <= 0.01
    if subtotal is not None and total is not None and subtotal > 0:
        checks["implied_tax"] = round(total - subtotal, 2)
        checks["implied_tax_pct"] = round((total - subtotal) / subtotal * 100, 2)
    return checks


def pretty(fields: Dict[str, List[str]]) -> str:
    return json.dumps(fields, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LayoutLMv3, Donut, and the Hybrid on a receipt image.")
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--layoutlmv3-path", default="./layoutlmv3-fixed")
    parser.add_argument("--tesseract-lang", default="eng")
    args = parser.parse_args()

    device, device_name = get_device()
    print(f"Device: {device_name}")
    print(f"Image:  {args.image_path}")

    image = Image.open(args.image_path).convert("RGB")

    print("\n─── Running LayoutLMv3 (Tesseract OCR → token classifier) ───")
    lv3 = predict_layoutlmv3(image, args.layoutlmv3_path, device, lang=args.tesseract_lang)
    print(pretty(lv3))

    print("\n─── Running Donut (OCR-free seq2seq) ───")
    dn = predict_donut(image, device)
    print(pretty(dn))

    hybrid = build_hybrid(lv3, dn)
    print("\n─── Hybrid (items←LayoutLMv3, aggregates←Donut) ───")
    print(pretty(hybrid))

    print("\n─── Verification ───")
    checks = verify(hybrid, lv3, dn)
    ok = "✓"
    no = "✗"
    print(f"  Items extracted:             {checks['items_count']}")
    print(f"  Sum of items:                ${checks['items_sum']:.2f}" if checks['items_sum'] is not None else "  Sum of items:                —")
    print(f"  Subtotal:                    ${checks['subtotal']:.2f}" if checks['subtotal'] is not None else "  Subtotal:                    —")
    print(f"  Total:                       ${checks['total']:.2f}" if checks['total'] is not None else "  Total:                       —")
    if checks["items_sum_matches_subtotal"] is not None:
        print(f"  Items-sum == subtotal?       {ok if checks['items_sum_matches_subtotal'] else no}")
    if checks["items_sum_matches_total"] is not None:
        print(f"  Items-sum == total?          {ok if checks['items_sum_matches_total'] else no}")
    if checks["implied_tax"] is not None:
        print(f"  Implied tax:                 ${checks['implied_tax']:.2f}  ({checks['implied_tax_pct']:.2f}%)")
    print(f"  LayoutLMv3 & Donut agree on subtotal? {ok if checks['layoutlmv3_donut_agree_subtotal'] else no}")
    print(f"  LayoutLMv3 & Donut agree on total?    {ok if checks['layoutlmv3_donut_agree_total'] else no}")


if __name__ == "__main__":
    main()
