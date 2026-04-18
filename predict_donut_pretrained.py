"""
Run the OFFICIAL Donut CORD-finetuned checkpoint on a receipt image.

Uses `naver-clova-ix/donut-base-finetuned-cord-v2` directly — no training needed.
This is the baseline the Donut authors released with the paper (Kim et al., ECCV 2022).

Example:
    python3 predict_donut_pretrained.py --image-path ./cord_examples/test_0.png
    python3 predict_donut_pretrained.py --image-path ~/Desktop/bill.png
"""

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel


MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"
TASK_START_TOKEN = "<s_cord-v2>"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def donut_output_to_json(sequence: str, processor) -> dict:
    """Strip Donut's special tokens and turn the tagged string into nested JSON."""
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
    return processor.token2json(sequence)


def flatten_to_target_fields(parsed: dict) -> dict:
    """Reduce the CORD schema to the 5 fields our project cares about."""
    out = {
        "menu.nm": [],
        "menu.price": [],
        "menu.cnt": [],
        "sub_total.subtotal_price": [],
        "total.total_price": [],
    }

    menu = parsed.get("menu")
    if isinstance(menu, dict):
        menu = [menu]
    if isinstance(menu, list):
        for entry in menu:
            if not isinstance(entry, dict):
                continue
            if entry.get("nm"):
                out["menu.nm"].append(str(entry["nm"]))
            if entry.get("price"):
                out["menu.price"].append(str(entry["price"]))
            if entry.get("cnt"):
                out["menu.cnt"].append(str(entry["cnt"]))

    sub = parsed.get("sub_total")
    if isinstance(sub, dict) and sub.get("subtotal_price"):
        out["sub_total.subtotal_price"].append(str(sub["subtotal_price"]))

    tot = parsed.get("total")
    if isinstance(tot, dict) and tot.get("total_price"):
        out["total.total_price"].append(str(tot["total_price"]))

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pre-finetuned Donut (CORD v2) on a receipt image.")
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--show-raw", action="store_true", help="Also print the raw Donut token stream.")
    args = parser.parse_args()

    device, device_name = get_device()
    print(f"Loading {args.model_name} on {device_name} ...")

    processor = DonutProcessor.from_pretrained(args.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    image = Image.open(args.image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    decoder_input_ids = processor.tokenizer(
        TASK_START_TOKEN, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            early_stopping=True,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

    raw = processor.batch_decode(outputs.sequences)[0]
    parsed = donut_output_to_json(raw, processor)
    flattened = flatten_to_target_fields(parsed)

    if args.show_raw:
        print("\nRaw Donut output:")
        print(raw)

    print("\nFull Donut parse (all CORD fields):")
    print(json.dumps(parsed, indent=2, ensure_ascii=False))

    print("\nFiltered to project's 5 target fields:")
    print(json.dumps(flattened, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
