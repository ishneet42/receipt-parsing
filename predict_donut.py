import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

from train_donut import TARGET_MODES, get_device_name, resize_with_padding, token_sequence_to_target_fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Donut inference on a single receipt image.")
    parser.add_argument("--model-path", type=Path, default=Path("./donut-local-run"))
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--task-start-token", default="<s_receipt_parse>")
    parser.add_argument("--target-mode", choices=["full", "total_only"], default="full")
    parser.add_argument("--target-width", type=int, default=960)
    parser.add_argument("--target-height", type=int, default=720)
    parser.add_argument("--max-length", type=int, default=768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_name = get_device_name()
    device = torch.device("mps" if device_name == "mps" else device_name)
    selected_fields = TARGET_MODES[args.target_mode]

    processor = DonutProcessor.from_pretrained(str(args.model_path), use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained(str(args.model_path))
    model.to(device)
    model.eval()

    image = Image.open(args.image_path).convert("RGB")
    image = resize_with_padding(image, (args.target_width, args.target_height))
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    decoder_input_ids = processor.tokenizer(
        args.task_start_token,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=args.max_length,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            early_stopping=False,
        )

    prediction = processor.batch_decode(outputs, skip_special_tokens=False)[0]
    structured = token_sequence_to_target_fields(prediction, selected_fields)

    print("Raw prediction:")
    print(prediction)
    print()
    print("Parsed JSON:")
    if not any(structured.values()):
        print("null")
    else:
        print(json.dumps(structured, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
