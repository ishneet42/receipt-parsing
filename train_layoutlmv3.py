import argparse
import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from PIL import Image
from transformers import (
    EarlyStoppingCallback,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    Trainer,
    TrainingArguments,
)

from data_prep import CORD_DATASET_NAME, parse_ground_truth


if sys.version_info >= (3, 14):
    raise RuntimeError(
        "This script depends on Hugging Face datasets/transformers components that should be "
        "run on Python 3.11, 3.12, or 3.13."
    )


MODEL_NAME = "microsoft/layoutlmv3-base"

# Five item-level CORD fields, identical to those used by Donut.
TARGET_CORD_LABELS = [
    "menu.nm",
    "menu.price",
    "menu.cnt",
    "sub_total.subtotal_price",
    "total.total_price",
]
TARGET_MODES = {
    "full": TARGET_CORD_LABELS,
    "total_only": ["total.total_price"],
}

OUTSIDE_LABEL = "O"


def build_label_maps(selected_fields: List[str]) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """BIO label scheme: O, B-<field>, I-<field>."""
    labels = [OUTSIDE_LABEL]
    for field in selected_fields:
        labels.append(f"B-{field}")
        labels.append(f"I-{field}")
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return labels, label2id, id2label


def get_device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def freeze_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def quad_to_bbox(quad: Dict[str, Any]) -> List[int]:
    xs = [quad["x1"], quad["x2"], quad["x3"], quad["x4"]]
    ys = [quad["y1"], quad["y2"], quad["y3"], quad["y4"]]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def normalize_bbox(bbox: List[int], width: int, height: int) -> List[int]:
    """LayoutLMv3 expects boxes scaled to the 0-1000 range."""
    if width <= 0 or height <= 0:
        return [0, 0, 0, 0]
    x0, y0, x1, y1 = bbox
    return [
        max(0, min(1000, int(1000 * x0 / width))),
        max(0, min(1000, int(1000 * y0 / height))),
        max(0, min(1000, int(1000 * x1 / width))),
        max(0, min(1000, int(1000 * y1 / height))),
    ]


def load_image_from_example(example: Dict[str, Any]) -> Image.Image:
    image = example.get("image")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict):
        if image.get("bytes"):
            return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
        if image.get("path"):
            return Image.open(image["path"]).convert("RGB")
    raise ValueError("Unsupported image payload in CORD example.")


def extract_word_level_records(
    example: Dict[str, Any],
    selected_fields: List[str],
) -> Tuple[Image.Image, List[str], List[List[int]], List[str]]:
    """Pull words, normalized bboxes, and BIO labels out of one CORD example."""
    image = load_image_from_example(example)
    width, height = image.size

    ground_truth = parse_ground_truth(example["ground_truth"])
    selected_set = set(selected_fields)

    words: List[str] = []
    boxes: List[List[int]] = []
    bio_labels: List[str] = []

    for line in ground_truth.get("valid_line", []):
        category = line.get("category")
        is_target = category in selected_set
        for position, word in enumerate(line.get("words", [])):
            text = (word.get("text") or "").strip()
            if not text:
                continue
            words.append(text)
            boxes.append(normalize_bbox(quad_to_bbox(word["quad"]), width, height))
            if is_target:
                bio_labels.append(f"{'B' if position == 0 else 'I'}-{category}")
            else:
                bio_labels.append(OUTSIDE_LABEL)

    return image, words, boxes, bio_labels


def build_token_classification_dataset(
    raw: DatasetDict,
    processor: LayoutLMv3Processor,
    label2id: Dict[str, int],
    selected_fields: List[str],
    max_length: int,
) -> DatasetDict:
    def encode(example: Dict[str, Any]) -> Dict[str, Any]:
        image, words, boxes, word_labels = extract_word_level_records(example, selected_fields)
        if not words:
            words = [""]
            boxes = [[0, 0, 0, 0]]
            word_labels = [OUTSIDE_LABEL]

        word_label_ids = [label2id[label] for label in word_labels]
        encoding = processor(
            image,
            words,
            boxes=boxes,
            word_labels=word_label_ids,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="np",
        )

        return {
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "bbox": encoding["bbox"][0],
            "pixel_values": encoding["pixel_values"][0],
            "labels": encoding["labels"][0],
        }

    encoded = DatasetDict()
    for split_name, split_dataset in raw.items():
        encoded[split_name] = split_dataset.map(
            encode,
            remove_columns=split_dataset.column_names,
            desc=f"Encoding {split_name} for LayoutLMv3",
        )
    return encoded


def load_cord_for_training(train_split: str, validation_split: Optional[str]) -> DatasetDict:
    raw = DatasetDict({"train": load_dataset(CORD_DATASET_NAME, split=train_split)})
    if validation_split:
        raw["validation"] = load_dataset(CORD_DATASET_NAME, split=validation_split)
    return raw


def build_model_and_processor(
    model_name: str,
    label_list: List[str],
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    freeze_visual_flag: bool,
):
    # apply_ocr=False because we feed CORD's gold word boxes ourselves.
    processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    if freeze_visual_flag and hasattr(model.layoutlmv3, "patch_embed"):
        freeze_module(model.layoutlmv3.patch_embed)

    return model, processor


def compute_token_metrics_factory(id2label: Dict[int, str]):
    def compute(eval_pred) -> Dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        valid_mask = labels != -100
        flat_preds = predictions[valid_mask]
        flat_labels = labels[valid_mask]

        if flat_labels.size == 0:
            return {"accuracy": 0.0, "non_o_accuracy": 0.0, "non_o_support": 0}

        token_accuracy = float((flat_preds == flat_labels).mean())

        outside_id = next((idx for idx, lbl in id2label.items() if lbl == OUTSIDE_LABEL), -1)
        non_o_mask = flat_labels != outside_id
        non_o_support = int(non_o_mask.sum())
        if non_o_support > 0:
            non_o_accuracy = float((flat_preds[non_o_mask] == flat_labels[non_o_mask]).mean())
        else:
            non_o_accuracy = 0.0

        return {
            "accuracy": token_accuracy,
            "non_o_accuracy": non_o_accuracy,
            "non_o_support": non_o_support,
        }

    return compute


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LayoutLMv3 on CORD for item-level token classification.")
    parser.add_argument("--output-dir", type=Path, default=Path("./layoutlmv3-receipt-model"))
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--target-mode", choices=["full", "total_only"], default="full")
    parser.add_argument("--profile", choices=["default", "local"], default="default")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--num-train-epochs", type=int, default=30)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--eval-strategy", default="epoch")
    parser.add_argument("--save-strategy", default="epoch")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--freeze-visual", action="store_true")
    args = parser.parse_args()

    if args.profile == "local":
        args.num_train_epochs = 5
        args.per_device_train_batch_size = 1
        args.per_device_eval_batch_size = 1
        args.gradient_accumulation_steps = 1
        args.max_steps = 75 if args.max_steps == -1 else args.max_steps
        args.eval_strategy = "steps"
        args.save_strategy = "steps"
        args.train_split = "train[:128]"
        args.validation_split = "validation[:16]"
        args.freeze_visual = True

    return args


def main() -> None:
    args = parse_args()
    device_name = get_device_name()
    selected_fields = TARGET_MODES[args.target_mode]
    label_list, label2id, id2label = build_label_maps(selected_fields)
    print(f"Using device: {device_name}")
    print(f"Selected fields ({args.target_mode}): {selected_fields}")
    print(f"Label space ({len(label_list)} classes): {label_list}")

    model, processor = build_model_and_processor(
        args.model_name,
        label_list,
        label2id,
        id2label,
        freeze_visual_flag=args.freeze_visual,
    )

    raw = load_cord_for_training(args.train_split, args.validation_split)
    encoded = build_token_classification_dataset(
        raw,
        processor=processor,
        label2id=label2id,
        selected_fields=selected_fields,
        max_length=args.max_length,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        optim="adamw_torch",
        max_steps=args.max_steps,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        remove_unused_columns=False,
        report_to="none",
        use_mps_device=device_name == "mps",
        use_cpu=device_name == "cpu",
        fp16=device_name == "cuda",
        dataloader_num_workers=0,
        dataloader_pin_memory=device_name == "cuda",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded.get("validation"),
        processing_class=processor,
        compute_metrics=compute_token_metrics_factory(id2label),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
