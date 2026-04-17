import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

if sys.version_info >= (3, 14):
    raise RuntimeError(
        "This script depends on Hugging Face datasets, which is currently incompatible with "
        f"Python {sys.version_info.major}.{sys.version_info.minor}. "
        "Create the virtual environment with Python 3.11, 3.12, or 3.13 and reinstall "
        "the requirements."
    )

from datasets import DatasetDict, Image, load_dataset


CORD_DATASET_NAME = "naver-clova-ix/cord-v2"
SROIE_DATASET_NAME = "jsdnrs/ICDAR2019-SROIE"

TARGET_CORD_LABELS = {
    "menu.nm",
    "menu.price",
    "menu.cnt",
    "sub_total.subtotal_price",
    "total.total_price",
}


def quad_to_bbox(quad: Dict[str, Any]) -> List[int]:
    xs = [quad["x1"], quad["x2"], quad["x3"], quad["x4"]]
    ys = [quad["y1"], quad["y2"], quad["y3"], quad["y4"]]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def extract_word_boxes(words: Iterable[Dict[str, Any]]) -> List[List[int]]:
    return [quad_to_bbox(word["quad"]) for word in words]


def join_word_text(words: Iterable[Dict[str, Any]]) -> str:
    return " ".join(word["text"] for word in words if word.get("text")).strip()


def normalize_menu_entries(menu_value: Any) -> List[Dict[str, Any]]:
    if isinstance(menu_value, list):
        return menu_value
    if isinstance(menu_value, dict):
        return [menu_value]
    return []


def ensure_list_of_strings(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def normalize_section_entries(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def extract_section_field(section_value: Any, field_name: str) -> List[str]:
    entries = normalize_section_entries(section_value)
    values = [entry.get(field_name) for entry in entries if entry.get(field_name) is not None]
    return ensure_list_of_strings(values)


def parse_ground_truth(raw_ground_truth: Any) -> Dict[str, Any]:
    if isinstance(raw_ground_truth, str):
        return json.loads(raw_ground_truth)
    if isinstance(raw_ground_truth, dict):
        return raw_ground_truth
    raise TypeError(f"Unsupported ground_truth type: {type(raw_ground_truth)!r}")


def build_cord_target_fields(gt_parse: Dict[str, Any]) -> Dict[str, Any]:
    menu_entries = normalize_menu_entries(gt_parse.get("menu"))

    return {
        "menu.nm": ensure_list_of_strings([entry.get("nm") for entry in menu_entries if entry.get("nm")]),
        "menu.price": ensure_list_of_strings([entry.get("price") for entry in menu_entries if entry.get("price")]),
        "menu.cnt": ensure_list_of_strings([entry.get("cnt") for entry in menu_entries if entry.get("cnt")]),
        "sub_total.subtotal_price": extract_section_field(gt_parse.get("sub_total"), "subtotal_price"),
        "total.total_price": extract_section_field(gt_parse.get("total"), "total_price"),
    }


def extract_image_ref(image_value: Any) -> Dict[str, Any]:
    if isinstance(image_value, dict):
        return {
            "path": image_value.get("path"),
            "bytes_present": image_value.get("bytes") is not None,
        }
    return {"path": None, "bytes_present": False}


def preprocess_cord_example(example: Dict[str, Any]) -> Dict[str, Any]:
    ground_truth = parse_ground_truth(example["ground_truth"])
    meta = ground_truth.get("meta", {})
    image_size = meta.get("image_size", {})

    annotations: List[Dict[str, Any]] = []
    for line in ground_truth.get("valid_line", []):
        category = line.get("category")
        if category not in TARGET_CORD_LABELS:
            continue

        words = line.get("words", [])
        annotations.append(
            {
                "label": category,
                "text": join_word_text(words),
                "word_texts": [word.get("text", "") for word in words],
                "word_boxes": extract_word_boxes(words),
                "group_id": line.get("group_id"),
                "sub_group_id": line.get("sub_group_id"),
                "row_ids": sorted({word.get("row_id") for word in words if word.get("row_id") is not None}),
                "is_key_flags": [int(word.get("is_key", 0)) for word in words],
            }
        )

    return {
        "image_ref": extract_image_ref(example["image"]),
        "image_id": meta.get("image_id"),
        "split": meta.get("split"),
        "image_size": {
            "width": image_size.get("width"),
            "height": image_size.get("height"),
        },
        "annotations": annotations,
        "target_fields": build_cord_target_fields(ground_truth.get("gt_parse", {})),
    }


def load_cord_filtered(
    splits: Optional[Iterable[str]] = None,
    keep_hf_dataset: bool = True,
) -> Dict[str, Any]:
    raw = load_dataset(CORD_DATASET_NAME)
    raw = DatasetDict(
        {
            split: dataset.cast_column("image", Image(decode=False))
            for split, dataset in raw.items()
        }
    )
    split_names = list(splits) if splits is not None else list(raw.keys())

    processed = {
        split: raw[split].map(
            preprocess_cord_example,
            remove_columns=raw[split].column_names,
            desc=f"Preprocessing {split}",
            writer_batch_size=16,
        )
        for split in split_names
    }

    if keep_hf_dataset:
        return DatasetDict(processed)

    return {split: list(dataset) for split, dataset in processed.items()}


def load_sroie_total_eval(split: str = "test", keep_hf_dataset: bool = True) -> Any:
    raw_split = load_dataset(SROIE_DATASET_NAME, split=split)

    def to_total_example(example: Dict[str, Any]) -> Dict[str, Any]:
        entities = example.get("entities", {})
        return {
            "image_ref": extract_image_ref(example["image"]),
            "image_size": example.get("image_size"),
            "words": example.get("words", []),
            "bboxes": example.get("bboxes", []),
            "target_total": entities.get("total"),
        }

    processed = raw_split.map(
        to_total_example,
        remove_columns=raw_split.column_names,
        desc=f"Preparing SROIE {split} total-only split",
    )
    return processed if keep_hf_dataset else list(processed)


def save_dataset_dict_to_jsonl(dataset_dict: DatasetDict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, dataset in dataset_dict.items():
        output_path = output_dir / f"{split}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for record in dataset:
                serializable = dict(record)
                handle.write(json.dumps(serializable, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and preprocess receipt datasets.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to save the filtered CORD splits as JSONL files.",
    )
    parser.add_argument(
        "--cord-splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="CORD splits to preprocess.",
    )
    parser.add_argument(
        "--sroie-split",
        default="test",
        help="SROIE split to load for total-field evaluation.",
    )
    args = parser.parse_args()

    cord_dataset = load_cord_filtered(splits=args.cord_splits, keep_hf_dataset=True)
    sroie_total_dataset = load_sroie_total_eval(split=args.sroie_split, keep_hf_dataset=True)

    print("CORD filtered splits:")
    for split, dataset in cord_dataset.items():
        print(f"  {split}: {len(dataset)} examples")

    print(f"SROIE total-eval split '{args.sroie_split}': {len(sroie_total_dataset)} examples")

    if args.output_dir is not None:
        save_dataset_dict_to_jsonl(cord_dataset, args.output_dir)
        print(f"Saved CORD JSONL files to: {args.output_dir}")


if __name__ == "__main__":
    main()
