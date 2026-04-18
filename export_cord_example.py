import argparse
import json
from pathlib import Path

from datasets import load_dataset

from data_prep import CORD_DATASET_NAME, build_cord_target_fields, parse_ground_truth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a single CORD example image and labels.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("./cord_examples"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(CORD_DATASET_NAME, split=args.split)
    example = dataset[args.index]
    ground_truth = parse_ground_truth(example["ground_truth"])
    target_fields = build_cord_target_fields(ground_truth.get("gt_parse", {}))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / f"{args.split}_{args.index}.png"
    label_path = args.output_dir / f"{args.split}_{args.index}.json"

    example["image"].save(image_path)
    label_path.write_text(json.dumps(target_fields, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved image to: {image_path}")
    print(f"Saved target fields to: {label_path}")


if __name__ == "__main__":
    main()
