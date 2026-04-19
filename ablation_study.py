"""
Ablation study for the receipt-parsing project.

Composes multiple system variants from the predictions cached by `evaluate.py`
(no retraining required) and reports a single comparison table that isolates
the contribution of each design decision.

Variants reported:
    - layoutlmv3              fine-tuned LayoutLMv3 alone (the trained model)
    - donut                   pre-finetuned Donut (Naver CORD-v2) alone
    - hybrid (ours)           items ← LayoutLMv3, aggregates ← Donut
    - hybrid_reversed         items ← Donut, aggregates ← LayoutLMv3 (negative control)
    - layoutlmv3 (no is_key fix)  historical reference, from git commit b5c0ca7

The historical "no is_key fix" numbers are hard-coded because the buggy model's
predictions were not preserved on disk; the values come from eval-results.json
at commit b5c0ca7 (before the fix in commit 7858050).

Usage:
    python3 ablation_study.py
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from evaluate import (
    CHEF_SOURCE,
    TARGET_FIELDS,
    build_chef_predictions,
    compute_metrics,
)


CACHE_DIR = Path("./eval-cache")
SPLIT = "test"
N = 100


# Hard-coded reference: LayoutLMv3 BEFORE the is_key labeling fix.
# Source: eval-results.json at git commit b5c0ca7 (before commit 7858050).
LEGACY_LAYOUTLMV3_NO_KEY_FIX = {
    "per_field": {
        "norm": {
            "menu.nm":                  {"f1": 0.938},
            "menu.price":               {"f1": 0.970},
            "menu.cnt":                 {"f1": 0.970},
            "sub_total.subtotal_price": {"f1": 0.074},
            "total.total_price":        {"f1": 0.010},
        },
        "exact": {
            "menu.nm":                  {"f1": 0.938},
            "menu.price":               {"f1": 0.970},
            "menu.cnt":                 {"f1": 0.970},
            "sub_total.subtotal_price": {"f1": 0.074},
            "total.total_price":        {"f1": 0.010},
        },
    },
    "micro":     {"norm": {"f1": 0.787}, "exact": {"f1": 0.787}},
    "line_item": {"norm": 0.878,         "exact": 0.878},
    "checksum":  {"consistency_rate": float("nan"), "mean_abs_error_pct": float("nan")},
}


def load_cached(name: str) -> List[Dict[str, List[str]]]:
    path = CACHE_DIR / f"{name}-{SPLIT}-{N}.json"
    return json.loads(path.read_text())


def reverse_chef_source() -> Dict[str, str]:
    """Items ← Donut, aggregates ← LayoutLMv3. The OPPOSITE of CHEF_SOURCE."""
    return {
        f: ("donut" if CHEF_SOURCE[f] == "layoutlmv3" else "layoutlmv3")
        for f in CHEF_SOURCE
    }


def build_with_routing(
    layoutlmv3_preds: List[Dict[str, List[str]]],
    donut_preds: List[Dict[str, List[str]]],
    routing: Dict[str, str],
) -> List[Dict[str, List[str]]]:
    out = []
    for lv3, dn in zip(layoutlmv3_preds, donut_preds):
        merged = {}
        for f in TARGET_FIELDS:
            merged[f] = (lv3 if routing[f] == "layoutlmv3" else dn).get(f, [])
        out.append(merged)
    return out


def main() -> None:
    print(f"Loading cached predictions from {CACHE_DIR} (split={SPLIT}, N={N}) ...")
    lv3 = load_cached("layoutlmv3")
    dn  = load_cached("donut")

    # Build the ground truth via the evaluation harness (gold parses are
    # deterministic from CORD, so no model is needed).
    from datasets import load_dataset
    from data_prep import CORD_DATASET_NAME
    from evaluate import build_ground_truth
    ds = load_dataset(CORD_DATASET_NAME, split=SPLIT)
    examples = list(ds.select(range(min(N, len(ds)))))
    golds = [build_ground_truth(ex) for ex in examples]

    # Compose the four live variants.
    variants: Dict[str, List[Dict[str, List[str]]]] = {
        "layoutlmv3 (alone)":     lv3,
        "donut (alone)":          dn,
        "CHEF (ours)":            build_chef_predictions(lv3, dn),
        "CHEF (reversed)":        build_with_routing(lv3, dn, reverse_chef_source()),
    }

    results = {name: compute_metrics(preds, golds) for name, preds in variants.items()}
    # Insert the historical reference at the top of the table for context.
    results = {"layoutlmv3 (no is_key fix)": LEGACY_LAYOUTLMV3_NO_KEY_FIX, **results}

    print_table(results)
    Path("./ablation-results.json").write_text(json.dumps(results, indent=2, default=str))
    print("\nWrote ablation-results.json")


def fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}" if not (v != v) else "  —  "  # NaN check
    return str(v)


def print_table(results: Dict[str, Dict[str, Any]]) -> None:
    names = list(results.keys())
    col_w = max(14, 90 // max(len(names), 1))

    def hdr() -> str:
        return f"{'':32s}" + "".join(f"{n:>{col_w}s}" for n in names)

    rows = []
    rows.append("=" * (32 + col_w * len(names)))
    rows.append("ABLATION STUDY — CORD test (n=100), entity-level F1, normalized matching")
    rows.append("=" * (32 + col_w * len(names)))
    rows.append(hdr())
    rows.append("-" * (32 + col_w * len(names)))
    for field in TARGET_FIELDS:
        row = f"{field:32s}"
        for n in names:
            row += f"{fmt(results[n]['per_field']['norm'][field]['f1']):>{col_w}s}"
        rows.append(row)
    rows.append("-" * (32 + col_w * len(names)))
    row = f"{'MICRO F1':32s}"
    for n in names:
        row += f"{fmt(results[n]['micro']['norm']['f1']):>{col_w}s}"
    rows.append(row)
    row = f"{'Line-item matching accuracy':32s}"
    for n in names:
        row += f"{fmt(results[n]['line_item']['norm']):>{col_w}s}"
    rows.append(row)
    row = f"{'Checksum consistency (±5%)':32s}"
    for n in names:
        row += f"{fmt(results[n]['checksum']['consistency_rate']):>{col_w}s}"
    rows.append(row)
    rows.append("=" * (32 + col_w * len(names)))
    print("\n" + "\n".join(rows))


if __name__ == "__main__":
    main()
