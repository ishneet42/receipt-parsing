# Receipt Parsing: OCR-Free vs. OCR-Dependent

CSCI 5922 — Neural Networks and Deep Learning
Ishneet Chadha, Manikandan Gunaseelan

A systematic comparison of two deep-learning paradigms for item-level receipt
parsing, motivated by automated bill splitting:

- **LayoutLMv3** — *OCR-dependent*. An external OCR engine (Tesseract) extracts
  words and bounding boxes; a multimodal transformer classifies each token into
  a semantic field.
- **Donut** — *OCR-free*. An end-to-end transformer maps raw image pixels
  directly to a structured JSON output, with no intermediate text recognition.

We fine-tune LayoutLMv3 on CORD v2 under MacBook Air / MPS compute, adopt the
official Naver Donut-CORD-v2 checkpoint, and evaluate both on the CORD test
set. We also compose a **hybrid** that routes fields to whichever architecture
wins each one (line items → LayoutLMv3, aggregates → Donut), and introduce a
**checksum-consistency** metric that measures whether extracted line items sum
to the extracted total.

## Headline results (CORD test set, 100 receipts, normalized matching)

| Metric | LayoutLMv3 | Donut | **Hybrid** |
| :-- | --: | --: | --: |
| `menu.nm` F1 | **0.899** | 0.791 | **0.899** |
| `menu.price` F1 | **0.968** | 0.866 | **0.968** |
| `menu.cnt` F1 | **0.980** | 0.968 | **0.980** |
| `sub_total.subtotal_price` F1 | 0.882 | **0.901** | **0.901** |
| `total.total_price` F1 | 0.898 | **0.917** | **0.917** |
| **Micro F1 (5 fields)** | 0.937 | 0.878 | **0.940** |
| **Line-item matching accuracy** | **0.886** | 0.732 | **0.886** |
| Checksum pass rate (±5%) | 0.549 | 0.536 | 0.526 |
| Mean \|sum − total\| error | **5.9%** | 22,691% | 22,687% |

The ground-truth checksum ceiling on CORD is ~53% (receipts include tax and
service charges). LayoutLMv3 hits this ceiling almost exactly. Donut's huge
mean error reflects occasional catastrophic hallucinations of the total.

## Repository layout

```
.
├── README.md                              ← you are here
├── requirements.txt                       ← pinned deps (Python 3.11-3.13)
│
├── paper/
│   ├── paper.typ                          ← Typst source
│   └── paper.pdf                          ← compiled PDF
│
├── cluster/                               ← CU Boulder Alpine Slurm scripts
│   ├── README.md                          ← Alpine runbook
│   ├── setup_alpine_env.sh
│   ├── donut/                             ← Mani's Donut training jobs
│   │   ├── alpine_smoke.sbatch
│   │   ├── alpine_full.sbatch
│   │   └── alpine_cpu_smoke.sbatch
│   └── layoutlmv3/                        ← our LayoutLMv3 training jobs
│       ├── alpine_smoke.sbatch
│       └── alpine_full.sbatch
│
├── cord_examples/                         ← one image + label JSON for sanity tests
├── processed_receipts/                    ← filtered CORD JSONL (from data_prep.py)
├── eval-cache/                            ← cached model predictions (speeds up re-scoring)
├── eval-results.json                      ← last metrics dump
│
├── data_prep.py                           ← CORD/SROIE loading + filtering
├── export_cord_example.py                 ← export one CORD sample as PNG+JSON
│
├── train_donut.py                         ← Donut fine-tune (unused; see note below)
├── predict_donut.py                       ← Donut inference using a locally trained ckpt
├── predict_donut_pretrained.py            ← Donut inference using Naver's CORD-v2 ckpt ✔
│
├── train_layoutlmv3.py                    ← LayoutLMv3 fine-tune on CORD ✔
├── predict_layoutlmv3.py                  ← LayoutLMv3 inference (Tesseract OCR) ✔
│
├── predict_hybrid.py                      ← run both models + compose + verify ✔
└── evaluate.py                            ← shared eval harness (metrics table) ✔
```

> **Note on Donut training.** We attempted to fine-tune Donut from scratch on
> both a MacBook Air and Google Colab free-tier. Both ran out of memory. For
> the final evaluation we adopted the publicly released
> `naver-clova-ix/donut-base-finetuned-cord-v2` checkpoint instead — this is a
> standard substitution in comparison studies. The `train_donut.py` script is
> preserved for reproducibility of the attempt.

## Setup

One-time, on macOS:

```bash
cd receipt-parsing
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Tesseract is only needed for predict_layoutlmv3.py / predict_hybrid.py
brew install tesseract
```

Every new terminal session:

```bash
cd receipt-parsing
source .venv/bin/activate
```

## How to run things

### Fine-tune LayoutLMv3 on CORD (on a MacBook)

```bash
python3 train_layoutlmv3.py \
  --output-dir ./layoutlmv3-fixed \
  --num-train-epochs 15 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --max-length 384 \
  --freeze-visual \
  --early-stopping-patience 4 \
  --seed 42
```

Expected: ~1.5 hours on Apple Silicon, ~50 min with a warm HuggingFace cache.
Early stopping typically fires around epoch 8.

### Test a single receipt (hybrid pipeline — recommended)

```bash
python3 predict_hybrid.py --image-path /path/to/receipt.png
```

Runs both LayoutLMv3 (Tesseract OCR) and Donut, composes the hybrid output,
prints each model's view, and verifies the arithmetic checksum and
cross-model agreement.

### Test a single receipt (one model only)

```bash
python3 predict_layoutlmv3.py --model-path ./layoutlmv3-fixed --image-path /path/to/receipt.png
python3 predict_donut_pretrained.py --image-path /path/to/receipt.png
```

### Evaluate on the full CORD test set

```bash
python3 evaluate.py --models layoutlmv3 donut --layoutlmv3-path ./layoutlmv3-fixed
```

Runs both models on 100 receipts, composes the hybrid, computes per-field F1,
micro-F1, line-item matching accuracy, and checksum consistency. About 3 min
total on MPS. Predictions are cached under `eval-cache/`; rerun with
`--cached` to skip inference and only recompute metrics.

### Train on the Alpine cluster (GPU)

See `cluster/README.md` for the full Slurm runbook.

```bash
sbatch --account=<ALLOCATION> cluster/layoutlmv3/alpine_smoke.sbatch   # 20-step smoke test
sbatch --account=<ALLOCATION> cluster/layoutlmv3/alpine_full.sbatch    # full training
```

## Key findings

1. **Complementary strengths.** LayoutLMv3 wins on per-item fields (menu name /
   price / count), Donut wins on aggregates (subtotal, total). Neither
   architecture dominates across the board.

2. **A field-level routed hybrid beats either single model.** Routing fields
   to their respective winning architecture yields the best overall micro-F1
   (0.940) without requiring any additional training.

3. **Token classifiers struggle with key-value structures.** LayoutLMv3
   initially got F1 ≈ 0.01 on total because it labeled both the key word
   ("TOTAL") and the value ("60.000") as part of the same field. A minor
   labeling fix (skip `is_key=1` tokens) brought F1 up to 0.898. Donut does
   not have this problem by construction: its JSON output schema forces
   key/value separation.

4. **OCR errors propagate.** A qualitative test on a photographed real-world
   grocery receipt shows LayoutLMv3 produces unusable output because
   Tesseract fails on the image, while Donut recovers recognizable items and
   prices from raw pixels. Swapping Tesseract for a modern OCR (PaddleOCR,
   Apple Vision) is expected to close most of this gap.

5. **Cross-architecture agreement is a free verification signal.** When both
   models independently predict the same total, that's stronger evidence of
   correctness than either model's internal confidence, because the two
   architectures fail in uncorrelated ways.

See `paper/paper.pdf` for the full write-up.

## Remaining work

- SROIE cross-dataset evaluation
- Qualitative error taxonomy on 50 sampled test receipts
- Retrain LayoutLMv3 with unfrozen visual backbone on GPU
- Swap Tesseract for PaddleOCR or Apple Vision
- Learn a router conditioned on image quality rather than hand-routing by field

## Dependencies

- Python 3.11, 3.12, or 3.13 (not 3.14 — HuggingFace `datasets` is incompatible)
- `torch==2.8.0`, `transformers==4.46.3`, `datasets==3.0.2`, `accelerate==1.0.1`
- `pytesseract>=0.3.10` + the Tesseract binary (`brew install tesseract`)

Exact pinned versions are in `requirements.txt`.
