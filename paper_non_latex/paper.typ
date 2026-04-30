#set document(
  title: "OCR-Free vs. OCR-Dependent Architectures for Item-Level Receipt Parsing: Preliminary Findings",
  author: ("Ishneet Chadha", "Manikandan Gunaseelan"),
)
#set page(
  paper: "us-letter",
  margin: (x: 1in, y: 1in),
  numbering: "1",
)
#set text(font: "New Computer Modern", size: 10pt, lang: "en")
#set par(justify: true, leading: 0.6em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): set text(size: 12pt, weight: "bold")
#show heading.where(level: 2): set text(size: 11pt, weight: "bold")
#show heading.where(level: 3): set text(size: 10.5pt, weight: "bold")
#show heading: it => [#v(0.6em) #it #v(0.3em)]
#show link: set text(fill: blue.darken(20%))

#align(center)[
  #text(size: 16pt, weight: "bold")[
    OCR-Free vs. OCR-Dependent Architectures \
    for Item-Level Receipt Parsing:
    Preliminary Findings
  ]
  #v(0.4em)
  #text(size: 11pt)[
    Ishneet Chadha, Manikandan Gunaseelan \
    University of Colorado Boulder \
    Course CSCI 5922 — Neural Networks and Deep Learning
  ]
]

#v(1em)

#align(center)[
  #block(width: 90%)[
    #align(left)[
      #text(weight: "bold")[Abstract.]
      Splitting shared expenses fairly requires accurate extraction of individual
      line items and prices from receipt images — a task that current bill-splitting
      applications handle poorly due to their reliance on black-box OCR APIs. This
      paper presents a systematic comparison of two contrasting deep learning paradigms
      for item-level receipt parsing — an OCR-free approach (Donut) and an OCR-dependent
      approach (LayoutLMv3) — and introduces *CHEF* — *Composed Hybrid for Extraction & Field-routing* — that composes the two.
      We fine-tune LayoutLMv3 on the CORD v2 dataset under resource-constrained
      conditions (MacBook Air, MPS backend, frozen visual backbone) and evaluate the
      pre-finetuned Donut checkpoint released by the original authors. On the 100-receipt
      CORD test set we report entity-level F1 per field, micro-F1, line-item matching
      accuracy, and a novel receipt-level #emph[checksum-consistency] metric. Our results
      reveal a complementary strength pattern: LayoutLMv3 wins on per-item line-level
      fields (menu.nm F1 0.899, menu.price F1 0.968, line-item matching 0.886) while
      Donut wins on aggregate fields (subtotal F1 0.901, total F1 0.917). CHEF, a field-routed hybrid that takes line items from LayoutLMv3 and aggregates from Donut,
      strictly beats either single model (micro-F1 0.940 vs 0.937 and 0.878). A reversed
      hybrid (items from Donut, aggregates from LayoutLMv3) performs worse than either
      single model (0.874), demonstrating that the routing direction itself is the source
      of the gain — not ensembling per se. An ablation of the training-label construction
      shows a simple `is_key` token filter contributes +0.150 micro-F1. Crucially,
      replacing CORD's gold-annotated OCR with the output of a real OCR engine
      (Tesseract) at inference time — same model, same images, same metrics —
      collapses LayoutLMv3's micro-F1 from 0.937 to 0.314, dropping it well below
      Donut's 0.878. This OCR-source ablation reframes the headline result: roughly
      two thirds of LayoutLMv3's apparent CORD performance is contributed by
      human-verified OCR annotations rather than by the trained classifier. A
      qualitative comparison on a real-world grocery receipt photograph further
      illustrates the OCR-propagation failure mode. We conclude that receipt parsing
      is a composite task whose sub-tasks reward different architectures, that
      cross-architecture output agreement provides a free verification signal, and
      that standard CORD evaluations of OCR-dependent models should be interpreted
      as upper bounds on deployable performance.

      #text(weight: "bold")[Keywords:]
      receipt parsing, optical character recognition, document understanding, Donut,
      LayoutLMv3, information extraction, bill splitting, deep learning.
    ]
  ]
]

#v(0.5em)

= Introduction

Splitting shared expenses is a ubiquitous social task that arises whenever groups of
people dine out, travel together, or share household costs. While mobile applications
such as Splitwise and Venmo have simplified the process of tracking who owes whom, they
still require users to manually enter itemized amounts — a tedious and error-prone step
that discourages adoption. A more desirable workflow would allow a user to simply
photograph a receipt and have the system automatically extract each line item and its
price, after which individuals can be assigned to specific items for fair, item-level
splitting. Achieving this vision requires robust receipt information extraction, a task
that sits at the intersection of computer vision, optical character recognition (OCR),
and natural language understanding.

Existing approaches to receipt understanding generally follow one of two paradigms. The
first and more established paradigm is _OCR-dependent_: a dedicated OCR engine first
localizes and transcribes all text regions in the receipt image, and a downstream model
then consumes these text tokens along with their spatial coordinates to classify each
token into semantic categories [3, 2]. The second, more recent paradigm is
_OCR-free_: an end-to-end model ingests the raw receipt image and directly produces
structured output without an intermediate text-recognition step [1]. Prior
evaluations of both paradigms predominantly report aggregate accuracy on high-level
fields such as the store name, date, and total amount [6, 5]. Comparatively
little attention has been paid to _item-level_ extraction — that is, correctly
identifying every individual menu item and its associated price — which is precisely
what a bill-splitting application requires.

In this work, we conduct a systematic comparison of OCR-free and OCR-dependent
architectures specifically evaluated on item-level receipt parsing. We select Donut
[1] as a representative OCR-free model and LayoutLMv3 [2] as a
representative OCR-dependent model, and evaluate both on the CORD v2 dataset [5].
Our preliminary contributions are threefold: (1) we fine-tune LayoutLMv3 for token
classification over five item-level CORD fields under practical compute constraints;
(2) we wire the pre-finetuned Donut-CORD checkpoint released by the original authors
into a directly comparable inference pipeline; and (3) we perform a qualitative
comparison on a real-world grocery receipt photograph that exposes the characteristic
failure modes of each paradigm. A complete quantitative evaluation on the CORD test set
and cross-dataset generalization to SROIE [6] are left as ongoing work.

= Related Work

*Receipt OCR and Information Extraction Benchmarks.* The ICDAR 2019 SROIE competition
[6] introduced a benchmark of 1,000 scanned receipt images annotated for text
localization, OCR, and key information extraction (company, date, address, total). The
CORD dataset [5] expanded on this with over 11,000 Indonesian receipt images
annotated with 30 semantic classes organized into five superclasses. WildReceipt
[7] contributed 1,768 receipt images with 25 key-value categories. Our work does
not propose a new dataset; we leverage CORD for training and evaluation and reserve
SROIE for cross-dataset generalization.

*OCR-Dependent Document Understanding.* LayoutLM [3] pioneered the joint
pre-training of text, layout, and image features for document understanding. LayoutLMv2
[4] and LayoutLMv3 [2] introduced improved pre-training objectives and
unified text-image masking strategies, achieving state-of-the-art results on CORD and
FUNSD. All of these methods depend on an external OCR engine.

*OCR-Free Document Understanding.* Donut [1] proposed an end-to-end transformer
that encodes document images using a Swin Transformer [8] vision encoder and
decodes structured output using a BART-based text decoder, entirely bypassing OCR.
Pix2Struct [9] extended this idea by pre-training on web page screenshots.

*Receipt Scanning for Bill-Splitting Applications.* Several mobile applications and
open-source projects implement receipt-scanning bill splitters, including OCRcpt
[10] and Receipt Hacker [11], all of which rely on third-party OCR APIs
as black boxes and do not conduct systematic evaluations of extraction accuracy.

= Methods

== Dataset and Preprocessing

We use the CORD v2 dataset [5], consisting of 800 training, 100 validation, and
100 test receipt images annotated with bounding boxes, transcribed text, and 30 semantic
labels. For our item-level evaluation we focus on five fields relevant to bill splitting:
`menu.nm` (item name), `menu.price` (item price), `menu.cnt` (item count),
`sub_total.subtotal_price`, and `total.total_price`.

For each CORD example, we extract word-level records by iterating through
`valid_line` entries: every word contributes its text, its quadrilateral-derived
bounding box (converted to an axis-aligned $[x_0, y_0, x_1, y_1]$ and normalized to the
LayoutLMv3-required 0–1000 range), and a BIO-scheme label. Words belonging to target
fields receive `B-<field>` on the first word and `I-<field>` on subsequent words; all
other words receive the `O` (outside) label. This produces an 11-class label space.

== Model 1: LayoutLMv3 (OCR-Dependent)

LayoutLMv3 [2] is a multimodal transformer that jointly encodes text tokens,
their 2D spatial positions, and image patch embeddings. We fine-tune the
`microsoft/layoutlmv3-base` checkpoint as a token classifier using
`LayoutLMv3ForTokenClassification` with an 11-class output head corresponding to our
BIO scheme. The processor is configured with `apply_ocr=False`, allowing us to supply
CORD's gold word boxes directly as training targets (standard practice for LayoutLMv3
CORD fine-tuning).

*Training configuration.* Training was conducted on a MacBook Air with Apple Silicon
using the MPS backend. Due to memory constraints observed in initial attempts (a full
unfrozen configuration with batch size 4 estimated over 30 hours and caused system
instability from unified-memory pressure), we adopt a constrained recipe: the visual
backbone is frozen, per-device batch size is 1, gradient accumulation is 8 (effective
batch size 8), maximum sequence length is 384 tokens, learning rate is
$5 times 10^(-5)$ with 10% warmup, weight decay is 0.01, and the optimizer is AdamW.
We train for up to 15 epochs with early stopping on `eval_loss` (patience = 4), save
the two most recent checkpoints, and load the best checkpoint at the end of training.
All randomness is seeded to 42.

== Model 2: Donut (OCR-Free)

Donut [1] employs a Swin Transformer [8] as a visual encoder that converts
the input image into a sequence of patch embeddings, followed by a BART-based
autoregressive decoder that generates structured output token by token. Initial attempts
to fine-tune Donut from the base checkpoint failed under our compute budget: the model's
1280 × 960 input resolution and autoregressive decoder require substantially more VRAM
than LayoutLMv3-base (approximately 14–18 GB per sample at batch size 1), and both a
MacBook Air and a Google Colab free-tier T4 session proved insufficient after
multi-hour attempts.

Rather than compromise the training recipe further, we adopt the pre-finetuned Donut
checkpoint `naver-clova-ix/donut-base-finetuned-cord-v2`, which was fine-tuned on
CORD v2 by the original Donut authors using their reference recipe. This decision
isolates architectural capability from training-setup variability and is standard
practice in comparison studies of document-understanding systems.

At inference time, images are fed to the Donut processor and the decoder is prompted
with the task-start token `<s_cord-v2>`. Generation uses greedy decoding
(`num_beams=1`). We post-process Donut's tagged output stream into a nested JSON
structure via `processor.token2json`, and further flatten it to our five target fields
to enable apples-to-apples comparison with the LayoutLMv3 output.

== Item-Level Parsing Pipeline

After each model produces its raw predictions, we apply a lightweight post-processing
step to group extracted fields into structured records. For Donut, the decoder output
is a JSON-like token stream that we parse into item records (name, price, count). For
LayoutLMv3, we walk the tokenizer's `word_ids` to attribute each subword prediction back
to its source word, then stitch adjacent tokens sharing the same field label into spans,
respecting the BIO scheme such that a `B-` tag always opens a new span. This grouping
logic yields the same canonical output schema — a dictionary keyed by the five target
fields — for both models, ensuring that downstream evaluation is architecture-agnostic.

At real-world inference time (as opposed to evaluation on CORD's pre-annotated word
boxes), LayoutLMv3 requires an external OCR step. We use Tesseract v5 via `pytesseract`
to extract word-level bounding boxes and text from the input image, then normalize
boxes to the 0–1000 range before feeding them to the model.

= Experiments and Results

== Training Dynamics on CORD

We fine-tune LayoutLMv3 according to the configuration described in Section 3.2. Total
training time was approximately 1 hour 43 minutes on the MacBook Air, averaging 4.75
seconds per optimizer step. Early stopping terminated training at epoch 12 of a planned
15 after `eval_loss` failed to improve for four consecutive epochs. The best checkpoint
was epoch 8, with `eval_loss` = 0.074.

#align(center)[
  #table(
    columns: (auto, auto, auto, auto),
    align: (center, right, right, right),
    stroke: 0.5pt,
    inset: 6pt,
    [*Epoch*], [*`eval_loss`*], [*Token acc.*], [*Non-O acc.*],
    [1],  [0.315], [88.6%], [91.2%],
    [2],  [0.216], [91.4%], [95.4%],
    [3],  [0.104], [97.5%], [96.2%],
    [*4*],  [*0.103*], [*96.7%*], [*96.2%*],
    [5],  [0.105], [97.5%], [97.1%],
    [6],  [0.116], [97.9%], [97.0%],
    [7],  [0.121], [97.1%], [96.2%],
    [8],  [0.127], [97.8%], [96.9%],
  )
  #text(size: 9pt)[
    *Table 1.* Per-epoch validation metrics for LayoutLMv3 on the CORD validation split
    (100 receipts, 1,169 non-background tokens under the final label scheme). Early
    stopping selects epoch 4 as the best checkpoint by `eval_loss`; training terminated
    at epoch 8 of a planned 15 after four consecutive epochs without improvement.
  ]
]

We report two token-level accuracies: overall accuracy across all predicted tokens and
non-O accuracy, which restricts the evaluation to tokens whose ground-truth label is
not the `O` background class. The latter is more informative because the class
distribution is imbalanced — most tokens in a receipt are background (store name,
dates, separators, etc.) and a trivial classifier that always predicts `O` would
already exceed 70% overall accuracy. The training curve is well-behaved: cross-entropy
loss decreases monotonically for the first eight epochs before entering a noisy plateau,
and early stopping correctly identifies the turning point.

== Qualitative Comparison on a Clean Receipt (CORD Sample)

To verify end-to-end correctness of both pipelines, we compare predictions on a
randomly selected CORD test image (`test_0.png`), whose ground-truth fields are
`menu.nm` = "-TICKET CP", `menu.price` = "60.000", `menu.cnt` = "2",
`sub_total.subtotal_price` = "60.000", `total.total_price` = "60.000".

*LayoutLMv3 (Tesseract OCR → trained classifier):* Correctly recovered the item name
and count, and correctly labeled the total as a total-price field. Tesseract misread
the price "60" as "£0", which propagated into the model's output
(`menu.price` = "£0,000"); the subtotal was tagged onto the word "Subtotal" rather
than the numeric value. Net: 3–4 of 5 fields substantially correct.

*Donut (pre-finetuned CORD checkpoint):* Correctly recovered all five fields with
minor punctuation variants (`60,000` vs. `60.000`) and a spurious leading minus sign
on the subtotal. Net: 4–5 of 5 fields substantially correct.

Both models perform well on this clean, in-distribution sample. The more informative
comparison is on real-world out-of-distribution images, presented next.

== CHEF: Composition and Routing Ablation

The per-field F1 scores in Table 2 suggest a complementary pattern: LayoutLMv3 wins
on per-item fields while Donut wins on aggregates. We exploit this directly by composing
CHEF, a #emph[field-routed hybrid]: for each receipt we take `menu.nm`, `menu.price`,
`menu.cnt` from LayoutLMv3 and `sub_total.subtotal_price`, `total.total_price` from
Donut. No additional training is required; the routing is applied to cached predictions.

To test whether CHEF's improvement is driven by the routing direction (rather
than by ensembling two models in general), we also evaluate a #emph[reversed] CHEF
that assigns each field to the model that loses it in Table 2: items come from Donut,
aggregates come from LayoutLMv3. If any composition of two independent systems were
inherently better, the reversed CHEF would also beat its components. It does not.

#align(center)[
  #table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    stroke: 0.5pt,
    inset: 5pt,
    [*System*], [*Micro F1*], [*Line-item*], [*subtotal F1*], [*total F1*],
    [LayoutLMv3 (alone)],    [0.937], [*0.886*], [0.882], [0.898],
    [Donut (alone)],         [0.878], [0.732],   [0.901], [0.917],
    [*CHEF (ours)*],       [*0.940*], [*0.886*], [*0.901*], [*0.917*],
    [CHEF (reversed)],     [0.874], [0.732],   [0.882], [0.898],
  )
  #text(size: 9pt)[
    *Table 3.* Hybrid composition and routing ablation on CORD test
    (normalized matching, $n=100$). CHEF (ours) routes each field to the winning
    model in Table 2 and strictly dominates both single models on micro-F1 and
    line-item matching accuracy. The reversed CHEF routes each field to the losing
    model and performs #emph[worse] than either single model — establishing that the
    gain comes from the routing direction itself, not from ensembling.
  ]
]

== Label-Construction Ablation: the `is_key` Filter

A subtle choice during training-target construction dominates LayoutLMv3's performance
on aggregate fields. The CORD annotation marks key words (e.g., the token "TOTAL"
preceding the numeric total) with an `is_key = 1` flag, distinguishing them from value
tokens in the same field. Our initial training script assigned the same BIO label to
both key and value tokens; at inference the grouping post-processor then concatenated
them, producing outputs of the form `"TOTAL 60.000"` for `total.total_price` rather
than the gold value `"60.000"`. Exact-match F1 on aggregate fields collapsed to
effectively zero.

Filtering out `is_key = 1` words at label-construction time (assigning them to the
background class `O`) and retraining with an otherwise identical configuration yields
the numbers reported throughout this section.

#align(center)[
  #table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    stroke: 0.5pt,
    inset: 5pt,
    [*Variant*], [*Micro F1*], [*Line-item*], [*menu.price F1*], [*subtotal F1*], [*total F1*],
    [LayoutLMv3, no `is_key` filter], [0.787],   [0.878],   [0.970],   [0.074],   [0.010],
    [*LayoutLMv3, with `is_key` filter*], [*0.937*], [*0.886*], [*0.968*], [*0.882*], [*0.898*],
    [$Delta$ (absolute)], [+0.150], [+0.008], [#sym.minus 0.002], [+0.808], [+0.888],
  )
  #text(size: 9pt)[
    *Table 4.* Label-construction ablation on CORD test (normalized matching, $n = 100$).
    A three-line change that excludes key tokens from target-field labels during
    training accounts for +0.150 absolute micro-F1, concentrated almost entirely in
    the two aggregate fields. Line-item fields are essentially unaffected (±0.01),
    confirming that the failure mode was specific to key-value structures where the
    key and value tokens are adjacent.
  ]
]

== Receipt-Level Checksum Consistency

Beyond field-wise F1, a bill-splitting application requires the extraction to be
#emph[internally coherent]: the sum of the extracted line-item prices should equal the
extracted total. We define a receipt-level #emph[checksum-consistency] metric as the
fraction of receipts for which $|sum_i italic("menu.price")_i - italic("total.total_price")|
#h(0.3em) lt.eq #h(0.3em) 5% dot italic("total.total_price")$. Receipts for which
either the predicted items or the predicted total is missing are excluded from the
denominator.

A natural concern is the ceiling: real receipts include taxes, service charges, and
discounts between the sum of items and the printed total, so even #emph[perfectly
extracted] ground truth will not pass this check on every receipt. Evaluating the
metric on CORD's own gold-standard parses, we measure a ground-truth ceiling of 0.532
on the 94 CORD test receipts that have both non-empty item and total fields, with a
mean absolute error of 6.0%.

#align(center)[
  #table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    stroke: 0.5pt,
    inset: 5pt,
    [*System*], [*Consistency*], [*Mean error*], [*Defined on*],
    [Ground-truth ceiling], [0.532], [6.0%], [94 receipts],
    [LayoutLMv3],           [0.549], [*5.9%*], [91 receipts],
    [Donut],                [0.536], [22,691%], [97 receipts],
    [CHEF (ours)],        [0.526], [22,687%], [97 receipts],
  )
  #text(size: 9pt)[
    *Table 5.* Receipt-level checksum consistency on CORD test. All three systems
    operate at the ground-truth ceiling on the pass/fail metric, indicating that this
    metric is saturated by the presence of taxes and service charges in CORD receipts.
    The mean-absolute-error breakdown exposes a failure mode invisible to F1: Donut
    occasionally hallucinates numerically implausible totals, producing extreme
    outliers that inflate the mean error by four orders of magnitude. LayoutLMv3's
    mean error is indistinguishable from the ground-truth ceiling.
  ]
]

We interpret this as evidence that (a) the simple sum-equals-total checksum is an
uninformative discriminator on CORD at the pass/fail level but (b) is highly
discriminating at the distributional level, exposing a previously unreported
catastrophic-hallucination failure mode of the OCR-free model. For a deployed
bill-splitting application, a single 22,000% error on the total field is functionally
catastrophic regardless of aggregate F1 scores.

A related free signal comes from inter-model agreement at inference. When LayoutLMv3
and Donut independently predict the same numeric value for subtotal or total on a
receipt, this cross-architecture agreement provides a verification signal stronger
than either model's internal confidence, because the two architectures fail in
structurally different ways (OCR-propagation errors vs. generative hallucination).
Quantifying this as an automated confidence score is left for future work.

== OCR-Source Ablation: Gold vs. Tesseract on CORD

The LayoutLMv3 numbers reported in Tables 2-4 use CORD's pre-annotated, human-verified
word boxes as the OCR input. This is the standard evaluation protocol on CORD and
matches how published baselines report results, but it conflates two questions: (1)
how good is the trained classifier? and (2) how good is a deployable
OCR$arrow$classifier pipeline? In any real bill-splitting application no human-verified
OCR is available; the model has to consume the output of an actual OCR engine.

To isolate the OCR-quality effect from the classifier itself, we re-evaluate
LayoutLMv3 on the same 100 CORD test images, the same trained model, and the same
metrics — changing only the source of the words and bounding boxes fed into the
classifier. Tesseract v5 reads each image at inference time; the rest of the pipeline
(box normalization, BIO grouping, post-processing) is identical.

#align(center)[
  #table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    stroke: 0.5pt,
    inset: 5pt,
    [*System*], [*Micro F1*], [*Line-item*], [*menu.price F1*], [*total F1*],
    [LayoutLMv3 + gold OCR (CORD)], [*0.937*], [*0.886*], [*0.968*], [*0.898*],
    [LayoutLMv3 + Tesseract],       [0.314],   [0.061],   [0.277],   [0.260],
    [Donut (OCR-free)],             [0.878],   [0.732],   [0.866],   [0.917],
  )
  #text(size: 9pt)[
    *Table 6.* OCR-source ablation on CORD test (normalized matching, $n = 100$). Holding
    the trained LayoutLMv3 model constant and swapping the source of input words
    from CORD's gold annotations to Tesseract collapses micro-F1 from 0.937 to 0.314
    and line-item matching accuracy from 0.886 to 0.061. The intermediate baseline
    Donut, which has no OCR step, scores 0.878 micro-F1 — substantially above
    LayoutLMv3-with-Tesseract.
  ]
]

This single ablation reframes the headline numbers. LayoutLMv3's nominal 0.937 micro-F1
overstates the real deployment behavior of the OCR-dependent paradigm by roughly
0.62 absolute F1 points; almost two thirds of the model's apparent performance is in
fact contributed by the human-verified OCR annotations that come with CORD, not by
the multimodal classifier itself. Once Tesseract is substituted in — i.e., once we
measure the full pipeline rather than just the classification head — Donut wins by
0.564 absolute micro-F1 on the same images.

The implication is that prior CORD evaluations of OCR-dependent models, including
ours in earlier sections of this paper, should be interpreted as upper bounds on
deployable performance. The OCR-free paradigm's real advantage on CORD becomes visible
only when both systems are evaluated under matched-realism input conditions.

This finding generalizes our earlier qualitative observation on a real-world grocery
receipt (`bill.png`, next section): the OCR-propagation effect is not a corner case
that appears only on degraded photographs but rather a baseline-shifting effect even
on the clean printed receipts that constitute the CORD benchmark.

== Qualitative Comparison on a Real-World Photograph

To probe the robustness of each architecture to real-world image conditions, we
collected a cell-phone photograph of a long printed grocery-store receipt from
Kroger / King Soopers, containing on the order of 15 line items plus taxes, fuel-points
messaging, and a store footer. The image was saved as a 2.4 MB PNG (`bill.png`). We ran
both models on this image with no preprocessing beyond the models' own input transforms.

*LayoutLMv3 on `bill.png`.* Tesseract's OCR output on this image is severely degraded:
the word "TOTAL" is transcribed as `TOVAL`, `Tal`, or `ioial`; "NUMBER OF ITEMS SOLD"
becomes `NUMBER OF ITENS SOL ni`; and large sections of the receipt are read as
nonsense token sequences such as `Pperrevecceresreernesenersneerennienny` and
`rrerviveeiveysreiseaTeerrrren`. Downstream, the LayoutLMv3 classifier faithfully
applies labels to this garbage: it tags `CHANGE` as `menu.nm`, labels the string
`999 384` as a price, and produces an empty `total.total_price` field. The parsed
output is not usable for a bill-splitting application.

*Donut on `bill.png`.* Donut produces recognizable — though still imperfect — item
names and prices. The `menu.nm` field contains entries such as `SAGE TOLLYCHIRT`
(apparently SAGE TORTELLINI), `ROMANE TEATCUE` (ROMAINE LETTUCE), `RCP PENS BELI`
(RCP [recipe] BELL peppers), `KOROGER EGOS` (KROGER EGGS), and `CHO BANASK 7lb`
(CHIQUITA BANANAS). The `menu.price` field contains numerically plausible values
including `10.99`, `1.58`, `0.98`, `3.79`, `10.99`, `2.09`, `0.70`, `47.73`, `1.73`,
`2.90`. The total field is filled (`63/36`) though the slash is a transcription
artifact. The output contains several hallucinations and non-English characters, but
a non-trivial fraction of the line items are correctly recovered at the
item-plus-price granularity.

*Interpretation.* This contrast is the qualitative signature of OCR error propagation.
Once Tesseract fails — as it does on thermal-paper, photographed-at-an-angle,
long-receipt images common in the bill-splitting use case — there is no recovery path
for the downstream classifier, because the text it is classifying no longer corresponds
to the image content. Donut, by attending directly to visual features, degrades more
gracefully: it still makes many errors, but the errors are character-level noise within
recognizable item strings rather than the wholesale fabrication of token content. This
single qualitative example is, of course, not a quantitative claim about
population-level behavior. It does, however, motivate Section 4.4's remaining work: a
full item-level evaluation on the CORD test set and a cross-dataset evaluation on
SROIE, both planned as the next step.

== Head-to-Head Quantitative Evaluation on CORD Test Set

We evaluated both models on the full 100-receipt CORD test split using a single shared
evaluation harness that consumes the canonical `{field: [values]}` output schema from
Section 3.4, ensuring identical scoring for both models. For each of the five target
fields we compute entity-level precision, recall, and F1 using bag-style (multiset)
matching, under two matching modes: _exact_ (string equality) and _normalized_
(lowercased, whitespace-stripped, punctuation normalized). We also report a micro-F1
over all five fields and a _line-item matching accuracy_ metric defined as the
fraction of ground-truth `(menu.nm, menu.price)` pairs for which both the item name
and its associated price are correctly and jointly extracted.

*Results.* Table 2 reports per-field and aggregate scores under normalized matching;
exact-match scores are nearly identical and are omitted for brevity. Inference took
approximately 75 seconds for LayoutLMv3 (~0.75 s/sample) and 159 seconds for Donut
(~1.59 s/sample) on the MacBook Air's MPS backend.

#align(center)[
  #table(
    columns: (auto, auto, auto),
    align: (left, right, right),
    stroke: 0.5pt,
    inset: 6pt,
    [*Field / Metric*], [*LayoutLMv3*], [*Donut*],
    [`menu.nm` (item name) F1],       [*0.938*], [0.791],
    [`menu.price` (item price) F1],   [*0.970*], [0.866],
    [`menu.cnt` (item count) F1],     [0.970],   [0.968],
    [`sub_total.subtotal_price` F1],  [0.074],   [*0.901*],
    [`total.total_price` F1],         [0.010],   [*0.917*],
    [],[],[],
    [*Micro-F1 (5 fields)*],          [0.787],   [*0.878*],
    [*Line-item matching accuracy*],  [*0.878*], [0.732],
  )
  #text(size: 9pt)[
    *Table 2.* Entity-level F1 and line-item matching accuracy on the CORD test set
    (100 receipts, normalized matching). Higher is better. *Bold* indicates the winning
    model for each row.
  ]
]

*Finding 1: Complementary strengths.* Contrary to the narrative in which one paradigm
uniformly dominates, the two models exhibit complementary patterns. LayoutLMv3 wins
decisively on per-item line-level fields (`menu.nm`, `menu.price`) and on line-item
matching accuracy (0.878 vs 0.732). Donut wins decisively on aggregate fields
(`sub_total.subtotal_price`, `total.total_price`), where LayoutLMv3 essentially fails
(F1 < 0.08). `menu.cnt` is a near-tie.

*Finding 2: Donut is better on the overall micro-F1 but LayoutLMv3 is better on the
metric that matters for bill splitting.* Donut's higher micro-F1 (0.878 vs 0.787) is
driven almost entirely by its advantage on the two aggregate fields. For the
bill-splitting use case, however, the critical metric is _line-item matching accuracy_
— the fraction of ground-truth (name, price) pairs correctly paired in the output —
and LayoutLMv3 is meaningfully better on this metric (0.878 vs 0.732, a 20% relative
improvement). A practitioner building a receipt-scanning bill splitter on CORD-like
data would reasonably choose LayoutLMv3 despite its lower aggregate F1.

*Finding 3: LayoutLMv3 collapses on key-value aggregate fields.* We inspected the
LayoutLMv3 predictions on `total.total_price` to understand the near-zero F1.
LayoutLMv3 consistently produces outputs of the form `"TOTAL 60,000"` rather than the
ground-truth `"60.000"`. The cause is a labeling convention in our training script: the
CORD annotation marks both the key word ("TOTAL") and the value word ("60,000") as
part of the same `total.total_price` line, distinguished only by an `is_key` flag. Our
training loop assigns the same BIO-scheme label to both, so the model learns to tag
the key word as part of the value span; our grouping post-processor then concatenates
them. Donut does not suffer from this failure mode because its sequence-to-JSON output
naturally separates keys from values (Donut emits `{"total": {"total_price": "60.000"}}`
directly, without ever representing "TOTAL" as part of the value). This is a real,
inherent advantage of the generative paradigm for key-value structures where keys and
values are adjacent tokens: the token-classification paradigm must rely on label
geometry (B/I tags) that cannot cleanly distinguish key tokens from value tokens when
both are annotated as part of the same field. We expect that filtering out tokens with
`is_key=1` during training-target construction would substantially improve LayoutLMv3's
performance on aggregate fields; this is straightforward to implement and is left as
an improvement for a subsequent training run.

== Remaining Work

Two further experiments remain as next-milestone work.

*Experiment 2: Cross-Dataset Generalization to SROIE.* We plan to run both models
zero-shot on the SROIE test set (400 images) and report field-level F1 for the four
available fields (company, date, address, total), with particular attention to
exact-match accuracy on the `total` field — the most critical field for a checksum
against extracted items.

*Experiment 3: Qualitative Error Taxonomy.* We plan to sample 50 CORD test receipts
and categorize errors into: (a) OCR misreads, (b) price misalignments, (c) missing
items, (d) hallucinated items, and (e) structural errors (merged or split items).
Comparing the distribution across the two models will allow us to characterize
_how_ each paradigm fails, not just how often.

= Discussion

*On the frozen-visual-backbone decision.* Published LayoutLMv3 fine-tunes on CORD
typically report ≈ 96–97% micro-F1 with the full model unfrozen on GPU hardware. Our
98.5% non-O token accuracy on the validation set is not directly comparable to that
number (it is a token-level metric, not field-level micro-F1, and is computed on 5
fields rather than 30), but the relative ordering suggests that freezing the visual
backbone did not materially hurt training on this dataset. A plausible explanation is
that CORD receipts are visually homogeneous: the printed Courier-like font, the rigid
column structure, and the modest aspect-ratio variation mean that the
`microsoft/layoutlmv3-base` pre-trained visual features transfer well out of the box.
This finding is of independent practical interest: practitioners with constrained
training budgets can obtain strong item-level results on CORD without needing a full
GPU.

*On using the pre-finetuned Donut checkpoint.* Our original plan was to fine-tune
Donut from `naver-clova-ix/donut-base` under identical conditions to LayoutLMv3.
Compute constraints made this infeasible, so we adopted the Naver-released
CORD-fine-tuned checkpoint instead. This is a standard substitution in comparison
studies and is arguably _more_ faithful to the Donut paper's intent — we compare
against the model the authors actually optimized — but it introduces an asymmetry:
LayoutLMv3 was fine-tuned by us (with our five target fields, our BIO scheme, and our
hyperparameters), whereas Donut was fine-tuned by the original authors (on the full 30
CORD fields with their recipe). The asymmetry is disclosed and will be discussed
explicitly in the final report.

*On the real-world photograph result.* The `bill.png` experiment is a single data
point and not a statistical claim. Its value is illustrative: it shows that the
well-documented OCR-error-propagation problem of OCR-dependent pipelines is not an
abstract concern but a regime-shifting one when the input distribution moves from
scanned benchmark images to photographed real receipts. For the bill-splitting
application, the real distribution _is_ photographs — this suggests that the
benchmark-accuracy gap between Donut and LayoutLMv3 understates the deployment-accuracy
gap.

*Limitations.* The preliminary results reported here are based on validation-set
metrics, not test-set metrics. Item-level F1, line-item matching accuracy, and tree
edit distance — the metrics explicitly specified in our proposal — have not yet been
computed. We also have not yet run SROIE generalization. These results will be reported
in the final milestone.

= Conclusion

We have fine-tuned LayoutLMv3 on CORD v2 for five-field item-level token classification
and integrated the pre-finetuned Donut-CORD checkpoint into a directly comparable
inference pipeline. On the 100-receipt CORD test set, Donut achieves higher overall
micro-F1 (0.878 vs 0.787), but LayoutLMv3 wins on the line-item matching accuracy
metric that is most relevant to bill splitting (0.878 vs 0.732). The two paradigms
exhibit complementary strengths: LayoutLMv3 excels at per-item line-level extraction
while Donut excels at aggregate key-value fields such as subtotal and total. A
qualitative real-world test additionally demonstrates that LayoutLMv3 is brittle to
OCR failures outside the clean benchmark distribution, whereas Donut degrades more
gracefully. For a bill-splitting application operating on photographs of arbitrary
receipts, these findings argue for either (a) an OCR-free architecture such as Donut,
or (b) an OCR-dependent architecture such as LayoutLMv3 paired with a more robust
upstream OCR engine than Tesseract. We also identified a concrete and easily-fixable
cause of LayoutLMv3's near-zero performance on aggregate fields — key-token inclusion
during label construction — whose remediation is a natural next step. Cross-dataset
generalization on SROIE and a quantitative error taxonomy are next-milestone work.

#heading(numbering: none)[References]

#set text(size: 9pt)
#set par(first-line-indent: 0pt)

[1] Kim, G., Hong, T., Yim, M., Nam, J., Park, J., Yim, J., Hwang, W., Yun, S., Han, D., Park, S. _OCR-Free Document Understanding Transformer._ ECCV 2022, pp. 498–517. \
[2] Huang, Y., Lv, T., Cui, L., Lu, Y., Wei, F. _LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking._ ACM Multimedia 2022, pp. 4083–4091. \
[3] Xu, Y., Li, M., Cui, L., Huang, S., Wei, F., Zhou, M. _LayoutLM: Pre-training of Text and Layout for Document Image Understanding._ KDD 2020, pp. 1192–1200. \
[4] Xu, Y., Xu, Y., Lv, T., Cui, L., Wei, F., et al. _LayoutLMv2: Multi-modal Pre-training for Visually-rich Document Understanding._ ACL 2021, pp. 2579–2591. \
[5] Park, S., Shin, S., Lee, B., Lee, J., Surh, J., Seo, M., Lee, H. _CORD: A Consolidated Receipt Dataset for Post-OCR Parsing._ Document Intelligence Workshop, NeurIPS 2019. \
[6] Huang, Z., Chen, K., He, J., Bai, X., Karatzas, D., Lu, S., Jawahar, C.V. _ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction._ ICDAR 2019, pp. 1516–1520. \
[7] Sun, H., Kuang, Z., Yue, X., Lin, C., Zhang, W. _Spatial Dual-Modality Graph Reasoning for Key Information Extraction._ arXiv:2103.14470, 2021. \
[8] Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B. _Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows._ ICCV 2021, pp. 10012–10022. \
[9] Lee, K., Joshi, M., Turc, I., Hu, H., Liu, F., Eisenschlos, J., Khandelwal, U., Shaw, P., Chang, M.W., Toutanova, K. _Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding._ ICML 2023, pp. 18893–18912. \
[10] Hu, C. _OCRcpt: Receipt Reader and Bill Splitting iOS App Using OCR via Google Vision API._ GitHub: coreyhu/OCRcpt, 2020. \
[11] Won, B. _Receipt Hacker: Mobile App for Instant Receipt-Based Cost Splitting._ GitHub: brendawon/receipt-hacker, 2021.
