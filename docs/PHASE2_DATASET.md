# Phase 2: Prepare the support-ticket dataset

Course equivalent: **"Prepare Support Ticket Dataset"** cell (run locally with
[`scripts/prepare_dataset.py`](../scripts/prepare_dataset.py) instead of the Colab upload widget).
Checks: [`scripts/data_report.py`](../scripts/data_report.py) →
[`results/phase2_data_report.json`](../results/phase2_data_report.json).

## What was done

1. Loaded `data/raw/support_tickets.csv` (585 rows, columns `category_truth`, `text`). All rows
   have one of the 7 known labels; no nulls, no empty texts, no duplicate texts.
2. Shuffled and split **80/20, stratified by label**, `random_state=42` (same seed and order as
   the course notebook, so the split is identical to a Colab run).
3. Converted the **training split only** to ShareGPT JSON → `data/processed/TRAIN.json`:
   ```json
   {"messages": [
     {"role": "system",    "content": "You are an IT helpdesk ticket routing assistant. Given a support ticket, respond with exactly one of the following categories: Support general, Fileservice, O365, EOL, Software, Active Directory, Computer-Services."},
     {"role": "user",      "content": "Support Ticket: <ticket text>"},
     {"role": "assistant", "content": "<label>"}]}
   ```
4. Copied it to `LLaMA-Factory/data/TRAIN.json` and registered it as **`support_tickets`** in
   `LLaMA-Factory/data/dataset_info.json` (entry saved in
   [`data/processed/dataset_info_entry.json`](../data/processed/dataset_info_entry.json)), so
   LLaMA Board lists it in its dataset drop-down.
5. Saved `train_split.csv` (468) and the held-out **`val_split.csv` (117)**.

## Class balance

![Class distribution](../results/phase2_class_distribution.png)

| Label | Total | Train | Val | Share |
|---|---:|---:|---:|---:|
| Support general | 150 | 120 | 30 | 25.6% |
| Fileservice | 138 | 110 | 28 | 23.6% |
| O365 | 92 | 74 | 18 | 15.7% |
| EOL | 58 | 46 | 12 | 9.9% |
| Software | 58 | 46 | 12 | 9.9% |
| Active Directory | 48 | 39 | 9 | 8.2% |
| Computer-Services | 41 | 33 | 8 | 7.0% |

- The split is properly stratified: no class's share differs by more than **0.6 percentage
  points** between train and validation.
- **Imbalance:** the two largest classes are ~3.5× the two smallest. A model that always says
  `Support general` would already score **25.6 %** accuracy. That is the floor to beat.
- **Small validation classes:** `Active Directory` has 9 validation tickets and
  `Computer-Services` 8, so one mistake moves their recall by 11–12.5 points. Per-class numbers
  for those classes should be read with that in mind.

## Leakage check (train vs validation)

| Check | Result |
|---|---|
| Exact text overlap (case/whitespace-normalised) | **0** |
| Nearest-train-ticket TF-IDF cosine similarity, median / p90 / max | 0.41 / 0.69 / 0.95 |
| Validation tickets with a near-paraphrase in train (sim ≥ 0.8) | **3 of 117** |
| Validation tickets with a close neighbour (sim ≥ 0.6) | 14 of 117 |
| Label agreement of the single most similar train ticket (1-NN) | **64.1 %** |

The dataset contains some templated paraphrases. The most similar pair (0.95) differs only by
"Hi there" / "Thanks!". This is mild (3 tickets) and is a property of the course dataset, so the
split is kept as-is for comparability. It is still worth remembering when reading the final
accuracy. The **1-NN agreement of 64 %** is a useful reference point: a pure word-overlap lookup
already gets about 64 %, so the fine-tuned model should land clearly above that.

**Label ambiguity:** several tickets that sound like account work are labelled `Support general`
rather than `Active Directory`. Example: *"create a light external user account… email access and
basic system access"*, or *"set up a new Windows account for a new team member… email through
Exchange, Skype, and standard file share access"*. The pattern appears to be that **multi-service
onboarding** goes to general triage while single identity tasks go to AD. The model has to learn
that from the labels; expect most errors at the `Support general` ↔ `Active Directory` boundary.

## LLaMA-Factory pipeline check (what training will actually see)

The dataset was loaded through LLaMA-Factory's own SFT data pipeline, with the settings LLaMA Board
uses by default for `Qwen3-1.7B-Base`: template **`qwen3`**, "Enable thinking" **on**,
`cutoff_len` **2048**.

- All **468** examples parse correctly.
- Training sequences are **90–190 tokens** (median 107, p95 126). The default `cutoff_len` of
  2048 truncates nothing. Batches are padded only to their longest example, so a smaller cutoff
  would not speed things up either.
- Only **7–10 tokens per example carry loss** (the answer); the system prompt and ticket are masked.

Exact sequence for one example (masked prompt | **trained response**):

```
<|im_start|>system
You are an IT helpdesk ticket routing assistant. ...<|im_end|>
<|im_start|>user
Support Ticket: Hi, could you please create a light external user account ...<|im_end|>
<|im_start|>assistant
───────────────────────────────── loss starts here ─────────────
<think>\n\n</think>\n\nSupport general<|im_end|>\n
```

### ⚠️ Template finding: the course's `classify()` would fail with these defaults

The `qwen3` template is a *reasoning* template. Because the training answers contain no
reasoning, it prepends an **empty `<think>\n\n</think>\n\n` block to every trained response**. The
course notebook's `classify()` then:

1. builds the prompt with `apply_chat_template(..., add_generation_prompt=True)`. Verified: this
   equals the training prompt **without** the think block;
2. lets the model generate, which it will do as trained: `<think>\n\n</think>\n\nO365`;
3. decodes with `skip_special_tokens=True`. **`<think>` / `</think>` are *not* special tokens**
   in the Qwen3 tokenizer (verified), so the text still starts with `<think>`;
4. checks `generated.startswith(label)`, which fails for every label, so it **falls back to
   `"Support general"` for every ticket** (≈ 25.6 % accuracy regardless of training). The
   confidence score is also computed on the `<think>` token instead of the label.

**Fix, used in Phase 5:** build the inference prompt with `enable_thinking=False`. Verified:
the Hugging Face template then appends exactly `<think>\n\n</think>\n\n`, so the prompt is
**token-identical to the training sequence right up to the label**. The first generated token is
the label's first token, and the confidence score measures the label decision. Training keeps
LLaMA Board's defaults; nothing about the course's training procedure changes.
