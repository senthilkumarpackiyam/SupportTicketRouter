# Support Ticket Router — Fine-tuning Qwen3-1.7B with LoRA

> The Gen Academy · Week 5 project. A general **Qwen3-1.7B-Base** model is fine-tuned with a
> **LoRA adapter (via LLaMA Factory / LLaMA Board)** to route internal IT support tickets to one
> of seven resolver queues. Then it is evaluated against the untrained base model on a held-out
> validation split.

**Status:** 🚧 in progress. See [Project progress](#project-progress).

---

## The business case

An internal IT support team receives tickets via email, Slack and a help portal. Today every
ticket is read by a human before it reaches the right queue. A small fine-tuned model can make
that routing decision first: cheaper and faster than calling a frontier model for every ticket.
**The model never writes a reply.** It predicts one label, and the rest of the tooling takes over.

| Routing label | Downstream queue |
|---|---|
| `Active Directory` | Identity & access: account setup, deactivation, permissions, directory access |
| `Computer-Services` | Endpoint & hardware: devices, printers, workstations, local computer issues |
| `EOL` | Lifecycle: end-of-life, retirement, decommissioning, replacement |
| `Fileservice` | File storage & access: shared drives, folders, file permissions, storage |
| `O365` | Microsoft 365: Outlook, Teams, mailbox, calendar, SharePoint, Office apps |
| `Software` | Application support: installs, licensing, updates, troubleshooting |
| `Support general` | General triage: needs human review / fits no specific queue |

## Approach

| Step | What happens |
|---|---|
| Data | `support_tickets.csv` (585 labelled tickets) → stratified 80/20 train/validation split → ShareGPT JSON registered as `support_tickets` |
| Train | LLaMA Board → `Qwen/Qwen3-1.7B-Base`, dataset `support_tickets`, finetuning type **LoRA** |
| Review | Training loss curve should drop, then level off |
| Merge | Fold the LoRA deltas into the base weights → one standalone model |
| Smoke test | 5 obvious tickets must route correctly |
| Evaluate | Classification report, confusion matrix, baseline (zero-shot A–G letter choice) vs fine-tuned |

**Hardware note:** the course notebook targets a free Colab Tesla T4. This run was done locally
on an **Apple M2 (16 GB unified memory)** using PyTorch's Apple-GPU (`mps`) backend. A
Colab-ready notebook is also included so the run can be repeated on a T4.

## Repository layout

```
.
├── data/
│   ├── raw/support_tickets.csv        # course dataset (585 rows, 7 labels)
│   └── processed/                     # TRAIN.json (ShareGPT), train/val split CSVs
├── notebooks/
│   └── original/                      # the unmodified course notebook
├── scripts/
│   ├── router_common.py               # shared labels, system prompt, paths
│   ├── check_env.py                   # Phase 1 environment / accelerator check
│   ├── prepare_dataset.py             # Phase 2 split + ShareGPT + dataset_info.json
│   ├── data_report.py                 # Phase 2 balance, leakage, LLaMA-Factory template check
│   ├── llamaboard_train.py            # Phase 3 drives LLaMA Board's Train tab (Playwright)
│   ├── llamaboard_chat.py             # Phase 3 loads the adapter in LLaMA Board's Chat tab
│   └── viz_style.py                   # one chart style for all figures
├── configs/                           # LLaMA Board training args          (Phase 3)
├── adapters/                          # trained LoRA adapter               (Phase 3)
├── results/                           # metrics, plots, logs               (Phases 1–6)
├── docs/                              # write-ups per phase
├── demo/                              # narrated demo video                (Phase 7)
├── setup.sh                           # one-shot reproducible setup
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/senthilkumarpackiyam/SupportTicketRouter.git
cd SupportTicketRouter
./setup.sh          # clones LLaMA-Factory @ pinned commit, creates .venv (Python 3.11),
                    # installs deps, downloads Qwen3-1.7B-Base, runs scripts/check_env.py
```

Pinned stack: LLaMA-Factory `f6355ab` (v0.9.6.dev0) · PyTorch 2.15.0 · Transformers 5.8.0 ·
PEFT 0.18.1 · Accelerate 1.11.0 · Gradio 5.50.0 · Python 3.11.

## Project progress

- [x] **Phase 1: Setup:** environment, LLaMA-Factory, base model, repo scaffold → [notes](docs/PHASE1_SETUP.md)
- [x] **Phase 2: Prepare dataset:** stratified split, ShareGPT JSON, `dataset_info.json` → [notes](docs/PHASE2_DATASET.md)
- [x] **Phase 3: Fine-tune via LLaMA Board:** LoRA on Qwen3-1.7B-Base, 90 steps, loss 7.16 → 0.19 in 69 min → [notes](docs/PHASE3_TRAINING.md)
- [ ] **Phase 4: Review training:** loss curve
- [ ] **Phase 5: Merge + smoke test:** baseline, merged model, `classify()`
- [ ] **Phase 6: Evaluate + compare:** report, confusion matrix, baseline vs fine-tuned
- [ ] **Phase 7: Demo video**

## Key findings so far

- **Chat template (Phases 2–3):** for a *Base* model LLaMA Board leaves the template at
  `default` (`System:/Human:/Assistant:`). LLaMA-Factory then rewrites the tokenizer's chat
  template to match and saves it with the adapter, so inference must load the tokenizer **from the
  adapter folder**, as the course notebook does. Picking `qwen3` by hand instead would make every
  answer start with an empty `<think></think>` block, and the course's `classify()` would label
  **every** ticket `Support general`. [Details](docs/PHASE2_DATASET.md#which-chat-template-does-llama-board-use-corrected-in-phase-3)
- **Course notebook bug:** `classification_report(..., target_names=LABEL_TOKENS)` is called
  without `labels=`, so sklearn sorts the classes alphabetically and the names are attached to
  the wrong rows (also in the per-class F1 chart). Fixed in this project by passing `labels=`.

## Results

_To be filled in after Phase 6._

## Credits

- Course material and dataset: [The Gen Academy, 5A Fine-Tune a Support Ticket Router](https://github.com/The-Gen-Academy/5A-Fine-Tune-a-Support-Ticket-Router)
- Fine-tuning framework: [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)
- Base model: [Qwen/Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base)
