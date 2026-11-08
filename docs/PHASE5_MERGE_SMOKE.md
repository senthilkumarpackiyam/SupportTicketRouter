# Phase 5: Baseline, merge and smoke test

Course equivalent: **"Merge Adapter Weights, Measure Baseline & Prepare for Inference"** and
**"Define `classify()` and Smoke Test the Merged Model"**.

Script: [`scripts/phase5_merge_and_smoke.py`](../scripts/phase5_merge_and_smoke.py). Inference
helpers ported 1:1 from the course cells: [`scripts/router_inference.py`](../scripts/router_inference.py).
Log: [`results/phase5_run.log`](../results/phase5_run.log).

Steps, in the course's order:

1. Load `Qwen/Qwen3-1.7B-Base` (fp16, Apple GPU) and the tokenizer **from the adapter folder**.
2. **Baseline first**, while the untouched base weights are in memory: zero-shot A–G letter
   choice on all 117 held-out validation tickets.
3. Attach the LoRA adapter, `merge_and_unload()`, save one standalone model.
4. Smoke-test `classify()` on the course's five tickets.

## 1. Baseline: the untrained model can't route

The prompt (built by the adapter tokenizer's chat template):

```
System: You are an IT helpdesk ticket routing assistant. Classify the ticket by responding with ONLY a single letter — nothing else:
A) Support general
B) Fileservice
…
G) Computer-Services<|endoftext|>
Human: Support Ticket: <ticket text><|endoftext|>
Assistant:
```

The seven letter tokens are compared at the first generated position, so the answer is always a
valid label.

| | Baseline |
|---|---|
| Accuracy | **24.8 %** (majority-class guess = 25.6 %) |
| Macro F1 | 0.078 |
| Predicted labels | `Support general` **113 / 117**, O365 2, EOL 1, Fileservice 1 |
| Mean letter probability | A 0.40 · B 0.14 · C 0.09 · D 0.06 · E 0.12 · F 0.09 · G 0.10 |

The base model mostly picks the **first option**, a position bias, not a routing decision. It
only gets EOL right when the ticket literally says "retire/decommission". The full report is in
the log, and per-ticket letter probabilities are in
[`results/phase5_baseline_predictions.csv`](../results/phase5_baseline_predictions.csv).

## 2. Merge

| Check | Result |
|---|---|
| LoRA tensors in `adapter_model.safetensors` / in the model | 392 / 392 (196 layers × A,B), **0 unexpected keys** |
| `merge_and_unload()` | 0.6 s |
| Adapter vs merged model, last-position logits on a test prompt | max abs difference 0.047 (fp16 rounding), **same top token** |
| Saved | `models.nosync/qwen3-1.7b-support-router-merged/` (**3.45 GB**: weights, config, tokenizer + chat template) |

The merged model is **not pushed to GitHub**. At 3.45 GB it's far over GitHub's 100 MB file
limit, and anyone can rebuild it in about a minute from the 35 MB adapter in this repo by running
this script. The `.nosync` suffix keeps it out of iCloud Drive, because this repo lives in a synced
`~/Documents` folder (see Phase 3).

## 3. Smoke test

`classify()` = training system prompt + `Support Ticket: …`, greedy decoding of ≤ 10 tokens,
prefix match against the labels. Confidence is the softmax over the seven labels' first tokens.

| Ticket | Expected | Fine-tuned | Conf. | Baseline |
|---|---|---|---|---|
| Please create a new user account for the new employee starting Monday. | Active Directory | Support general ❌ | 80.8 % | Support general |
| I cannot access the shared network folder — I keep getting Access Denied. | Fileservice | **Fileservice** ✅ | 62.3 % | Support general |
| Outlook keeps asking me to re-enter my password and Teams won't sync. | O365 | **O365** ✅ | 77.7 % | Support general |
| We're ready to retire [SERVER] — please remove it from monitoring. | EOL | **EOL** ✅ | 96.0 % | EOL |
| Adobe Acrobat fails to install, error code 1603. | Software | **Software** ✅ | 92.2 % | Software |
| *extra:* printer on the 2nd floor needs a new driver installed … | Computer-Services | **Computer-Services** ✅ | 46.1 % | Support general |
| *extra:* create a Light (external) user account for a contractor … | Support general | **Support general** ✅ | 59.4 % | Support general |

**Course smoke test: 4 / 5.** The two extra tickets cover the classes the course's set skips,
and both are correct. The model always answers with a clean label (`raw_generation` in
[`results/phase5_smoke_test.json`](../results/phase5_smoke_test.json)), which shows the chat template, the
checkpoint and the merge are all correct. Latency on the M2 GPU is about 200–350 ms per ticket.

### Why the Active Directory smoke ticket fails

The smoke test's job is to catch a broken pipeline, and this miss isn't one. It's a
**disagreement between the course's expectation and the dataset's labels**:

| Training tickets that ask for a *new user account* | Count |
|---|---|
| labelled **Fileservice** | 9 |
| labelled **Support general** | 3 |
| labelled **Active Directory** | 2 |

Requests for *light/basic/external* accounts are also labelled `Support general`. The model
learned the convention in the data, so `Support general` with 81 % confidence is a faithful
reading of it. Fixing this means **agreeing on a labelling rule and relabelling those tickets**.
More training wouldn't help. The course flags Active Directory recall as the production-critical
metric, and Phase 6 measures it on the whole validation split.
