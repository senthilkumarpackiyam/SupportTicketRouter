"""Phase 5: Baseline, merge the LoRA adapter, smoke test (course: "Merge Adapter Weights, Measure
Baseline & Prepare for Inference" + "Define classify() and Smoke Test the Merged Model").

Order matters, exactly as in the course:
  1. load the base model + the adapter's tokenizer
  2. zero-shot BASELINE on the held-out validation split (A–G letter choice): the control,
     measured now while the untouched base weights are in memory
  3. attach the LoRA adapter, record a reference output, merge_and_unload(), check the merged
     model gives the same output, save the standalone model (gitignored, 3.4 GB)
  4. smoke test classify() on the course's 5 tickets (+ 2 extra for the remaining classes)

Usage:   python scripts/phase5_merge_and_smoke.py
Outputs: results/phase5_baseline_predictions.csv, results/phase5_smoke_test.json,
         results/phase5_merge_and_baseline.json, models.nosync/qwen3-1.7b-support-router-merged/
"""

import gc
import json
import time

import pandas as pd
import safetensors.torch as st
import torch
from peft import PeftConfig, get_peft_model
from peft.utils import set_peft_model_state_dict
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm.auto import tqdm

from router_common import PROCESSED_DIR, RESULTS_DIR
from router_inference import (
    ADAPTER_DIR, CHOICES, EXTRA_SMOKE_TICKETS, LABEL_TOKENS, MERGED_DIR, SMOKE_TICKETS,
    BaselineRouter, Router, get_device, load_base, sync,
)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    device, dtype = get_device()
    print(f"Device: {device}  dtype: {dtype}")
    summary = {"device": device, "dtype": str(dtype)}

    # ── 1. Base model + tokenizer (from the adapter folder) ───────────────────────────────
    t0 = time.time()
    base_model, tok = load_base(device, dtype)
    summary["load_base_s"] = round(time.time() - t0, 1)
    print(f"Base model loaded in {summary['load_base_s']} s")

    # ── 2. Baseline on the validation split ───────────────────────────────────────────────
    baseline = BaselineRouter(base_model, tok)
    print("Baseline prompt example:\n" + baseline.prompt("<ticket text>"))
    df_val = pd.read_csv(PROCESSED_DIR / "val_split.csv")
    y_true = df_val["label"].tolist()
    rows = []
    t0 = time.time()
    for text in tqdm(df_val["text"], desc="Baseline inference"):
        label, probs = baseline(text)
        rows.append({"pred_base": label, **{f"p_{ch}": round(p, 4) for ch, p in zip(CHOICES, probs)}})
    summary["baseline_seconds"] = round(time.time() - t0, 1)
    base_df = pd.concat([df_val, pd.DataFrame(rows)], axis=1)
    base_df.to_csv(RESULTS_DIR / "phase5_baseline_predictions.csv", index=False)
    y_pred_base = base_df["pred_base"].tolist()

    print("\n=== Baseline (no fine-tuning) ===")
    # labels= keeps the row names aligned with LABEL_TOKENS (the course cell omits it; see README)
    print(classification_report(y_true, y_pred_base, labels=LABEL_TOKENS, target_names=LABEL_TOKENS,
                                digits=3, zero_division=0))
    summary["baseline"] = {
        "accuracy": round(accuracy_score(y_true, y_pred_base), 4),
        "macro_f1": round(f1_score(y_true, y_pred_base, labels=LABEL_TOKENS, average="macro", zero_division=0), 4),
        "predicted_label_counts": pd.Series(y_pred_base).value_counts().to_dict(),
    }
    # the base model on the smoke tickets too, for the before/after table
    base_smoke = {t: baseline(t)[0] for t, _ in SMOKE_TICKETS + EXTRA_SMOKE_TICKETS}

    # ── 3. Attach LoRA (course method), reference output, merge, save ─────────────────────
    print("Attaching LoRA adapter...")
    peft_cfg = PeftConfig.from_pretrained(str(ADAPTER_DIR))
    peft_model = get_peft_model(base_model, peft_cfg)
    adapter_weights = st.load_file(str(ADAPTER_DIR / "adapter_model.safetensors"), device=device)
    load_result = set_peft_model_state_dict(peft_model, adapter_weights)
    n_lora_in_model = sum(1 for k, _ in peft_model.named_parameters() if "lora_" in k)
    summary["adapter_load"] = {
        "tensors_in_file": len(adapter_weights),
        "lora_tensors_in_model": n_lora_in_model,
        "unexpected_keys": len(load_result.unexpected_keys),
    }
    assert len(load_result.unexpected_keys) == 0 and len(adapter_weights) == n_lora_in_model, summary["adapter_load"]
    peft_model.eval()

    probe = Router(peft_model, tok).prompt(SMOKE_TICKETS[2][0])
    probe_ids = tok(probe, return_tensors="pt").to(device)
    with torch.no_grad():
        ref_logits = peft_model(**probe_ids).logits[0, -1].float().cpu()

    print("Merging ...")
    t0 = time.time()
    model = peft_model.merge_and_unload()
    model.eval()
    with torch.no_grad():
        merged_logits = model(**probe_ids).logits[0, -1].float().cpu()
    summary["merge"] = {
        "seconds": round(time.time() - t0, 1),
        "max_abs_logit_diff_adapter_vs_merged": round(float((ref_logits - merged_logits).abs().max()), 4),
        "same_top1_token": bool(ref_logits.argmax() == merged_logits.argmax()),
    }
    print("merge check:", summary["merge"])

    t0 = time.time()
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MERGED_DIR)
    tok.save_pretrained(MERGED_DIR)
    stale = MERGED_DIR / "adapter_config.json"
    if stale.exists():
        stale.unlink()
    size_gb = sum(f.stat().st_size for f in MERGED_DIR.iterdir()) / 1e9
    summary["merged_model"] = {"path": str(MERGED_DIR.relative_to(MERGED_DIR.parents[1])), "size_gb": round(size_gb, 2),
                               "save_seconds": round(time.time() - t0, 1)}
    print(f"Saved -> {MERGED_DIR} ({size_gb:.2f} GB)")

    del peft_model, base_model, adapter_weights
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    # ── 4. Smoke test ─────────────────────────────────────────────────────────────────────
    router = Router(model, tok)
    router("warm-up ticket")  # first call compiles MPS kernels; keep it out of the latency numbers
    sync(device)
    smoke = []
    print(f"\n{'Ticket (truncated)':<60} {'Expected':<20} {'Predicted':<20} {'Conf':>6}")
    print("-" * 112)
    for group, tickets in (("course", SMOKE_TICKETS), ("extra", EXTRA_SMOKE_TICKETS)):
        for ticket, expected in tickets:
            p = router(ticket)
            ok = p.label == expected
            print(f"{ticket[:58]:<60} {expected:<20} {p.label:<20} {p.confidence:>6.1%}  {'OK' if ok else '--'}")
            smoke.append({"group": group, "ticket": ticket, "expected": expected, "predicted": p.label,
                          "raw_generation": p.raw, "confidence": round(p.confidence, 4), "ok": ok,
                          "latency_ms": round(p.latency_ms, 1), "baseline_predicted": base_smoke[ticket]})
    course_ok = sum(s["ok"] for s in smoke if s["group"] == "course")
    print(f"\nCourse smoke test: {course_ok}/{len(SMOKE_TICKETS)} correct")
    summary["smoke_course_correct"] = f"{course_ok}/{len(SMOKE_TICKETS)}"

    (RESULTS_DIR / "phase5_smoke_test.json").write_text(json.dumps(smoke, indent=2, ensure_ascii=False) + "\n")
    (RESULTS_DIR / "phase5_merge_and_baseline.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
