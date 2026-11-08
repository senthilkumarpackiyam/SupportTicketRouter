"""Phase 6: Evaluate the fine-tuned router on the held-out validation split and compare it with the
baseline (course: "Evaluate the Fine-tuned Model", "Classification report", "Confusion matrix",
"Baseline vs Fine-tuned Comparison").

1. Loads the MERGED model saved in Phase 5 (fresh process, so this also proves the standalone model works)
2. classify() on all 117 validation tickets: label, confidence, raw generation, latency
3. The course's three views: classification report (with the label-order bug fixed), confusion
   matrix, baseline-vs-fine-tuned chart
4. Extras that make the numbers interpretable:
   - context baselines: majority class, TF-IDF nearest neighbour, TF-IDF + logistic regression
   - baseline sensitivity: the A–G baseline re-run with Qwen's ORIGINAL chat template and both
     letter tokenisations ("A" / " A"), to make sure the baseline wasn't handicapped
   - confidence thresholds: auto-route only confident tickets, send the rest to a human
   - latency on the Apple GPU (and a small CPU sample)
   - error analysis: every misrouted ticket

Usage:   python scripts/phase6_evaluate.py [--cpu-latency 10]
Outputs: results/phase6_*.{csv,json,txt,png}
"""

import argparse
import gc
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm.auto import tqdm

import viz_style as vs
from router_common import BASE_MODEL, PROCESSED_DIR, RESULTS_DIR
from router_inference import CHOICES, LABEL_TOKENS, BaselineRouter, Router, get_device, load_merged, sync

THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]


def macro_f1(y_t, y_p) -> float:
    return f1_score(y_t, y_p, labels=LABEL_TOKENS, average="macro", zero_division=0)


def per_class_f1(y_t, y_p) -> dict:
    # FIXED version of the course helper: labels= pins the row order to LABEL_TOKENS
    rep = classification_report(y_t, y_p, labels=LABEL_TOKENS, target_names=LABEL_TOKENS, output_dict=True,
                                zero_division=0)
    return {l: rep[l]["f1-score"] for l in LABEL_TOKENS}


# ── fine-tuned evaluation ─────────────────────────────────────────────────────────────────
def evaluate_finetuned(df_val: pd.DataFrame, device, dtype) -> pd.DataFrame:
    model, tok = load_merged(device, dtype)
    router = Router(model, tok)
    for _ in range(2):
        router("warm-up ticket")
    rows = []
    for text in tqdm(df_val["text"], desc="Fine-tuned inference"):
        p = router(text)
        rows.append({"pred": p.label, "confidence": round(p.confidence, 4), "raw_generation": p.raw,
                     "latency_ms": round(p.latency_ms, 1)})
    del model, router
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return pd.DataFrame(rows)


# ── baseline sensitivity (original Qwen template, both letter tokenisations) ──────────────
def baseline_variants(df_val: pd.DataFrame, device, dtype) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from router_inference import ADAPTER_DIR
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype, device_map=device).eval()
    out = {}
    for tmpl_name, tok_src in (("default (course)", ADAPTER_DIR), ("qwen original", BASE_MODEL)):
        tok = AutoTokenizer.from_pretrained(str(tok_src))
        br = BaselineRouter(model, tok)
        plain = [tok.encode(ch, add_special_tokens=False)[0] for ch in CHOICES]
        spaced = [tok.encode(f" {ch}", add_special_tokens=False)[0] for ch in CHOICES]
        preds = {"plain": [], "spaced": []}
        for text in tqdm(df_val["text"], desc=f"Baseline variants [{tmpl_name}]"):
            ids = tok(br.prompt(text), return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**ids).logits[0, -1].float()
            for k, idx in (("plain", plain), ("spaced", spaced)):
                preds[k].append(LABEL_TOKENS[int(torch.argmax(logits[idx]))])
        for k, p in preds.items():
            out[f"{tmpl_name} | letters {k!s}"] = {
                "accuracy": round(accuracy_score(df_val["label"], p), 4),
                "macro_f1": round(macro_f1(df_val["label"], p), 4),
                "share_predicted_A": round(float(np.mean([x == LABEL_TOKENS[0] for x in p])), 3),
            }
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return out


# ── classical context baselines ───────────────────────────────────────────────────────────
def classical_baselines(df_train: pd.DataFrame, df_val: pd.DataFrame) -> dict:
    y_t = df_val["label"]
    majority = df_train["label"].value_counts().idxmax()
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    x_tr, x_va = vec.fit_transform(df_train["text"]), vec.transform(df_val["text"])
    nn = df_train["label"].to_numpy()[(x_va @ x_tr.T).toarray().argmax(axis=1)]
    lr = LogisticRegression(max_iter=5000).fit(x_tr, df_train["label"]).predict(x_va)  # defaults, no tuning on val
    res = {}
    for name, pred in (("majority class", [majority] * len(df_val)), ("TF-IDF 1-nearest-neighbour", nn),
                       ("TF-IDF + logistic regression", lr)):
        res[name] = {"accuracy": round(accuracy_score(y_t, pred), 4), "macro_f1": round(macro_f1(y_t, pred), 4)}
    return res


def error_neighbour_check(df_train: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """For each misrouted ticket: does the model's answer equal the label of the most similar
    training ticket (TF-IDF cosine)? If yes, the "error" mirrors an inconsistency in the labels."""
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    x_tr = vec.fit_transform(df_train["text"])
    err = pred[pred["pred"] != pred["label"]]
    sims = (vec.transform(err["text"]) @ x_tr.T).toarray()
    nn = sims.argmax(axis=1)
    nn_label = df_train["label"].to_numpy()[nn]
    return {
        "errors": len(err),
        "model_answer_equals_nearest_train_label": int((nn_label == err["pred"].to_numpy()).sum()),
        "nearest_train_label_equals_true_label": int((nn_label == err["label"].to_numpy()).sum()),
        "median_similarity_to_nearest_train": round(float(np.median(sims.max(axis=1))), 3),
    }


def cpu_latency(df_val: pd.DataFrame, n: int) -> dict:
    model, tok = load_merged("cpu", torch.float32)
    router = Router(model, tok)
    router("warm-up ticket")
    lat = [router(t).latency_ms for t in df_val["text"].head(n)]
    del model, router
    gc.collect()
    return {"tickets": n, "dtype": "float32", "median_ms": round(float(np.median(lat)), 1),
            "p95_ms": round(float(np.percentile(lat, 95)), 1)}


# ── charts ────────────────────────────────────────────────────────────────────────────────
def plot_confusion(cm: np.ndarray, path) -> None:
    vs.apply()
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("seq_blue", vs.SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(8.6, 7))
    ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
    n = len(LABEL_TOKENS)
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                dark = cm_norm[i, j] > 0.55
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=11,
                        fontweight="bold" if i == j else "normal", color=vs.SURFACE if dark else vs.INK)
    ax.set_xticks(range(n), LABEL_TOKENS, rotation=30, ha="right")
    ax.set_yticks(range(n), [f"{l}  (n={cm[i].sum()})" for i, l in enumerate(LABEL_TOKENS)])
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color=vs.SURFACE, linewidth=2)  # 2px surface gap between cells
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Fine-tuned router: confusion matrix on 117 held-out tickets\n"
                 "numbers = ticket counts · colour = share of the true class (row recall)", fontsize=11.5)
    fig.savefig(path)
    plt.close(fig)


def plot_comparison(base_f1, ft_f1, base_acc, ft_acc, path) -> None:
    vs.apply()
    labels = LABEL_TOKENS + ["Overall\naccuracy"]
    b = [base_f1[l] for l in LABEL_TOKENS] + [base_acc]
    f = [ft_f1[l] for l in LABEL_TOKENS] + [ft_acc]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    bb = ax.bar(x - w / 2, b, w, color=vs.SERIES_2, edgecolor=vs.SURFACE, linewidth=2,
                label="Base model, zero-shot A–G (no fine-tuning)")
    fb = ax.bar(x + w / 2, f, w, color=vs.SERIES_1, edgecolor=vs.SURFACE, linewidth=2,
                label="Fine-tuned (LLaMA Board LoRA, merged)")
    for bars in (bb, fb):
        for r in bars:
            ax.annotate(f"{r.get_height():.2f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=vs.INK_2)
    ax.axvline(len(LABEL_TOKENS) - 0.5, color=vs.AXIS, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_xticks(x, [l.replace(" ", "\n") if "\n" not in l else l for l in labels], fontsize=9)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("F1 score per class  ·  accuracy (right)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncols=2, bbox_to_anchor=(0, 1.02))
    ax.set_title(f"Baseline vs fine-tuned on 117 held-out tickets: accuracy {base_acc:.1%} → {ft_acc:.1%}", pad=28)
    fig.savefig(path)
    plt.close(fig)


def plot_thresholds(tbl: pd.DataFrame, path) -> None:
    vs.apply()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(tbl["threshold"], 100 * tbl["accuracy_auto_routed"], color=vs.SERIES_1, marker="o", markersize=6,
            markeredgecolor=vs.SURFACE, markeredgewidth=1.5, label="Accuracy on auto-routed tickets")
    ax.plot(tbl["threshold"], 100 * tbl["share_auto_routed"], color=vs.SERIES_2, marker="o", markersize=6,
            markeredgecolor=vs.SURFACE, markeredgewidth=1.5, label="Share of tickets auto-routed")
    for col, dy in (("accuracy_auto_routed", 8), ("share_auto_routed", -14)):
        for t in (0.0, 0.85):
            r = tbl[tbl["threshold"] == t].iloc[0]
            ax.annotate(f"{100 * r[col]:.0f}%", (t, 100 * r[col]), xytext=(0, dy), textcoords="offset points",
                        ha="center", fontsize=9, color=vs.INK, fontweight="bold")
    ax.set_xlabel("Confidence threshold (below it, the ticket goes to a human)")
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 108)
    ax.legend(loc="lower left")
    ax.set_title("Route automatically only when the model is confident")
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-latency", type=int, default=10, help="tickets for the CPU latency sample (0 = skip)")
    args = ap.parse_args()
    device, dtype = get_device()
    df_train = pd.read_csv(PROCESSED_DIR / "train_split.csv")
    df_val = pd.read_csv(PROCESSED_DIR / "val_split.csv")
    base_pred = pd.read_csv(RESULTS_DIR / "phase5_baseline_predictions.csv")
    assert (base_pred["text"] == df_val["text"]).all()

    ft = evaluate_finetuned(df_val, device, dtype)
    pred = pd.concat([df_val, ft, base_pred[["pred_base"]]], axis=1)
    pred.to_csv(RESULTS_DIR / "phase6_predictions.csv", index=False)
    y_true, y_pred, y_base = pred["label"].tolist(), pred["pred"].tolist(), pred["pred_base"].tolist()
    print(f"Evaluated {len(y_pred)} samples.")

    # 1. classification report: fixed vs course version
    fixed = classification_report(y_true, y_pred, labels=LABEL_TOKENS, target_names=LABEL_TOKENS, digits=3,
                                  zero_division=0)
    course = classification_report(y_true, y_pred, target_names=LABEL_TOKENS, digits=3, zero_division=0)
    print(fixed)
    (RESULTS_DIR / "phase6_classification_report.txt").write_text(
        "Fine-tuned router, 117 held-out validation tickets\n"
        "classification_report(y_true, y_pred, labels=LABEL_TOKENS, target_names=LABEL_TOKENS)\n\n" + fixed
        + "\n\n---- For reference: the course cell's call WITHOUT labels= (row names are WRONG) ----\n"
        "sklearn sorts the classes alphabetically and attaches target_names in LABEL_TOKENS order:\n"
        + "\n".join(f"  row named {n!r:21} actually holds {a!r}" for n, a in zip(LABEL_TOKENS, sorted(set(y_true))))
        + "\n\n" + course + "\n")

    # 2. confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_TOKENS)
    plot_confusion(cm, RESULTS_DIR / "phase6_confusion_matrix.png")
    recall = {l: cm[i, i] / cm[i].sum() for i, l in enumerate(LABEL_TOKENS)}
    print(f"Active Directory recall : {recall['Active Directory']:.1%}")
    print(f"Fileservice recall      : {recall['Fileservice']:.1%}")

    # 3. baseline vs fine-tuned
    ft_f1, base_f1 = per_class_f1(y_true, y_pred), per_class_f1(y_true, y_base)
    ft_acc, base_acc = accuracy_score(y_true, y_pred), accuracy_score(y_true, y_base)
    plot_comparison(base_f1, ft_f1, base_acc, ft_acc, RESULTS_DIR / "phase6_baseline_vs_finetuned.png")
    print(f"Base accuracy  : {base_acc:.1%}\nFine-tuned acc : {ft_acc:.1%}\nDelta          : +{ft_acc - base_acc:.1%}")

    # 4. confidence thresholds
    conf = pred["confidence"].to_numpy()
    correct = (pred["pred"] == pred["label"]).to_numpy()
    tbl = pd.DataFrame([{"threshold": t, "share_auto_routed": float((conf >= t).mean()),
                         "accuracy_auto_routed": float(correct[conf >= t].mean()) if (conf >= t).any() else np.nan,
                         "tickets_to_human": int((conf < t).sum())} for t in THRESHOLDS])
    plot_thresholds(tbl, RESULTS_DIR / "phase6_confidence_thresholds.png")
    print(tbl.round(3).to_string(index=False))

    # 5. errors
    errors = pred[~correct].sort_values("confidence", ascending=False)
    errors[["label", "pred", "confidence", "pred_base", "text"]].to_csv(RESULTS_DIR / "phase6_errors.csv", index=False)
    pairs = errors.groupby(["label", "pred"]).size().sort_values(ascending=False)

    # 6. context: classical models + baseline sensitivity + latency
    classical = classical_baselines(df_train, df_val)
    err_nn = error_neighbour_check(df_train, pred)
    variants = baseline_variants(df_val, device, dtype)
    lat = pred["latency_ms"]
    latency = {"device": device, "dtype": str(dtype), "median_ms": round(float(lat.median()), 1),
               "p95_ms": round(float(lat.quantile(0.95)), 1), "mean_ms": round(float(lat.mean()), 1)}
    if args.cpu_latency:
        try:
            latency["cpu_sample"] = cpu_latency(df_val, args.cpu_latency)
        except Exception as e:  # noqa: BLE001  (memory pressure on a 16 GB laptop)
            latency["cpu_sample"] = {"error": repr(e)}

    rep = classification_report(y_true, y_pred, labels=LABEL_TOKENS, target_names=LABEL_TOKENS, output_dict=True,
                                zero_division=0)
    metrics = {
        "n_val": len(pred),
        "finetuned": {"accuracy": round(ft_acc, 4), "macro_f1": round(macro_f1(y_true, y_pred), 4),
                      "weighted_f1": round(rep["weighted avg"]["f1-score"], 4),
                      "per_class": {l: {k: round(rep[l][k], 4) for k in ("precision", "recall", "f1-score", "support")}
                                    for l in LABEL_TOKENS},
                      "clean_label_outputs": int((pred["raw_generation"] == pred["pred"]).sum())},
        "baseline": {"accuracy": round(base_acc, 4), "macro_f1": round(macro_f1(y_true, y_base), 4),
                     "per_class_f1": {l: round(v, 4) for l, v in base_f1.items()}},
        "delta_accuracy_pct_points": round(100 * (ft_acc - base_acc), 1),
        "active_directory_recall": round(recall["Active Directory"], 4),
        "fileservice_recall": round(recall["Fileservice"], 4),
        "confusion_matrix": {"labels": LABEL_TOKENS, "counts": cm.tolist()},
        "errors_vs_nearest_training_ticket": err_nn,
        "top_confusions": [{"true": t, "pred": p, "count": int(c)} for (t, p), c in pairs.head(8).items()],
        "confidence_thresholds": tbl.round(4).to_dict(orient="records"),
        "context_baselines": classical,
        "baseline_variants": variants,
        "latency": latency,
        "course_examples": [{"i": i, "true": pred.loc[i, "label"], "pred": pred.loc[i, "pred"],
                             "confidence": float(pred.loc[i, "confidence"])} for i in (0, 10, 20, 30)],
    }
    (RESULTS_DIR / "phase6_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: metrics[k] for k in ("errors_vs_nearest_training_ticket", "top_confusions", "context_baselines", "baseline_variants", "latency",
                                              "course_examples")}, indent=2))
    sync(device)


if __name__ == "__main__":
    main()
