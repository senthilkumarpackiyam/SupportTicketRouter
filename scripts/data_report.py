"""Phase 2: Data report and pre-training checks for the support_tickets dataset.

Checks
  1. class balance per split (is the stratified split really stratified?)
  2. leakage: exact overlap and near-duplicates between train and validation (TF-IDF cosine)
  3. the dataset loads through LLaMA-Factory's own SFT pipeline, compared for three chat templates
     (`default` = what LLaMA Board leaves selected for a Base model, `qwen3`, `qwen3_nothink`):
     exact token sequences, which tokens carry loss, sequence lengths (to choose cutoff_len)
  4. template alignment: the training prompt vs Hugging Face `apply_chat_template` (what the
     course's classify() uses at inference time)

Usage:  python scripts/data_report.py
Outputs: results/phase2_data_report.json, results/phase2_class_distribution.png
"""

import json
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import viz_style as vs
from router_common import (
    BASE_MODEL, DATASET_NAME, LABELS, LLAMA_FACTORY_DIR, PROCESSED_DIR, RESULTS_DIR, SYSTEM_PROMPT, format_user,
)

IGNORE_INDEX = -100


def class_balance(df_train: pd.DataFrame, df_val: pd.DataFrame) -> pd.DataFrame:
    tbl = pd.DataFrame({
        "train": df_train["label"].value_counts(),
        "val": df_val["label"].value_counts(),
    }).reindex(LABELS).fillna(0).astype(int)
    tbl["total"] = tbl["train"] + tbl["val"]
    tbl["share_of_all_%"] = (100 * tbl["total"] / tbl["total"].sum()).round(1)
    tbl["train_%"] = (100 * tbl["train"] / tbl["train"].sum()).round(1)
    tbl["val_%"] = (100 * tbl["val"] / tbl["val"].sum()).round(1)
    return tbl.sort_values("total", ascending=False)


def leakage(df_train: pd.DataFrame, df_val: pd.DataFrame) -> dict:
    norm = lambda s: s.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()  # noqa: E731
    exact = len(set(norm(df_train["text"])) & set(norm(df_val["text"])))

    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    x_train = vec.fit_transform(df_train["text"])
    x_val = vec.transform(df_val["text"])
    sims = linear_kernel(x_val, x_train)
    nn_idx = sims.argmax(axis=1)
    nn_sim = sims.max(axis=1)
    nn_same_label = (df_train["label"].to_numpy()[nn_idx] == df_val["label"].to_numpy())

    top = np.argsort(-nn_sim)[:3]
    examples = [
        {
            "similarity": round(float(nn_sim[i]), 3),
            "val_label": df_val["label"].iloc[i],
            "val_text": df_val["text"].iloc[i],
            "nearest_train_label": df_train["label"].iloc[nn_idx[i]],
            "nearest_train_text": df_train["text"].iloc[nn_idx[i]],
        }
        for i in top
    ]
    return {
        "exact_text_overlap": exact,
        "nearest_train_similarity": {
            "median": round(float(np.median(nn_sim)), 3),
            "p90": round(float(np.percentile(nn_sim, 90)), 3),
            "max": round(float(nn_sim.max()), 3),
            "val_tickets_ge_0.8": int((nn_sim >= 0.8).sum()),
            "val_tickets_ge_0.6": int((nn_sim >= 0.6).sum()),
        },
        "nearest_neighbour_label_agreement": round(float(nn_same_label.mean()), 3),
        "most_similar_pairs": examples,
    }


def llama_factory_pipeline_check(template_name: str, enable_thinking: bool = True) -> dict:
    """Tokenise TRAIN.json exactly the way a LLaMA Board training run with this template will."""
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer

    with tempfile.TemporaryDirectory() as tmp:
        model_args, data_args, training_args, _, _ = get_train_args(dict(
            stage="sft", do_train=True, model_name_or_path=BASE_MODEL, finetuning_type="lora",
            dataset=DATASET_NAME, dataset_dir=str(LLAMA_FACTORY_DIR / "data"),
            template=template_name,
            enable_thinking=enable_thinking,  # LLaMA Board's checkbox defaults to True
            cutoff_len=2048,           # LLaMA Board's default
            output_dir=tmp, overwrite_cache=True, report_to="none",
        ))
        tokenizer_module = load_tokenizer(model_args)
        tok = tokenizer_module["tokenizer"]
        template = get_template_and_fix_tokenizer(tok, data_args)
        ds = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)["train_dataset"]

    lengths = np.array([len(x) for x in ds["input_ids"]])
    loss_tokens = np.array([sum(l != IGNORE_INDEX for l in x) for x in ds["labels"]])

    # Split example 0 into the masked prompt part and the trained response part.
    ids, labels = ds[0]["input_ids"], ds[0]["labels"]
    first_loss = next(i for i, l in enumerate(labels) if l != IGNORE_INDEX)
    lf_prompt = tok.decode(ids[:first_loss], skip_special_tokens=False)
    lf_response = tok.decode(ids[first_loss:], skip_special_tokens=False)

    # What Hugging Face's chat template produces at inference time for the same ticket.
    train0 = json.loads((PROCESSED_DIR / "TRAIN.json").read_text())[0]["messages"]
    msgs = train0[:2]
    hf_default = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    hf_nothink = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    think_ids = tok.convert_tokens_to_ids(["<think>", "</think>"])
    # The model's ORIGINAL chat template (fresh tokenizer, untouched by LLaMA-Factory).
    from transformers import AutoTokenizer
    orig_prompt = AutoTokenizer.from_pretrained(BASE_MODEL).apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )

    return {
        "template": template_name, "enable_thinking": enable_thinking,
        "num_examples": len(ds),
        "sequence_tokens": {"min": int(lengths.min()), "median": int(np.median(lengths)),
                            "p95": int(np.percentile(lengths, 95)), "max": int(lengths.max())},
        "loss_tokens_per_example": {"min": int(loss_tokens.min()), "max": int(loss_tokens.max())},
        "example0_prompt_masked": lf_prompt,
        "example0_response_trained": lf_response,
        "hf_prompt_default": hf_default,
        "hf_prompt_enable_thinking_false": hf_nothink,
        # LLaMA-Factory may overwrite tokenizer.chat_template with the training template
        # (replace_jinja_template); that tokenizer is saved next to the adapter.
        "tokenizer_chat_template_replaced_by_llamafactory": bool(template.replace_jinja_template),
        "check_lf_prompt_equals_original_qwen_template": lf_prompt == orig_prompt,
        "check_lf_prompt_equals_hf_default": lf_prompt == hf_default,
        "check_lf_full_starts_with_hf_nothink": (lf_prompt + lf_response).startswith(hf_nothink),
        "check_lf_response_is_bare_label": lf_response.strip().removesuffix(tok.eos_token).strip() == train0[2]["content"],
        "think_tokens_are_special": [tok.convert_ids_to_tokens(i) in tok.all_special_tokens for i in think_ids],
        "decoded_response_with_skip_special_tokens": tok.decode(ids[first_loss:], skip_special_tokens=True),
    }


def plot_class_distribution(tbl: pd.DataFrame, path) -> None:
    vs.apply()
    tbl = tbl.iloc[::-1]  # largest at the top
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(tbl))
    ax.barh(y, tbl["train"], color=vs.SERIES_1, edgecolor=vs.SURFACE, linewidth=2, height=0.62, label="Train (80%)")
    ax.barh(y, tbl["val"], left=tbl["train"], color=vs.SERIES_2, edgecolor=vs.SURFACE, linewidth=2,
            height=0.62, label="Validation, held out (20%)")
    for yi, (tr, va, tot) in zip(y, tbl[["train", "val", "total"]].to_numpy()):
        ax.annotate(f"{tot}", (tot, yi), xytext=(5, 0), textcoords="offset points", va="center", ha="left",
                    fontsize=10, color=vs.INK, fontweight="bold")
        ax.annotate(f"{tr} / {va}", (tot, yi), xytext=(30, 0), textcoords="offset points", va="center",
                    ha="left", fontsize=9, color=vs.INK_2)
    ax.set_yticks(y, tbl.index)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, tbl["total"].max() * 1.22)
    ax.set_xlabel("Tickets (labels: total, then train / val)")
    ax.set_title("Support tickets per routing label: stratified 80/20 split")
    ax.legend(loc="lower right", ncols=1)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    df_train = pd.read_csv(PROCESSED_DIR / "train_split.csv")
    df_val = pd.read_csv(PROCESSED_DIR / "val_split.csv")

    tbl = class_balance(df_train, df_val)
    print("Class balance\n", tbl.to_string(), "\n")
    max_drift = float((tbl["train_%"] - tbl["val_%"]).abs().max())

    leak = leakage(df_train, df_val)
    print("Leakage:", json.dumps({k: v for k, v in leak.items() if k != "most_similar_pairs"}, indent=2), "\n")

    # "default"      = what LLaMA Board actually leaves selected for a *Base* model (it only auto-fills
    #                  templates for -Chat/-Instruct/-Thinking models)
    # "qwen3"        = Qwen3 reasoning template (adds an empty <think></think> block to every answer)
    # "qwen3_nothink"= Qwen3 ChatML without thinking (original Qwen format, bare-label answers)
    # Training used LLaMA Board's choice, "default" (see docs/PHASE2_DATASET.md).
    lf = {name: llama_factory_pipeline_check(name) for name in ("default", "qwen3", "qwen3_nothink")}
    for name, res in lf.items():
        print(f"\n===== template={name} =====")
        print(json.dumps({k: v for k, v in res.items() if k.startswith("check") or k == "sequence_tokens"}, indent=2))
        print("--- prompt (no loss) ---\n" + res["example0_prompt_masked"])
        print("--- response (loss) ---\n" + repr(res["example0_response_trained"]))

    RESULTS_DIR.mkdir(exist_ok=True)
    plot_class_distribution(tbl, RESULTS_DIR / "phase2_class_distribution.png")
    report = {
        "rows": {"train": len(df_train), "val": len(df_val)},
        "class_balance": tbl.reset_index(names="label").to_dict(orient="records"),
        "max_class_share_drift_train_vs_val_pct_points": round(max_drift, 2),
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt_format": format_user("<ticket text>"),
        "leakage": leak,
        "llama_factory_pipeline": lf,
    }
    (RESULTS_DIR / "phase2_data_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\nSaved -> {RESULTS_DIR}/phase2_data_report.json, phase2_class_distribution.png")


if __name__ == "__main__":
    main()
