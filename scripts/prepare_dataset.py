"""Phase 2: Prepare the support-ticket dataset for LLaMA Factory.

Mirrors the course notebook cell "Prepare Support Ticket Dataset", running locally instead of
using the Colab upload widget:

  1. load support_tickets.csv, rename category_truth -> label, keep the 7 known labels,
     shuffle (random_state=42)
  2. stratified 80/20 train/validation split (random_state=42)
  3. convert the TRAIN split to ShareGPT JSON (system / user / assistant messages)
  4. write TRAIN.json into LLaMA-Factory/data and register it as `support_tickets`
     in LLaMA-Factory/data/dataset_info.json (so LLaMA Board lists it)
  5. save the held-out validation split for evaluation (never used in training)

Usage:  python scripts/prepare_dataset.py
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from router_common import (
    DATASET_NAME, LABEL2ID, LLAMA_FACTORY_DIR, PROCESSED_DIR, RAW_CSV, SEED, SYSTEM_PROMPT, VAL_SIZE, format_user,
)

DATASET_INFO_ENTRY = {
    "file_name": "TRAIN.json",
    "formatting": "sharegpt",
    "columns": {"messages": "messages"},
    "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
        "system_tag": "system",
    },
}


def to_sharegpt(df: pd.DataFrame) -> list[dict]:
    return [
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format_user(row.text)},
                {"role": "assistant", "content": row.label},
            ]
        }
        for row in df.itertuples(index=False)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=RAW_CSV)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--llama-factory-dir", type=Path, default=LLAMA_FACTORY_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load + filter + shuffle
    df = pd.read_csv(args.csv).rename(columns={"category_truth": "label"})
    n_raw = len(df)
    df = df[df["label"].isin(LABEL2ID)].sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"Loaded {n_raw:,} rows, kept {len(df):,} with a known label")

    # 2. Stratified split
    df_train, df_val = train_test_split(df, test_size=VAL_SIZE, stratify=df["label"], random_state=SEED)
    df_train, df_val = df_train.reset_index(drop=True), df_val.reset_index(drop=True)
    print(f"Train: {len(df_train):,} rows | Val (held-out): {len(df_val):,} rows")

    # 3. ShareGPT JSON for the training split only
    records = to_sharegpt(df_train)
    train_json = args.out_dir / "TRAIN.json"
    train_json.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(records):,} ShareGPT records -> {train_json}")

    # 4. Copy into LLaMA-Factory/data and register in dataset_info.json
    lf_data = args.llama_factory_dir / "data"
    shutil.copyfile(train_json, lf_data / "TRAIN.json")
    info_path = lf_data / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info[DATASET_NAME] = DATASET_INFO_ENTRY
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out_dir / "dataset_info_entry.json").write_text(
        json.dumps({DATASET_NAME: DATASET_INFO_ENTRY}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Registered '{DATASET_NAME}' in {info_path}")

    # 5. Save both splits as CSV (val is the honest, never-trained-on evaluation set)
    df_train.to_csv(args.out_dir / "train_split.csv", index=False)
    df_val.to_csv(args.out_dir / "val_split.csv", index=False)
    print(f"Saved splits -> {args.out_dir}/train_split.csv, val_split.csv")


if __name__ == "__main__":
    main()
