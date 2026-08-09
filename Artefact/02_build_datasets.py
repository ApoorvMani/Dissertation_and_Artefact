"""
Step 2 - build the three frozen datasets and save them as CSV.
Run once. Everything downstream reads these CSVs, so the data never changes.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")  # windows terminal defaults to cp1252

import os
import pandas as pd
from huggingface_hub import hf_hub_download
from datasets import load_dataset

SEED = 42          # fixed everywhere so results are reproducible
MIN_WORDS = 8      # skip very short comments
N_SAMPLE = 250     # 250 toxic + 250 benign
N_WIKITEXT = 10000 # null corpus size

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)


def build_jigsaw():
    """Download Jigsaw train.csv and sample 250 toxic + 250 benign comments."""
    print("downloading jigsaw train.csv...")
    csv_path = hf_hub_download(
        repo_id="thesofakillers/jigsaw-toxic-comment-classification-challenge",
        filename="train.csv",
        repo_type="dataset",
    )
    df = pd.read_csv(csv_path)
    print("  rows:", len(df), "| columns:", list(df.columns))

    # keep only comments with at least MIN_WORDS words
    df["word_count"] = df["comment_text"].str.split().str.len()
    df = df[df["word_count"] >= MIN_WORDS]

    # split by the binary 'toxic' label and sample with a fixed seed
    toxic = df[df["toxic"] == 1].sample(n=N_SAMPLE, random_state=SEED)
    benign = df[df["toxic"] == 0].sample(n=N_SAMPLE, random_state=SEED)

    # keep only the columns we need, and give each row a stable id
    for name, subset in [("toxic", toxic), ("benign", benign)]:
        out = subset[["id", "comment_text", "toxic", "word_count"]].copy()
        out = out.rename(columns={"id": "sample_id", "comment_text": "text"})
        path = f"{OUT_DIR}/jigsaw_{name}_250.csv"
        out.to_csv(path, index=False)
        print(f"  saved {path}  ({len(out)} rows)")


def build_wikitext():
    """Download WikiText-2 and split 10,000 clean lines into calibration / held-out."""
    print("downloading wikitext-2...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    df = pd.DataFrame(ds)

    # raw wikitext has blank lines and ' = Heading = ' lines - drop both
    df["text"] = df["text"].str.strip()
    df = df[df["text"] != ""]
    df = df[~df["text"].str.startswith("=")]

    # same minimum length rule as jigsaw, so the corpora are comparable
    df["word_count"] = df["text"].str.split().str.len()
    df = df[df["word_count"] >= MIN_WORDS]
    print("  usable lines:", len(df))

    # sample 10,000 then cut in half: half sets thresholds, half reports FPR
    sample = df.sample(n=N_WIKITEXT, random_state=SEED).reset_index(drop=True)
    sample["sample_id"] = ["wt_" + str(i) for i in range(len(sample))]

    calib = sample.iloc[:5000]
    heldout = sample.iloc[5000:]

    for name, subset in [("calibration", calib), ("heldout", heldout)]:
        path = f"{OUT_DIR}/wikitext_{name}_5000.csv"
        subset[["sample_id", "text", "word_count"]].to_csv(path, index=False)
        print(f"  saved {path}  ({len(subset)} rows)")


if __name__ == "__main__":
    build_jigsaw()
    build_wikitext()
    print("\nstep 2 done - datasets frozen in ./data")