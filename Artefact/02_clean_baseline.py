"""
Step 3 - run the victim classifier on all 500 frozen Jigsaw rows.
This gives the clean baseline and defines the eligible attack set.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")  # windows terminal defaults to cp1252

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

VICTIM = "martin-ha/toxic-comment-model"
MAX_LEN = 512   # distilbert's hard limit - longer comments get truncated
BATCH = 16      # small batches so 4gb vram is enough

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

tok = AutoTokenizer.from_pretrained(VICTIM)
model = AutoModelForSequenceClassification.from_pretrained(VICTIM).to(device)
model.eval()  # inference only, never training this


def predict(texts):
    """Return the toxic probability for a list of strings."""
    probs = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        # index 1 = toxic, confirmed from config.id2label in step 1
        p = torch.softmax(logits, dim=-1)[:, 1]
        probs.extend(p.cpu().tolist())
        print(f"  {min(i + BATCH, len(texts))}/{len(texts)}", end="\r")
    print()
    return probs


def run(name):
    """Score one frozen dataset and save the results."""
    df = pd.read_csv(f"data/jigsaw_{name}_250.csv")
    print(f"scoring {name}...")
    df["clean_toxic_prob"] = predict(df["text"].tolist())
    # 0.5 is the standard binary decision threshold
    df["clean_prediction"] = (df["clean_toxic_prob"] >= 0.5).astype(int)
    df.to_csv(f"data/clean_baseline_{name}.csv", index=False)
    return df


if __name__ == "__main__":
    toxic_df = run("toxic")
    benign_df = run("benign")

    # eligible set = toxic rows the clean classifier actually catches.
    # an already-missed row cannot count as a successful attack later.
    eligible = int(toxic_df["clean_prediction"].sum())
    false_pos = int(benign_df["clean_prediction"].sum())

    print("\n--- clean baseline ---")
    print(f"toxic detected as toxic  : {eligible}/250   (recall {eligible/250:.3f})")
    print(f"benign flagged as toxic  : {false_pos}/250  (fpr {false_pos/250:.3f})")
    print(f"mean toxic prob, toxic rows : {toxic_df['clean_toxic_prob'].mean():.4f}")
    print(f"mean toxic prob, benign rows: {benign_df['clean_toxic_prob'].mean():.4f}")
    print(f"\nELIGIBLE ATTACK SET = {eligible} rows")