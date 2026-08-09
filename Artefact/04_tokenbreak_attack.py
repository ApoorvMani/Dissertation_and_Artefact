"""
Step 4 - TokenBreak attack, reimplementing Algorithm 1 (BreakPrompt).
Each word is judged on its own; suspicious words get a single letter
prepended until that word alone looks confidently benign.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")  # windows terminal defaults to cp1252

import json
import string
import time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from attack_schema import build_record

VICTIM = "martin-ha/toxic-comment-model"
MAX_LEN = 512
BATCH = 64
BASE_THRESHOLD = 0.995   # paper line 5
DEEP_CHECK = True        # paper lines 21-26 - set False to skip the retry loop

# paper line 10: try uppercase first, then lowercase
LETTERS = list(string.ascii_uppercase + string.ascii_lowercase)

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(VICTIM)
model = AutoModelForSequenceClassification.from_pretrained(VICTIM).to(device)
model.eval()

cache = {}  # word -> (class, confidence). words repeat a lot, so this saves hours


def classify(texts):
    """Return (predicted_class, confidence_of_that_class) for each string."""
    out = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1)
        conf, cls = probs.max(dim=-1)
        out.extend(zip(cls.cpu().tolist(), conf.cpu().tolist()))
    return out


def classify_cached(words):
    """Same as classify() but reuses results for words we've already seen."""
    missing = [w for w in set(words) if w not in cache]
    if missing:
        for w, r in zip(missing, classify(missing)):
            cache[w] = r
    return [cache[w] for w in words]


def is_detected(text):
    """paper: test_model() - True when the full prompt is still flagged toxic."""
    cls, _ = classify([text])[0]
    return cls == 1


def break_prompt(text, threshold):
    """Algorithm 1 lines 6-20. Returns (new_text, edits) where each edit is
    (word_index, original_word, inserted_character)."""
    words = text.split(" ")   # paper line 6: split by spaces
    edits = []

    # classify every word on its own first (paper line 8)
    real = [w for w in words if w.strip()]
    scores = dict(zip(real, classify_cached(real)))

    for i, word in enumerate(words):
        if not word.strip():   # skip the empty strings from double spaces
            continue
        cls, conf = scores[word]

        # paper line 9: target words that are toxic, or not confidently benign
        if cls == 1 or conf < threshold:
            variants = [ch + word for ch in LETTERS]
            results = classify_cached(variants)

            # paper lines 13-15: keep the FIRST letter that makes the word
            # look confidently benign, then stop trying letters
            for ch, (c, con) in zip(LETTERS, results):
                if c == 0 and con >= threshold:
                    words[i] = ch + word
                    edits.append((i, word, ch))
                    break

    return " ".join(words), edits   # paper line 20


def attack(text):
    """Full BreakPrompt including the deep_check retry.

    NOTE: the paper's deep_check recurses with the threshold reset to 0.995,
    which as written never terminates. We escalate iteratively instead."""
    if not is_detected(text):
        return False, text, [], BASE_THRESHOLD   # paper lines 2-3

    threshold = BASE_THRESHOLD
    new_text, edits = break_prompt(text, threshold)

    if DEEP_CHECK:
        while is_detected(new_text) and threshold < 0.9999:
            threshold += 0.0001
            new_text, edits = break_prompt(text, threshold)

    return not is_detected(new_text), new_text, edits, threshold


if __name__ == "__main__":
    df = pd.read_csv("data/clean_baseline_toxic.csv")
    eligible = df[df["clean_prediction"] == 1].reset_index(drop=True)
    print(f"attacking {len(eligible)} eligible rows...")

    rows = []
    for n, r in eligible.iterrows():
        t0 = time.perf_counter()
        success, attacked_text, edits, thr = attack(r["text"])
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # re-score the final text so we can report the toxic probability
        enc = tok(attacked_text, return_tensors="pt",
                  truncation=True, max_length=MAX_LEN).to(device)
        with torch.no_grad():
            attacked_prob = torch.softmax(model(**enc).logits, dim=-1)[0, 1].item()

        rows.append(build_record(
            attack="tokenbreak",
            sample_id=r["sample_id"],
            original_text=r["text"],
            attacked_text=attacked_text,
            modified_words=[w for _i, w, _c in edits],
            inserted_characters=[c for _i, _w, c in edits],
            clean_p_toxic=r["clean_toxic_prob"],
            attacked_p_toxic=attacked_prob,
            clean_tokens=tok.tokenize(r["text"]),
            attacked_tokens=tok.tokenize(attacked_text),
            clean_ids=tok(r["text"], add_special_tokens=False)["input_ids"],
            attacked_ids=tok(attacked_text, add_special_tokens=False)["input_ids"],
            search_latency_ms=elapsed_ms,
            extras={
                "n_modified_words": len(edits),
                "n_added_chars": len(edits),   # exactly one letter per modified word
                "modified_positions": json.dumps([i for i, _w, _c in edits]),
                "final_threshold": thr,
            },
        ))
        print(f"  {n + 1}/{len(eligible)}  cache={len(cache)}", end="\r")

    out = pd.DataFrame(rows)
    out.to_csv("data/tokenbreak_results.csv", index=False)

    s = out["attack_success"].sum()
    print(f"\n\n--- tokenbreak (algorithm 1) ---")
    print(f"eligible        : {len(out)}")
    print(f"successful      : {s}")
    print(f"ASR             : {s / len(out):.3f}")
    print(f"mean words hit  : {out['n_modified_words'].mean():.1f}")
    print(f"mean conf drop  : {(out['clean_toxic_prob'] - out['attacked_toxic_prob']).mean():.4f}")