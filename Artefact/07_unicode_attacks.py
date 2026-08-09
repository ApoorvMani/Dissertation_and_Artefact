"""
Step 7 - unicode attack family. Four sub-attacks, one script.

Follows Boucher, Shumailov, Anderson & Papernot, "Bad Characters: Imperceptible
NLP Attacks" (IEEE S&P 2022), Algorithm 1: a black-box differential-evolution
search picks where each perturbation goes and which character it uses, under a
fixed perturbation budget. Population 32, 10 iterations, as in the paper.

Deviations, deliberate:
  - objective is min P(toxic | x) instead of the paper's levenshtein distance on
    model output - the target is a binary classifier, not a seq2seq model
  - the paper's fourth class is deletions (visible char + backspace); we run
    compatibility characters instead. that family is OUR OWN EXTENSION and must
    be labelled experimental in the write-up, like the CPT defences.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")  # windows terminal defaults to cp1252

import json
import os
import time
import unicodedata
import urllib.request

import numpy as np
import pandas as pd
import torch
from scipy.optimize import differential_evolution
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from attack_schema import build_record

VICTIM = "martin-ha/toxic-comment-model"
MAX_LEN = 512
BATCH = 64
SEED = 42
BUDGET = 5      # paper sweeps 0-5; we fix at the top of that range
POP_SIZE = 32   # paper: population size 32
MAX_ITERS = 10  # paper: maximum 10 evolution iterations
LIMIT = None    # set to e.g. 5 for a smoke test before the full run

OUT_DIR = "data"

# paper: invisible characters chosen from ZWSP, ZWNJ, ZWJ
INVISIBLE = ["\u200b", "\u200c", "\u200d"]

# paper Algorithm 2 - the bidi control characters used to nest one swap
LRO, RLO, PDF = "\u202d", "\u202e", "\u202c"
LRI, PDI = "\u2066", "\u2069"

# paper uses the UTS #39 "intentional" mapping for homoglyphs
INTENTIONAL_URL = "https://www.unicode.org/Public/security/latest/intentional.txt"
INTENTIONAL_CACHE = f"{OUT_DIR}/intentional.txt"

rng = np.random.default_rng(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(VICTIM)
model = AutoModelForSequenceClassification.from_pretrained(VICTIM).to(device)
model.eval()


# --------------------------------------------------------------- character maps
def load_homoglyphs():
    """ascii char -> list of visually confusable characters, from UTS #39."""
    if not os.path.exists(INTENTIONAL_CACHE):
        print("downloading UTS #39 intentional.txt...")
        urllib.request.urlretrieve(INTENTIONAL_URL, INTENTIONAL_CACHE)

    table = {}
    with open(INTENTIONAL_CACHE, encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()   # drop the comment half
            parts = [p.strip() for p in line.split(";") if p.strip()]
            if len(parts) < 2:
                continue
            try:
                a, b = chr(int(parts[0], 16)), chr(int(parts[1], 16))
            except ValueError:
                continue                        # multi-codepoint entries, skip
            # the file lists pairs in both directions, so check each side
            for src, tgt in [(a, b), (b, a)]:
                if src.isascii() and src.isalpha() and not tgt.isascii():
                    table.setdefault(src, []).append(tgt)
    return table


def load_compatibility():
    """ascii char -> list of characters that NFKC-normalise back to it.

    OUR OWN construction, not from the paper. picks up fullwidth forms,
    mathematical alphanumerics, superscripts and so on."""
    table = {}
    for cp in range(0x80, 0x30000):             # skip ascii itself
        ch = chr(cp)
        nf = unicodedata.normalize("NFKC", ch)
        if len(nf) == 1 and nf.isascii() and nf.isalnum():
            table.setdefault(nf, []).append(ch)
    return table


HOMOGLYPHS = load_homoglyphs()
COMPATIBILITY = load_compatibility()


# --------------------------------------------------------------- perturbations
def swap_sequence(one, two):
    """Paper Algorithm 2, ENCODE of a single Swap. Ten bidi control characters
    wrap the two letters so they render in their original order while the bytes
    are reversed."""
    return LRO + LRI + RLO + LRI + one + PDI + LRI + two + PDI + PDF + PDI + PDF


def sites(text, family):
    """Character positions this family can actually act on."""
    if family in ("invisible",):
        return list(range(len(text)))           # insert anywhere
    if family == "reorder":
        return list(range(len(text) - 1))       # needs a pair to swap
    table = HOMOGLYPHS if family == "homoglyph" else COMPATIBILITY
    return [i for i, c in enumerate(text) if c in table]


def apply_one(text, family, pos, frac):
    """Apply a single perturbation. Returns (new_text, injected_string)."""
    if family == "invisible":
        ch = INVISIBLE[int(frac * len(INVISIBLE)) % len(INVISIBLE)]
        return text[:pos] + ch + text[pos:], ch

    if family == "reorder":
        a, b = text[pos], text[pos + 1]
        seq = swap_sequence(b, a)               # paper: Swap := {body[i+1], body[i]}
        return text[:pos] + seq + text[pos + 2:], seq

    table = HOMOGLYPHS if family == "homoglyph" else COMPATIBILITY
    pool = table[text[pos]]
    ch = pool[int(frac * len(pool)) % len(pool)]
    return text[:pos] + ch + text[pos + 1:], ch


def build(text, family, valid, params):
    """Turn a parameter vector into perturbed text.

    params is 2*BUDGET floats in [0,1): (site fraction, character fraction) per
    perturbation. Applied right to left so earlier positions stay valid."""
    picks = {}
    for k in range(BUDGET):
        site = valid[int(params[2 * k] * len(valid)) % len(valid)]
        picks[site] = params[2 * k + 1]         # same site twice = one edit

    out, injected = text, []
    for pos in sorted(picks, reverse=True):
        out, ch = apply_one(out, family, pos, picks[pos])
        injected.append(ch)
    return out, sorted(picks), injected


# --------------------------------------------------------------- scoring
@torch.no_grad()
def p_toxic(texts):
    """P(toxic) for a list of strings. index 1 = toxic, from step 1."""
    out = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN).to(device)
        probs = torch.softmax(model(**enc).logits, dim=-1)[:, 1]
        out.extend(probs.cpu().tolist())
    return out


# --------------------------------------------------------------- the attack
def attack(text, family):
    """Differential evolution over perturbation sites and characters."""
    valid = sites(text, family)
    if not valid:
        return text, p_toxic([text])[0], [], []

    def objective(x):
        """x arrives as (n_params, n_candidates) because vectorized=True."""
        cands = [build(text, family, valid, x[:, j])[0] for j in range(x.shape[1])]
        return np.array(p_toxic(cands))

    n_params = 2 * BUDGET
    init = rng.random((POP_SIZE, n_params))     # paper: randomly initialised

    res = differential_evolution(
        objective,
        bounds=[(0.0, 0.999999)] * n_params,
        init=init,
        maxiter=MAX_ITERS,
        seed=SEED,
        vectorized=True,
        updating="deferred",                    # required when vectorized
        polish=False,                           # params are discrete, don't refine
        tol=0.0,                                # let it use the full iteration budget
    )

    new_text, positions, injected = build(text, family, valid, res.x)
    return new_text, res.fun, positions, injected


# --------------------------------------------------------------- run
FAMILIES = ["invisible", "homoglyph", "compat", "reorder"]

if __name__ == "__main__":
    df = pd.read_csv(f"{OUT_DIR}/clean_baseline_toxic.csv")
    eligible = df[df["clean_prediction"] == 1].reset_index(drop=True)
    if LIMIT:
        eligible = eligible.head(LIMIT)
    print(f"attacking {len(eligible)} eligible rows x {len(FAMILIES)} families")
    print(f"homoglyph map: {len(HOMOGLYPHS)} letters | "
          f"compatibility map: {len(COMPATIBILITY)} letters\n")

    all_rows = []
    for family in FAMILIES:
        rows = []
        for n, r in eligible.iterrows():
            t0 = time.perf_counter()
            new_text, p_adv, positions, injected = attack(r["text"], family)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            rows.append(build_record(
                attack=f"unicode_{family}",
                sample_id=r["sample_id"],
                original_text=r["text"],
                attacked_text=new_text,
                # unicode attacks work on characters, not words - left empty so
                # the column stays comparable with the other two attacks
                modified_words=[],
                inserted_characters=injected,
                clean_p_toxic=r["clean_toxic_prob"],
                attacked_p_toxic=p_adv,
                clean_tokens=tok.tokenize(r["text"]),
                attacked_tokens=tok.tokenize(new_text),
                clean_ids=tok(r["text"], add_special_tokens=False)["input_ids"],
                attacked_ids=tok(new_text, add_special_tokens=False)["input_ids"],
                search_latency_ms=elapsed_ms,
                extras={
                    "family": family,
                    "budget": BUDGET,
                    "n_perturbations": len(positions),
                    "positions": json.dumps(positions),
                    # codepoints are easier to read than the raw characters
                    "codepoints": json.dumps(
                        ["U+%04X" % ord(c) for s in injected for c in s]),
                },
            ))
            print(f"  {family}: {n + 1}/{len(eligible)}", end="\r")

        out = pd.DataFrame(rows)
        out.to_csv(f"{OUT_DIR}/unicode_{family}.csv", index=False)
        all_rows.extend(rows)

        s = out["attack_success"].sum()
        print(f"\n--- unicode / {family} ---")
        print(f"eligible       : {len(out)}")
        print(f"successful     : {s}")
        print(f"ASR            : {s / len(out):.3f}")
        print(f"mean conf drop : "
              f"{(out['clean_toxic_prob'] - out['attacked_toxic_prob']).mean():.4f}")
        print(f"mean edits     : {out['n_perturbations'].mean():.1f}")
        print(f"tokens clean/adv: {out['clean_token_count'].mean():.1f} / "
              f"{out['attacked_token_count'].mean():.1f}\n")

    combined = pd.DataFrame(all_rows)
    combined.to_csv(f"{OUT_DIR}/unicode_results.csv", index=False)
    print(f"wrote {OUT_DIR}/unicode_results.csv ({len(combined)} rows) "
          f"plus one csv per family")