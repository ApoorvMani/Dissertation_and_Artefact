"""
Step 8 - defence library and CPT calibration.

Every defence has the same signature:

    defend(text, ids) -> (out_ids, flagged, mechanism)

text
    the visible string submitted by the attacker

ids
    raw token ids supplied directly by the attacker, or None when the
    interface received ordinary text

out_ids
    token ids that reach the victim model.
    None means the request was rejected before model inference.

flagged
    1 if the defence detected something suspicious, otherwise 0

mechanism
    records WHY the result happened:
        none             defence did nothing
        acted            defence itself changed/detected the input
        reencode         supplied ids disappeared only because text was re-encoded
        acted+reencode   both happened

Defences:
    D1 tokenizer translation
    D2 canonical token-id reject
    D3 canonical token-id replace
    D4 targeted unicode sanitisation
    D5 NFKC + confusable handling
    D6 global CPT
    D7 sliding-window CPT

CPT note:
    Broken-Token defines average characters-per-token and demonstrates
    sliding-window CPT with a 5-token window. Its experiments focus on BPE.
    Here CPT is deliberately adapted to the victim's WordPiece tokenizer.

Threshold note:
    the 1% WikiText calibration FPR is our evaluation policy, not a threshold
    copied from Broken-Token. Attack data is never used to choose it.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")  # windows terminal defaults to cp1252

import json
import os
import unicodedata

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from attack_schema import piece_len


# --------------------------------------------------------------- config

VICTIM = "martin-ha/toxic-comment-model"
UNIGRAM = "xlnet-base-cased"   # fast tokenizer works without sentencepiece

MAX_TOKENS = 510               # leaves room for [CLS] and [SEP]
# Broken-Token uses sliding-window CPT with a 5-token example.
# 5 tokens degenerates to the CPT floor on our WordPiece tokenizer,
# so we use 10 as a deliberate WordPiece adaptation.
WINDOW = 10
TARGET_FPR = 0.01              # our calibration policy: 1% clean WikiText FPR

DATA = "data"
THRESHOLD_FILE = f"{DATA}/cpt_thresholds.json"
INTENTIONAL_CACHE = f"{DATA}/intentional.txt"

# short labels so the matrix groups them cleanly
NONE = "none"
ACTED = "acted"
REENCODE = "reencode"
BOTH = "acted+reencode"


# --------------------------------------------------------------- tokenizers

tok = AutoTokenizer.from_pretrained(VICTIM)
uni = AutoTokenizer.from_pretrained(UNIGRAM)

VOCAB = tok.get_vocab()


# --------------------------------------------------------------- confusables

def load_confusables():
    """Load non-ascii -> ascii mappings from the cached UTS #39 table."""
    if not os.path.exists(INTENTIONAL_CACHE):
        raise FileNotFoundError(
            f"{INTENTIONAL_CACHE} not found. Run step 7 first."
        )

    table = {}

    with open(INTENTIONAL_CACHE, encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            parts = [p.strip() for p in line.split(";") if p.strip()]

            if len(parts) < 2:
                continue

            try:
                a = chr(int(parts[0], 16))
                b = chr(int(parts[1], 16))
            except ValueError:
                # keep the same single-codepoint rule as step 7
                continue

            # the file contains pairs in both directions
            if b.isascii() and not a.isascii():
                table[a] = b
            elif a.isascii() and not b.isascii():
                table[b] = a

    return table


CONFUSABLES = load_confusables()


# --------------------------------------------------------------- unicode sets

# deletion-style controls named in the Bad Characters attack family
DELETIONS = {
    "\u0008",   # backspace
    "\u007f",   # delete
    "\r",       # carriage return
}

# unicode bidi formatting controls - these can change display order rather
# than simply inserting an ignorable character, so detecting one means reject
BIDI_CLASSES = {
    "LRE", "RLE", "LRO", "RLO", "PDF",
    "LRI", "RLI", "FSI", "PDI",
}


def has_bidi_control(text):
    """True when the text contains a unicode bidi formatting control."""
    return any(
        unicodedata.bidirectional(ch) in BIDI_CLASSES
        for ch in text
    )


def is_targeted(ch):
    """True for invisible, bidi, or deletion/control characters."""
    if ch in DELETIONS:
        return True

    cat = unicodedata.category(ch)

    # Cf covers zero-width and bidi format controls.
    # preserve ordinary tab/newline so normal text structure survives.
    return cat == "Cf" or (cat == "Cc" and ch not in "\t\n")


# --------------------------------------------------------------- helpers

def encode(text):
    """Canonical victim ids that can safely be sent to DistilBERT."""
    return tok(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_TOKENS,
    )["input_ids"]


def tokenize_full(text):
    """Tokenize without truncation for measurements such as CPT."""
    encoding = tok.backend_tokenizer.encode(
        text,
        add_special_tokens=False,
    )
    return encoding.tokens


def mech(acted, reencoded):
    """Combine defence action and interface re-encoding into one label."""
    if acted and reencoded:
        return BOTH

    if acted:
        return ACTED

    if reencoded:
        return REENCODE

    return NONE


# --------------------------------------------------------------- CPT scores

def cpt_global(tokens, text):
    """Broken-Token: average input characters per token."""
    if not tokens:
        return 0.0

    return len(text) / len(tokens)


def cpt_window(tokens, text):
    """Lowest CPT over a 5-token window.

    Broken-Token applies CPT to token subsets and demonstrates a 5-token
    sliding window. piece_len() is our WordPiece adaptation for measuring
    how many visible characters each token contributes.
    """
    if not tokens:
        return 0.0

    if len(tokens) < WINDOW:
        return cpt_global(tokens, text)

    lengths = [piece_len(t) for t in tokens]

    scores = []

    for i in range(len(lengths) - WINDOW + 1):
        chars = sum(lengths[i:i + WINDOW])
        scores.append(chars / WINDOW)

    return min(scores)


# --------------------------------------------------------------- D1

def tokenizer_translation(text, ids):
    """TokenBreak Algorithm 2: Unigram first, then map into WordPiece ids."""
    out = []

    # use the fast backend directly so long strings do not produce
    # misleading model-length warnings during preprocessing
    uni_tokens = uni.backend_tokenizer.encode(
        text,
        add_special_tokens=False,
    ).tokens

    for piece in uni_tokens:
        # XLNet marks word starts using the sentencepiece ▁ character
        piece = piece.replace("\u2581", "")

        if not piece:
            continue

        # victim is uncased but XLNet is cased
        piece = piece.lower()

        # TokenBreak Algorithm 2: use it directly if target vocab has it
        if piece in VOCAB:
            out.append(VOCAB[piece])
            continue

        # otherwise let the victim WordPiece tokenizer split that piece
        subtokens = tok.tokenize(piece)
        sub_ids = tok.convert_tokens_to_ids(subtokens)
        out.extend(sub_ids)

    out = out[:MAX_TOKENS]

    # translation "acted" when it produces a representation different
    # from the victim's ordinary canonical encoding
    acted = out != encode(text)

    # supplied attacker ids are discarded because this defence starts
    # from the visible string
    reencoded = ids is not None

    return out, 0, mech(acted, reencoded)


# --------------------------------------------------------------- D2

def canonical_reject(text, ids):
    """Reject supplied ids when they differ from canonical ids for the text."""
    if ids is None:
        # ordinary string input gives us nothing noncanonical to check
        return encode(text), 0, NONE

    supplied = list(ids)
    canonical = encode(text)

    bad = supplied != canonical

    if bad:
        # reject means nothing reaches the classifier
        return None, 1, ACTED

    return supplied, 0, NONE


# --------------------------------------------------------------- D3

def canonical_replace(text, ids):
    """Replace noncanonical supplied ids with canonical victim-tokenizer ids."""
    canonical = encode(text)

    if ids is None:
        return canonical, 0, NONE

    supplied = list(ids)
    bad = supplied != canonical

    # unlike reject, the corrected canonical ids continue to the model
    return canonical, int(bad), mech(bad, False)


# --------------------------------------------------------------- D4

def unicode_sanitise(text, ids):
    """Remove invisible/deletion controls, but reject bidi manipulation."""
    reencoded = ids is not None

    # bidi attacks reorder the underlying characters. simply deleting the
    # controls can leave damaged/reversed text, so reject instead.
    if has_bidi_control(text):
        return None, 1, mech(True, reencoded)

    # ordinary invisible/control characters can safely be stripped
    clean = "".join(
        ch for ch in text
        if not is_targeted(ch)
    )

    acted = clean != text

    return (
        encode(clean),
        int(acted),
        mech(acted, reencoded),
    )


# --------------------------------------------------------------- D5

def nfkc_confusable(text, ids):
    """Apply NFKC and map known UTS #39 confusables back toward ASCII."""
    # NFKC reverses our compatibility-character attack by construction
    folded = unicodedata.normalize("NFKC", text)

    # map the same one-codepoint confusable family used by step 7
    mapped = "".join(
        CONFUSABLES.get(ch, ch)
        for ch in folded
    )

    acted = mapped != text
    reencoded = ids is not None

    return (
        encode(mapped),
        int(acted),
        mech(acted, reencoded),
    )


# --------------------------------------------------------------- D6 / D7

def cpt_detector(text, ids, thresholds, which):
    """Detect abnormally fragmented tokenization without rewriting it."""
    if ids is None:
        # score the COMPLETE string, not a 510-token truncation.
        # only the ids that later reach DistilBERT are truncated.
        tokens = tokenize_full(text)
        out_ids = encode(text)

    else:
        # AdvTok supplied the representation directly, so CPT inspects it
        # rather than silently destroying the attack by retokenizing.
        out_ids = list(ids)[:MAX_TOKENS]
        tokens = tok.convert_ids_to_tokens(out_ids)

    if which == "global":
        score = cpt_global(tokens, text)

    elif which == "window":
        score = cpt_window(tokens, text)

    else:
        raise ValueError(f"unknown CPT type: {which}")

    # fragmented inputs live in the low-CPT tail
    flagged = int(score < thresholds[which])

    return (
        out_ids,
        flagged,
        mech(flagged, False),
    )


# --------------------------------------------------------------- registry

def build_defences(thresholds):
    """Return every defence with the same (text, ids) calling convention."""
    return {
        "tokenizer_translation": tokenizer_translation,
        "canonical_reject": canonical_reject,
        "canonical_replace": canonical_replace,
        "unicode_sanitise": unicode_sanitise,
        "nfkc_confusable": nfkc_confusable,
        "cpt_global": lambda text, ids: cpt_detector(
            text, ids, thresholds, "global"
        ),
        "cpt_window": lambda text, ids: cpt_detector(
            text, ids, thresholds, "window"
        ),
    }


# --------------------------------------------------------------- calibration

def calibrate():
    """Choose CPT thresholds using only clean WikiText calibration rows."""
    df = pd.read_csv(
        f"{DATA}/wikitext_calibration_5000.csv"
    )

    global_scores = []
    window_scores = []

    for n, text in enumerate(df["text"]):
        tokens = tokenize_full(text)

        if not tokens:
            continue

        global_scores.append(
            cpt_global(tokens, text)
        )

        window_scores.append(
            cpt_window(tokens, text)
        )

        print(
            f"  {n + 1}/{len(df)}",
            end="\r",
        )

    print()

    # low CPT means unusual fragmentation, so use the left 1% tail.
    # this is our pre-registered calibration policy, not attack tuning.
    global_threshold = float(
        np.percentile(
            global_scores,
            TARGET_FPR * 100,
        )
    )

    window_threshold = float(
        np.percentile(
            window_scores,
            TARGET_FPR * 100,
        )
    )

    thresholds = {
        "global": global_threshold,
        "window": window_threshold,
    }

    # sanity check the FPR achieved on the calibration set itself
    global_fpr = float(
        np.mean(
            np.array(global_scores) < global_threshold
        )
    )

    window_fpr = float(
        np.mean(
            np.array(window_scores) < window_threshold
        )
    )

    return thresholds, global_fpr, window_fpr


def load_thresholds():
    """Load previously frozen CPT thresholds."""
    with open(
        THRESHOLD_FILE,
        encoding="utf-8",
    ) as f:
        return json.load(f)


# --------------------------------------------------------------- smoke tests

def smoke_test(defences):
    """Check ordinary string/id inputs plus one real AdvTok attack."""
    probe = "you are a cfucking aidiot"
    probe_ids = encode(probe)

    print("\n--- basic smoke test ---")

    for name, fn in defences.items():
        _out, flag_text, mech_text = fn(
            probe,
            None,
        )

        _out, flag_ids, mech_ids = fn(
            probe,
            probe_ids,
        )

        print(
            f"{name:22s} "
            f"text: flag={flag_text} {mech_text:16s} "
            f"ids: flag={flag_ids} {mech_ids}"
        )

    # targeted unicode should definitely notice a zero-width character
    unicode_probe = "you are a\u200b bad person"

    _out, flag, mechanism = defences["unicode_sanitise"](
        unicode_probe,
        None,
    )

    print("\nunicode sanitiser probe")
    print(f"  flagged   : {flag}")
    print(f"  mechanism : {mechanism}")

    # fullwidth characters are compatibility-equivalent under NFKC
    nfkc_probe = "you are ｂａｄ"

    _out, flag, mechanism = defences["nfkc_confusable"](
        nfkc_probe,
        None,
    )

    print("\nNFKC probe")
    print(f"  flagged   : {flag}")
    print(f"  mechanism : {mechanism}")


def advtok_smoke_test(defences):
    """Verify canonical enforcement using a real noncanonical AdvTok row."""
    path = f"{DATA}/advtok_results.csv"

    if not os.path.exists(path):
        print("\nAdvTok smoke test skipped - results file not found.")
        return

    df = pd.read_csv(path)

    # use only a row already validated by step 5
    if "valid" in df.columns:
        df = df[df["valid"] == 1]

    if "noncanonical" in df.columns:
        df = df[df["noncanonical"] == 1]

    if len(df) == 0:
        print("\nAdvTok smoke test skipped - no valid row found.")
        return

    row = df.iloc[0]

    text = row["attacked_text"]
    adv_ids = json.loads(row["attacked_ids"])

    print("\n--- real AdvTok smoke test ---")
    print(f"sample_id : {row['sample_id']}")
    print(f"canonical : {adv_ids == encode(text)}")

    out_ids, flagged, mechanism = defences["canonical_reject"](
        text,
        adv_ids,
    )

    print("\ncanonical_reject")
    print(f"  flagged   : {flagged}")
    print(f"  mechanism : {mechanism}")
    print(f"  blocked   : {out_ids is None}")

    out_ids, flagged, mechanism = defences["canonical_replace"](
        text,
        adv_ids,
    )

    print("\ncanonical_replace")
    print(f"  flagged   : {flagged}")
    print(f"  mechanism : {mechanism}")
    print(f"  canonical : {out_ids == encode(text)}")


# --------------------------------------------------------------- run

if __name__ == "__main__":
    print("calibrating CPT thresholds on clean WikiText...")

    thresholds, global_fpr, window_fpr = calibrate()

    with open(
        THRESHOLD_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            thresholds,
            f,
            indent=2,
        )

    print("\n--- calibration ---")
    print(f"target fpr        : {TARGET_FPR:.3f}")
    print(f"global threshold  : {thresholds['global']:.4f}")
    print(f"window threshold  : {thresholds['window']:.4f}  (window {WINDOW})")
    print(f"global calib fpr  : {global_fpr:.4f}")
    print(f"window calib fpr  : {window_fpr:.4f}")
    print(f"confusable map    : {len(CONFUSABLES)} characters")

    defences = build_defences(thresholds)

    smoke_test(defences)
    advtok_smoke_test(defences)

    print(f"\nwrote {THRESHOLD_FILE}")
    print("step 8 done")