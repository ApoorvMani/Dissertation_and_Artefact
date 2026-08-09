"""
Step 9 - paired clean-vs-attacked defence matrix.

For every attack example and every defence, run TWO inputs:

    1. the original clean toxic input
    2. the attacked version of that same input

This prevents a defence from getting credit when the defence itself already
causes the clean toxic input to become non-toxic.

A defended attack success therefore requires:

    clean + defence    -> still detected toxic
    attacked + defence -> bypasses as non-toxic

For AdvTok, the clean side supplies canonical token ids while the attacked
side supplies the noncanonical adversarial ids.

Outputs:
    data/defence_matrix.csv
    data/defence_matrix_summary.csv
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")  # windows terminal defaults to cp1252

import importlib.util
import json
import os
import time

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification


# --------------------------------------------------------------- config

VICTIM = "martin-ha/toxic-comment-model"

MAX_TOKENS = 510   # leaves room for [CLS] and [SEP]
BATCH = 32

DATA = "data"
DEFENCE_FILE = "08_defences.py"

OUT_FILE = f"{DATA}/defence_matrix.csv"
SUMMARY_FILE = f"{DATA}/defence_matrix_summary.csv"

# these are detection/filter defences.
# if they flag an input, deployment blocks it before inference.
FILTER_DEFENCES = {
    "canonical_reject",
    "cpt_global",
    "cpt_window",
}


# --------------------------------------------------------------- load step 8

def load_defence_module():
    """Load 08_defences.py even though its filename starts with a number."""
    if not os.path.exists(DEFENCE_FILE):
        raise FileNotFoundError(
            f"{DEFENCE_FILE} not found in the project root."
        )

    spec = importlib.util.spec_from_file_location(
        "defence_library",
        DEFENCE_FILE,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


defence_lib = load_defence_module()

thresholds = defence_lib.load_thresholds()
defences = defence_lib.build_defences(thresholds)

# reuse exactly the tokenizer used by the defence library
tok = defence_lib.tok


# --------------------------------------------------------------- victim model

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForSequenceClassification.from_pretrained(
    VICTIM
).to(device)

model.eval()

CLS = tok.cls_token_id
SEP = tok.sep_token_id
PAD = tok.pad_token_id

TOXIC = 1   # confirmed in step 1


# --------------------------------------------------------------- helpers

def parse_ids(value):
    """Read token ids stored as JSON in the attack CSV."""
    if value is None:
        return None

    if isinstance(value, list):
        return value

    if pd.isna(value):
        return None

    return json.loads(value)


def high_level_attack(name):
    """Collapse the unicode sub-attacks into one broad attack family."""
    if name.startswith("unicode_"):
        return "unicode"

    return name


def is_blocked(defence_name, flagged, out_ids):
    """Apply the deployment policy for one defence."""
    # reject-style defences return no model input
    if out_ids is None:
        return 1

    # CPT is detection-only, so a positive alarm means block
    if defence_name in FILTER_DEFENCES and flagged:
        return 1

    return 0


def verify_bidi_policy():
    """Stop immediately unless unicode bidi controls are rejected."""
    probe = "ab\u202ecd\u202c"

    out_ids, flagged, _mechanism = defences[
        "unicode_sanitise"
    ](
        probe,
        None,
    )

    if flagged != 1 or out_ids is not None:
        raise RuntimeError(
            "unicode_sanitise is not rejecting bidi controls. "
            "patch 08_defences.py before running step 9."
        )

    print("bidi rejection check passed")


# --------------------------------------------------------------- load attacks

def load_attacks():
    """Load the frozen outputs from all three attack families."""
    print("loading attacks...")

    tokenbreak = pd.read_csv(
        f"{DATA}/tokenbreak_results.csv"
    )

    advtok = pd.read_csv(
        f"{DATA}/advtok_results.csv"
    )

    unicode_df = pd.read_csv(
        f"{DATA}/unicode_results.csv"
    )

    # only genuinely valid noncanonical AdvTok examples are evaluated
    if "valid" in advtok.columns:
        advtok = advtok[
            advtok["valid"] == 1
        ].copy()

    if "noncanonical" in advtok.columns:
        advtok = advtok[
            advtok["noncanonical"] == 1
        ].copy()

    print(f"  tokenbreak : {len(tokenbreak)}")
    print(f"  advtok     : {len(advtok)}")
    print(f"  unicode    : {len(unicode_df)}")

    # frozen experiment sizes
    if len(tokenbreak) != 156:
        raise RuntimeError(
            f"expected 156 TokenBreak rows, got {len(tokenbreak)}"
        )

    if len(advtok) != 156:
        raise RuntimeError(
            f"expected 156 valid AdvTok rows, got {len(advtok)}"
        )

    if len(unicode_df) != 624:
        raise RuntimeError(
            f"expected 624 Unicode rows, got {len(unicode_df)}"
        )

    return [
        ("tokenbreak", tokenbreak, "text"),
        ("advtok", advtok, "ids"),
        ("unicode", unicode_df, "text"),
    ]


# --------------------------------------------------------------- scoring

@torch.no_grad()
def p_toxic(id_seqs):
    """P(toxic) for raw token-id sequences."""
    probs = []

    for i in range(0, len(id_seqs), BATCH):
        chunk = id_seqs[i:i + BATCH]

        chunk = [
            list(ids)[:MAX_TOKENS]
            for ids in chunk
        ]

        width = max(
            len(ids) for ids in chunk
        ) + 2

        input_ids = torch.full(
            (len(chunk), width),
            PAD,
            dtype=torch.long,
        )

        attention = torch.zeros(
            (len(chunk), width),
            dtype=torch.long,
        )

        for j, ids in enumerate(chunk):
            seq = [CLS] + ids + [SEP]

            input_ids[j, :len(seq)] = torch.tensor(
                seq,
                dtype=torch.long,
            )

            attention[j, :len(seq)] = 1

        logits = model(
            input_ids=input_ids.to(device),
            attention_mask=attention.to(device),
        ).logits

        p = torch.softmax(
            logits,
            dim=-1,
        )[:, TOXIC]

        probs.extend(
            p.cpu().tolist()
        )

    return probs


# --------------------------------------------------------------- matrix

def run_matrix(attack_sets):
    """Run paired clean and attacked inputs through every defence."""
    rows = []

    total = sum(
        len(df) for _name, df, _kind in attack_sets
    ) * len(defences)

    done = 0

    for family_name, df, input_kind in attack_sets:
        print(f"\n{family_name} x defences...")

        for _, r in df.iterrows():
            # original_text is the correct clean counterpart for each attack.
            # this matters for AdvTok because very long examples were trimmed
            # to the model's usable token range during step 5.
            clean_text = r["original_text"]
            attacked_text = r["attacked_text"]

            if input_kind == "ids":
                # clean AdvTok comparison uses the canonical ids for the
                # exact same visible text.
                clean_ids = defence_lib.encode(
                    clean_text
                )

                # attacked side gets the actual noncanonical ids.
                attacked_ids = parse_ids(
                    r["attacked_ids"]
                )

            else:
                # TokenBreak and Unicode enter through the string interface.
                clean_ids = None
                attacked_ids = None

            for defence_name, defend in defences.items():
                # ------------------------------------------------ clean side

                clean_t0 = time.perf_counter()

                clean_out, clean_flag, clean_mechanism = defend(
                    clean_text,
                    clean_ids,
                )

                clean_latency = (
                    time.perf_counter() - clean_t0
                ) * 1000

                clean_blocked = is_blocked(
                    defence_name,
                    clean_flag,
                    clean_out,
                )

                # ------------------------------------------------ attack side

                attack_t0 = time.perf_counter()

                attack_out, attack_flag, attack_mechanism = defend(
                    attacked_text,
                    attacked_ids,
                )

                attack_latency = (
                    time.perf_counter() - attack_t0
                ) * 1000

                attack_blocked = is_blocked(
                    defence_name,
                    attack_flag,
                    attack_out,
                )

                rows.append({
                    "sample_id": r["sample_id"],

                    "attack_group": high_level_attack(
                        r["attack"]
                    ),

                    "attack": r["attack"],
                    "defence": defence_name,

                    # -------------------------------------- original attack
                    "undefended_toxic_prob": float(
                        r["attacked_toxic_prob"]
                    ),

                    "undefended_attack_success": int(
                        r["attack_success"]
                    ),

                    # -------------------------------------- clean + defence
                    "clean_defence_flagged": int(
                        clean_flag
                    ),

                    "clean_mechanism": clean_mechanism,

                    "clean_blocked_by_policy": int(
                        clean_blocked
                    ),

                    "clean_out_token_count": (
                        len(clean_out)
                        if clean_out is not None
                        else 0
                    ),

                    "clean_defence_latency_ms": clean_latency,

                    "clean_defended_toxic_prob": None,
                    "clean_defended_label": None,

                    # -------------------------------------- attack + defence
                    "attack_defence_flagged": int(
                        attack_flag
                    ),

                    "attack_mechanism": attack_mechanism,

                    "attack_blocked_by_policy": int(
                        attack_blocked
                    ),

                    "attack_out_token_count": (
                        len(attack_out)
                        if attack_out is not None
                        else 0
                    ),

                    "attack_defence_latency_ms": attack_latency,

                    "defended_toxic_prob": None,
                    "defended_label": None,

                    # -------------------------------------- filled later
                    "clean_retained": None,
                    "attack_bypass_after_policy": None,
                    "defended_attack_success": None,

                    # temporary raw ids for batched scoring
                    "_clean_out": clean_out,
                    "_attack_out": attack_out,
                })

                done += 1

                print(
                    f"  {done}/{total}",
                    end="\r",
                )

        print()

    return rows


# --------------------------------------------------------------- score sides

def score_side(rows, id_field, prob_field, label_field, name):
    """Score every non-rejected output for one side of the pair."""
    row_indices = []
    sequences = []

    for i, row in enumerate(rows):
        ids = row[id_field]

        if ids is None:
            continue

        row_indices.append(i)
        sequences.append(ids)

    print(
        f"\nscoring {name}: {len(sequences)} model inputs..."
    )

    probs = []

    for i in range(0, len(sequences), BATCH):
        chunk = sequences[i:i + BATCH]

        probs.extend(
            p_toxic(chunk)
        )

        print(
            f"  {min(i + BATCH, len(sequences))}/{len(sequences)}",
            end="\r",
        )

    print()

    for row_index, prob in zip(
        row_indices,
        probs,
    ):
        rows[row_index][prob_field] = prob
        rows[row_index][label_field] = int(
            prob >= 0.5
        )


# --------------------------------------------------------------- final labels

def finish_rows(rows):
    """Decide whether each attack truly bypassed its defence."""
    for row in rows:
        # ------------------------------------------------------ clean retention
        #
        # the clean input must:
        #   1. not be blocked by the defence
        #   2. still be classified toxic
        #
        # otherwise the defence damaged the clean baseline itself.

        clean_retained = int(
            row["clean_blocked_by_policy"] == 0
            and row["clean_defended_label"] == 1
        )

        row["clean_retained"] = clean_retained

        # ------------------------------------------------------ attacked bypass

        attack_bypass = int(
            row["attack_blocked_by_policy"] == 0
            and row["defended_label"] == 0
        )

        row["attack_bypass_after_policy"] = attack_bypass

        # ------------------------------------------------------ paired success
        #
        # only call this an attack success when:
        #
        # clean + defence    -> toxic
        # attacked + defence -> non-toxic
        #
        # if clean + defence was already broken, the attack does not get
        # credit for the failure.

        row["defended_attack_success"] = int(
            clean_retained
            and attack_bypass
        )

        # temporary lists should not be written to the CSV
        del row["_clean_out"]
        del row["_attack_out"]


# --------------------------------------------------------------- summary

def make_summary(matrix):
    """Summarise each attack x defence cell."""
    records = []

    group_cols = [
        "attack_group",
        "attack",
        "defence",
    ]

    for keys, g in matrix.groupby(group_cols):
        attack_group, attack, defence = keys

        n = len(g)

        retained_n = int(
            g["clean_retained"].sum()
        )

        successes = int(
            g["defended_attack_success"].sum()
        )

        # main defended ASR uses only examples where the defence preserved
        # the classifier's clean toxic decision.
        defended_asr = (
            successes / retained_n
            if retained_n
            else float("nan")
        )

        reencoded = (
            g["attack_mechanism"]
            .str.contains(
                "reencode",
                regex=False,
            )
            .mean()
        )

        records.append({
            "attack_group": attack_group,
            "attack": attack,
            "defence": defence,

            "n": n,

            "undefended_asr": g[
                "undefended_attack_success"
            ].mean(),

            "clean_retained_n": retained_n,

            "clean_retention": g[
                "clean_retained"
            ].mean(),

            # an un-attacked input being flagged is a defence false alarm
            "clean_flag_rate": g[
                "clean_defence_flagged"
            ].mean(),

            "defended_successes": successes,

            "defended_asr": defended_asr,

            # useful secondary number: bypasses over the original 156,
            # even before conditioning on clean retention
            "overall_bypass_rate": g[
                "defended_attack_success"
            ].mean(),

            "attack_detection_rate": g[
                "attack_defence_flagged"
            ].mean(),

            "attack_block_rate": g[
                "attack_blocked_by_policy"
            ].mean(),

            "reencode_rate": reencoded,

            "mean_clean_defence_latency_ms": g[
                "clean_defence_latency_ms"
            ].mean(),

            "mean_attack_defence_latency_ms": g[
                "attack_defence_latency_ms"
            ].mean(),
        })

    return pd.DataFrame(records)


# --------------------------------------------------------------- run

if __name__ == "__main__":
    print("device:", device)

    verify_bidi_policy()

    attack_sets = load_attacks()

    total_attack_rows = sum(
        len(df)
        for _name, df, _kind in attack_sets
    )

    expected_rows = (
        total_attack_rows
        * len(defences)
    )

    print(f"\nattack rows : {total_attack_rows}")
    print(f"defences    : {len(defences)}")
    print(f"matrix rows : {expected_rows}")

    rows = run_matrix(
        attack_sets
    )

    if len(rows) != expected_rows:
        raise RuntimeError(
            f"expected {expected_rows} rows, got {len(rows)}"
        )

    # score both members of every clean/attack pair
    score_side(
        rows,
        "_clean_out",
        "clean_defended_toxic_prob",
        "clean_defended_label",
        "clean side",
    )

    score_side(
        rows,
        "_attack_out",
        "defended_toxic_prob",
        "defended_label",
        "attacked side",
    )

    finish_rows(rows)

    matrix = pd.DataFrame(rows)

    matrix.to_csv(
        OUT_FILE,
        index=False,
    )

    summary = make_summary(
        matrix
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print("\n--- paired defence matrix ---")
    print(f"attack rows  : {total_attack_rows}")
    print(f"defences     : {len(defences)}")
    print(f"matrix rows  : {len(matrix)}")
    print(f"matrix cells : {len(summary)}")
    print(f"output       : {OUT_FILE}")
    print(f"summary      : {SUMMARY_FILE}")

    print("\n--- quick results ---")

    for _, r in summary.iterrows():
        print(
            f"{r['attack']:20s} | "
            f"{r['defence']:22s} | "
            f"ASR {r['undefended_asr']:.3f}"
            f" -> {r['defended_asr']:.3f} | "
            f"clean keep {r['clean_retention']:.3f} | "
            f"detect {r['attack_detection_rate']:.3f} | "
            f"reencode {r['reencode_rate']:.3f}"
        )

    print("\nstep 9 done")