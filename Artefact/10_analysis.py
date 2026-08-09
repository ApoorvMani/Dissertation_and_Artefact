"""
Step 10 - final WordPiece analysis, clean controls, bootstrap CIs and plots.

This script DOES NOT rerun the attacks.

It uses:
    data/defence_matrix.csv
    data/wikitext_heldout_5000.csv
    data/jigsaw_benign_250.csv
    data/cpt_thresholds.json
    08_defences.py

It produces:
    results/step10/attack_metrics.csv
    results/step10/clean_controls.csv
    results/step10/clean_summary.csv
    results/step10/detection_metrics.csv
    results/step10/literature_notes.txt

    results/step10/figures/*.png
    results/step10/figures/*.pdf

    results/step10/report.html


------------------------------------------------------------
literature basis
------------------------------------------------------------

TokenBreak
Schulz, Yeung & Evans, 2025.
"TokenBreak: Bypassing Text Classification Models Through
Token Manipulation."
https://arxiv.org/abs/2506.07948

    - motivates tokenizer translation
    - their Algorithm 2 maps Unigram pieces back into the
      classifier's original vocabulary

AdvTok
Geh, Shao & Van den Broeck, ACL 2025.
"Adversarial Tokenization."
https://aclanthology.org/2025.acl-long.1012/

    - attacks noncanonical token sequences without changing text
    - Section 10 discusses retokenizing all inputs
    - also discusses restricting APIs to string-only inputs

Bad Characters
Boucher et al., IEEE S&P 2022.
"Bad Characters: Imperceptible NLP Attacks."
https://arxiv.org/abs/2106.09898

    - motivates Unicode input sanitisation
    - studies invisible, homoglyph, reordering and deletion attacks

Unicode UTS #39
https://www.unicode.org/reports/tr39/

    - specifies confusable-character detection mechanisms

Broken-Token
Zychlinski & Kainan, 2025.
"Broken-Token: Filtering Obfuscated Prompts by Counting
Characters-Per-Token."
https://arxiv.org/abs/2510.26847

    - provides the CPT filtering mechanism
    - thresholds are tokenizer-specific
    - demonstrates sliding-window CPT

IMPORTANT ADAPTATIONS IN THIS PROJECT:

    1. Broken-Token evaluates BPE tokenizers.
       Our victim uses WordPiece.

    2. Broken-Token selects optimal thresholds using clean +
       obfuscated examples.

       We instead froze the threshold using CLEAN WikiText only
       at approximately 1% FPR, avoiding attack-data leakage.

    3. Broken-Token demonstrates a 5-token window.
       This collapsed to the WordPiece score floor in our experiment,
       so we use a 10-token window as a documented adaptation.

Statistics:

    Bootstrap confidence intervals are an analysis method, not a
    tokenizer defence. We use paired nonparametric bootstrap
    resampling so clean/attacked observations stay paired.

    Efron, B. (1979).
    "Bootstrap Methods: Another Look at the Jackknife."
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import importlib.util
import html
import json
import math
import os
import time
import webbrowser
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from transformers import AutoModelForSequenceClassification


# --------------------------------------------------------------- config

SEED = 42

VICTIM = "martin-ha/toxic-comment-model"

MAX_TOKENS = 510
BATCH = 32
TOXIC_THRESHOLD = 0.5

# 5000 is enough for stable percentile intervals here while still fast.
N_BOOTSTRAP = 5000

DATA = Path("data")

MATRIX_FILE = DATA / "defence_matrix.csv"
DEFENCE_FILE = Path("08_defences.py")

WIKITEXT_FILE = DATA / "wikitext_heldout_5000.csv"
JIGSAW_FILE = DATA / "jigsaw_benign_250.csv"

OUT = Path("results") / "step10"
FIG_DIR = OUT / "figures"

ATTACK_METRICS_FILE = OUT / "attack_metrics.csv"
CLEAN_CONTROLS_FILE = OUT / "clean_controls.csv"
CLEAN_SUMMARY_FILE = OUT / "clean_summary.csv"
DETECTION_FILE = OUT / "detection_metrics.csv"
LITERATURE_FILE = OUT / "literature_notes.txt"
REPORT_FILE = OUT / "report.html"

# change this to True only if you deliberately want to rerun
# the 5,250 clean-control examples after they have been cached.
FORCE_CLEAN_RECOMPUTE = False

# automatically open the final HTML report in the browser
OPEN_REPORT = True


# these are detectors whose alarm means the request is blocked.
#
# unicode_sanitise can also block Bidi input, but it does that by
# returning None, so it does not need to be listed here.
FILTER_DEFENCES = {
    "canonical_reject",
    "cpt_global",
    "cpt_window",
}


ATTACK_ORDER = [
    "tokenbreak",
    "advtok",
    "unicode_compat",
    "unicode_homoglyph",
    "unicode_invisible",
    "unicode_reorder",
]


DEFENCE_ORDER = [
    "tokenizer_translation",
    "canonical_reject",
    "canonical_replace",
    "unicode_sanitise",
    "nfkc_confusable",
    "cpt_global",
    "cpt_window",
]


ATTACK_LABELS = {
    "tokenbreak": "TokenBreak",
    "advtok": "AdvTok",
    "unicode_compat": "Unicode\ncompatibility",
    "unicode_homoglyph": "Unicode\nhomoglyph",
    "unicode_invisible": "Unicode\ninvisible",
    "unicode_reorder": "Unicode\nreorder",
}


DEFENCE_LABELS = {
    "tokenizer_translation": "Tokenizer\ntranslation",
    "canonical_reject": "Canonical\nreject",
    "canonical_replace": "Canonical\nreplace",
    "unicode_sanitise": "Unicode\nsanitiser",
    "nfkc_confusable": "NFKC +\nconfusables",
    "cpt_global": "Global\nCPT",
    "cpt_window": "Window\nCPT",
}


OUT.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------- literature

LITERATURE_TEXT = """
LITERATURE AND METHOD PROVENANCE
================================

1. TokenBreak
   Kasimir Schulz, Kenneth Yeung, Kieran Evans.
   TokenBreak: Bypassing Text Classification Models Through Token Manipulation.
   arXiv:2506.07948, 2025.
   https://arxiv.org/abs/2506.07948

   Relevant defence:
   Tokenizer Translation / MapUnigramToWordPiece.

   Project adaptation:
   XLNet supplies the Unigram segmentation while the victim is an uncased
   DistilBERT WordPiece classifier.


2. Adversarial Tokenization
   Renato Lui Geh, Zilei Shao, Guy Van den Broeck.
   Adversarial Tokenization.
   ACL 2025.
   https://aclanthology.org/2025.acl-long.1012/

   Relevant defence discussion:
   Retokenize inputs and/or restrict the interface to strings.

   Project variants:
   - canonical_reject
   - canonical_replace

   canonical_replace is closest to retokenization.
   canonical_reject is a stricter deployment-policy variant.


3. Bad Characters
   Nicholas Boucher, Ilia Shumailov, Ross Anderson, Nicolas Papernot.
   Bad Characters: Imperceptible NLP Attacks.
   IEEE Symposium on Security and Privacy.
   https://arxiv.org/abs/2106.09898

   Project relevance:
   motivates defence against invisible characters, homoglyphs,
   Unicode control characters and reordering attacks.


4. Unicode Technical Standard #39
   Unicode Security Mechanisms.
   https://www.unicode.org/reports/tr39/

   Project relevance:
   confusable-character mapping/detection.


5. Broken-Token
   Shaked Zychlinski, Yuval Kainan.
   Broken-Token: Filtering Obfuscated Prompts by Counting Characters-Per-Token.
   arXiv:2510.26847, 2025.
   https://arxiv.org/abs/2510.26847

   Project relevance:
   - global characters-per-token filtering
   - sliding-window CPT filtering

   Important project adaptations:
   - paper focuses on BPE; this project evaluates WordPiece
   - paper optimises thresholds using regular + obfuscated data
   - project threshold is calibrated using CLEAN WikiText only
   - target calibration FPR = 1%
   - 5-token sliding CPT degenerated on WordPiece
   - project therefore uses a documented 10-token WordPiece adaptation


6. Bootstrap statistics
   Bradley Efron.
   Bootstrap Methods: Another Look at the Jackknife.
   Annals of Statistics, 1979.

   Project use:
   95% percentile bootstrap confidence intervals.

   For paired ASR comparisons, clean and attacked outcomes are resampled
   together so the correspondence between the two observations is preserved.
"""


# --------------------------------------------------------------- load defence library

def load_defence_module():
    """Load 08_defences.py despite the filename starting with a number."""
    if not DEFENCE_FILE.exists():
        raise FileNotFoundError(
            f"{DEFENCE_FILE} was not found."
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

tok = defence_lib.tok


# --------------------------------------------------------------- model

device = "cuda" if torch.cuda.is_available() else "cpu"

_model = None


def get_model():
    """Load the victim only when clean-control inference is required."""
    global _model

    if _model is None:
        print("loading victim model...")
        print("device:", device)

        _model = AutoModelForSequenceClassification.from_pretrained(
            VICTIM
        ).to(device)

        _model.eval()

    return _model


CLS = tok.cls_token_id
SEP = tok.sep_token_id
PAD = tok.pad_token_id


# --------------------------------------------------------------- helpers

def stable_seed(*parts):
    """Deterministic seed independent of Python's randomized hash()."""
    text = "|".join(str(x) for x in parts)

    return (
        SEED
        + zlib.crc32(text.encode("utf-8"))
    ) % (2**32 - 1)


def is_blocked(defence_name, flagged, out_ids):
    """Apply the same deployment policy used in Step 9."""
    if out_ids is None:
        return 1

    if (
        defence_name in FILTER_DEFENCES
        and int(flagged) == 1
    ):
        return 1

    return 0


@torch.no_grad()
def p_toxic(id_seqs):
    """Return victim P(toxic) for raw token-id sequences."""
    if not id_seqs:
        return []

    model = get_model()

    probs = []

    for i in range(0, len(id_seqs), BATCH):
        chunk = id_seqs[i:i + BATCH]

        chunk = [
            list(ids)[:MAX_TOKENS]
            for ids in chunk
        ]

        width = max(
            len(ids)
            for ids in chunk
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
        )[:, 1]

        probs.extend(
            p.cpu().tolist()
        )

    return probs


def percentile_ci(values):
    """Simple 95% percentile interval."""
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return np.nan, np.nan

    return (
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def bootstrap_rate(values, seed):
    """Bootstrap CI for a binary/rate statistic without huge memory use."""
    values = np.asarray(values, dtype=float)

    n = len(values)

    if n == 0:
        return np.nan, np.nan, np.nan

    point = float(values.mean())

    rng = np.random.default_rng(seed)

    boot = np.empty(
        N_BOOTSTRAP,
        dtype=float,
    )

    # chunking avoids allocating a giant 5000 x 5000 matrix
    chunk_size = 200

    for start in range(
        0,
        N_BOOTSTRAP,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            N_BOOTSTRAP,
        )

        m = end - start

        idx = rng.integers(
            0,
            n,
            size=(m, n),
        )

        boot[start:end] = (
            values[idx].mean(axis=1)
        )

    lo, hi = percentile_ci(boot)

    return point, lo, hi


def bootstrap_paired(undefended, defended, seed):
    """Paired bootstrap for before-vs-after attack success."""
    u = np.asarray(
        undefended,
        dtype=float,
    )

    d = np.asarray(
        defended,
        dtype=float,
    )

    if len(u) != len(d):
        raise ValueError(
            "paired bootstrap arrays differ in length"
        )

    n = len(u)

    if n == 0:
        return {
            "u": np.nan,
            "u_lo": np.nan,
            "u_hi": np.nan,

            "d": np.nan,
            "d_lo": np.nan,
            "d_hi": np.nan,

            "reduction": np.nan,
            "reduction_lo": np.nan,
            "reduction_hi": np.nan,
        }

    point_u = float(u.mean())
    point_d = float(d.mean())

    point_reduction = (
        point_u - point_d
    )

    rng = np.random.default_rng(seed)

    boot_u = np.empty(N_BOOTSTRAP)
    boot_d = np.empty(N_BOOTSTRAP)
    boot_r = np.empty(N_BOOTSTRAP)

    chunk_size = 500

    for start in range(
        0,
        N_BOOTSTRAP,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            N_BOOTSTRAP,
        )

        m = end - start

        # SAME indices for both arrays.
        # this is the important paired part.
        idx = rng.integers(
            0,
            n,
            size=(m, n),
        )

        bu = u[idx].mean(axis=1)
        bd = d[idx].mean(axis=1)

        boot_u[start:end] = bu
        boot_d[start:end] = bd
        boot_r[start:end] = bu - bd

    u_lo, u_hi = percentile_ci(boot_u)
    d_lo, d_hi = percentile_ci(boot_d)
    r_lo, r_hi = percentile_ci(boot_r)

    return {
        "u": point_u,
        "u_lo": u_lo,
        "u_hi": u_hi,

        "d": point_d,
        "d_lo": d_lo,
        "d_hi": d_hi,

        "reduction": point_reduction,
        "reduction_lo": r_lo,
        "reduction_hi": r_hi,
    }


# --------------------------------------------------------------- validate matrix

def load_matrix():
    """Load the frozen paired Step 9 matrix."""
    if not MATRIX_FILE.exists():
        raise FileNotFoundError(
            f"{MATRIX_FILE} not found. "
            "Run the final paired Step 9 first."
        )

    matrix = pd.read_csv(
        MATRIX_FILE
    )

    required = {
        "sample_id",
        "attack",
        "defence",
        "undefended_attack_success",
        "defended_attack_success",
        "clean_retained",
        "attack_defence_flagged",
        "attack_blocked_by_policy",
        "attack_mechanism",
        "attack_defence_latency_ms",
    }

    missing = required - set(
        matrix.columns
    )

    if missing:
        raise RuntimeError(
            "defence matrix is missing columns: "
            + ", ".join(sorted(missing))
        )

    print("\nloaded paired matrix")
    print("  rows:", len(matrix))
    print("  attacks:", matrix["attack"].nunique())
    print("  defences:", matrix["defence"].nunique())

    if len(matrix) != 6552:
        print(
            "WARNING: expected 6552 rows from the frozen experiment."
        )

    return matrix


# --------------------------------------------------------------- attack analysis

def analyse_attack_matrix(matrix):
    """Compute paired attack metrics and bootstrap confidence intervals."""
    print("\ncomputing paired attack statistics...")

    records = []

    groups = matrix.groupby(
        ["attack", "defence"],
        sort=False,
    )

    total = len(groups)

    for number, ((attack, defence), g) in enumerate(
        groups,
        start=1,
    ):
        # ------------------------------------------------ paired denominator
        #
        # only use rows where this defence preserved the clean toxic
        # decision. This guarantees that undefended and defended ASR
        # are compared on exactly the same samples.

        retained = g[
            g["clean_retained"] == 1
        ].copy()

        paired = bootstrap_paired(
            retained[
                "undefended_attack_success"
            ].to_numpy(),

            retained[
                "defended_attack_success"
            ].to_numpy(),

            stable_seed(
                attack,
                defence,
                "paired",
            ),
        )

        # ------------------------------------------------ clean retention

        clean_point, clean_lo, clean_hi = bootstrap_rate(
            g["clean_retained"].to_numpy(),

            stable_seed(
                attack,
                defence,
                "clean",
            ),
        )

        # ------------------------------------------------ detection rate

        detect_point, detect_lo, detect_hi = bootstrap_rate(
            g[
                "attack_defence_flagged"
            ].to_numpy(),

            stable_seed(
                attack,
                defence,
                "detect",
            ),
        )

        # ------------------------------------------------ block rate

        block_point, block_lo, block_hi = bootstrap_rate(
            g[
                "attack_blocked_by_policy"
            ].to_numpy(),

            stable_seed(
                attack,
                defence,
                "block",
            ),
        )

        # ------------------------------------------------ re-encoding rate

        reencoded = (
            g["attack_mechanism"]
            .fillna("")
            .str.contains(
                "reencode",
                regex=False,
            )
            .astype(int)
            .to_numpy()
        )

        reencode_point, reencode_lo, reencode_hi = bootstrap_rate(
            reencoded,

            stable_seed(
                attack,
                defence,
                "reencode",
            ),
        )

        records.append({
            "attack": attack,
            "defence": defence,

            "n_total": len(g),
            "n_clean_retained": len(retained),

            # the original ASR over all 156 is kept only as context
            "original_full_undefended_asr": float(
                g[
                    "undefended_attack_success"
                ].mean()
            ),

            # these are the MAIN paired results
            "paired_undefended_asr": paired["u"],
            "paired_undefended_ci_low": paired["u_lo"],
            "paired_undefended_ci_high": paired["u_hi"],

            "paired_defended_asr": paired["d"],
            "paired_defended_ci_low": paired["d_lo"],
            "paired_defended_ci_high": paired["d_hi"],

            "asr_reduction": paired["reduction"],
            "asr_reduction_ci_low": paired["reduction_lo"],
            "asr_reduction_ci_high": paired["reduction_hi"],

            # percentage-point version for plots/reporting
            "asr_reduction_pp": paired["reduction"] * 100,
            "asr_reduction_ci_low_pp": paired["reduction_lo"] * 100,
            "asr_reduction_ci_high_pp": paired["reduction_hi"] * 100,

            "clean_retention": clean_point,
            "clean_retention_ci_low": clean_lo,
            "clean_retention_ci_high": clean_hi,

            "attack_detection_rate": detect_point,
            "attack_detection_ci_low": detect_lo,
            "attack_detection_ci_high": detect_hi,

            "attack_block_rate": block_point,
            "attack_block_ci_low": block_lo,
            "attack_block_ci_high": block_hi,

            "reencode_rate": reencode_point,
            "reencode_ci_low": reencode_lo,
            "reencode_ci_high": reencode_hi,

            "median_attack_defence_latency_ms": float(
                g[
                    "attack_defence_latency_ms"
                ].median()
            ),

            "mean_attack_defence_latency_ms": float(
                g[
                    "attack_defence_latency_ms"
                ].mean()
            ),
        })

        print(
            f"  {number}/{total}",
            end="\r",
        )

    print()

    out = pd.DataFrame(records)

    # stable order for tables and plots
    out["attack"] = pd.Categorical(
        out["attack"],
        ATTACK_ORDER,
        ordered=True,
    )

    out["defence"] = pd.Categorical(
        out["defence"],
        DEFENCE_ORDER,
        ordered=True,
    )

    out = (
        out
        .sort_values(
            ["attack", "defence"]
        )
        .reset_index(drop=True)
    )

    out.to_csv(
        ATTACK_METRICS_FILE,
        index=False,
    )

    print(
        "wrote",
        ATTACK_METRICS_FILE,
    )

    return out


# --------------------------------------------------------------- clean controls

def prepare_corpora():
    """Load the two clean held-out corpora."""
    if not WIKITEXT_FILE.exists():
        raise FileNotFoundError(
            f"{WIKITEXT_FILE} not found."
        )

    if not JIGSAW_FILE.exists():
        raise FileNotFoundError(
            f"{JIGSAW_FILE} not found."
        )

    wiki = pd.read_csv(
        WIKITEXT_FILE
    )

    jig = pd.read_csv(
        JIGSAW_FILE
    )

    return [
        (
            "wikitext_heldout",
            wiki[
                ["sample_id", "text"]
            ].copy(),
        ),
        (
            "jigsaw_benign",
            jig[
                ["sample_id", "text"]
            ].copy(),
        ),
    ]


def run_clean_controls():
    """Run every defence on held-out clean inputs."""
    if (
        CLEAN_CONTROLS_FILE.exists()
        and not FORCE_CLEAN_RECOMPUTE
    ):
        print(
            "\nusing cached clean controls:",
            CLEAN_CONTROLS_FILE,
        )

        return pd.read_csv(
            CLEAN_CONTROLS_FILE
        )

    corpora = prepare_corpora()

    all_rows = []

    for corpus_name, df in corpora:
        print(
            f"\nclean corpus: {corpus_name} "
            f"({len(df)} rows)"
        )

        texts = (
            df["text"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        # ------------------------------------------------ baseline classifier

        print("  scoring baseline...")

        canonical_ids = [
            defence_lib.encode(text)
            for text in texts
        ]

        baseline_probs = p_toxic(
            canonical_ids
        )

        baseline_labels = [
            int(p >= TOXIC_THRESHOLD)
            for p in baseline_probs
        ]

        # ------------------------------------------------ each defence

        for defence_number, defence_name in enumerate(
            DEFENCE_ORDER,
            start=1,
        ):
            print(
                f"  defence {defence_number}/"
                f"{len(DEFENCE_ORDER)}: {defence_name}"
            )

            defend = defences[
                defence_name
            ]

            local_rows = []
            sequences = []
            sequence_rows = []

            for i, text in enumerate(texts):
                t0 = time.perf_counter()

                # held-out controls use the ordinary STRING interface.
                #
                # canonical-ID enforcement therefore has nothing
                # noncanonical to reject here.
                out_ids, flagged, mechanism = defend(
                    text,
                    None,
                )

                latency_ms = (
                    time.perf_counter()
                    - t0
                ) * 1000

                blocked = is_blocked(
                    defence_name,
                    flagged,
                    out_ids,
                )

                row = {
                    "corpus": corpus_name,
                    "sample_id": df.iloc[i]["sample_id"],
                    "defence": defence_name,

                    "baseline_toxic_prob": baseline_probs[i],
                    "baseline_label": baseline_labels[i],

                    "flagged": int(flagged),
                    "blocked": int(blocked),
                    "mechanism": mechanism,

                    "defence_latency_ms": latency_ms,

                    "defended_toxic_prob": np.nan,
                    "defended_label": np.nan,

                    "label_agreement": np.nan,
                    "operational_preservation": 0,

                    "benign_usable": np.nan,
                    "absolute_probability_shift": np.nan,
                }

                local_rows.append(row)

                # even a CPT-flagged input still has IDs.
                #
                # score it so we can distinguish detector blocking
                # from changes in classifier behaviour.
                if out_ids is not None:
                    sequences.append(
                        out_ids
                    )

                    sequence_rows.append(i)

            # batch inference for this defence
            defended_probs = p_toxic(
                sequences
            )

            for local_index, prob in zip(
                sequence_rows,
                defended_probs,
            ):
                row = local_rows[
                    local_index
                ]

                label = int(
                    prob >= TOXIC_THRESHOLD
                )

                row[
                    "defended_toxic_prob"
                ] = prob

                row[
                    "defended_label"
                ] = label

                row[
                    "label_agreement"
                ] = int(
                    label
                    == row["baseline_label"]
                )

                row[
                    "absolute_probability_shift"
                ] = abs(
                    prob
                    - row[
                        "baseline_toxic_prob"
                    ]
                )

                # operational preservation isolates harm introduced
                # by the defence:
                #
                #     not blocked
                #     AND
                #     same classifier decision as before defence
                row[
                    "operational_preservation"
                ] = int(
                    row["blocked"] == 0
                    and label
                    == row["baseline_label"]
                )

                # Jigsaw benign has a true benign label.
                if corpus_name == "jigsaw_benign":
                    row[
                        "benign_usable"
                    ] = int(
                        row["blocked"] == 0
                        and label == 0
                    )

            all_rows.extend(
                local_rows
            )

    controls = pd.DataFrame(
        all_rows
    )

    controls.to_csv(
        CLEAN_CONTROLS_FILE,
        index=False,
    )

    print(
        "\nwrote",
        CLEAN_CONTROLS_FILE,
    )

    return controls


# --------------------------------------------------------------- clean summary

def analyse_clean_controls(controls):
    """Summarise clean false alarms, false blocks and utility."""
    print(
        "\ncomputing held-out clean statistics..."
    )

    records = []

    groups = controls.groupby(
        ["corpus", "defence"],
        sort=False,
    )

    for (
        corpus,
        defence,
    ), g in groups:

        # ------------------------------------------------ false alarm

        alarm, alarm_lo, alarm_hi = bootstrap_rate(
            g["flagged"].to_numpy(),

            stable_seed(
                corpus,
                defence,
                "alarm",
            ),
        )

        # ------------------------------------------------ false block

        block, block_lo, block_hi = bootstrap_rate(
            g["blocked"].to_numpy(),

            stable_seed(
                corpus,
                defence,
                "false_block",
            ),
        )

        # ------------------------------------------------ baseline preservation

        preservation, preservation_lo, preservation_hi = bootstrap_rate(
            g[
                "operational_preservation"
            ].to_numpy(),

            stable_seed(
                corpus,
                defence,
                "preservation",
            ),
        )

        scored = g[
            g["defended_label"].notna()
        ].copy()

        if len(scored):
            agreement = float(
                scored[
                    "label_agreement"
                ].mean()
            )

            prob_shift = float(
                scored[
                    "absolute_probability_shift"
                ].mean()
            )

            defended_toxic_rate = float(
                scored[
                    "defended_label"
                ].mean()
            )

        else:
            agreement = np.nan
            prob_shift = np.nan
            defended_toxic_rate = np.nan

        # true-benign utility is only available for Jigsaw.
        if corpus == "jigsaw_benign":
            usable = float(
                g[
                    "benign_usable"
                ].mean()
            )
        else:
            usable = np.nan

        records.append({
            "corpus": corpus,
            "defence": defence,
            "n": len(g),

            # original classifier behaviour before defence
            "baseline_toxic_rate": float(
                g[
                    "baseline_label"
                ].mean()
            ),

            # defence detection FPR on clean input
            "false_alarm_rate": alarm,
            "false_alarm_ci_low": alarm_lo,
            "false_alarm_ci_high": alarm_hi,

            # deployment false-block rate on clean input
            "false_block_rate": block,
            "false_block_ci_low": block_lo,
            "false_block_ci_high": block_hi,

            # did the defence preserve the original classifier decision?
            "operational_preservation": preservation,
            "operational_preservation_ci_low": preservation_lo,
            "operational_preservation_ci_high": preservation_hi,

            "label_agreement_when_scored": agreement,

            "defended_toxic_rate_when_scored": defended_toxic_rate,

            "mean_abs_probability_shift": prob_shift,

            # only meaningful for Jigsaw true-benign controls
            "jigsaw_benign_usable_rate": usable,

            "median_defence_latency_ms": float(
                g[
                    "defence_latency_ms"
                ].median()
            ),
        })

    summary = pd.DataFrame(
        records
    )

    summary["defence"] = pd.Categorical(
        summary["defence"],
        DEFENCE_ORDER,
        ordered=True,
    )

    summary = (
        summary
        .sort_values(
            ["corpus", "defence"]
        )
        .reset_index(drop=True)
    )

    summary.to_csv(
        CLEAN_SUMMARY_FILE,
        index=False,
    )

    print(
        "wrote",
        CLEAN_SUMMARY_FILE,
    )

    return summary


# --------------------------------------------------------------- detector precision / F1

def calculate_detection_metrics(matrix, controls):
    """Combine attack flags with true-benign Jigsaw controls."""
    print(
        "\ncomputing detector precision/F1..."
    )

    benign = controls[
        controls["corpus"]
        == "jigsaw_benign"
    ]

    records = []

    for (
        attack,
        defence,
    ), attack_rows in matrix.groupby(
        ["attack", "defence"],
        sort=False,
    ):
        clean_rows = benign[
            benign["defence"]
            == defence
        ]

        tp = int(
            attack_rows[
                "attack_defence_flagged"
            ].sum()
        )

        fn = (
            len(attack_rows)
            - tp
        )

        fp = int(
            clean_rows[
                "flagged"
            ].sum()
        )

        tn = (
            len(clean_rows)
            - fp
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn)
            else np.nan
        )

        fpr = (
            fp / (fp + tn)
            if (fp + tn)
            else np.nan
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp)
            else np.nan
        )

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
            if (
                not np.isnan(precision)
                and not np.isnan(recall)
                and precision + recall > 0
            )
            else np.nan
        )

        records.append({
            "attack": attack,
            "defence": defence,

            "attack_n": len(
                attack_rows
            ),

            "benign_n": len(
                clean_rows
            ),

            "true_positive": tp,
            "false_negative": fn,
            "false_positive": fp,
            "true_negative": tn,

            "detection_rate_recall": recall,
            "benign_false_alarm_rate": fpr,
            "precision": precision,
            "f1": f1,
        })

    result = pd.DataFrame(
        records
    )

    result["attack"] = pd.Categorical(
        result["attack"],
        ATTACK_ORDER,
        ordered=True,
    )

    result["defence"] = pd.Categorical(
        result["defence"],
        DEFENCE_ORDER,
        ordered=True,
    )

    result = (
        result
        .sort_values(
            ["attack", "defence"]
        )
        .reset_index(drop=True)
    )

    result.to_csv(
        DETECTION_FILE,
        index=False,
    )

    print(
        "wrote",
        DETECTION_FILE,
    )

    return result


# --------------------------------------------------------------- plotting style

def set_plot_style():
    """Simple publication-friendly matplotlib defaults."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,

        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,

        "xtick.labelsize": 9,
        "ytick.labelsize": 9,

        "figure.facecolor": "white",
        "axes.facecolor": "white",

        "axes.spines.top": False,
        "axes.spines.right": False,

        "axes.grid": False,
    })


def save_figure(fig, name):
    """Save every figure as PNG and vector PDF."""
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"

    fig.savefig(
        png,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(fig)


def matrix_for_plot(
    df,
    value,
):
    """Attack x defence matrix in frozen order."""
    pivot = df.pivot(
        index="attack",
        columns="defence",
        values=value,
    )

    return pivot.reindex(
        index=ATTACK_ORDER,
        columns=DEFENCE_ORDER,
    )


def draw_heatmap(
    values,
    title,
    colour_map,
    output_name,
    percent=True,
    vmin=0,
    vmax=1,
    norm=None,
    colourbar_label=None,
):
    """Draw an annotated heatmap without requiring seaborn."""
    arr = values.to_numpy(
        dtype=float
    )

    fig, ax = plt.subplots(
        figsize=(12.5, 6.5)
    )

    if norm is None:
        image = ax.imshow(
            arr,
            aspect="auto",
            cmap=colour_map,
            vmin=vmin,
            vmax=vmax,
        )
    else:
        image = ax.imshow(
            arr,
            aspect="auto",
            cmap=colour_map,
            norm=norm,
        )

    ax.set_title(
        title,
        pad=14,
        fontweight="bold",
    )

    ax.set_xticks(
        np.arange(
            len(DEFENCE_ORDER)
        )
    )

    ax.set_xticklabels(
        [
            DEFENCE_LABELS[x]
            for x in DEFENCE_ORDER
        ]
    )

    ax.set_yticks(
        np.arange(
            len(ATTACK_ORDER)
        )
    )

    ax.set_yticklabels(
        [
            ATTACK_LABELS[x]
            for x in ATTACK_ORDER
        ]
    )

    for i in range(
        len(ATTACK_ORDER)
    ):
        for j in range(
            len(DEFENCE_ORDER)
        ):
            value = arr[i, j]

            if np.isnan(value):
                text = "—"
            elif percent:
                text = (
                    f"{value * 100:.1f}%"
                )
            else:
                text = f"{value:.1f}"

            # simple contrast choice
            colour = (
                "white"
                if (
                    not np.isnan(value)
                    and abs(value) > 0.55
                )
                else "black"
            )

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color=colour,
            )

    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.025,
        pad=0.025,
    )

    if colourbar_label:
        cbar.set_label(
            colourbar_label
        )

    fig.tight_layout()

    save_figure(
        fig,
        output_name,
    )


# --------------------------------------------------------------- figures

def make_plots(
    attack_metrics,
    clean_summary,
    detection_metrics,
    matrix,
):
    """Create all main publication figures."""
    print(
        "\ncreating figures..."
    )

    set_plot_style()

    # ----------------------------------------------------------- figure 1
    #
    # main outcome:
    # attack success remaining after defence

    defended = matrix_for_plot(
        attack_metrics,
        "paired_defended_asr",
    )

    draw_heatmap(
        defended,
        "Defended attack success rate",
        "viridis_r",
        "fig01_defended_asr_heatmap",
        percent=True,
        vmin=0,
        vmax=1,
        colourbar_label="Attack success rate",
    )

    # ----------------------------------------------------------- figure 2
    #
    # positive values = defence helped
    # negative values = defence made things worse

    reduction = matrix_for_plot(
        attack_metrics,
        "asr_reduction_pp",
    )

    arr = reduction.to_numpy(
        dtype=float
    )

    max_abs = np.nanmax(
        np.abs(arr)
    )

    if (
        not np.isfinite(max_abs)
        or max_abs == 0
    ):
        max_abs = 1

    draw_heatmap(
        reduction,
        "Change in attack success after defence",
        "RdYlGn",
        "fig02_asr_reduction_heatmap",
        percent=False,
        norm=TwoSlopeNorm(
            vmin=-max_abs,
            vcenter=0,
            vmax=max_abs,
        ),
        colourbar_label="ASR reduction (percentage points)",
    )

    # ----------------------------------------------------------- figure 3

    detection = matrix_for_plot(
        attack_metrics,
        "attack_detection_rate",
    )

    draw_heatmap(
        detection,
        "Attack detection rate",
        "Blues",
        "fig03_detection_heatmap",
        percent=True,
        vmin=0,
        vmax=1,
        colourbar_label="Detection rate",
    )

    # ----------------------------------------------------------- figure 4
    #
    # this is the mechanism-aware AdvTok evidence:
    # defence success caused by throwing away supplied IDs

    reencode = matrix_for_plot(
        attack_metrics,
        "reencode_rate",
    )

    draw_heatmap(
        reencode,
        "How often the defence re-encoded the input",
        "Purples",
        "fig04_reencoding_heatmap",
        percent=True,
        vmin=0,
        vmax=1,
        colourbar_label="Re-encoding rate",
    )

    # ----------------------------------------------------------- figure 5

    retention = matrix_for_plot(
        attack_metrics,
        "clean_retention",
    )

    draw_heatmap(
        retention,
        "Clean toxic decision retained before attack comparison",
        "YlGn",
        "fig05_clean_retention_heatmap",
        percent=True,
        vmin=0,
        vmax=1,
        colourbar_label="Clean retention",
    )

    # ----------------------------------------------------------- figure 6
    #
    # 95% paired bootstrap CI for the ASR reduction

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9),
    )

    axes = axes.flatten()

    global_low = float(
        attack_metrics[
            "asr_reduction_ci_low_pp"
        ].min()
    )

    global_high = float(
        attack_metrics[
            "asr_reduction_ci_high_pp"
        ].max()
    )

    padding = 5

    for ax, attack in zip(
        axes,
        ATTACK_ORDER,
    ):
        sub = (
            attack_metrics[
                attack_metrics[
                    "attack"
                ].astype(str)
                == attack
            ]
            .copy()
        )

        sub["defence_str"] = (
            sub["defence"]
            .astype(str)
        )

        sub = (
            sub
            .set_index(
                "defence_str"
            )
            .reindex(
                DEFENCE_ORDER
            )
        )

        y = np.arange(
            len(sub)
        )

        x = (
            sub[
                "asr_reduction_pp"
            ].to_numpy()
        )

        low = (
            sub[
                "asr_reduction_ci_low_pp"
            ].to_numpy()
        )

        high = (
            sub[
                "asr_reduction_ci_high_pp"
            ].to_numpy()
        )

        xerr = np.vstack([
            x - low,
            high - x,
        ])

        ax.errorbar(
            x,
            y,
            xerr=xerr,
            fmt="o",
            capsize=3,
            linewidth=1.3,
        )

        ax.axvline(
            0,
            linestyle="--",
            linewidth=1,
            alpha=0.6,
        )

        ax.set_yticks(y)

        ax.set_yticklabels([
            DEFENCE_LABELS[d]
            .replace("\n", " ")
            for d in DEFENCE_ORDER
        ])

        ax.invert_yaxis()

        ax.set_title(
            ATTACK_LABELS[attack]
            .replace("\n", " "),
            fontweight="bold",
        )

        ax.set_xlim(
            global_low - padding,
            global_high + padding,
        )

        ax.set_xlabel(
            "ASR reduction (percentage points)"
        )

        ax.grid(
            axis="x",
            alpha=0.2,
        )

    fig.suptitle(
        "Paired ASR reduction with 95% bootstrap confidence intervals",
        fontsize=15,
        fontweight="bold",
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    save_figure(
        fig,
        "fig06_asr_reduction_bootstrap_ci",
    )

    # ----------------------------------------------------------- figure 7
    #
    # held-out clean false alarms / blocking / preservation

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5.5),
    )

    x = np.arange(
        len(DEFENCE_ORDER)
    )

    width = 0.36

    corpora = [
        "jigsaw_benign",
        "wikitext_heldout",
    ]

    corpus_labels = {
        "jigsaw_benign": "Jigsaw benign",
        "wikitext_heldout": "WikiText held-out",
    }

    fields = [
        (
            "false_alarm_rate",
            "Clean false-alarm rate",
        ),
        (
            "false_block_rate",
            "Clean false-block rate",
        ),
        (
            "operational_preservation",
            "Baseline decision preservation",
        ),
    ]

    for ax, (
        field,
        title,
    ) in zip(
        axes,
        fields,
    ):
        for c_index, corpus in enumerate(
            corpora
        ):
            sub = (
                clean_summary[
                    clean_summary[
                        "corpus"
                    ] == corpus
                ]
                .copy()
            )

            sub["defence_str"] = (
                sub["defence"]
                .astype(str)
            )

            sub = (
                sub
                .set_index(
                    "defence_str"
                )
                .reindex(
                    DEFENCE_ORDER
                )
            )

            values = (
                sub[field]
                .to_numpy()
                * 100
            )

            offset = (
                -width / 2
                if c_index == 0
                else width / 2
            )

            ax.bar(
                x + offset,
                values,
                width,
                label=corpus_labels[
                    corpus
                ],
            )

        ax.set_title(
            title,
            fontweight="bold",
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            [
                DEFENCE_LABELS[d]
                .replace("\n", " ")
                for d in DEFENCE_ORDER
            ],
            rotation=45,
            ha="right",
        )

        ax.set_ylabel(
            "Percent"
        )

        ax.set_ylim(
            0,
            105,
        )

        ax.grid(
            axis="y",
            alpha=0.2,
        )

    axes[0].legend(
        frameon=False,
    )

    fig.suptitle(
        "Held-out clean-input cost of each defence",
        fontsize=15,
        fontweight="bold",
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    save_figure(
        fig,
        "fig07_clean_control_rates",
    )

    # ----------------------------------------------------------- figure 8
    #
    # latency of the defence itself, excluding classifier inference

    latency = (
        matrix
        .groupby(
            "defence"
        )[
            "attack_defence_latency_ms"
        ]
        .median()
        .reindex(
            DEFENCE_ORDER
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 5.5)
    )

    ax.bar(
        np.arange(
            len(latency)
        ),
        latency.to_numpy(),
    )

    ax.set_xticks(
        np.arange(
            len(latency)
        )
    )

    ax.set_xticklabels(
        [
            DEFENCE_LABELS[x]
            .replace("\n", " ")
            for x in DEFENCE_ORDER
        ],
        rotation=35,
        ha="right",
    )

    ax.set_ylabel(
        "Median preprocessing latency (ms)"
    )

    ax.set_title(
        "Defence preprocessing cost",
        fontweight="bold",
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "fig08_defence_latency",
    )

    # ----------------------------------------------------------- figure 9
    #
    # precision/F1 is secondary because it depends on the attack/benign
    # class mixture. TPR/FPR remain the primary detector metrics.

    f1 = matrix_for_plot(
        detection_metrics,
        "f1",
    )

    draw_heatmap(
        f1,
        "Flag-based attack-detection F1 against Jigsaw benign controls",
        "YlGnBu",
        "fig09_detection_f1_heatmap",
        percent=False,
        vmin=0,
        vmax=1,
        colourbar_label="F1",
    )

    # ----------------------------------------------------------- figure 10
    #
    # high-level effectiveness / utility trade-off.
    #
    # exclude attacks whose paired undefended ASR is zero because there is
    # no positive ASR for a defence to reduce.

    active = attack_metrics[
        attack_metrics[
            "paired_undefended_asr"
        ] > 0
    ]

    mean_reduction = (
        active
        .groupby(
            "defence",
            observed=True,
        )[
            "asr_reduction_pp"
        ]
        .mean()
        .reindex(
            DEFENCE_ORDER
        )
    )

    jig = (
        clean_summary[
            clean_summary[
                "corpus"
            ] == "jigsaw_benign"
        ]
        .copy()
    )

    jig["defence_str"] = (
        jig["defence"]
        .astype(str)
    )

    jig = (
        jig
        .set_index(
            "defence_str"
        )
        .reindex(
            DEFENCE_ORDER
        )
    )

    preservation = (
        jig[
            "operational_preservation"
        ]
        * 100
    )

    fig, ax = plt.subplots(
        figsize=(9, 6.5)
    )

    ax.scatter(
        preservation,
        mean_reduction,
        s=70,
    )

    for defence in DEFENCE_ORDER:
        x_value = preservation.loc[
            defence
        ]

        y_value = mean_reduction.loc[
            defence
        ]

        ax.annotate(
            DEFENCE_LABELS[
                defence
            ].replace("\n", " "),
            (
                x_value,
                y_value,
            ),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
        alpha=0.5,
    )

    ax.set_xlabel(
        "Jigsaw baseline-decision preservation (%)"
    )

    ax.set_ylabel(
        "Mean paired ASR reduction (percentage points)"
    )

    ax.set_title(
        "Defence effectiveness vs clean utility",
        fontweight="bold",
    )

    ax.grid(
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "fig10_effectiveness_utility_tradeoff",
    )

    print(
        "figures written to",
        FIG_DIR,
    )


# --------------------------------------------------------------- report helpers

def pct(value, digits=1):
    if pd.isna(value):
        return "—"

    return (
        f"{float(value) * 100:.{digits}f}%"
    )


def number(value, digits=2):
    if pd.isna(value):
        return "—"

    return (
        f"{float(value):.{digits}f}"
    )


def attack_table_html(metrics):
    """Readable main-results table for HTML report."""
    table = metrics.copy()

    table["Attack"] = (
        table["attack"]
        .astype(str)
        .map(
            lambda x: ATTACK_LABELS[x]
            .replace("\n", " ")
        )
    )

    table["Defence"] = (
        table["defence"]
        .astype(str)
        .map(
            lambda x: DEFENCE_LABELS[x]
            .replace("\n", " ")
        )
    )

    table["N retained"] = (
        table["n_clean_retained"]
    )

    table["Undefended ASR"] = (
        table[
            "paired_undefended_asr"
        ]
        .map(pct)
    )

    table["Defended ASR"] = (
        table[
            "paired_defended_asr"
        ]
        .map(pct)
    )

    table["ASR reduction"] = (
        table[
            "asr_reduction_pp"
        ]
        .map(
            lambda x:
            "—"
            if pd.isna(x)
            else f"{x:.1f} pp"
        )
    )

    table["95% CI reduction"] = [
        (
            "—"
            if pd.isna(lo)
            else (
                f"[{lo:.1f}, "
                f"{hi:.1f}] pp"
            )
        )
        for lo, hi in zip(
            table[
                "asr_reduction_ci_low_pp"
            ],
            table[
                "asr_reduction_ci_high_pp"
            ],
        )
    ]

    table["Detect"] = (
        table[
            "attack_detection_rate"
        ]
        .map(pct)
    )

    table["Block"] = (
        table[
            "attack_block_rate"
        ]
        .map(pct)
    )

    table["Clean keep"] = (
        table[
            "clean_retention"
        ]
        .map(pct)
    )

    table["Re-encode"] = (
        table[
            "reencode_rate"
        ]
        .map(pct)
    )

    display_cols = [
        "Attack",
        "Defence",
        "N retained",
        "Undefended ASR",
        "Defended ASR",
        "ASR reduction",
        "95% CI reduction",
        "Detect",
        "Block",
        "Clean keep",
        "Re-encode",
    ]

    return table[
        display_cols
    ].to_html(
        index=False,
        classes="data-table",
        border=0,
    )


def clean_table_html(summary):
    table = summary.copy()

    table["Corpus"] = (
        table["corpus"]
        .map({
            "jigsaw_benign":
                "Jigsaw benign",
            "wikitext_heldout":
                "WikiText held-out",
        })
    )

    table["Defence"] = (
        table["defence"]
        .astype(str)
        .map(
            lambda x: DEFENCE_LABELS[x]
            .replace("\n", " ")
        )
    )

    table["False alarm"] = (
        table[
            "false_alarm_rate"
        ]
        .map(pct)
    )

    table["False block"] = (
        table[
            "false_block_rate"
        ]
        .map(pct)
    )

    table["Decision preservation"] = (
        table[
            "operational_preservation"
        ]
        .map(pct)
    )

    table["Mean |ΔP|"] = (
        table[
            "mean_abs_probability_shift"
        ]
        .map(
            lambda x:
            number(x, 4)
        )
    )

    table["Median latency"] = (
        table[
            "median_defence_latency_ms"
        ]
        .map(
            lambda x:
            f"{x:.3f} ms"
        )
    )

    return table[[
        "Corpus",
        "Defence",
        "False alarm",
        "False block",
        "Decision preservation",
        "Mean |ΔP|",
        "Median latency",
    ]].to_html(
        index=False,
        classes="data-table",
        border=0,
    )


def automatic_findings(metrics):
    """Generate purely data-derived report bullets."""
    findings = []

    # best outcome for each attack
    for attack in ATTACK_ORDER:
        sub = metrics[
            metrics["attack"].astype(str)
            == attack
        ]

        if len(sub) == 0:
            continue

        best_value = sub[
            "paired_defended_asr"
        ].min()

        best = sub[
            sub[
                "paired_defended_asr"
            ] == best_value
        ]

        names = ", ".join(
            DEFENCE_LABELS[
                str(x)
            ].replace("\n", " ")
            for x in best[
                "defence"
            ]
        )

        findings.append(
            f"<li><strong>{html.escape(ATTACK_LABELS[attack].replace(chr(10), ' '))}</strong>: "
            f"lowest paired defended ASR = "
            f"<strong>{best_value * 100:.1f}%</strong> "
            f"({html.escape(names)}).</li>"
        )

    # identify any cases where a defence increases paired ASR
    worse = metrics[
        metrics[
            "asr_reduction"
        ] < 0
    ]

    if len(worse):
        examples = []

        for _, row in worse.iterrows():
            examples.append(
                f"{ATTACK_LABELS[str(row['attack'])].replace(chr(10), ' ')} "
                f"+ {DEFENCE_LABELS[str(row['defence'])].replace(chr(10), ' ')} "
                f"({row['asr_reduction_pp']:.1f} pp)"
            )

        findings.append(
            "<li><strong>Defence-induced regressions:</strong> "
            + html.escape(
                "; ".join(examples)
            )
            + ".</li>"
        )

    # interface effect for AdvTok
    adv = metrics[
        metrics["attack"].astype(str)
        == "advtok"
    ]

    interface = adv[
        adv[
            "reencode_rate"
        ] >= 0.99
    ]

    if len(interface):
        names = ", ".join(
            DEFENCE_LABELS[
                str(x)
            ].replace("\n", " ")
            for x in interface[
                "defence"
            ]
        )

        findings.append(
            "<li><strong>AdvTok interface effect:</strong> "
            + html.escape(names)
            + " discarded supplied adversarial IDs through re-encoding rather than detecting the noncanonical tokenization.</li>"
        )

    return "\n".join(
        findings
    )


# --------------------------------------------------------------- HTML report

def make_report(
    attack_metrics,
    clean_summary,
    detection_metrics,
):
    """Create one browsable report containing every major plot."""
    figure_files = [
        (
            "Defended ASR",
            "fig01_defended_asr_heatmap.png",
        ),
        (
            "ASR reduction",
            "fig02_asr_reduction_heatmap.png",
        ),
        (
            "Attack detection",
            "fig03_detection_heatmap.png",
        ),
        (
            "Re-encoding mechanism",
            "fig04_reencoding_heatmap.png",
        ),
        (
            "Clean retention",
            "fig05_clean_retention_heatmap.png",
        ),
        (
            "Bootstrap confidence intervals",
            "fig06_asr_reduction_bootstrap_ci.png",
        ),
        (
            "Held-out clean controls",
            "fig07_clean_control_rates.png",
        ),
        (
            "Defence latency",
            "fig08_defence_latency.png",
        ),
        (
            "Detection F1",
            "fig09_detection_f1_heatmap.png",
        ),
        (
            "Effectiveness / utility trade-off",
            "fig10_effectiveness_utility_tradeoff.png",
        ),
    ]

    figures_html = ""

    for title, filename in figure_files:
        figures_html += f"""
        <section class="figure-card">
            <h3>{html.escape(title)}</h3>
            <img src="figures/{html.escape(filename)}"
                 alt="{html.escape(title)}">
        </section>
        """

    findings = automatic_findings(
        attack_metrics
    )

    attack_html = attack_table_html(
        attack_metrics
    )

    clean_html = clean_table_html(
        clean_summary
    )

    report = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport"
      content="width=device-width, initial-scale=1">

<title>Tokenizer Defence Evaluation - Step 10</title>

<style>
    :root {{
        --bg: #f5f7fb;
        --card: #ffffff;
        --text: #172033;
        --muted: #667085;
        --border: #e4e7ec;
        --accent: #3448c5;
        --code: #eef2ff;
    }}

    * {{
        box-sizing: border-box;
    }}

    body {{
        margin: 0;
        font-family:
            Inter,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
        background: var(--bg);
        color: var(--text);
        line-height: 1.55;
    }}

    .wrap {{
        max-width: 1450px;
        margin: auto;
        padding: 36px 28px 80px;
    }}

    h1 {{
        margin-bottom: 5px;
        font-size: 34px;
    }}

    h2 {{
        margin-top: 50px;
        font-size: 25px;
    }}

    h3 {{
        margin-top: 0;
    }}

    .subtitle {{
        color: var(--muted);
        margin-bottom: 30px;
    }}

    .cards {{
        display: grid;
        grid-template-columns:
            repeat(auto-fit, minmax(190px, 1fr));
        gap: 16px;
        margin: 26px 0;
    }}

    .metric {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 18px;
    }}

    .metric .big {{
        font-size: 26px;
        font-weight: 700;
    }}

    .metric .small {{
        color: var(--muted);
        font-size: 13px;
    }}

    .note {{
        background: #eef2ff;
        border-left: 4px solid var(--accent);
        padding: 16px 19px;
        border-radius: 9px;
        margin: 20px 0;
    }}

    .warning {{
        background: #fff7ed;
        border-left: 4px solid #f97316;
        padding: 16px 19px;
        border-radius: 9px;
        margin: 20px 0;
    }}

    .figure-grid {{
        display: grid;
        grid-template-columns:
            repeat(auto-fit, minmax(520px, 1fr));
        gap: 22px;
    }}

    .figure-card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px;
        overflow: hidden;
    }}

    .figure-card img {{
        display: block;
        width: 100%;
        height: auto;
    }}

    .table-wrap {{
        overflow-x: auto;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 8px;
    }}

    .data-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }}

    .data-table th {{
        text-align: left;
        background: #f9fafb;
        border-bottom: 2px solid var(--border);
        padding: 10px;
        white-space: nowrap;
    }}

    .data-table td {{
        border-bottom: 1px solid var(--border);
        padding: 9px 10px;
        white-space: nowrap;
    }}

    code {{
        background: var(--code);
        padding: 2px 5px;
        border-radius: 5px;
    }}

    a {{
        color: var(--accent);
    }}

    .references li {{
        margin-bottom: 13px;
    }}

    @media(max-width: 650px) {{
        .wrap {{
            padding: 20px 14px;
        }}

        .figure-grid {{
            grid-template-columns: 1fr;
        }}
    }}
</style>
</head>

<body>
<div class="wrap">

<h1>Tokenizer Attack × Defence Evaluation</h1>

<div class="subtitle">
    WordPiece victim · paired clean controls ·
    bootstrap uncertainty · mechanism-aware evaluation
</div>

<div class="cards">
    <div class="metric">
        <div class="big">936</div>
        <div class="small">attack instances</div>
    </div>

    <div class="metric">
        <div class="big">7</div>
        <div class="small">defence variants</div>
    </div>

    <div class="metric">
        <div class="big">6,552</div>
        <div class="small">attack × defence rows</div>
    </div>

    <div class="metric">
        <div class="big">{N_BOOTSTRAP:,}</div>
        <div class="small">bootstrap resamples per cell</div>
    </div>

    <div class="metric">
        <div class="big">5,250</div>
        <div class="small">held-out clean controls</div>
    </div>
</div>


<h2>How the final ASR is measured</h2>

<div class="note">
    The before/after ASR comparison uses the
    <strong>same defence-conditioned sample set</strong>.
    A sample enters the paired ASR calculation only when the
    clean toxic version remains toxic after the defence.
    This prevents defence-induced classifier damage from being
    incorrectly counted as an adversarial success.
</div>

<p>
For each retained sample:
</p>

<p>
<code>clean + defence → toxic</code><br>
then compare<br>
<code>attacked without defence</code>
against
<code>attacked + defence</code>.
</p>


<h2>Data-derived findings</h2>

<ul>
{findings}
</ul>


<h2>Main attack × defence results</h2>

<div class="table-wrap">
{attack_html}
</div>


<h2>Visual results</h2>

<div class="figure-grid">
{figures_html}
</div>


<h2>Held-out clean controls</h2>

<p>
Two clean corpora are used:
<strong>5,000 unseen WikiText lines</strong> and
<strong>250 Jigsaw comments labelled benign</strong>.
The WikiText rows used to calibrate CPT are not reused here.
</p>

<div class="table-wrap">
{clean_html}
</div>


<h2>Interpretation of the clean metrics</h2>

<div class="note">

<strong>False alarm rate</strong>:
clean input is flagged by the defence.

<br><br>

<strong>False block rate</strong>:
clean input would actually be rejected by the deployment policy.

<br><br>

<strong>Baseline-decision preservation</strong>:
the input is not blocked and the victim classifier gives the same
decision it gave before adding the defence.

This is deliberately different from the victim model's own toxicity
false-positive rate.

</div>


<h2>Literature provenance</h2>

<ul class="references">

<li>
<strong>TokenBreak</strong> —
Schulz, Yeung & Evans (2025).
Tokenizer Translation is based on the paper's
Unigram-to-original-tokenizer defence.
<a href="https://arxiv.org/abs/2506.07948">
arXiv:2506.07948
</a>
</li>

<li>
<strong>Adversarial Tokenization</strong> —
Geh, Shao & Van den Broeck, ACL 2025.
The paper motivates retokenisation and string-only interfaces
as defences against noncanonical token sequences.
<a href="https://aclanthology.org/2025.acl-long.1012/">
ACL Anthology
</a>
</li>

<li>
<strong>Bad Characters</strong> —
Boucher et al.
Provides the Unicode adversarial-input motivation behind the
sanitisation family.
<a href="https://arxiv.org/abs/2106.09898">
arXiv:2106.09898
</a>
</li>

<li>
<strong>Unicode UTS #39</strong> —
used as the standards basis for confusable-character detection.
<a href="https://www.unicode.org/reports/tr39/">
Unicode Security Mechanisms
</a>
</li>

<li>
<strong>Broken-Token</strong> —
Zychlinski & Kainan.
Provides global and sliding-window CPT filtering.
The WordPiece calibration strategy used in this dissertation
is explicitly an adaptation rather than an exact reproduction.
<a href="https://arxiv.org/abs/2510.26847">
arXiv:2510.26847
</a>
</li>

<li>
<strong>Bootstrap uncertainty</strong> —
Efron (1979), Bootstrap Methods: Another Look at the Jackknife.
The analysis uses paired percentile bootstrap confidence intervals.
</li>

</ul>


<h2>Important interpretation warning</h2>

<div class="warning">

A low defended ASR does not automatically mean that the named
defence detected the attack.

For AdvTok in particular,
<code>reencode = 100%</code> means the attacker-supplied token IDs
were discarded because the defence reconstructed input from the
visible string.

That is an <strong>interface effect</strong>, not attack detection.

The detection-rate and re-encoding plots must therefore be read
alongside the defended-ASR plot.

</div>


<h2>Scope</h2>

<p>
These results currently establish the main experiment for one
WordPiece toxicity classifier. They do not yet establish
cross-tokenizer or cross-model generality.
Those are separate experiments.
</p>

</div>
</body>
</html>
"""

    REPORT_FILE.write_text(
        report,
        encoding="utf-8",
    )

    print(
        "wrote",
        REPORT_FILE,
    )


# --------------------------------------------------------------- console summary

def print_key_results(
    attack_metrics,
    clean_summary,
):
    """Short terminal output after all analysis finishes."""
    print(
        "\n"
        + "=" * 76
    )

    print(
        "FINAL PAIRED RESULTS"
    )

    print(
        "=" * 76
    )

    for attack in ATTACK_ORDER:
        print(
            "\n",
            ATTACK_LABELS[
                attack
            ].replace(
                "\n",
                " ",
            ),
        )

        sub = attack_metrics[
            attack_metrics[
                "attack"
            ].astype(str)
            == attack
        ]

        for _, r in sub.iterrows():
            defence = str(
                r["defence"]
            )

            print(
                f"  "
                f"{DEFENCE_LABELS[defence].replace(chr(10), ' '):24s}"
                f" | "
                f"{r['paired_undefended_asr']:.3f}"
                f" -> "
                f"{r['paired_defended_asr']:.3f}"
                f" | Δ "
                f"{r['asr_reduction_pp']:+6.1f} pp"
                f" | detect "
                f"{r['attack_detection_rate']:.3f}"
                f" | clean "
                f"{r['clean_retention']:.3f}"
            )

    print(
        "\n"
        + "=" * 76
    )

    print(
        "HELD-OUT CLEAN FALSE-BLOCK RATES"
    )

    print(
        "=" * 76
    )

    for corpus in [
        "jigsaw_benign",
        "wikitext_heldout",
    ]:
        print(
            "\n",
            corpus,
        )

        sub = clean_summary[
            clean_summary[
                "corpus"
            ] == corpus
        ]

        for _, r in sub.iterrows():
            defence = str(
                r["defence"]
            )

            print(
                f"  "
                f"{DEFENCE_LABELS[defence].replace(chr(10), ' '):24s}"
                f" | alarm "
                f"{r['false_alarm_rate']:.3f}"
                f" | block "
                f"{r['false_block_rate']:.3f}"
                f" | preserve "
                f"{r['operational_preservation']:.3f}"
            )


# --------------------------------------------------------------- run

if __name__ == "__main__":
    print(
        "Step 10 - statistical analysis and visualisation"
    )

    print(
        f"bootstrap resamples: {N_BOOTSTRAP}"
    )

    print(
        "CPT thresholds:",
        json.dumps(
            thresholds,
            indent=2,
        ),
    )

    # ----------------------------------------------------------- provenance

    LITERATURE_FILE.write_text(
        LITERATURE_TEXT.strip()
        + "\n",
        encoding="utf-8",
    )

    print(
        "wrote",
        LITERATURE_FILE,
    )

    # ----------------------------------------------------------- attack matrix

    matrix = load_matrix()

    attack_metrics = analyse_attack_matrix(
        matrix
    )

    # ----------------------------------------------------------- held-out controls

    clean_controls = run_clean_controls()

    clean_summary = analyse_clean_controls(
        clean_controls
    )

    # ----------------------------------------------------------- detection metrics

    detection_metrics = calculate_detection_metrics(
        matrix,
        clean_controls,
    )

    # ----------------------------------------------------------- visualisations

    make_plots(
        attack_metrics,
        clean_summary,
        detection_metrics,
        matrix,
    )

    # ----------------------------------------------------------- HTML report

    make_report(
        attack_metrics,
        clean_summary,
        detection_metrics,
    )

    # ----------------------------------------------------------- terminal

    print_key_results(
        attack_metrics,
        clean_summary,
    )

    print(
        "\n"
        + "=" * 76
    )

    print(
        "STEP 10 DONE"
    )

    print(
        "=" * 76
    )

    print(
        "\nMain outputs:"
    )

    print(
        " ",
        ATTACK_METRICS_FILE,
    )

    print(
        " ",
        CLEAN_SUMMARY_FILE,
    )

    print(
        " ",
        DETECTION_FILE,
    )

    print(
        " ",
        FIG_DIR,
    )

    print(
        " ",
        REPORT_FILE,
    )

    if OPEN_REPORT:
        try:
            webbrowser.open(
                REPORT_FILE
                .resolve()
                .as_uri()
            )

            print(
                "\nopened HTML report in browser"
            )

        except Exception as exc:
            print(
                "\ncould not automatically open report:",
                exc,
            )

            print(
                "open this manually:",
                REPORT_FILE.resolve(),
            )