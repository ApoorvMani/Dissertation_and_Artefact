"""
Step 12 - cross-tokenizer clean baselines.

Adds two toxicity classifiers representing the tokenizer families not covered
by the main WordPiece experiment:

    BPE      -> RoBERTa
    Unigram  -> DeBERTa-v3

The existing WordPiece DistilBERT baseline is reused for comparison.

Why this step comes before attacks:
    TokenBreak evaluates attack success only on samples that the target model
    correctly detects before the attack. Different models therefore require
    different eligible sets.

Outputs:
    data/cross_bpe_clean_toxic.csv
    data/cross_bpe_clean_benign.csv

    data/cross_unigram_clean_toxic.csv
    data/cross_unigram_clean_benign.csv

    data/cross_tokenizer_eligibility.csv
    data/cross_tokenizer_shared_eligible.csv
    data/cross_tokenizer_metadata.json
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")  # windows terminal defaults to cp1252

import gc
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# --------------------------------------------------------------- config

SEED = 42

DATA = Path("data")

TOXIC_FILE = DATA / "jigsaw_toxic_250.csv"
BENIGN_FILE = DATA / "jigsaw_benign_250.csv"

WORDPIECE_TOXIC = DATA / "clean_baseline_toxic.csv"

MAX_LEN = 512
THRESHOLD = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


MODELS = {
    "bpe": {
        # binary RoBERTa classifier:
        # config maps neutral=0, toxic=1
        "name": "s-nlp/roberta_toxicity_classifier",
        "expected_tokenizer": "BPE",
        "score_type": "softmax",
        "toxic_index": 1,
        "batch": 8,
    },

    "unigram": {
        # DeBERTa-v3-small fine-tuned on Jigsaw toxicity labels.
        # config defines toxic as output head 0.
        "name": (
            "Emmytheo/"
            "Deberta-v3-finetuned-hate-speech-jigsaw-toxic-comments"
        ),
        "expected_tokenizer": "Unigram",
        "score_type": "sigmoid",
        "toxic_index": 0,
        "batch": 8,
    },
}


# --------------------------------------------------------------- tokenizer check

def tokenizer_algorithm(tok):
    """Return the actual fast-tokenizer model type."""
    if not tok.is_fast:
        return "not-fast"

    backend_model = tok.backend_tokenizer.model

    return type(backend_model).__name__


def load_tokenizer(model_name):
    """Load only the fast tokenizer; never require sentencepiece Python."""
    try:
        tok = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"\nCould not load the FAST tokenizer for {model_name}.\n"
            "Do NOT install sentencepiece in this project environment.\n"
            "The experiment requires tokenizer.json / fast-tokenizer loading."
        ) from exc

    if not tok.is_fast:
        raise RuntimeError(
            f"{model_name} loaded a slow tokenizer. "
            "Do not continue with the experiment."
        )

    return tok


# --------------------------------------------------------------- scoring

@torch.no_grad()
def toxic_probs(
    texts,
    tok,
    model,
    score_type,
    toxic_index,
    batch_size,
):
    """Score P(toxic) for a list of strings."""
    probs = []

    total = len(texts)

    for start in range(
        0,
        total,
        batch_size,
    ):
        chunk = texts[
            start:start + batch_size
        ]

        encoded = tok(
            chunk,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }

        logits = model(
            **encoded
        ).logits

        if score_type == "softmax":
            batch_probs = torch.softmax(
                logits,
                dim=-1,
            )[:, toxic_index]

        elif score_type == "sigmoid":
            batch_probs = torch.sigmoid(
                logits,
            )[:, toxic_index]

        else:
            raise ValueError(
                f"unknown score type: {score_type}"
            )

        probs.extend(
            batch_probs
            .detach()
            .cpu()
            .tolist()
        )

        print(
            f"  {min(start + batch_size, total)}/{total}",
            end="\r",
        )

    print()

    return probs


# --------------------------------------------------------------- one model

def evaluate_model(
    family,
    config,
    toxic_df,
    benign_df,
):
    """Run one tokenizer-family classifier on the frozen Jigsaw sets."""
    print(
        f"\n--- {family.upper()} ---"
    )

    print(
        "model      :",
        config["name"],
    )

    tok = load_tokenizer(
        config["name"]
    )

    actual_algorithm = tokenizer_algorithm(
        tok
    )

    print(
        "tokenizer  :",
        actual_algorithm,
    )

    print(
        "expected   :",
        config["expected_tokenizer"],
    )

    # this is a methodology check, not just decoration.
    # the experiment is meaningless if the model does not use the intended
    # tokenizer family.
    if (
        actual_algorithm.lower()
        != config[
            "expected_tokenizer"
        ].lower()
    ):
        raise RuntimeError(
            f"{family}: expected "
            f"{config['expected_tokenizer']}, "
            f"but tokenizer.json reports {actual_algorithm}"
        )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            config["name"]
        )
        .to(DEVICE)
    )

    model.eval()

    print(
        "model type :",
        model.config.model_type,
    )

    print(
        "id2label   :",
        model.config.id2label,
    )

    print(
        "device     :",
        DEVICE,
    )

    # ----------------------------------------------------------- toxic

    print(
        "\nscoring 250 toxic..."
    )

    toxic_probs_list = toxic_probs(
        toxic_df["text"].tolist(),
        tok,
        model,
        config["score_type"],
        config["toxic_index"],
        config["batch"],
    )

    toxic_out = toxic_df.copy()

    toxic_out[
        "clean_toxic_prob"
    ] = toxic_probs_list

    toxic_out[
        "clean_prediction"
    ] = (
        toxic_out[
            "clean_toxic_prob"
        ]
        >= THRESHOLD
    ).astype(int)

    toxic_out[
        "tokenizer_family"
    ] = family

    toxic_out[
        "model_name"
    ] = config["name"]

    toxic_out.to_csv(
        DATA
        / f"cross_{family}_clean_toxic.csv",
        index=False,
    )

    # ----------------------------------------------------------- benign

    print(
        "\nscoring 250 benign..."
    )

    benign_probs_list = toxic_probs(
        benign_df["text"].tolist(),
        tok,
        model,
        config["score_type"],
        config["toxic_index"],
        config["batch"],
    )

    benign_out = benign_df.copy()

    benign_out[
        "clean_toxic_prob"
    ] = benign_probs_list

    benign_out[
        "clean_prediction"
    ] = (
        benign_out[
            "clean_toxic_prob"
        ]
        >= THRESHOLD
    ).astype(int)

    benign_out[
        "tokenizer_family"
    ] = family

    benign_out[
        "model_name"
    ] = config["name"]

    benign_out.to_csv(
        DATA
        / f"cross_{family}_clean_benign.csv",
        index=False,
    )

    # ----------------------------------------------------------- metrics

    eligible = int(
        toxic_out[
            "clean_prediction"
        ].sum()
    )

    recall = (
        eligible
        / len(toxic_out)
    )

    false_positives = int(
        benign_out[
            "clean_prediction"
        ].sum()
    )

    fpr = (
        false_positives
        / len(benign_out)
    )

    print(
        "\nclean results"
    )

    print(
        f"  toxic detected : {eligible}/{len(toxic_out)}"
    )

    print(
        f"  recall         : {recall:.3f}"
    )

    print(
        f"  benign FP      : {false_positives}/{len(benign_out)}"
    )

    print(
        f"  benign FPR     : {fpr:.3f}"
    )

    print(
        f"  mean P(toxic) toxic  : "
        f"{toxic_out['clean_toxic_prob'].mean():.4f}"
    )

    print(
        f"  mean P(toxic) benign : "
        f"{benign_out['clean_toxic_prob'].mean():.4f}"
    )

    metadata = {
        "family": family,
        "model": config["name"],
        "tokenizer_algorithm": actual_algorithm,
        "score_type": config["score_type"],
        "toxic_index": config["toxic_index"],
        "threshold": THRESHOLD,
        "toxic_n": len(toxic_out),
        "eligible_n": eligible,
        "recall": recall,
        "benign_n": len(benign_out),
        "false_positive_n": false_positives,
        "benign_fpr": fpr,
    }

    # free VRAM before loading the next model
    del model
    del tok

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return toxic_out, benign_out, metadata


# --------------------------------------------------------------- eligibility

def build_eligibility(
    bpe_toxic,
    unigram_toxic,
):
    """Create per-model and shared eligible sets."""
    if not WORDPIECE_TOXIC.exists():
        raise FileNotFoundError(
            f"{WORDPIECE_TOXIC} not found."
        )

    wp = pd.read_csv(
        WORDPIECE_TOXIC
    )

    wp = wp[[
        "sample_id",
        "clean_prediction",
    ]].rename(
        columns={
            "clean_prediction":
                "eligible_wordpiece"
        }
    )

    bpe = bpe_toxic[[
        "sample_id",
        "clean_prediction",
    ]].rename(
        columns={
            "clean_prediction":
                "eligible_bpe"
        }
    )

    unigram = unigram_toxic[[
        "sample_id",
        "clean_prediction",
    ]].rename(
        columns={
            "clean_prediction":
                "eligible_unigram"
        }
    )

    merged = (
        wp
        .merge(
            bpe,
            on="sample_id",
            how="inner",
        )
        .merge(
            unigram,
            on="sample_id",
            how="inner",
        )
    )

    merged[
        "eligible_all_three"
    ] = (
        (
            merged[
                "eligible_wordpiece"
            ]
            == 1
        )
        &
        (
            merged[
                "eligible_bpe"
            ]
            == 1
        )
        &
        (
            merged[
                "eligible_unigram"
            ]
            == 1
        )
    ).astype(int)

    merged.to_csv(
        DATA
        / "cross_tokenizer_eligibility.csv",
        index=False,
    )

    shared_ids = set(
        merged.loc[
            merged[
                "eligible_all_three"
            ] == 1,
            "sample_id",
        ]
    )

    toxic_source = pd.read_csv(
        TOXIC_FILE
    )

    shared = toxic_source[
        toxic_source[
            "sample_id"
        ].isin(
            shared_ids
        )
    ].copy()

    shared.to_csv(
        DATA
        / "cross_tokenizer_shared_eligible.csv",
        index=False,
    )

    return merged, shared


# --------------------------------------------------------------- run

if __name__ == "__main__":
    torch.manual_seed(
        SEED
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            SEED
        )

    if not TOXIC_FILE.exists():
        raise FileNotFoundError(
            f"{TOXIC_FILE} not found."
        )

    if not BENIGN_FILE.exists():
        raise FileNotFoundError(
            f"{BENIGN_FILE} not found."
        )

    toxic_df = pd.read_csv(
        TOXIC_FILE
    )

    benign_df = pd.read_csv(
        BENIGN_FILE
    )

    print(
        "cross-tokenizer baseline"
    )

    print(
        "device :",
        DEVICE,
    )

    print(
        "toxic  :",
        len(toxic_df),
    )

    print(
        "benign :",
        len(benign_df),
    )

    results = {}
    metadata = {}

    for family in [
        "bpe",
        "unigram",
    ]:
        toxic_out, benign_out, meta = evaluate_model(
            family,
            MODELS[family],
            toxic_df,
            benign_df,
        )

        results[
            family
        ] = toxic_out

        metadata[
            family
        ] = meta

    eligibility, shared = build_eligibility(
        results["bpe"],
        results["unigram"],
    )

    # include the frozen WordPiece baseline in metadata for comparison
    wp = pd.read_csv(
        WORDPIECE_TOXIC
    )

    metadata[
        "wordpiece"
    ] = {
        "model":
            "martin-ha/toxic-comment-model",

        "tokenizer_algorithm":
            "WordPiece",

        "toxic_n":
            len(wp),

        "eligible_n":
            int(
                wp[
                    "clean_prediction"
                ].sum()
            ),

        "recall":
            float(
                wp[
                    "clean_prediction"
                ].mean()
            ),
    }

    metadata[
        "shared_eligible_all_three"
    ] = len(
        shared
    )

    with open(
        DATA
        / "cross_tokenizer_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        "\n--- cross-tokenizer baseline ---"
    )

    print(
        f"WordPiece eligible : "
        f"{metadata['wordpiece']['eligible_n']}/250"
    )

    print(
        f"BPE eligible       : "
        f"{metadata['bpe']['eligible_n']}/250"
    )

    print(
        f"Unigram eligible   : "
        f"{metadata['unigram']['eligible_n']}/250"
    )

    print(
        f"Shared all three   : "
        f"{len(shared)}/250"
    )

    print(
        "\noutputs:"
    )

    print(
        "  data/cross_bpe_clean_toxic.csv"
    )

    print(
        "  data/cross_bpe_clean_benign.csv"
    )

    print(
        "  data/cross_unigram_clean_toxic.csv"
    )

    print(
        "  data/cross_unigram_clean_benign.csv"
    )

    print(
        "  data/cross_tokenizer_eligibility.csv"
    )

    print(
        "  data/cross_tokenizer_shared_eligible.csv"
    )

    print(
        "  data/cross_tokenizer_metadata.json"
    )

    print(
        "\nstep 12 baseline done"
    )