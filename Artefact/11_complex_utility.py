"""
Step 11 - complex legitimate-text utility stress test.

Goal
----
The normal WikiText/Jigsaw clean evaluation mostly contains ordinary English.
That is not enough to measure collateral damage from tokenizer defences.

This experiment therefore evaluates legitimate complex inputs:

    100 code snippets
    100 URLs
    100 emoji-containing comments
    100 benign texts with realistic single-word typos
    100 non-English sentences
    100 legitimate mixed-script bilingual strings

Total:
    600 legitimate complex inputs

Each input is evaluated under all 7 defences.

Important:
    This is a UTILITY / STABILITY test, not another attack test.

For non-English, code, URLs, etc. the victim toxicity classifier is not
necessarily an appropriate semantic classifier. Therefore we DO NOT claim
that its baseline label is "correct".

Instead we ask:

    - did the defence trigger on legitimate text?
    - did the defence block legitimate text?
    - did the defence change the representation?
    - did the classifier's decision change because of the defence?
    - how much did P(toxic) change?
    - how much did tokenization change?
    - what was the preprocessing latency?

Sources
-------
Code / URLs:
    CodeSearchNet
    Husain et al. 2019, arXiv:1909.09436

Emoji:
    GoEmotions
    Demszky et al. 2020, arXiv:2005.00547
    We preferentially use examples labelled neutral.

Non-English:
    FLORES-200 / FLORES benchmark family.
    We use professionally translated multilingual benchmark sentences.

Mixed-script:
    Constructed bilingual strings using aligned English and FLORES
    translations. They are intentionally legitimate mixed-script examples,
    not spoof attacks.

Misspellings:
    Constructed from the project's Jigsaw BENIGN examples using exactly one
    deterministic internal typo. These are deliberately labelled constructed.

Literature motivation
---------------------
Unicode UTS #39 explicitly distinguishes mixed-script and confusable text.

Broken-Token reports multilingual complications for CPT, particularly for
non-alphanumeric scripts such as Chinese and Arabic.

Bad Characters motivates Unicode sanitisation, but sanitisation also requires
utility evaluation because unusual Unicode is not automatically malicious.

Outputs
-------
data/complex_legitimate_600.csv

results/step11/complex_utility_rows.csv
results/step11/complex_utility_summary.csv
results/step11/complex_utility_overall.csv

results/step11/figures/*.png
results/step11/figures/*.pdf

results/step11/report.html
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import ast
import hashlib
import html
import importlib.util
import itertools
import json
import os
import random
import re
import time
import webbrowser

from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification


# --------------------------------------------------------------- config

SEED = 42

N_PER_CATEGORY = 100
N_BOOTSTRAP = 5000

VICTIM = "martin-ha/toxic-comment-model"

MAX_TOKENS = 510
BATCH = 32
TOXIC_THRESHOLD = 0.5

DATA = Path("data")

DEFENCE_FILE = Path("08_defences.py")

JIGSAW_BENIGN = DATA / "jigsaw_benign_250.csv"

DATASET_FILE = DATA / "complex_legitimate_600.csv"

OUT = Path("results") / "step11"
FIG_DIR = OUT / "figures"

ROWS_FILE = OUT / "complex_utility_rows.csv"
SUMMARY_FILE = OUT / "complex_utility_summary.csv"
OVERALL_FILE = OUT / "complex_utility_overall.csv"

REPORT_FILE = OUT / "report.html"
PROVENANCE_FILE = OUT / "dataset_provenance.txt"

# set True only if you deliberately want to rebuild the 600 examples
FORCE_REBUILD_DATASET = False

# set True only if you deliberately want to rerun model inference
FORCE_RECOMPUTE = False

OPEN_REPORT = True


CATEGORIES = [
    "code",
    "url",
    "emoji",
    "misspelling",
    "non_english",
    "mixed_script",
]


CATEGORY_LABELS = {
    "code": "Code",
    "url": "URLs",
    "emoji": "Emoji",
    "misspelling": "Misspellings",
    "non_english": "Non-English",
    "mixed_script": "Mixed-script",
}


DEFENCE_ORDER = [
    "tokenizer_translation",
    "canonical_reject",
    "canonical_replace",
    "unicode_sanitise",
    "nfkc_confusable",
    "cpt_global",
    "cpt_window",
]


DEFENCE_LABELS = {
    "tokenizer_translation": "Tokenizer\ntranslation",
    "canonical_reject": "Canonical\nreject",
    "canonical_replace": "Canonical\nreplace",
    "unicode_sanitise": "Unicode\nsanitiser",
    "nfkc_confusable": "NFKC +\nconfusables",
    "cpt_global": "Global\nCPT",
    "cpt_window": "Window\nCPT",
}


# only these defences treat "flagged" as a blocking detector policy.
#
# unicode_sanitise can also reject Bidi text, but it does that by
# returning None, so is_blocked() catches it separately.
FILTER_DEFENCES = {
    "canonical_reject",
    "cpt_global",
    "cpt_window",
}


# languages deliberately cover several scripts
FLORES_LANGS = [
    ("hin_Deva", "Hindi / Devanagari"),
    ("arb_Arab", "Arabic"),
    ("rus_Cyrl", "Russian / Cyrillic"),
    ("zho_Hans", "Chinese / Han"),
    ("ben_Beng", "Bengali"),
]


OUT.mkdir(
    parents=True,
    exist_ok=True,
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------------------- provenance

PROVENANCE = """
COMPLEX LEGITIMATE-TEXT UTILITY SET
===================================

Purpose
-------
This dataset is not an attack dataset.

It measures whether tokenizer defences create collateral effects on legitimate
inputs that differ from ordinary English prose.

Categories
----------
1. code
   Source: CodeSearchNet Python functions.
   Original corpus: Husain et al., CodeSearchNet Challenge, 2019.

2. url
   Source: GitHub source URLs attached to CodeSearchNet examples.

3. emoji
   Source: GoEmotions Reddit comments containing emoji.
   Neutral examples are preferred.
   Original corpus: Demszky et al., GoEmotions, 2020.

4. misspelling
   Constructed from the project's Jigsaw BENIGN comments.
   Exactly one internal word-level typo is introduced.
   This category is explicitly synthetic/constructed.

5. non_english
   Source: FLORES multilingual benchmark sentences.
   Five languages/scripts are sampled:
       Hindi / Devanagari
       Arabic
       Russian / Cyrillic
       Chinese / Han
       Bengali

6. mixed_script
   Constructed from aligned FLORES English + non-English translations.
   Format:
       English sentence — equivalent translated sentence

   These are legitimate bilingual mixed-script strings, not spoof attacks.
   This category is explicitly constructed.

Literature rationale
--------------------
Unicode UTS #39 distinguishes single-script, mixed-script and whole-script
confusables and defines mixed-script detection mechanisms.

Broken-Token notes that CPT behaviour differs substantially for languages such
as Chinese and Arabic. This makes multilingual clean utility testing directly
relevant to CPT filtering.

Bad Characters shows why Unicode sanitisation is security-relevant, but the
existence of legitimate Unicode means a defence must also be evaluated for
collateral utility damage.

Important interpretation
------------------------
The victim is an English toxicity classifier.

For non-English, code and URLs, the classifier's baseline prediction is not
treated as semantic ground truth.

The meaningful utility quantity is therefore defence-induced CHANGE relative
to the exact same input without the defence.
"""


PROVENANCE_FILE.write_text(
    PROVENANCE.strip() + "\n",
    encoding="utf-8",
)


# --------------------------------------------------------------- load defences

def load_defence_module():
    """Load 08_defences.py."""
    if not DEFENCE_FILE.exists():
        raise FileNotFoundError(
            f"{DEFENCE_FILE} not found."
        )

    spec = importlib.util.spec_from_file_location(
        "defence_library",
        DEFENCE_FILE,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


defence_lib = load_defence_module()

thresholds = defence_lib.load_thresholds()
defences = defence_lib.build_defences(
    thresholds
)

tok = defence_lib.tok


# --------------------------------------------------------------- victim

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

_model = None


def get_model():
    """Load victim only when inference is required."""
    global _model

    if _model is None:
        print(
            "\nloading victim model..."
        )

        print(
            "device:",
            device,
        )

        _model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                VICTIM
            )
            .to(device)
        )

        _model.eval()

    return _model


CLS = tok.cls_token_id
SEP = tok.sep_token_id
PAD = tok.pad_token_id


# --------------------------------------------------------------- ids / model

@torch.no_grad()
def p_toxic(id_seqs):
    """Return P(toxic) for raw token-id sequences."""
    if not id_seqs:
        return []

    model = get_model()

    probabilities = []

    for start in range(
        0,
        len(id_seqs),
        BATCH,
    ):
        chunk = id_seqs[
            start:start + BATCH
        ]

        chunk = [
            list(ids)[:MAX_TOKENS]
            for ids in chunk
        ]

        width = (
            max(
                len(ids)
                for ids in chunk
            )
            + 2
        )

        input_ids = torch.full(
            (
                len(chunk),
                width,
            ),
            PAD,
            dtype=torch.long,
        )

        attention = torch.zeros(
            (
                len(chunk),
                width,
            ),
            dtype=torch.long,
        )

        for i, ids in enumerate(
            chunk
        ):
            sequence = (
                [CLS]
                + ids
                + [SEP]
            )

            input_ids[
                i,
                :len(sequence),
            ] = torch.tensor(
                sequence,
                dtype=torch.long,
            )

            attention[
                i,
                :len(sequence),
            ] = 1

        logits = model(
            input_ids=input_ids.to(
                device
            ),
            attention_mask=attention.to(
                device
            ),
        ).logits

        probs = torch.softmax(
            logits,
            dim=-1,
        )[:, 1]

        probabilities.extend(
            probs.cpu().tolist()
        )

    return probabilities


# --------------------------------------------------------------- general helpers

def sample_id(category, text):
    """Stable anonymous identifier."""
    digest = hashlib.sha256(
        (
            category
            + "\n"
            + text
        ).encode(
            "utf-8"
        )
    ).hexdigest()

    return digest[:16]


def safe_text(value):
    if value is None:
        return ""

    return str(value).strip()


def is_blocked(
    defence_name,
    flagged,
    out_ids,
):
    """Same deployment policy used in Steps 9 and 10."""
    if out_ids is None:
        return 1

    if (
        defence_name
        in FILTER_DEFENCES
        and int(flagged) == 1
    ):
        return 1

    return 0


def mechanism_acted(mechanism):
    """Did the defence itself act?"""
    return int(
        "acted"
        in str(mechanism)
    )


# --------------------------------------------------------------- emoji detection

EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]"
)


def has_emoji(text):
    return bool(
        EMOJI_RE.search(text)
    )


# --------------------------------------------------------------- CodeSearchNet

def load_code_and_urls():
    """Collect 100 real Python snippets and 100 associated GitHub URLs."""
    print(
        "\nbuilding code + URL categories..."
    )

    # streaming avoids downloading the full ~hundreds-of-MB corpus
    dataset = load_dataset(
        "Nan-Do/code-search-net-python",
        split="train",
        streaming=True,
    )

    code_rows = []
    url_rows = []

    seen_code = set()
    seen_url = set()

    for index, row in enumerate(
        dataset
    ):
        code = safe_text(
            row.get("code")
            or row.get(
                "original_string"
            )
        )

        url = safe_text(
            row.get("url")
        )

        # moderate-size real functions are enough for the tokenizer test
        if (
            len(code_rows)
            < N_PER_CATEGORY
            and 80 <= len(code) <= 1800
            and code not in seen_code
        ):
            seen_code.add(
                code
            )

            code_rows.append({
                "category": "code",
                "text": code,
                "source_dataset":
                    "CodeSearchNet",
                "source_detail":
                    "Python function",
                "source_id":
                    url or f"row_{index}",
                "construction":
                    "natural/source",
            })

        if (
            len(url_rows)
            < N_PER_CATEGORY
            and url.startswith(
                ("http://", "https://")
            )
            and url not in seen_url
        ):
            seen_url.add(
                url
            )

            url_rows.append({
                "category": "url",
                "text": url,
                "source_dataset":
                    "CodeSearchNet",
                "source_detail":
                    "GitHub source URL",
                "source_id":
                    f"row_{index}",
                "construction":
                    "natural/source",
            })

        if (
            len(code_rows)
            >= N_PER_CATEGORY
            and len(url_rows)
            >= N_PER_CATEGORY
        ):
            break

    if len(code_rows) < N_PER_CATEGORY:
        raise RuntimeError(
            "Could not collect 100 CodeSearchNet code samples."
        )

    if len(url_rows) < N_PER_CATEGORY:
        raise RuntimeError(
            "Could not collect 100 CodeSearchNet URLs."
        )

    print(
        f"  code : {len(code_rows)}"
    )

    print(
        f"  urls : {len(url_rows)}"
    )

    return (
        code_rows,
        url_rows,
    )


# --------------------------------------------------------------- GoEmotions

def load_emoji():
    """Collect legitimate neutral Reddit comments containing emoji."""
    print(
        "\nbuilding emoji category..."
    )

    dataset = load_dataset(
        "SetFit/go_emotions",
        split="train",
        streaming=True,
    )

    rows = []
    seen = set()

    for index, row in enumerate(
        dataset
    ):
        text = safe_text(
            row.get("text")
        )

        # neutral==1 gives us a cleaner legitimate control subset.
        neutral = int(
            row.get(
                "neutral",
                0,
            )
        )

        if (
            neutral == 1
            and has_emoji(text)
            and 3 <= len(text) <= 700
            and text not in seen
        ):
            seen.add(
                text
            )

            rows.append({
                "category": "emoji",
                "text": text,
                "source_dataset":
                    "GoEmotions",
                "source_detail":
                    "neutral Reddit comment containing emoji",
                "source_id":
                    f"train_{index}",
                "construction":
                    "natural/source",
            })

        if len(rows) >= N_PER_CATEGORY:
            break

    # if neutral-only does not provide enough examples,
    # widen to all legitimate source comments containing emoji.
    #
    # these remain non-adversarial inputs, but provenance records
    # the fallback explicitly.
    if len(rows) < N_PER_CATEGORY:
        print(
            "  neutral emoji rows were insufficient;"
            " widening to other GoEmotions labels"
        )

        dataset = load_dataset(
            "SetFit/go_emotions",
            split="train",
            streaming=True,
        )

        for index, row in enumerate(
            dataset
        ):
            text = safe_text(
                row.get("text")
            )

            if (
                has_emoji(text)
                and 3 <= len(text) <= 700
                and text not in seen
            ):
                seen.add(
                    text
                )

                rows.append({
                    "category": "emoji",
                    "text": text,
                    "source_dataset":
                        "GoEmotions",
                    "source_detail":
                        "Reddit comment containing emoji",
                    "source_id":
                        f"train_{index}",
                    "construction":
                        "natural/source",
                })

            if len(rows) >= N_PER_CATEGORY:
                break

    if len(rows) < N_PER_CATEGORY:
        raise RuntimeError(
            f"Only found {len(rows)} emoji examples."
        )

    rows = rows[
        :N_PER_CATEGORY
    ]

    print(
        f"  emoji: {len(rows)}"
    )

    return rows


# --------------------------------------------------------------- misspellings

WORD_RE = re.compile(
    r"\b[A-Za-z]{5,}\b"
)


def make_one_typo(
    text,
    rng,
):
    """Insert exactly one small human-readable typo."""
    matches = list(
        WORD_RE.finditer(text)
    )

    if not matches:
        return None, None

    # randomly choose an eligible word
    match = matches[
        rng.randrange(
            len(matches)
        )
    ]

    word = match.group(0)

    if len(word) < 5:
        return None, None

    modes = [
        "swap",
        "delete",
        "duplicate",
    ]

    rng.shuffle(
        modes
    )

    changed = None
    used_mode = None

    for mode in modes:

        # ------------------------------------------------------- swap
        if mode == "swap":
            positions = [
                i
                for i in range(
                    1,
                    len(word) - 2
                )
                if word[i]
                != word[i + 1]
            ]

            if positions:
                pos = rng.choice(
                    positions
                )

                chars = list(
                    word
                )

                chars[pos], chars[pos + 1] = (
                    chars[pos + 1],
                    chars[pos],
                )

                changed = "".join(
                    chars
                )

                used_mode = "adjacent_swap"
                break

        # ------------------------------------------------------- delete
        if mode == "delete":
            pos = rng.randrange(
                1,
                len(word) - 1,
            )

            changed = (
                word[:pos]
                + word[pos + 1:]
            )

            used_mode = "internal_delete"
            break

        # ------------------------------------------------------- duplicate
        if mode == "duplicate":
            pos = rng.randrange(
                1,
                len(word) - 1,
            )

            changed = (
                word[:pos]
                + word[pos]
                + word[pos:]
            )

            used_mode = "internal_duplicate"
            break

    if (
        changed is None
        or changed == word
    ):
        return None, None

    new_text = (
        text[:match.start()]
        + changed
        + text[match.end():]
    )

    return (
        new_text,
        used_mode,
    )


def load_misspellings():
    """Construct one-typo versions of Jigsaw benign comments."""
    print(
        "\nbuilding misspelling category..."
    )

    if not JIGSAW_BENIGN.exists():
        raise FileNotFoundError(
            f"{JIGSAW_BENIGN} not found."
        )

    df = pd.read_csv(
        JIGSAW_BENIGN
    )

    rng = random.Random(
        SEED
    )

    rows = []
    seen = set()

    for index, row in df.iterrows():
        original = safe_text(
            row["text"]
        )

        typo_text, typo_type = make_one_typo(
            original,
            rng,
        )

        if (
            typo_text
            and typo_text not in seen
        ):
            seen.add(
                typo_text
            )

            rows.append({
                "category":
                    "misspelling",

                "text":
                    typo_text,

                "source_dataset":
                    "Jigsaw benign",

                "source_detail":
                    typo_type,

                "source_id":
                    str(
                        row.get(
                            "sample_id",
                            index,
                        )
                    ),

                "construction":
                    "constructed: one benign word typo",

                "original_clean_text":
                    original,
            })

        if len(rows) >= N_PER_CATEGORY:
            break

    if len(rows) < N_PER_CATEGORY:
        raise RuntimeError(
            f"Only created {len(rows)} misspelling examples."
        )

    print(
        f"  misspellings: {len(rows)}"
    )

    return rows


# --------------------------------------------------------------- FLORES

def get_flores_value(
    row,
    language_code,
):
    """Support both official and consolidated FLORES column naming."""
    candidates = [
        language_code,
        f"sentence_{language_code}",
    ]

    for key in candidates:
        if key in row:
            value = safe_text(
                row[key]
            )

            if value:
                return value

    return ""


def load_flores_source():
    """Load a small FLORES dev stream with a fallback mirror."""
    print(
        "\nloading FLORES..."
    )

    attempts = [
        (
            "facebook/flores",
            "all",
        ),
        (
            "yash9439/flores200",
            None,
        ),
    ]

    last_error = None

    for dataset_name, config in attempts:
        try:
            if config:
                dataset = load_dataset(
                    dataset_name,
                    config,
                    split="dev",
                    streaming=True,
                )
            else:
                dataset = load_dataset(
                    dataset_name,
                    split="dev",
                    streaming=True,
                )

            # only need 50 aligned rows:
            # first 20 non-English,
            # next 20 mixed-script,
            # remaining rows give buffer.
            rows = list(
                itertools.islice(
                    dataset,
                    60,
                )
            )

            if len(rows) < 40:
                raise RuntimeError(
                    "FLORES returned too few rows."
                )

            # verify English and our selected languages exist
            test = rows[0]

            if not get_flores_value(
                test,
                "eng_Latn",
            ):
                raise RuntimeError(
                    "English FLORES field not found."
                )

            for code, _label in FLORES_LANGS:
                if not get_flores_value(
                    test,
                    code,
                ):
                    raise RuntimeError(
                        f"FLORES field {code} not found."
                    )

            print(
                "  source:",
                dataset_name,
            )

            return (
                rows,
                dataset_name,
            )

        except Exception as exc:
            last_error = exc

            print(
                f"  failed {dataset_name}: {exc}"
            )

    raise RuntimeError(
        "Could not load FLORES."
    ) from last_error


def load_multilingual():
    """Build non-English and legitimate mixed-script categories."""
    flores_rows, source_name = (
        load_flores_source()
    )

    print(
        "\nbuilding non-English + mixed-script categories..."
    )

    non_english = []
    mixed_script = []

    # 5 languages x 20 examples = 100
    per_language = (
        N_PER_CATEGORY
        // len(FLORES_LANGS)
    )

    # ----------------------------------------------------------- non-English

    for language_code, language_name in FLORES_LANGS:
        count = 0

        for row_index, row in enumerate(
            flores_rows[:30]
        ):
            text = get_flores_value(
                row,
                language_code,
            )

            if not text:
                continue

            non_english.append({
                "category":
                    "non_english",

                "text":
                    text,

                "source_dataset":
                    "FLORES-200",

                "source_detail":
                    language_name,

                "source_id":
                    f"{language_code}_{row_index}",

                "construction":
                    "natural/professional translation",

                "language":
                    language_code,
            })

            count += 1

            if count >= per_language:
                break

        if count < per_language:
            raise RuntimeError(
                f"Not enough FLORES rows for {language_code}"
            )

    # ----------------------------------------------------------- mixed-script
    #
    # use different aligned rows where possible so these are not just
    # repetitions of the non-English category.

    for language_code, language_name in FLORES_LANGS:
        count = 0

        for row_index, row in enumerate(
            flores_rows[20:60],
            start=20,
        ):
            english = get_flores_value(
                row,
                "eng_Latn",
            )

            translated = get_flores_value(
                row,
                language_code,
            )

            if (
                not english
                or not translated
            ):
                continue

            text = (
                english
                + " — "
                + translated
            )

            mixed_script.append({
                "category":
                    "mixed_script",

                "text":
                    text,

                "source_dataset":
                    "FLORES-200",

                "source_detail":
                    f"English + {language_name}",

                "source_id":
                    f"eng_{language_code}_{row_index}",

                "construction":
                    "constructed legitimate bilingual pair",

                "language":
                    language_code,
            })

            count += 1

            if count >= per_language:
                break

        if count < per_language:
            raise RuntimeError(
                f"Not enough mixed-script FLORES rows for {language_code}"
            )

    non_english = non_english[
        :N_PER_CATEGORY
    ]

    mixed_script = mixed_script[
        :N_PER_CATEGORY
    ]

    print(
        f"  non-English : {len(non_english)}"
    )

    print(
        f"  mixed-script: {len(mixed_script)}"
    )

    return (
        non_english,
        mixed_script,
    )


# --------------------------------------------------------------- build dataset

def build_dataset():
    """Build exactly 600 legitimate complex examples."""
    if (
        DATASET_FILE.exists()
        and not FORCE_REBUILD_DATASET
    ):
        print(
            "\nusing existing dataset:",
            DATASET_FILE,
        )

        dataset = pd.read_csv(
            DATASET_FILE
        )

        validate_dataset(
            dataset
        )

        return dataset

    code_rows, url_rows = (
        load_code_and_urls()
    )

    emoji_rows = load_emoji()

    misspelling_rows = (
        load_misspellings()
    )

    (
        non_english_rows,
        mixed_script_rows,
    ) = load_multilingual()

    rows = (
        code_rows
        + url_rows
        + emoji_rows
        + misspelling_rows
        + non_english_rows
        + mixed_script_rows
    )

    # make optional fields consistent
    for row in rows:
        row.setdefault(
            "language",
            "",
        )

        row.setdefault(
            "original_clean_text",
            "",
        )

        row["sample_id"] = sample_id(
            row["category"],
            row["text"],
        )

    dataset = pd.DataFrame(
        rows
    )

    dataset = dataset[[
        "sample_id",
        "category",
        "text",
        "source_dataset",
        "source_detail",
        "source_id",
        "construction",
        "language",
        "original_clean_text",
    ]]

    validate_dataset(
        dataset
    )

    dataset.to_csv(
        DATASET_FILE,
        index=False,
    )

    print(
        "\nwrote",
        DATASET_FILE,
    )

    print(
        "\ncategory counts:"
    )

    print(
        dataset[
            "category"
        ].value_counts()
    )

    return dataset


def validate_dataset(dataset):
    """Fail early if category sizes or IDs are wrong."""
    expected_total = (
        len(CATEGORIES)
        * N_PER_CATEGORY
    )

    if len(dataset) != expected_total:
        raise RuntimeError(
            f"expected {expected_total} rows, "
            f"got {len(dataset)}"
        )

    for category in CATEGORIES:
        count = int(
            (
                dataset["category"]
                == category
            ).sum()
        )

        if count != N_PER_CATEGORY:
            raise RuntimeError(
                f"{category}: expected "
                f"{N_PER_CATEGORY}, got {count}"
            )

    if dataset[
        "sample_id"
    ].duplicated().any():
        raise RuntimeError(
            "duplicate sample IDs found"
        )

    if dataset[
        "text"
    ].isna().any():
        raise RuntimeError(
            "empty text found"
        )


# --------------------------------------------------------------- evaluate

def run_utility_test(dataset):
    """Run every complex legitimate input through every defence."""
    if (
        ROWS_FILE.exists()
        and not FORCE_RECOMPUTE
    ):
        print(
            "\nusing cached utility rows:",
            ROWS_FILE,
        )

        return pd.read_csv(
            ROWS_FILE
        )

    print(
        "\nscoring baseline representations..."
    )

    texts = (
        dataset["text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    canonical_ids = [
        defence_lib.encode(
            text
        )
        for text in texts
    ]

    baseline_probs = p_toxic(
        canonical_ids
    )

    baseline_labels = [
        int(
            p >= TOXIC_THRESHOLD
        )
        for p in baseline_probs
    ]

    rows = []

    total = (
        len(dataset)
        * len(defences)
    )

    done = 0

    # store model sequences so they can all be scored in batches
    sequences = []
    sequence_row_indices = []

    for data_index, data_row in dataset.iterrows():
        text = safe_text(
            data_row["text"]
        )

        base_ids = canonical_ids[
            data_index
        ]

        for defence_name in DEFENCE_ORDER:
            defend = defences[
                defence_name
            ]

            start = time.perf_counter()

            # legitimate deployment input uses ordinary string interface
            out_ids, flagged, mechanism = defend(
                text,
                None,
            )

            latency_ms = (
                time.perf_counter()
                - start
            ) * 1000

            blocked = is_blocked(
                defence_name,
                flagged,
                out_ids,
            )

            acted = mechanism_acted(
                mechanism
            )

            if out_ids is None:
                representation_changed = 1
                out_token_count = 0

            else:
                out_ids = list(
                    out_ids
                )

                representation_changed = int(
                    out_ids
                    != list(base_ids)
                )

                out_token_count = len(
                    out_ids
                )

            base_token_count = len(
                base_ids
            )

            if base_token_count:
                token_ratio = (
                    out_token_count
                    / base_token_count
                )
            else:
                token_ratio = np.nan

            result = {
                "sample_id":
                    data_row[
                        "sample_id"
                    ],

                "category":
                    data_row[
                        "category"
                    ],

                "source_dataset":
                    data_row[
                        "source_dataset"
                    ],

                "source_detail":
                    data_row[
                        "source_detail"
                    ],

                "construction":
                    data_row[
                        "construction"
                    ],

                "language":
                    data_row.get(
                        "language",
                        "",
                    ),

                "defence":
                    defence_name,

                # -------------------------------- baseline
                "baseline_toxic_prob":
                    baseline_probs[
                        data_index
                    ],

                "baseline_label":
                    baseline_labels[
                        data_index
                    ],

                "baseline_token_count":
                    base_token_count,

                # -------------------------------- defence mechanism
                "triggered":
                    int(flagged),

                "acted":
                    acted,

                "mechanism":
                    mechanism,

                "blocked":
                    int(blocked),

                "representation_changed":
                    representation_changed,

                "out_token_count":
                    out_token_count,

                "token_count_ratio":
                    token_ratio,

                "token_count_delta":
                    (
                        out_token_count
                        - base_token_count
                    ),

                "defence_latency_ms":
                    latency_ms,

                # -------------------------------- filled after scoring
                "defended_toxic_prob":
                    np.nan,

                "defended_label":
                    np.nan,

                "model_label_changed":
                    np.nan,

                "absolute_probability_shift":
                    np.nan,

                "operational_preservation":
                    0,
            }

            row_index = len(
                rows
            )

            rows.append(
                result
            )

            # even CPT-flagged input can be scored separately from
            # the operational blocking policy.
            if out_ids is not None:
                sequences.append(
                    out_ids
                )

                sequence_row_indices.append(
                    row_index
                )

            done += 1

            print(
                f"  preparing {done}/{total}",
                end="\r",
            )

    print()

    print(
        f"\nscoring {len(sequences)} defended representations..."
    )

    defended_probs = []

    for start in range(
        0,
        len(sequences),
        BATCH,
    ):
        chunk = sequences[
            start:start + BATCH
        ]

        defended_probs.extend(
            p_toxic(
                chunk
            )
        )

        print(
            f"  "
            f"{min(start + BATCH, len(sequences))}"
            f"/{len(sequences)}",
            end="\r",
        )

    print()

    for row_index, prob in zip(
        sequence_row_indices,
        defended_probs,
    ):
        row = rows[
            row_index
        ]

        defended_label = int(
            prob >= TOXIC_THRESHOLD
        )

        row[
            "defended_toxic_prob"
        ] = prob

        row[
            "defended_label"
        ] = defended_label

        row[
            "model_label_changed"
        ] = int(
            defended_label
            != row[
                "baseline_label"
            ]
        )

        row[
            "absolute_probability_shift"
        ] = abs(
            prob
            - row[
                "baseline_toxic_prob"
            ]
        )

        # main utility metric:
        #
        # legitimate input is preserved only if:
        #   - defence does not block it
        #   - classifier decision remains the same
        row[
            "operational_preservation"
        ] = int(
            row["blocked"] == 0
            and row[
                "model_label_changed"
            ] == 0
        )

    output = pd.DataFrame(
        rows
    )

    output.to_csv(
        ROWS_FILE,
        index=False,
    )

    print(
        "\nwrote",
        ROWS_FILE,
    )

    return output


# --------------------------------------------------------------- bootstrap

def stable_seed(*parts):
    text = "|".join(
        str(x)
        for x in parts
    )

    digest = hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).digest()

    return (
        SEED
        + int.from_bytes(
            digest[:4],
            "little",
        )
    ) % (
        2**32 - 1
    )


def bootstrap_rate(
    values,
    seed,
):
    """Mean binary rate + 95% percentile bootstrap CI."""
    values = np.asarray(
        values,
        dtype=float,
    )

    n = len(
        values
    )

    if n == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    point = float(
        values.mean()
    )

    rng = np.random.default_rng(
        seed
    )

    results = np.empty(
        N_BOOTSTRAP
    )

    batch = 500

    for start in range(
        0,
        N_BOOTSTRAP,
        batch,
    ):
        end = min(
            start + batch,
            N_BOOTSTRAP,
        )

        size = end - start

        indices = rng.integers(
            0,
            n,
            size=(
                size,
                n,
            ),
        )

        results[
            start:end
        ] = (
            values[
                indices
            ].mean(
                axis=1
            )
        )

    low = float(
        np.percentile(
            results,
            2.5,
        )
    )

    high = float(
        np.percentile(
            results,
            97.5,
        )
    )

    return (
        point,
        low,
        high,
    )


# --------------------------------------------------------------- summarise

def summarise(rows):
    """Category x defence utility table."""
    print(
        "\ncalculating utility summary..."
    )

    records = []

    grouped = rows.groupby(
        [
            "category",
            "defence",
        ],
        sort=False,
    )

    for (
        category,
        defence,
    ), group in grouped:

        (
            trigger,
            trigger_low,
            trigger_high,
        ) = bootstrap_rate(
            group[
                "triggered"
            ].to_numpy(),

            stable_seed(
                category,
                defence,
                "trigger",
            ),
        )

        (
            action,
            action_low,
            action_high,
        ) = bootstrap_rate(
            group[
                "acted"
            ].to_numpy(),

            stable_seed(
                category,
                defence,
                "action",
            ),
        )

        (
            block,
            block_low,
            block_high,
        ) = bootstrap_rate(
            group[
                "blocked"
            ].to_numpy(),

            stable_seed(
                category,
                defence,
                "block",
            ),
        )

        (
            rep_change,
            rep_low,
            rep_high,
        ) = bootstrap_rate(
            group[
                "representation_changed"
            ].to_numpy(),

            stable_seed(
                category,
                defence,
                "representation",
            ),
        )

        (
            preserve,
            preserve_low,
            preserve_high,
        ) = bootstrap_rate(
            group[
                "operational_preservation"
            ].to_numpy(),

            stable_seed(
                category,
                defence,
                "preservation",
            ),
        )

        scored = group[
            group[
                "defended_label"
            ].notna()
        ]

        if len(scored):
            label_flip = float(
                scored[
                    "model_label_changed"
                ].mean()
            )

            prob_shift = float(
                scored[
                    "absolute_probability_shift"
                ].mean()
            )

            median_prob_shift = float(
                scored[
                    "absolute_probability_shift"
                ].median()
            )

        else:
            label_flip = np.nan
            prob_shift = np.nan
            median_prob_shift = np.nan

        records.append({
            "category":
                category,

            "defence":
                defence,

            "n":
                len(group),

            # ----------------------------------------- defence triggering
            "trigger_rate":
                trigger,

            "trigger_ci_low":
                trigger_low,

            "trigger_ci_high":
                trigger_high,

            "action_rate":
                action,

            "action_ci_low":
                action_low,

            "action_ci_high":
                action_high,

            # ----------------------------------------- operational blocking
            "block_rate":
                block,

            "block_ci_low":
                block_low,

            "block_ci_high":
                block_high,

            # ----------------------------------------- representation
            "representation_change_rate":
                rep_change,

            "representation_change_ci_low":
                rep_low,

            "representation_change_ci_high":
                rep_high,

            # ----------------------------------------- main utility
            "operational_preservation":
                preserve,

            "operational_preservation_ci_low":
                preserve_low,

            "operational_preservation_ci_high":
                preserve_high,

            # ----------------------------------------- classifier stability
            "model_label_flip_rate_when_scored":
                label_flip,

            "mean_abs_probability_shift":
                prob_shift,

            "median_abs_probability_shift":
                median_prob_shift,

            # ----------------------------------------- tokenization
            "mean_token_count_ratio":
                float(
                    group[
                        "token_count_ratio"
                    ].mean()
                ),

            "median_token_count_ratio":
                float(
                    group[
                        "token_count_ratio"
                    ].median()
                ),

            "mean_token_count_delta":
                float(
                    group[
                        "token_count_delta"
                    ].mean()
                ),

            # ----------------------------------------- model context
            "baseline_toxic_rate":
                float(
                    group[
                        "baseline_label"
                    ].mean()
                ),

            # ----------------------------------------- latency
            "median_latency_ms":
                float(
                    group[
                        "defence_latency_ms"
                    ].median()
                ),

            "mean_latency_ms":
                float(
                    group[
                        "defence_latency_ms"
                    ].mean()
                ),
        })

    summary = pd.DataFrame(
        records
    )

    summary["category"] = pd.Categorical(
        summary["category"],
        CATEGORIES,
        ordered=True,
    )

    summary["defence"] = pd.Categorical(
        summary["defence"],
        DEFENCE_ORDER,
        ordered=True,
    )

    summary = (
        summary
        .sort_values(
            [
                "category",
                "defence",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print(
        "wrote",
        SUMMARY_FILE,
    )

    return summary


def summarise_overall(rows):
    """Overall utility per defence over all 600 examples."""
    records = []

    for defence in DEFENCE_ORDER:
        group = rows[
            rows[
                "defence"
            ] == defence
        ]

        trigger = float(
            group[
                "triggered"
            ].mean()
        )

        action = float(
            group[
                "acted"
            ].mean()
        )

        blocked = float(
            group[
                "blocked"
            ].mean()
        )

        representation = float(
            group[
                "representation_changed"
            ].mean()
        )

        preservation = float(
            group[
                "operational_preservation"
            ].mean()
        )

        scored = group[
            group[
                "absolute_probability_shift"
            ].notna()
        ]

        records.append({
            "defence":
                defence,

            "n":
                len(group),

            "trigger_rate":
                trigger,

            "action_rate":
                action,

            "block_rate":
                blocked,

            "representation_change_rate":
                representation,

            "operational_preservation":
                preservation,

            "model_label_flip_rate":
                float(
                    scored[
                        "model_label_changed"
                    ].mean()
                )
                if len(scored)
                else np.nan,

            "mean_abs_probability_shift":
                float(
                    scored[
                        "absolute_probability_shift"
                    ].mean()
                )
                if len(scored)
                else np.nan,

            "median_latency_ms":
                float(
                    group[
                        "defence_latency_ms"
                    ].median()
                ),
        })

    overall = pd.DataFrame(
        records
    )

    overall.to_csv(
        OVERALL_FILE,
        index=False,
    )

    print(
        "wrote",
        OVERALL_FILE,
    )

    return overall


# --------------------------------------------------------------- plots

def set_plot_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,

        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,

        "xtick.labelsize": 9,
        "ytick.labelsize": 9,

        "figure.facecolor":
            "white",

        "axes.facecolor":
            "white",

        "axes.spines.top":
            False,

        "axes.spines.right":
            False,
    })


def save_figure(
    fig,
    name,
):
    fig.savefig(
        FIG_DIR
        / f"{name}.png",
        bbox_inches="tight",
    )

    fig.savefig(
        FIG_DIR
        / f"{name}.pdf",
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def pivot_metric(
    summary,
    metric,
):
    pivot = summary.pivot(
        index="category",
        columns="defence",
        values=metric,
    )

    return pivot.reindex(
        index=CATEGORIES,
        columns=DEFENCE_ORDER,
    )


def heatmap(
    values,
    title,
    output_name,
    cmap,
    percentage=True,
    vmin=0,
    vmax=1,
    colourbar_label=None,
):
    array = values.to_numpy(
        dtype=float
    )

    fig, ax = plt.subplots(
        figsize=(
            12.5,
            6.3,
        )
    )

    image = ax.imshow(
        array,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(
        np.arange(
            len(DEFENCE_ORDER)
        )
    )

    ax.set_xticklabels([
        DEFENCE_LABELS[
            defence
        ]
        for defence
        in DEFENCE_ORDER
    ])

    ax.set_yticks(
        np.arange(
            len(CATEGORIES)
        )
    )

    ax.set_yticklabels([
        CATEGORY_LABELS[
            category
        ]
        for category
        in CATEGORIES
    ])

    ax.set_title(
        title,
        fontweight="bold",
        pad=14,
    )

    for i in range(
        len(CATEGORIES)
    ):
        for j in range(
            len(DEFENCE_ORDER)
        ):
            value = array[
                i,
                j,
            ]

            if np.isnan(
                value
            ):
                text = "—"

            elif percentage:
                text = (
                    f"{value * 100:.1f}%"
                )

            else:
                text = (
                    f"{value:.3f}"
                )

            colour = (
                "white"
                if (
                    np.isfinite(
                        value
                    )
                    and value > (
                        (
                            vmax
                            + vmin
                        )
                        / 2
                    )
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

    colourbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.025,
        pad=0.025,
    )

    if colourbar_label:
        colourbar.set_label(
            colourbar_label
        )

    fig.tight_layout()

    save_figure(
        fig,
        output_name,
    )


def make_plots(
    summary,
    overall,
):
    """Generate publication-style utility plots."""
    print(
        "\ncreating figures..."
    )

    set_plot_style()

    # ----------------------------------------------------------- 1 trigger

    heatmap(
        pivot_metric(
            summary,
            "trigger_rate",
        ),
        "Legitimate-input defence trigger rate",
        "fig01_trigger_rate",
        "YlOrRd",
        colourbar_label=
            "Trigger rate",
    )

    # ----------------------------------------------------------- 2 block

    heatmap(
        pivot_metric(
            summary,
            "block_rate",
        ),
        "Legitimate-input block rate",
        "fig02_block_rate",
        "Reds",
        colourbar_label=
            "Block rate",
    )

    # ----------------------------------------------------------- 3 preservation

    heatmap(
        pivot_metric(
            summary,
            "operational_preservation",
        ),
        "Legitimate-input operational preservation",
        "fig03_operational_preservation",
        "YlGn",
        colourbar_label=
            "Preservation rate",
    )

    # ----------------------------------------------------------- 4 representation

    heatmap(
        pivot_metric(
            summary,
            "representation_change_rate",
        ),
        "Representation change on legitimate inputs",
        "fig04_representation_change",
        "Purples",
        colourbar_label=
            "Representation change rate",
    )

    # ----------------------------------------------------------- 5 probability shift

    probability = pivot_metric(
        summary,
        "mean_abs_probability_shift",
    )

    max_probability = float(
        np.nanmax(
            probability.to_numpy(
                dtype=float
            )
        )
    )

    if (
        not np.isfinite(
            max_probability
        )
        or max_probability <= 0
    ):
        max_probability = 1

    heatmap(
        probability,
        "Mean defence-induced change in toxicity probability",
        "fig05_probability_shift",
        "Blues",
        percentage=False,
        vmin=0,
        vmax=max_probability,
        colourbar_label=
            "Mean |Δ P(toxic)|",
    )

    # ----------------------------------------------------------- 6 token ratio

    token_ratio = pivot_metric(
        summary,
        "mean_token_count_ratio",
    )

    max_ratio = float(
        np.nanmax(
            token_ratio.to_numpy(
                dtype=float
            )
        )
    )

    max_ratio = max(
        max_ratio,
        1.0,
    )

    heatmap(
        token_ratio,
        "Mean defended / canonical token-count ratio",
        "fig06_token_count_ratio",
        "viridis",
        percentage=False,
        vmin=0,
        vmax=max_ratio,
        colourbar_label=
            "Token-count ratio",
    )

    # ----------------------------------------------------------- 7 latency

    ordered = (
        overall
        .set_index(
            "defence"
        )
        .reindex(
            DEFENCE_ORDER
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            5.5,
        )
    )

    x = np.arange(
        len(
            DEFENCE_ORDER
        )
    )

    ax.bar(
        x,
        ordered[
            "median_latency_ms"
        ],
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            DEFENCE_LABELS[
                defence
            ].replace(
                "\n",
                " ",
            )
            for defence
            in DEFENCE_ORDER
        ],
        rotation=35,
        ha="right",
    )

    ax.set_ylabel(
        "Median preprocessing latency (ms)"
    )

    ax.set_title(
        "Defence preprocessing cost on complex legitimate text",
        fontweight="bold",
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "fig07_latency",
    )

    # ----------------------------------------------------------- 8 overall

    fig, ax = plt.subplots(
        figsize=(
            10.5,
            6,
        )
    )

    ordered = (
        overall
        .set_index(
            "defence"
        )
        .reindex(
            DEFENCE_ORDER
        )
    )

    x = np.arange(
        len(
            DEFENCE_ORDER
        )
    )

    width = 0.25

    ax.bar(
        x - width,
        ordered[
            "trigger_rate"
        ] * 100,
        width,
        label="Trigger",
    )

    ax.bar(
        x,
        ordered[
            "block_rate"
        ] * 100,
        width,
        label="Block",
    )

    ax.bar(
        x + width,
        (
            1
            - ordered[
                "operational_preservation"
            ]
        ) * 100,
        width,
        label="Utility loss",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            DEFENCE_LABELS[
                defence
            ].replace(
                "\n",
                " ",
            )
            for defence
            in DEFENCE_ORDER
        ],
        rotation=35,
        ha="right",
    )

    ax.set_ylabel(
        "Percent of 600 legitimate inputs"
    )

    ax.set_title(
        "Overall complex-input defence cost",
        fontweight="bold",
    )

    ax.legend(
        frameon=False,
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "fig08_overall_utility_cost",
    )

    print(
        "figures written to",
        FIG_DIR,
    )


# --------------------------------------------------------------- HTML helpers

def pct(value):
    if pd.isna(
        value
    ):
        return "—"

    return (
        f"{float(value) * 100:.1f}%"
    )


def make_summary_table(
    summary,
):
    table = summary.copy()

    table[
        "Category"
    ] = (
        table[
            "category"
        ]
        .astype(str)
        .map(
            CATEGORY_LABELS
        )
    )

    table[
        "Defence"
    ] = (
        table[
            "defence"
        ]
        .astype(str)
        .map(
            lambda value:
                DEFENCE_LABELS[
                    value
                ].replace(
                    "\n",
                    " ",
                )
        )
    )

    table[
        "Trigger"
    ] = table[
        "trigger_rate"
    ].map(
        pct
    )

    table[
        "Block"
    ] = table[
        "block_rate"
    ].map(
        pct
    )

    table[
        "Representation changed"
    ] = table[
        "representation_change_rate"
    ].map(
        pct
    )

    table[
        "Decision preserved"
    ] = table[
        "operational_preservation"
    ].map(
        pct
    )

    table[
        "Label flip"
    ] = table[
        "model_label_flip_rate_when_scored"
    ].map(
        pct
    )

    table[
        "Mean |ΔP|"
    ] = table[
        "mean_abs_probability_shift"
    ].map(
        lambda value:
            "—"
            if pd.isna(value)
            else f"{value:.4f}"
    )

    return table[[
        "Category",
        "Defence",
        "Trigger",
        "Block",
        "Representation changed",
        "Decision preserved",
        "Label flip",
        "Mean |ΔP|",
    ]].to_html(
        index=False,
        classes="data-table",
        border=0,
    )


# --------------------------------------------------------------- report

def make_report(
    dataset,
    summary,
    overall,
):
    """Generate local browser report."""
    counts = (
        dataset[
            "category"
        ]
        .value_counts()
        .reindex(
            CATEGORIES
        )
    )

    count_html = ""

    for category in CATEGORIES:
        count_html += f"""
        <div class="metric">
            <div class="big">{int(counts[category])}</div>
            <div class="small">{html.escape(CATEGORY_LABELS[category])}</div>
        </div>
        """

    table_html = make_summary_table(
        summary
    )

    figures = [
        (
            "Defence trigger rate",
            "fig01_trigger_rate.png",
        ),
        (
            "Legitimate block rate",
            "fig02_block_rate.png",
        ),
        (
            "Operational preservation",
            "fig03_operational_preservation.png",
        ),
        (
            "Representation changes",
            "fig04_representation_change.png",
        ),
        (
            "Probability shift",
            "fig05_probability_shift.png",
        ),
        (
            "Token-count change",
            "fig06_token_count_ratio.png",
        ),
        (
            "Latency",
            "fig07_latency.png",
        ),
        (
            "Overall utility cost",
            "fig08_overall_utility_cost.png",
        ),
    ]

    figures_html = ""

    for title, filename in figures:
        figures_html += f"""
        <section class="figure-card">
            <h3>{html.escape(title)}</h3>
            <img
                src="figures/{html.escape(filename)}"
                alt="{html.escape(title)}"
            >
        </section>
        """

    report = f"""
<!DOCTYPE html>
<html lang="en">

<head>
<meta charset="utf-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1"
>

<title>
Complex Legitimate-Text Utility Stress Test
</title>

<style>

:root {{
    --bg: #f5f7fb;
    --card: white;
    --text: #172033;
    --muted: #667085;
    --border: #e4e7ec;
    --accent: #3448c5;
}}

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
    line-height: 1.55;
}}

.wrap {{
    max-width: 1450px;
    margin: auto;
    padding: 36px 28px 80px;
}}

h1 {{
    margin-bottom: 6px;
}}

h2 {{
    margin-top: 48px;
}}

.subtitle {{
    color: var(--muted);
}}

.cards {{
    display: grid;
    grid-template-columns:
        repeat(
            auto-fit,
            minmax(160px, 1fr)
        );
    gap: 15px;
    margin: 28px 0;
}}

.metric {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 17px;
}}

.big {{
    font-size: 27px;
    font-weight: 700;
}}

.small {{
    color: var(--muted);
}}

.note {{
    margin: 20px 0;
    padding: 16px 18px;
    background: #eef2ff;
    border-left:
        4px solid var(--accent);
    border-radius: 9px;
}}

.warning {{
    margin: 20px 0;
    padding: 16px 18px;
    background: #fff7ed;
    border-left:
        4px solid #f97316;
    border-radius: 9px;
}}

.figure-grid {{
    display: grid;
    grid-template-columns:
        repeat(
            auto-fit,
            minmax(500px, 1fr)
        );
    gap: 22px;
}}

.figure-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 18px;
}}

.figure-card img {{
    display: block;
    width: 100%;
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
    padding: 10px;
    background: #f9fafb;
    border-bottom:
        2px solid var(--border);
    white-space: nowrap;
}}

.data-table td {{
    padding: 9px 10px;
    border-bottom:
        1px solid var(--border);
    white-space: nowrap;
}}

code {{
    background: #eef2ff;
    padding: 2px 5px;
    border-radius: 5px;
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

<h1>
Complex Legitimate-Text Utility Stress Test
</h1>

<div class="subtitle">
600 legitimate complex inputs × 7 tokenizer defences
</div>

<div class="cards">
{count_html}
</div>


<h2>Why this experiment exists</h2>

<div class="note">

The earlier WikiText and Jigsaw clean controls mostly measure
ordinary English prose.

This experiment instead asks whether a defence that works against
tokenization attacks accidentally harms legitimate inputs containing
code, URLs, emoji, typos, non-English scripts or multiple scripts.

</div>


<h2>What “utility” means here</h2>

<p>
For every input we first obtain the victim's ordinary canonical
prediction.
We then apply each defence to the <strong>same exact input</strong>.
</p>

<p>
The main metric is:
</p>

<p>
<code>
not blocked AND classifier decision unchanged
</code>
</p>

<div class="warning">

The victim is an English toxicity classifier.
Therefore its prediction on Chinese, Arabic, code, URLs, etc. is
<strong>not treated as semantic ground truth</strong>.

For these categories we measure defence-induced stability,
not classification accuracy.

</div>


<h2>Dataset provenance</h2>

<ul>

<li>
<strong>Code:</strong>
100 real Python functions from CodeSearchNet.
</li>

<li>
<strong>URLs:</strong>
100 real GitHub source URLs from CodeSearchNet metadata.
</li>

<li>
<strong>Emoji:</strong>
100 GoEmotions Reddit comments containing emoji;
neutral examples are preferred.
</li>

<li>
<strong>Misspellings:</strong>
100 constructed one-typo versions of Jigsaw benign comments.
</li>

<li>
<strong>Non-English:</strong>
100 FLORES sentences across Hindi, Arabic, Russian,
Chinese and Bengali.
</li>

<li>
<strong>Mixed-script:</strong>
100 constructed but legitimate bilingual FLORES pairs:
English plus an aligned translation.
</li>

</ul>


<h2>Results</h2>

<div class="table-wrap">
{table_html}
</div>


<h2>Visual results</h2>

<div class="figure-grid">
{figures_html}
</div>


<h2>How to interpret the metrics</h2>

<div class="note">

<strong>Trigger:</strong>
the defence explicitly marked or transformed the legitimate input.

<br><br>

<strong>Block:</strong>
the legitimate input would be rejected by the deployment policy.

<br><br>

<strong>Representation changed:</strong>
the token IDs reaching the victim differ from ordinary canonical
WordPiece encoding.

<br><br>

<strong>Decision preserved:</strong>
the defence did not block the input and did not change the victim's
original classification.

<br><br>

<strong>Label flip:</strong>
among inputs that still reached the classifier, the defence changed
the classifier's binary decision.

</div>


<h2>Literature rationale</h2>

<ul>

<li>
<strong>Unicode UTS #39:</strong>
distinguishes mixed-script and confusable text and provides
mixed-script/confusable detection mechanisms.
</li>

<li>
<strong>Bad Characters:</strong>
demonstrates the security relevance of invisible characters,
homoglyphs and Unicode reordering, motivating sanitisation.
</li>

<li>
<strong>Broken-Token:</strong>
reports that CPT behaviour changes for multilingual inputs,
particularly Chinese and Arabic, making multilingual clean testing
important.
</li>

<li>
<strong>CodeSearchNet:</strong>
provides real open-source program text rather than synthetic code.
</li>

<li>
<strong>GoEmotions:</strong>
provides naturally occurring Reddit language from which
emoji-containing legitimate inputs are sampled.
</li>

<li>
<strong>FLORES:</strong>
provides professionally translated aligned multilingual text.
</li>

</ul>


<h2>Scope</h2>

<p>
This is a structured utility stress test rather than a population
estimate of all possible user input.
The misspelling and mixed-script categories are constructed by design
and are explicitly labelled as such in the exported dataset.
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


# --------------------------------------------------------------- console results

def print_results(
    summary,
    overall,
):
    print(
        "\n"
        + "=" * 88
    )

    print(
        "COMPLEX LEGITIMATE-TEXT UTILITY"
    )

    print(
        "=" * 88
    )

    for category in CATEGORIES:
        print(
            "\n",
            CATEGORY_LABELS[
                category
            ],
        )

        group = summary[
            summary[
                "category"
            ].astype(str)
            == category
        ]

        for _, row in group.iterrows():
            defence = str(
                row["defence"]
            )

            print(
                f"  "
                f"{DEFENCE_LABELS[defence].replace(chr(10), ' '):24s}"
                f" | trigger "
                f"{row['trigger_rate']:.3f}"
                f" | block "
                f"{row['block_rate']:.3f}"
                f" | preserve "
                f"{row['operational_preservation']:.3f}"
                f" | rep-change "
                f"{row['representation_change_rate']:.3f}"
                f" | ΔP "
                f"{row['mean_abs_probability_shift']:.4f}"
            )

    print(
        "\n"
        + "=" * 88
    )

    print(
        "OVERALL ACROSS ALL 600 INPUTS"
    )

    print(
        "=" * 88
    )

    for _, row in overall.iterrows():
        defence = str(
            row[
                "defence"
            ]
        )

        print(
            f"  "
            f"{DEFENCE_LABELS[defence].replace(chr(10), ' '):24s}"
            f" | trigger "
            f"{row['trigger_rate']:.3f}"
            f" | block "
            f"{row['block_rate']:.3f}"
            f" | preserve "
            f"{row['operational_preservation']:.3f}"
            f" | rep-change "
            f"{row['representation_change_rate']:.3f}"
        )


# --------------------------------------------------------------- run

if __name__ == "__main__":

    print(
        "Step 11 - complex legitimate-text utility stress test"
    )

    print(
        "\nCPT thresholds remain frozen:"
    )

    print(
        json.dumps(
            thresholds,
            indent=2,
        )
    )

    print(
        "\nNo attack data will be used to tune any defence."
    )

    # ----------------------------------------------------------- dataset

    dataset = build_dataset()

    # ----------------------------------------------------------- experiment

    rows = run_utility_test(
        dataset
    )

    # ----------------------------------------------------------- statistics

    summary = summarise(
        rows
    )

    overall = summarise_overall(
        rows
    )

    # ----------------------------------------------------------- visualisations

    make_plots(
        summary,
        overall,
    )

    # ----------------------------------------------------------- report

    make_report(
        dataset,
        summary,
        overall,
    )

    # ----------------------------------------------------------- terminal

    print_results(
        summary,
        overall,
    )

    print(
        "\n"
        + "=" * 88
    )

    print(
        "STEP 11 DONE"
    )

    print(
        "=" * 88
    )

    print(
        "\nOutputs:"
    )

    print(
        " ",
        DATASET_FILE,
    )

    print(
        " ",
        ROWS_FILE,
    )

    print(
        " ",
        SUMMARY_FILE,
    )

    print(
        " ",
        OVERALL_FILE,
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
                "\nopened Step 11 report in browser"
            )

        except Exception as exc:
            print(
                "\ncould not automatically open report:",
                exc,
            )

            print(
                "open manually:",
                REPORT_FILE.resolve(),
            )