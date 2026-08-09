"""
final current-study analysis before dissertation writing.

no model calls.
no new attacks.
no threshold tuning.

makes:
1. defence coverage matrix
2. attack success by clean model confidence
3. small factual observations for notes/observations.md
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------------------- config

SEED = 42
BOOTSTRAPS = 5000

OUT = Path("results/final_current")
FIG = OUT / "figures"
NOTES = Path("notes/observations.md")

OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
NOTES.parent.mkdir(parents=True, exist_ok=True)


ATTACK_ORDER = [
    "TokenBreak",
    "AdvTok",
    "Unicode invisible",
    "Unicode homoglyph",
    "Unicode compatibility",
    "Unicode reorder",
]


DEFENCE_ORDER = [
    "Tokenizer translation",
    "Canonical reject",
    "Canonical replace",
    "Unicode sanitiser",
    "NFKC + confusables",
    "Global CPT",
    "Window CPT",
]


# fixed before looking at this analysis
MARGIN_BINS = [
    0.5,
    0.6,
    0.8,
    1.0000001,
]

MARGIN_LABELS = [
    "(0.5, 0.6]",
    "(0.6, 0.8]",
    "(0.8, 1.0]",
]


# --------------------------------------------------------------- helpers

def find_file(*candidates):
    """find a file either in its normal folder or project root."""
    for candidate in candidates:
        path = Path(candidate)

        if path.exists():
            return path

    tried = "\n  ".join(
        str(x)
        for x in candidates
    )

    raise FileNotFoundError(
        f"could not find file. tried:\n  {tried}"
    )


def attack_name(value):
    """make attack names consistent across old csvs."""
    x = (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
    )

    mapping = {
        "tokenbreak":
            "TokenBreak",

        "advtok":
            "AdvTok",

        "unicode invisible":
            "Unicode invisible",

        "invisible":
            "Unicode invisible",

        "unicode homoglyph":
            "Unicode homoglyph",

        "homoglyph":
            "Unicode homoglyph",

        "unicode compat":
            "Unicode compatibility",

        "unicode compatibility":
            "Unicode compatibility",

        "compat":
            "Unicode compatibility",

        "compatibility":
            "Unicode compatibility",

        "unicode reorder":
            "Unicode reorder",

        "reorder":
            "Unicode reorder",
    }

    return mapping.get(
        x,
        str(value).strip(),
    )


def defence_name(value):
    """make defence names consistent."""
    x = (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
    )

    mapping = {
        "tokenizer translation":
            "Tokenizer translation",

        "canonical reject":
            "Canonical reject",

        "canonical replace":
            "Canonical replace",

        "unicode sanitise":
            "Unicode sanitiser",

        "unicode sanitiser":
            "Unicode sanitiser",

        "nfkc confusable":
            "NFKC + confusables",

        "nfkc + confusables":
            "NFKC + confusables",

        "global cpt":
            "Global CPT",

        "cpt global":
            "Global CPT",

        "window cpt":
            "Window CPT",

        "cpt window":
            "Window CPT",
    }

    return mapping.get(
        x,
        str(value).strip(),
    )


def wilson_rate(values):
    """wilson 95% ci for a binary success rate."""
    values = np.asarray(
        values,
        dtype=int,
    )

    n = len(values)

    if n == 0:
        return np.nan, np.nan

    successes = int(
        values.sum()
    )

    p = successes / n

    z = 1.959963984540054

    denominator = (
        1
        + (z ** 2 / n)
    )

    centre = (
        p
        + (z ** 2 / (2 * n))
    ) / denominator

    half = (
        z
        * np.sqrt(
            (
                p * (1 - p) / n
            )
            + (
                z ** 2
                / (4 * n ** 2)
            )
        )
        / denominator
    )

    lo = centre - half
    hi = centre + half

    # keep the interval numerically consistent with p.
    # this only corrects floating-point error at boundaries.
    lo = float(
        np.clip(
            lo,
            0.0,
            p,
        )
    )

    hi = float(
        np.clip(
            hi,
            p,
            1.0,
        )
    )

    return lo, hi


# --------------------------------------------------------------- coverage

def coverage_status(row):
    """
    classify the final paired defence result.

    complete:
        a real attack existed and defended asr became zero.

    partial:
        the entire reduction ci is above zero,
        but some attacks still succeed.

    harmful:
        defended asr became significantly worse.

    no base attack:
        the original attack already had zero asr.

    no clear effect:
        no clear improvement from the paired ci.
    """

    undefended = float(
        row["paired_undefended_asr"]
    )

    defended = float(
        row["paired_defended_asr"]
    )

    reduction_lo = float(
        row["asr_reduction_ci_low_pp"]
    )

    reduction_hi = float(
        row["asr_reduction_ci_high_pp"]
    )

    # negative reduction means the defence made things worse
    if reduction_hi < 0:
        return "HARMFUL"

    # important for invisible unicode
    if (
        undefended == 0
        and defended == 0
    ):
        return "NO BASE ATTACK"

    if (
        undefended > 0
        and defended == 0
    ):
        return "COMPLETE"

    # ci entirely above zero
    if reduction_lo > 0:
        return "PARTIAL"

    return "NO CLEAR EFFECT"


def mechanism_note(row):
    """
    small mechanism note.

    this does not decide whether a defence is good.
    it only says how it acted.
    """

    detect = float(
        row["attack_detection_rate"]
    )

    block = float(
        row["attack_block_rate"]
    )

    reencode = float(
        row["reencode_rate"]
    )

    if (
        detect >= 0.95
        and block >= 0.95
    ):
        return "detect+block"

    if detect >= 0.95:
        return "detect/repair"

    if block >= 0.95:
        return "block"

    if reencode >= 0.95:
        return "reencode"

    return ""


def build_coverage():
    metrics_path = find_file(
        "results/step10/attack_metrics.csv",
        "attack_metrics.csv",
    )

    df = pd.read_csv(
        metrics_path
    )

    needed = {
        "attack",
        "defence",
        "n_clean_retained",
        "paired_undefended_asr",
        "paired_defended_asr",
        "asr_reduction_pp",
        "asr_reduction_ci_low_pp",
        "asr_reduction_ci_high_pp",
        "clean_retention",
        "attack_detection_rate",
        "attack_block_rate",
        "reencode_rate",
    }

    missing = sorted(
        needed
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "attack_metrics.csv is missing columns: "
            f"{missing}"
        )

    df = df.copy()

    df["attack"] = (
        df["attack"]
        .map(attack_name)
    )

    df["defence"] = (
        df["defence"]
        .map(defence_name)
    )

    df["coverage_status"] = df.apply(
        coverage_status,
        axis=1,
    )

    df["mechanism_note"] = df.apply(
        mechanism_note,
        axis=1,
    )

    keep = [
        "attack",
        "defence",
        "n_clean_retained",
        "paired_undefended_asr",
        "paired_defended_asr",
        "asr_reduction_pp",
        "asr_reduction_ci_low_pp",
        "asr_reduction_ci_high_pp",
        "clean_retention",
        "attack_detection_rate",
        "attack_block_rate",
        "reencode_rate",
        "coverage_status",
        "mechanism_note",
    ]

    out = df[keep].copy()

    out.to_csv(
        OUT / "coverage_matrix.csv",
        index=False,
    )

    # ----------------------------------------------------------- figure

    status_code = {
        "NO BASE ATTACK": 0,
        "NO CLEAR EFFECT": 1,
        "PARTIAL": 2,
        "COMPLETE": 3,
        "HARMFUL": 4,
    }

    grid = np.full(
        (
            len(DEFENCE_ORDER),
            len(ATTACK_ORDER),
        ),
        np.nan,
        dtype=float,
    )

    annotations = [
        [
            ""
            for _ in ATTACK_ORDER
        ]
        for _ in DEFENCE_ORDER
    ]

    for _, row in out.iterrows():

        if row["attack"] not in ATTACK_ORDER:
            continue

        if row["defence"] not in DEFENCE_ORDER:
            continue

        y = DEFENCE_ORDER.index(
            row["defence"]
        )

        x = ATTACK_ORDER.index(
            row["attack"]
        )

        status = row[
            "coverage_status"
        ]

        grid[y, x] = status_code[
            status
        ]

        defended = (
            100
            * float(
                row["paired_defended_asr"]
            )
        )

        mechanism = row[
            "mechanism_note"
        ]

        text = (
            f"{status}\n"
            f"ASR {defended:.1f}%"
        )

        if mechanism:
            text += (
                f"\n{mechanism}"
            )

        annotations[y][x] = text

    fig, ax = plt.subplots(
        figsize=(15, 8),
        constrained_layout=True,
    )

    image = ax.imshow(
        grid,
        aspect="auto",
        vmin=0,
        vmax=4,
    )

    ax.set_xticks(
        range(
            len(ATTACK_ORDER)
        )
    )

    ax.set_xticklabels(
        ATTACK_ORDER,
        rotation=25,
        ha="right",
    )

    ax.set_yticks(
        range(
            len(DEFENCE_ORDER)
        )
    )

    ax.set_yticklabels(
        DEFENCE_ORDER
    )

    ax.set_xlabel(
        "Attack"
    )

    ax.set_ylabel(
        "Defence"
    )

    ax.set_title(
        "Attack × defence coverage "
        "on the final paired WordPiece evaluation"
    )

    for y in range(
        len(DEFENCE_ORDER)
    ):
        for x in range(
            len(ATTACK_ORDER)
        ):
            if annotations[y][x]:
                ax.text(
                    x,
                    y,
                    annotations[y][x],
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.025,
        pad=0.02,
    )

    cbar.set_ticks(
        [0, 1, 2, 3, 4]
    )

    cbar.set_ticklabels(
        [
            "no base attack",
            "no clear effect",
            "partial",
            "complete",
            "harmful",
        ]
    )

    fig.savefig(
        FIG / "coverage_matrix.png",
        dpi=220,
    )

    fig.savefig(
        FIG / "coverage_matrix.pdf",
    )

    plt.close(fig)

    return out


# --------------------------------------------------------------- attack rows

def load_attack_rows():

    tokenbreak = pd.read_csv(
        find_file(
            "data/tokenbreak_results.csv",
            "tokenbreak_results.csv",
        )
    )

    advtok = pd.read_csv(
        find_file(
            "data/advtok_results.csv",
            "advtok_results.csv",
        )
    )

    try:
        unicode_df = pd.read_csv(
            find_file(
                "data/unicode_results.csv",
                "unicode_results.csv",
            )
        )

    except FileNotFoundError:

        parts = []

        for family in [
            "invisible",
            "homoglyph",
            "compat",
            "reorder",
        ]:

            parts.append(
                pd.read_csv(
                    find_file(
                        f"data/unicode_{family}.csv",
                        f"unicode_{family}.csv",
                    )
                )
            )

        unicode_df = pd.concat(
            parts,
            ignore_index=True,
        )

    all_rows = pd.concat(
        [
            tokenbreak,
            advtok,
            unicode_df,
        ],
        ignore_index=True,
        sort=False,
    )

    needed = {
        "attack",
        "clean_toxic_prob",
        "attacked_toxic_prob",
        "attack_success",
    }

    missing = sorted(
        needed
        - set(all_rows.columns)
    )

    if missing:
        raise RuntimeError(
            "attack result csvs are missing columns: "
            f"{missing}"
        )

    all_rows = all_rows.copy()

    all_rows["attack"] = (
        all_rows["attack"]
        .map(attack_name)
    )

    all_rows[
        "clean_probability_bin"
    ] = pd.cut(
        all_rows[
            "clean_toxic_prob"
        ],
        bins=MARGIN_BINS,
        labels=MARGIN_LABELS,
        right=True,
        include_lowest=False,
    )

    # every eligible row should be above the 0.5 classifier threshold
    if (
        all_rows[
            "clean_probability_bin"
        ]
        .isna()
        .any()
    ):

        bad = all_rows.loc[
            all_rows[
                "clean_probability_bin"
            ].isna(),
            [
                "attack",
                "clean_toxic_prob",
            ],
        ]

        raise RuntimeError(
            "some eligible rows fell outside "
            "the frozen confidence bins:\n"
            + bad.to_string(
                index=False
            )
        )

    return all_rows


# --------------------------------------------------------------- margins

def build_margin_analysis():

    rows = load_attack_rows()

    records = []

    for attack in ATTACK_ORDER:

        attack_rows = rows[
            rows["attack"]
            == attack
        ]

        for label in MARGIN_LABELS:

            group = attack_rows[
                attack_rows[
                    "clean_probability_bin"
                ].astype(str)
                == label
            ]

            successes = (
                group["attack_success"]
                .astype(int)
                .to_numpy()
            )

            lo, hi = wilson_rate(
                successes
            )

            records.append(
                {
                    "attack":
                        attack,

                    "clean_probability_bin":
                        label,

                    "n":
                        len(group),

                    "successes":
                        int(
                            successes.sum()
                        ),

                    "asr":
                        float(
                            successes.mean()
                        )
                        if len(successes)
                        else np.nan,

                    "asr_ci_low":
                        lo,

                    "asr_ci_high":
                        hi,

                    "mean_clean_toxic_prob":
                        float(
                            group[
                                "clean_toxic_prob"
                            ].mean()
                        ),

                    "median_clean_toxic_prob":
                        float(
                            group[
                                "clean_toxic_prob"
                            ].median()
                        ),

                    "mean_attacked_toxic_prob":
                        float(
                            group[
                                "attacked_toxic_prob"
                            ].mean()
                        ),
                }
            )

    stratified = pd.DataFrame(
        records
    )

    stratified.to_csv(
        OUT
        / "margin_stratified_asr.csv",
        index=False,
    )

    # ----------------------------------------------------------- overall

    overall_records = []

    for attack in ATTACK_ORDER:

        group = rows[
            rows["attack"]
            == attack
        ]

        high = group[
            group[
                "clean_toxic_prob"
            ] > 0.8
        ]

        low = group[
            group[
                "clean_toxic_prob"
            ] <= 0.6
        ]

        overall_records.append(
            {
                "attack":
                    attack,

                "n":
                    len(group),

                "successes":
                    int(
                        group[
                            "attack_success"
                        ].sum()
                    ),

                "asr":
                    float(
                        group[
                            "attack_success"
                        ].mean()
                    ),

                "n_clean_prob_le_0_60":
                    len(low),

                "successes_clean_prob_le_0_60":
                    int(
                        low[
                            "attack_success"
                        ].sum()
                    ),

                "n_clean_prob_gt_0_80":
                    len(high),

                "successes_clean_prob_gt_0_80":
                    int(
                        high[
                            "attack_success"
                        ].sum()
                    ),

                "asr_clean_prob_gt_0_80":
                    float(
                        high[
                            "attack_success"
                        ].mean()
                    ),

                "min_clean_toxic_prob":
                    float(
                        group[
                            "clean_toxic_prob"
                        ].min()
                    ),

                "median_clean_toxic_prob":
                    float(
                        group[
                            "clean_toxic_prob"
                        ].median()
                    ),
            }
        )

    overall = pd.DataFrame(
        overall_records
    )

    overall.to_csv(
        OUT
        / "margin_overall.csv",
        index=False,
    )

    # ----------------------------------------------------------- figure

    fig, ax = plt.subplots(
        figsize=(13, 7),
        constrained_layout=True,
    )

    x = np.arange(
        len(ATTACK_ORDER),
        dtype=float,
    )

    width = 0.24

    for i, label in enumerate(
        MARGIN_LABELS
    ):

        part = (
            stratified[
                stratified[
                    "clean_probability_bin"
                ]
                == label
            ]
            .set_index("attack")
            .reindex(ATTACK_ORDER)
        )

        y = (
            part["asr"]
            .to_numpy(
                dtype=float
            )
        )

        lo = (
            part[
                "asr_ci_low"
            ]
            .to_numpy(
                dtype=float
            )
        )

        hi = (
            part[
                "asr_ci_high"
            ]
            .to_numpy(
                dtype=float
            )
        )

        error = np.vstack(
            [
                y - lo,
                hi - y,
            ]
        )

        ax.bar(
            x
            + (i - 1)
            * width,
            y,
            width=width,
            label=label,
            yerr=error,
            capsize=3,
        )

    ax.axhline(
        0.5,
        linewidth=1,
        linestyle="--",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        ATTACK_ORDER,
        rotation=20,
        ha="right",
    )

    ax.set_ylim(
        0,
        1.08,
    )

    ax.set_ylabel(
        "Attack success rate"
    )

    ax.set_xlabel(
        "Attack"
    )

    ax.set_title(
        "Attack success by clean model confidence"
    )

    ax.legend(
        title="Clean P(toxic)"
    )

    fig.savefig(
        FIG
        / "margin_stratified_asr.png",
        dpi=220,
    )

    fig.savefig(
        FIG
        / "margin_stratified_asr.pdf",
    )

    plt.close(fig)

    return (
        stratified,
        overall,
    )


# --------------------------------------------------------------- observations

def write_observations(
    coverage,
    overall,
):
    """
    add only short factual findings.

    do not keep appending the same block
    if the script is run twice.
    """

    marker = (
        "## final current-study "
        "margin/coverage observations"
    )

    if NOTES.exists():
        old = NOTES.read_text(
            encoding="utf-8"
        )

    else:
        old = (
            "# observations\n\n"
        )

    if marker in old:
        print(
            "observations block already exists; "
            "not appending it again"
        )
        return

    lines = [
        "",
        marker,
    ]

    tb = overall[
        overall["attack"]
        == "TokenBreak"
    ].iloc[0]

    adv = overall[
        overall["attack"]
        == "AdvTok"
    ].iloc[0]

    lines.append(
        "- clean-confidence distribution for the "
        "156 eligible WordPiece toxic rows: "
        "10 are at P(toxic)<=0.60, "
        "18 are in (0.60,0.80], "
        "and 128 are above 0.80."
    )

    lines.append(
        "- TokenBreak succeeds on "
        f"{int(tb['successes_clean_prob_gt_0_80'])}/"
        f"{int(tb['n_clean_prob_gt_0_80'])} "
        "high-confidence rows "
        f"({100 * tb['asr_clean_prob_gt_0_80']:.1f}%); "
        "its high ASR is therefore not explained "
        "only by borderline clean examples."
    )

    lines.append(
        "- AdvTok succeeds on "
        f"{int(adv['successes_clean_prob_gt_0_80'])}/"
        f"{int(adv['n_clean_prob_gt_0_80'])} "
        "high-confidence rows "
        f"({100 * adv['asr_clean_prob_gt_0_80']:.1f}%)."
    )

    harmful = coverage[
        coverage[
            "coverage_status"
        ]
        == "HARMFUL"
    ]

    for _, row in harmful.iterrows():

        lines.append(
            "- harmful defence interaction: "
            f"{row['defence']} changes "
            f"{row['attack']} paired ASR from "
            f"{100 * row['paired_undefended_asr']:.1f}% "
            "to "
            f"{100 * row['paired_defended_asr']:.1f}%."
        )

    with NOTES.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            "\n".join(lines)
            + "\n"
        )


# --------------------------------------------------------------- summary

def write_summary(
    coverage,
    stratified,
    overall,
):

    lines = []

    lines.append(
        "FINAL CURRENT-STUDY ANALYSIS"
    )

    lines.append(
        "=" * 54
    )

    lines.append("")

    lines.append(
        "coverage status counts"
    )

    lines.append(
        coverage[
            "coverage_status"
        ]
        .value_counts()
        .to_string()
    )

    lines.append("")

    lines.append(
        "margin-stratified attack success"
    )

    for attack in ATTACK_ORDER:

        lines.append(
            f"\n{attack}"
        )

        part = stratified[
            stratified[
                "attack"
            ]
            == attack
        ]

        for _, row in part.iterrows():

            lines.append(
                f"  "
                f"{row['clean_probability_bin']:>11} : "
                f"{int(row['successes'])}/"
                f"{int(row['n'])} "
                f"= {100 * row['asr']:.1f}% "
                f"[95% CI "
                f"{100 * row['asr_ci_low']:.1f}, "
                f"{100 * row['asr_ci_high']:.1f}]"
            )

    lines.append(
        "\n\nhigh-confidence check: "
        "P(toxic) > 0.80"
    )

    for _, row in overall.iterrows():

        lines.append(
            f"  "
            f"{row['attack']:<22} "
            f"{int(row['successes_clean_prob_gt_0_80'])}/"
            f"{int(row['n_clean_prob_gt_0_80'])} "
            f"= "
            f"{100 * row['asr_clean_prob_gt_0_80']:.1f}%"
        )

    text = (
        "\n".join(lines)
        + "\n"
    )

    (
        OUT
        / "headline_summary.txt"
    ).write_text(
        text,
        encoding="utf-8",
    )

    print(text)


# --------------------------------------------------------------- run

if __name__ == "__main__":

    print(
        "final current-study analysis"
    )

    print(
        "python :",
        sys.executable,
    )

    print(
        "seed   :",
        SEED,
    )

    print(
        "margin ci : Wilson 95%"
    )

    print(
        "note   : no model calls, "
        "no tuning, existing frozen results only"
    )

    coverage = build_coverage()

    stratified, overall = (
        build_margin_analysis()
    )

    write_observations(
        coverage,
        overall,
    )

    write_summary(
        coverage,
        stratified,
        overall,
    )

    print(
        "\noutputs:"
    )

    print(
        " ",
        OUT
        / "coverage_matrix.csv",
    )

    print(
        " ",
        OUT
        / "margin_stratified_asr.csv",
    )

    print(
        " ",
        OUT
        / "margin_overall.csv",
    )

    print(
        " ",
        OUT
        / "headline_summary.txt",
    )

    print(
        " ",
        FIG
        / "coverage_matrix.png",
    )

    print(
        " ",
        FIG
        / "margin_stratified_asr.png",
    )

    print(
        " ",
        NOTES,
    )

    print(
        "\nfinal current-study analysis done"
    )