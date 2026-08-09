"""
build final dissertation figures and tables.

no model calls.
no attacks.
no defence reruns.
no threshold tuning.

reads only frozen result artefacts and produces
publication-ready figures/tables.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


# ---------------------------------------------------------------- config

ROOT = Path(".")

OUT = Path("results/dissertation")
FIG = OUT / "figures"
TABLE = OUT / "tables"

OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
TABLE.mkdir(parents=True, exist_ok=True)


ATTACK_ORDER = [
    "TokenBreak",
    "AdvTok",
    "Unicode invisible",
    "Unicode homoglyph",
    "Unicode compatibility",
    "Unicode reorder",
]


SHORT_ATTACK = {
    "TokenBreak": "TokenBreak",
    "AdvTok": "AdvTok",
    "Unicode invisible": "Invisible",
    "Unicode homoglyph": "Homoglyph",
    "Unicode compatibility": "Compatibility",
    "Unicode reorder": "Reorder",
}


DEFENCE_ORDER = [
    "Tokenizer translation",
    "Canonical reject",
    "Canonical replace",
    "Unicode sanitiser",
    "NFKC + confusables",
    "Global CPT",
    "Window CPT",
]


MARGIN_ORDER = [
    "(0.5, 0.6]",
    "(0.6, 0.8]",
    "(0.8, 1.0]",
]


STATUS_ORDER = [
    "NO BASE ATTACK",
    "NO CLEAR EFFECT",
    "PARTIAL",
    "COMPLETE",
    "HARMFUL",
]


# keep all figures visually consistent
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ---------------------------------------------------------------- helpers

def find_file(*candidates):
    """return first existing file."""
    for candidate in candidates:
        path = Path(candidate)

        if path.exists():
            return path

    raise FileNotFoundError(
        "could not find any of:\n"
        + "\n".join(
            f"  {x}"
            for x in candidates
        )
    )


def save_figure(fig, name):
    """save png and pdf."""
    fig.savefig(
        FIG / f"{name}.png",
        dpi=300,
    )

    fig.savefig(
        FIG / f"{name}.pdf",
    )

    plt.close(fig)


def normalise_attack(value):
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

        "invisible":
            "Unicode invisible",

        "unicode invisible":
            "Unicode invisible",

        "homoglyph":
            "Unicode homoglyph",

        "unicode homoglyph":
            "Unicode homoglyph",

        "compat":
            "Unicode compatibility",

        "compatibility":
            "Unicode compatibility",

        "unicode compat":
            "Unicode compatibility",

        "unicode compatibility":
            "Unicode compatibility",

        "reorder":
            "Unicode reorder",

        "unicode reorder":
            "Unicode reorder",
    }

    return mapping.get(
        x,
        str(value).strip(),
    )


def normalise_defence(value):
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

        "nfkc confusables":
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


def as_percent(values):
    """
    convert fractions to percentages.

    accepts either:
    0.488 -> 48.8
    or
    48.8 -> 48.8
    """

    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    finite = values.dropna()

    if len(finite) == 0:
        return values

    if finite.abs().max() <= 1.5:
        return values * 100

    return values


def choose_column(df, candidates):
    """find first available column from a list."""
    for name in candidates:
        if name in df.columns:
            return name

    return None


def annotate_heatmap(ax, matrix, fmt=".1f"):
    """write values inside numeric heatmap."""
    for y in range(
        matrix.shape[0]
    ):
        for x in range(
            matrix.shape[1]
        ):
            value = matrix[y, x]

            if np.isnan(value):
                continue

            ax.text(
                x,
                y,
                format(
                    value,
                    fmt,
                ),
                ha="center",
                va="center",
                fontsize=8,
            )


# ---------------------------------------------------------------- load data

def load_attack_rows():
    """load all six frozen attack variants."""

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

    unicode_df = pd.read_csv(
        find_file(
            "data/unicode_results.csv",
            "unicode_results.csv",
        )
    )

    rows = pd.concat(
        [
            tokenbreak,
            advtok,
            unicode_df,
        ],
        ignore_index=True,
        sort=False,
    )

    rows = rows.copy()

    rows["attack"] = (
        rows["attack"]
        .map(normalise_attack)
    )

    return rows


def load_coverage():
    """load frozen final coverage matrix."""

    df = pd.read_csv(
        find_file(
            "results/final_current/coverage_matrix.csv",
            "coverage_matrix.csv",
        )
    )

    df = df.copy()

    df["attack"] = (
        df["attack"]
        .map(normalise_attack)
    )

    df["defence"] = (
        df["defence"]
        .map(normalise_defence)
    )

    return df


def load_margin():
    df = pd.read_csv(
        find_file(
            "results/final_current/margin_stratified_asr.csv",
            "margin_stratified_asr.csv",
        )
    )

    df = df.copy()

    df["attack"] = (
        df["attack"]
        .map(normalise_attack)
    )

    return df


def load_complex_utility():
    """
    load operational preservation by defence and
    legitimate-input category.

    first tries the summary file.
    if needed, falls back to row-level data.
    """

    summary = pd.read_csv(
        find_file(
            "results/step11/complex_utility_summary.csv",
            "complex_utility_summary.csv",
        )
    )

    defence_col = choose_column(
        summary,
        [
            "defence",
            "defense",
        ],
    )

    category_col = choose_column(
        summary,
        [
            "category",
            "input_category",
            "input_type",
            "type",
        ],
    )

    preserve_col = choose_column(
        summary,
        [
            "operational_preservation",
            "operational_preservation_rate",
            "operational_preserve_rate",
            "preservation_rate",
            "operational_retention",
        ],
    )

    if (
        defence_col
        and category_col
        and preserve_col
    ):
        out = summary[
            [
                defence_col,
                category_col,
                preserve_col,
            ]
        ].copy()

        out.columns = [
            "defence",
            "category",
            "operational_preservation",
        ]

    else:
        # fallback to row-level results
        rows = pd.read_csv(
            find_file(
                "results/step11/complex_utility_rows.csv",
                "complex_utility_rows.csv",
            )
        )

        defence_col = choose_column(
            rows,
            [
                "defence",
                "defense",
            ],
        )

        category_col = choose_column(
            rows,
            [
                "category",
                "input_category",
                "input_type",
                "type",
            ],
        )

        preserve_col = choose_column(
            rows,
            [
                "operational_preserved",
                "operational_preservation",
                "operational_keep",
                "preserved",
            ],
        )

        if not all(
            [
                defence_col,
                category_col,
                preserve_col,
            ]
        ):
            raise RuntimeError(
                "could not locate complex-utility "
                "columns.\n"
                "summary columns:\n"
                f"{list(summary.columns)}\n"
                "row-level columns:\n"
                f"{list(rows.columns)}"
            )

        out = (
            rows.groupby(
                [
                    defence_col,
                    category_col,
                ],
                as_index=False,
            )[preserve_col]
            .mean()
        )

        out.columns = [
            "defence",
            "category",
            "operational_preservation",
        ]

    out["defence"] = (
        out["defence"]
        .map(normalise_defence)
    )

    out[
        "operational_preservation"
    ] = as_percent(
        out[
            "operational_preservation"
        ]
    )

    return out


# ---------------------------------------------------------------- figure 1

def figure_pipeline():
    """main experiment-design diagram."""

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    ax.set_xlim(
        0,
        12,
    )

    ax.set_ylim(
        0,
        10,
    )

    ax.axis("off")

    def box(
        x,
        y,
        w,
        h,
        text,
        fontsize=10,
    ):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03",
            linewidth=1.4,
            fill=False,
        )

        ax.add_patch(
            patch
        )

        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
        )

    def arrow(
        x1,
        y1,
        x2,
        y2,
    ):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={
                "arrowstyle": "->",
                "linewidth": 1.4,
            },
        )

    box(
        4.5,
        8.7,
        3,
        0.8,
        "Jigsaw source data\n250 toxic + 250 benign",
    )

    box(
        4.5,
        7.2,
        3,
        0.8,
        "WordPiece victim\n156/250 toxic eligible",
    )

    arrow(
        6,
        8.7,
        6,
        8.0,
    )

    box(
        0.8,
        5.4,
        2.5,
        1,
        "TokenBreak\n156 cases",
    )

    box(
        4.75,
        5.4,
        2.5,
        1,
        "AdvTok\n156 cases",
    )

    box(
        8.7,
        5.4,
        2.5,
        1,
        "Unicode\n4 × 156 cases",
    )

    arrow(
        5.3,
        7.2,
        2.1,
        6.4,
    )

    arrow(
        6,
        7.2,
        6,
        6.4,
    )

    arrow(
        6.7,
        7.2,
        9.9,
        6.4,
    )

    box(
        4.25,
        3.8,
        3.5,
        0.9,
        "936 attack instances",
    )

    arrow(
        2.1,
        5.4,
        5.1,
        4.7,
    )

    arrow(
        6,
        5.4,
        6,
        4.7,
    )

    arrow(
        9.9,
        5.4,
        6.9,
        4.7,
    )

    box(
        4.25,
        2.3,
        3.5,
        0.9,
        "7 defences\n6,552 paired evaluations",
    )

    arrow(
        6,
        3.8,
        6,
        3.2,
    )

    box(
        0.6,
        0.4,
        3,
        1,
        "Security\nASR + paired CIs",
    )

    box(
        4.5,
        0.4,
        3,
        1,
        "Mechanism\nDetect / block / reencode",
    )

    box(
        8.4,
        0.4,
        3,
        1,
        "Utility\nClean + complex inputs",
    )

    arrow(
        5.2,
        2.3,
        2.1,
        1.4,
    )

    arrow(
        6,
        2.3,
        6,
        1.4,
    )

    arrow(
        6.8,
        2.3,
        9.9,
        1.4,
    )

    # supporting branch
    box(
        8.7,
        7.4,
        2.8,
        1.2,
        "Supporting clean baseline\n"
        "WP 156 | BPE 214 | Uni 242\n"
        "shared = 155",
        fontsize=9,
    )

    ax.text(
        10.1,
        7.0,
        "No BPE/Unigram attacks run",
        ha="center",
        va="center",
        fontsize=8,
    )

    ax.set_title(
        "Experimental design of the primary dissertation study",
        pad=15,
    )

    save_figure(
        fig,
        "fig01_experimental_design",
    )


# ---------------------------------------------------------------- figure 2

def figure_undefended_attacks(
    attack_rows,
):
    records = []

    for attack in ATTACK_ORDER:
        part = attack_rows[
            attack_rows["attack"]
            == attack
        ]

        records.append(
            {
                "attack":
                    SHORT_ATTACK[attack],

                "n":
                    len(part),

                "successes":
                    int(
                        part[
                            "attack_success"
                        ].astype(int)
                        .sum()
                    ),

                "asr":
                    100
                    * part[
                        "attack_success"
                    ].astype(float)
                    .mean(),
            }
        )

    result = pd.DataFrame(
        records
    )

    result.to_csv(
        TABLE
        / "table2_undefended_attacks.csv",
        index=False,
    )

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    plot = result.iloc[::-1]

    bars = ax.barh(
        plot["attack"],
        plot["asr"],
    )

    ax.set_xlim(
        0,
        105,
    )

    ax.set_xlabel(
        "Attack success rate (%)"
    )

    ax.set_title(
        "Undefended attack effectiveness"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    for bar, value in zip(
        bars,
        plot["asr"],
    ):
        ax.text(
            min(
                value + 1.2,
                101,
            ),
            bar.get_y()
            + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=9,
        )

    save_figure(
        fig,
        "fig02_undefended_attack_strength",
    )

    return result


# ---------------------------------------------------------------- figure 3

def figure_coverage_heatmap(
    coverage,
):
    status_code = {
        status: i
        for i, status
        in enumerate(
            STATUS_ORDER
        )
    }

    matrix = np.full(
        (
            len(DEFENCE_ORDER),
            len(ATTACK_ORDER),
        ),
        np.nan,
    )

    labels = [
        [
            ""
            for _ in ATTACK_ORDER
        ]
        for _ in DEFENCE_ORDER
    ]

    for _, row in coverage.iterrows():

        if (
            row["attack"]
            not in ATTACK_ORDER
        ):
            continue

        if (
            row["defence"]
            not in DEFENCE_ORDER
        ):
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

        matrix[y, x] = (
            status_code[
                status
            ]
        )

        defended = float(
            as_percent(
                pd.Series(
                    [
                        row[
                            "paired_defended_asr"
                        ]
                    ]
                )
            ).iloc[0]
        )

        labels[y][x] = (
            f"{status}\n"
            f"{defended:.1f}%"
        )

    fig, ax = plt.subplots(
        figsize=(14.5, 8)
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
        vmin=0,
        vmax=len(
            STATUS_ORDER
        ) - 1,
    )

    ax.set_xticks(
        range(
            len(ATTACK_ORDER)
        )
    )

    ax.set_xticklabels(
        [
            SHORT_ATTACK[x]
            for x
            in ATTACK_ORDER
        ],
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
        "Attack × defence coverage matrix"
    )

    for y in range(
        len(DEFENCE_ORDER)
    ):
        for x in range(
            len(ATTACK_ORDER)
        ):
            if labels[y][x]:
                ax.text(
                    x,
                    y,
                    labels[y][x],
                    ha="center",
                    va="center",
                    fontsize=7.5,
                )

    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.027,
        pad=0.02,
    )

    cbar.set_ticks(
        range(
            len(STATUS_ORDER)
        )
    )

    cbar.set_ticklabels(
        [
            "No base attack",
            "No clear effect",
            "Partial",
            "Complete",
            "Harmful",
        ]
    )

    save_figure(
        fig,
        "fig03_attack_defence_coverage",
    )


# ---------------------------------------------------------------- figure 4a-c

def mechanism_heatmap(
    coverage,
    column,
    title,
    filename,
):

    matrix = np.full(
        (
            len(DEFENCE_ORDER),
            len(ATTACK_ORDER),
        ),
        np.nan,
    )

    values = as_percent(
        coverage[column]
    )

    temp = coverage.copy()

    temp["_plot_value"] = (
        values
    )

    for _, row in temp.iterrows():

        if (
            row["attack"]
            not in ATTACK_ORDER
            or
            row["defence"]
            not in DEFENCE_ORDER
        ):
            continue

        y = DEFENCE_ORDER.index(
            row["defence"]
        )

        x = ATTACK_ORDER.index(
            row["attack"]
        )

        matrix[y, x] = row[
            "_plot_value"
        ]

    fig, ax = plt.subplots(
        figsize=(12.5, 7)
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
        vmin=0,
        vmax=100,
    )

    ax.set_xticks(
        range(
            len(ATTACK_ORDER)
        )
    )

    ax.set_xticklabels(
        [
            SHORT_ATTACK[x]
            for x
            in ATTACK_ORDER
        ],
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

    ax.set_title(
        title
    )

    ax.set_xlabel(
        "Attack"
    )

    ax.set_ylabel(
        "Defence"
    )

    annotate_heatmap(
        ax,
        matrix,
        ".0f",
    )

    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.03,
        pad=0.02,
    )

    cbar.set_label(
        "Rate (%)"
    )

    save_figure(
        fig,
        filename,
    )


def figure_mechanisms(
    coverage,
):

    mechanism_heatmap(
        coverage,
        "attack_detection_rate",
        "Attack detection rate",
        "fig04a_detection_heatmap",
    )

    mechanism_heatmap(
        coverage,
        "attack_block_rate",
        "Attack blocking rate",
        "fig04b_block_heatmap",
    )

    mechanism_heatmap(
        coverage,
        "reencode_rate",
        "Attack re-encoding rate",
        "fig04c_reencode_heatmap",
    )


# ---------------------------------------------------------------- figure 5

def figure_tokenbreak_defences(
    coverage,
):

    part = coverage[
        coverage["attack"]
        == "TokenBreak"
    ].copy()

    part["defence"] = pd.Categorical(
        part["defence"],
        categories=DEFENCE_ORDER,
        ordered=True,
    )

    part = (
        part.sort_values(
            "defence"
        )
        .reset_index(
            drop=True
        )
    )

    reduction = as_percent(
        part[
            "asr_reduction_pp"
        ]
    )

    low = as_percent(
        part[
            "asr_reduction_ci_low_pp"
        ]
    )

    high = as_percent(
        part[
            "asr_reduction_ci_high_pp"
        ]
    )

    defended = as_percent(
        part[
            "paired_defended_asr"
        ]
    )

    y = np.arange(
        len(part)
    )

    lower_error = np.maximum(
        reduction.to_numpy()
        - low.to_numpy(),
        0,
    )

    upper_error = np.maximum(
        high.to_numpy()
        - reduction.to_numpy(),
        0,
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.errorbar(
        reduction,
        y,
        xerr=np.vstack(
            [
                lower_error,
                upper_error,
            ]
        ),
        fmt="o",
        capsize=4,
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        part["defence"]
    )

    ax.set_xlabel(
        "Paired ASR reduction (percentage points)"
    )

    ax.set_title(
        "TokenBreak defence effectiveness with 95% confidence intervals"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    for i in range(
        len(part)
    ):
        ax.text(
            high.iloc[i] + 1,
            i,
            f"defended ASR "
            f"{defended.iloc[i]:.1f}%",
            va="center",
            fontsize=8,
        )

    save_figure(
        fig,
        "fig05_tokenbreak_defence_effectiveness",
    )


# ---------------------------------------------------------------- figure 6

def figure_complex_utility(
    utility,
):
    category_order = [
        "code",
        "urls",
        "emoji",
        "misspelling",
        "non-English",
        "mixed-script",
    ]

    # map likely category spellings
    category_map = {
        "code": "code",
        "url": "urls",
        "urls": "urls",
        "emoji": "emoji",
        "misspelling": "misspelling",
        "misspellings": "misspelling",
        "non-english": "non-English",
        "non english": "non-English",
        "non_english": "non-English",
        "mixed-script": "mixed-script",
        "mixed script": "mixed-script",
        "mixed_script": "mixed-script",
    }

    utility = utility.copy()

    utility["category"] = (
        utility["category"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            lambda x:
                category_map.get(
                    x,
                    x,
                )
        )
    )

    pivot = utility.pivot_table(
        index="defence",
        columns="category",
        values="operational_preservation",
        aggfunc="mean",
    )

    pivot = pivot.reindex(
        DEFENCE_ORDER
    )

    available_categories = [
        x
        for x in category_order
        if x in pivot.columns
    ]

    # if spellings differ, keep every remaining category too
    for col in pivot.columns:
        if col not in available_categories:
            available_categories.append(
                col
            )

    pivot = pivot[
        available_categories
    ]

    pivot.reset_index().to_csv(
        TABLE
        / "table4_complex_utility.csv",
        index=False,
    )

    matrix = pivot.to_numpy(
        dtype=float
    )

    fig, ax = plt.subplots(
        figsize=(11.5, 6.5)
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
        vmin=0,
        vmax=100,
    )

    ax.set_xticks(
        range(
            len(
                available_categories
            )
        )
    )

    ax.set_xticklabels(
        available_categories,
        rotation=20,
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

    ax.set_title(
        "Operational preservation on complex legitimate inputs"
    )

    ax.set_xlabel(
        "Legitimate-input category"
    )

    ax.set_ylabel(
        "Defence"
    )

    annotate_heatmap(
        ax,
        matrix,
        ".0f",
    )

    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.03,
        pad=0.02,
    )

    cbar.set_label(
        "Operational preservation (%)"
    )

    save_figure(
        fig,
        "fig06_complex_legitimate_utility",
    )


# ---------------------------------------------------------------- figure 7

def figure_margin(
    margin,
):

    fig, ax = plt.subplots(
        figsize=(12, 6.5)
    )

    x = np.arange(
        len(ATTACK_ORDER),
        dtype=float,
    )

    width = 0.24

    for i, label in enumerate(
        MARGIN_ORDER
    ):

        part = (
            margin[
                margin[
                    "clean_probability_bin"
                ]
                == label
            ]
            .set_index("attack")
            .reindex(
                ATTACK_ORDER
            )
        )

        y = (
            as_percent(
                part["asr"]
            )
            .to_numpy(
                dtype=float
            )
        )

        lo = (
            as_percent(
                part[
                    "asr_ci_low"
                ]
            )
            .to_numpy(
                dtype=float
            )
        )

        hi = (
            as_percent(
                part[
                    "asr_ci_high"
                ]
            )
            .to_numpy(
                dtype=float
            )
        )

        lower = np.maximum(
            y - lo,
            0,
        )

        upper = np.maximum(
            hi - y,
            0,
        )

        ax.bar(
            x
            + (i - 1)
            * width,
            y,
            width=width,
            label=label,
            yerr=np.vstack(
                [
                    lower,
                    upper,
                ]
            ),
            capsize=3,
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            SHORT_ATTACK[x]
            for x
            in ATTACK_ORDER
        ],
        rotation=20,
        ha="right",
    )

    ax.set_ylim(
        0,
        108,
    )

    ax.set_ylabel(
        "Attack success rate (%)"
    )

    ax.set_xlabel(
        "Attack"
    )

    ax.set_title(
        "Attack success by clean victim confidence"
    )

    ax.legend(
        title="Clean P(toxic)"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    save_figure(
        fig,
        "fig07_margin_stratified_asr",
    )


# ---------------------------------------------------------------- figure 8

def figure_harmful_regression(
    coverage,
):

    row = coverage[
        (
            coverage["attack"]
            == "Unicode invisible"
        )
        &
        (
            coverage["defence"]
            == "Tokenizer translation"
        )
    ]

    if len(row) != 1:
        raise RuntimeError(
            "could not uniquely locate "
            "Unicode invisible × Tokenizer translation"
        )

    row = row.iloc[0]

    undefended = float(
        as_percent(
            pd.Series(
                [
                    row[
                        "paired_undefended_asr"
                    ]
                ]
            )
        ).iloc[0]
    )

    defended = float(
        as_percent(
            pd.Series(
                [
                    row[
                        "paired_defended_asr"
                    ]
                ]
            )
        ).iloc[0]
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    labels = [
        "No defence",
        "Tokenizer translation",
    ]

    values = [
        undefended,
        defended,
    ]

    bars = ax.bar(
        labels,
        values,
    )

    ax.set_ylim(
        0,
        max(
            20,
            defended + 5,
        ),
    )

    ax.set_ylabel(
        "Attack success rate (%)"
    )

    ax.set_title(
        "Defence-induced regression on invisible Unicode"
    )

    for bar, value in zip(
        bars,
        values,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.5,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    save_figure(
        fig,
        "fig08_defence_induced_regression",
    )


# ---------------------------------------------------------------- figure 9

def figure_cross_tokenizer():
    df = pd.read_csv(
        find_file(
            "data/cross_tokenizer_eligibility.csv",
            "cross_tokenizer_eligibility.csv",
        )
    )

    required = [
        "eligible_wordpiece",
        "eligible_bpe",
        "eligible_unigram",
        "eligible_all_three",
    ]

    missing = [
        x
        for x in required
        if x not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "cross-tokenizer eligibility "
            f"missing columns: {missing}"
        )

    total = len(df)

    counts = [
        int(
            df[
                "eligible_wordpiece"
            ].astype(int)
            .sum()
        ),
        int(
            df[
                "eligible_bpe"
            ].astype(int)
            .sum()
        ),
        int(
            df[
                "eligible_unigram"
            ].astype(int)
            .sum()
        ),
        int(
            df[
                "eligible_all_three"
            ].astype(int)
            .sum()
        ),
    ]

    labels = [
        "WordPiece",
        "BPE",
        "Unigram",
        "Shared all three",
    ]

    percentages = [
        100 * x / total
        for x in counts
    ]

    output = pd.DataFrame(
        {
            "tokenizer_group":
                labels,

            "eligible_n":
                counts,

            "total_n":
                total,

            "eligible_percent":
                percentages,
        }
    )

    output.to_csv(
        TABLE
        / "table_cross_tokenizer_baseline.csv",
        index=False,
    )

    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    bars = ax.bar(
        labels,
        percentages,
    )

    ax.set_ylim(
        0,
        105,
    )

    ax.set_ylabel(
        "Clean toxic examples eligible (%)"
    )

    ax.set_title(
        "Cross-tokenizer clean-baseline eligibility"
    )

    ax.text(
        0.5,
        -0.18,
        "Supporting baseline only — no BPE/Unigram attacks were executed.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )

    for bar, count, pct in zip(
        bars,
        counts,
        percentages,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            pct + 1.5,
            f"{count}/{total}\n{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    save_figure(
        fig,
        "fig09_cross_tokenizer_clean_baseline",
    )


# ---------------------------------------------------------------- figure 10

def figure_research_evolution():
    """
    historical project evolution.

    this is deliberately conceptual.
    historical and current numerical results are not pooled.
    """

    labels = [
        (
            "15 Jul",
            "Cross-tokenizer\n"
            "disagreement programme",
        ),
        (
            "27–28 Jul",
            "LMSYS / WildChat /\n"
            "ToxicChat freezing",
        ),
        (
            "30 Jul",
            "293,034-row\n"
            "paired attack corpus",
        ),
        (
            "31 Jul",
            "Environment incident\n"
            "→ quarantine + envguard",
        ),
        (
            "31 Jul",
            "Pre-registered E1\n"
            "NOT SUPERIOR",
        ),
        (
            "7 Aug",
            "Current attack ×\n"
            "defence study",
        ),
        (
            "8 Aug",
            "Results + evidence\n"
            "freeze",
        ),
    ]

    x = np.arange(
        len(labels)
    )

    fig, ax = plt.subplots(
        figsize=(14, 5)
    )

    ax.plot(
        x,
        np.zeros_like(x),
        marker="o",
        linewidth=1.5,
    )

    for i, (
        date,
        text,
    ) in enumerate(labels):

        y = (
            0.25
            if i % 2 == 0
            else -0.25
        )

        ax.plot(
            [i, i],
            [0, y],
            linewidth=1,
        )

        ax.text(
            i,
            y,
            f"{date}\n{text}",
            ha="center",
            va=(
                "bottom"
                if y > 0
                else "top"
            ),
            fontsize=8.5,
        )

    ax.axvline(
        4.5,
        linestyle="--",
        linewidth=1,
    )

    ax.text(
        2,
        0.58,
        "Historical / preliminary programme",
        ha="center",
        fontsize=10,
    )

    ax.text(
        5.5,
        0.58,
        "Primary dissertation",
        ha="center",
        fontsize=10,
    )

    ax.set_ylim(
        -0.85,
        0.85,
    )

    ax.set_xlim(
        -0.5,
        len(labels) - 0.5,
    )

    ax.axis("off")

    ax.set_title(
        "Research evolution: negative preliminary result to current empirical study"
    )

    ax.text(
        0.5,
        -0.02,
        "Historical results are shown as research-development evidence "
        "and are not pooled with the primary study.",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.5,
    )

    save_figure(
        fig,
        "fig10_research_evolution",
    )


# ---------------------------------------------------------------- tables

def table_experimental_configuration():
    thresholds_path = find_file(
        "data/cpt_thresholds.json",
    )

    with thresholds_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        thresholds = json.load(
            f
        )

    rows = [
        [
            "Primary victim",
            "martin-ha/toxic-comment-model",
        ],
        [
            "Victim tokenizer family",
            "WordPiece",
        ],
        [
            "Random seed",
            "42",
        ],
        [
            "Source toxic rows",
            "250",
        ],
        [
            "Source benign rows",
            "250",
        ],
        [
            "Eligible toxic rows",
            "156",
        ],
        [
            "WikiText calibration",
            "5,000",
        ],
        [
            "WikiText held-out",
            "5,000",
        ],
        [
            "Classifier threshold",
            "0.5",
        ],
        [
            "Max model tokens",
            "510 + special tokens",
        ],
        [
            "TokenBreak threshold",
            "0.995",
        ],
        [
            "AdvTok max iterations",
            "25",
        ],
        [
            "AdvTok neighbourhood sample",
            "128",
        ],
        [
            "Global CPT threshold",
            str(
                thresholds.get(
                    "global",
                    "UNKNOWN",
                )
            ),
        ],
        [
            "Window CPT threshold",
            str(
                thresholds.get(
                    "window",
                    "UNKNOWN",
                )
            ),
        ],
        [
            "Window CPT size",
            "10",
        ],
        [
            "Attack instances",
            "936",
        ],
        [
            "Defences",
            "7",
        ],
        [
            "Paired defence evaluations",
            "6,552",
        ],
    ]

    df = pd.DataFrame(
        rows,
        columns=[
            "parameter",
            "value",
        ],
    )

    df.to_csv(
        TABLE
        / "table1_experimental_configuration.csv",
        index=False,
    )


def table_attack_detail(
    attack_rows,
):

    records = []

    for attack in ATTACK_ORDER:

        part = attack_rows[
            attack_rows["attack"]
            == attack
        ].copy()

        record = {
            "attack":
                SHORT_ATTACK[
                    attack
                ],

            "n":
                len(part),

            "successes":
                int(
                    part[
                        "attack_success"
                    ].astype(int)
                    .sum()
                ),

            "asr_percent":
                100
                * part[
                    "attack_success"
                ].astype(float)
                .mean(),
        }

        if (
            "clean_toxic_prob"
            in part.columns
        ):
            record[
                "mean_clean_toxic_prob"
            ] = (
                pd.to_numeric(
                    part[
                        "clean_toxic_prob"
                    ],
                    errors="coerce",
                )
                .mean()
            )

        if (
            "attacked_toxic_prob"
            in part.columns
        ):
            record[
                "mean_attacked_toxic_prob"
            ] = (
                pd.to_numeric(
                    part[
                        "attacked_toxic_prob"
                    ],
                    errors="coerce",
                )
                .mean()
            )

        if (
            "clean_toxic_prob"
            in part.columns
            and
            "attacked_toxic_prob"
            in part.columns
        ):
            clean = pd.to_numeric(
                part[
                    "clean_toxic_prob"
                ],
                errors="coerce",
            )

            attacked = pd.to_numeric(
                part[
                    "attacked_toxic_prob"
                ],
                errors="coerce",
            )

            record[
                "mean_probability_drop"
            ] = (
                clean - attacked
            ).mean()

        if (
            "token_edit_distance"
            in part.columns
        ):
            record[
                "mean_token_edit_distance"
            ] = (
                pd.to_numeric(
                    part[
                        "token_edit_distance"
                    ],
                    errors="coerce",
                )
                .mean()
            )

        records.append(
            record
        )

    pd.DataFrame(
        records
    ).to_csv(
        TABLE
        / "table2_attack_results_detailed.csv",
        index=False,
    )


def table_defence_outcomes(
    coverage,
):

    cols = [
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

    existing = [
        x
        for x in cols
        if x in coverage.columns
    ]

    out = coverage[
        existing
    ].copy()

    percent_cols = [
        "paired_undefended_asr",
        "paired_defended_asr",
        "asr_reduction_pp",
        "asr_reduction_ci_low_pp",
        "asr_reduction_ci_high_pp",
        "clean_retention",
        "attack_detection_rate",
        "attack_block_rate",
        "reencode_rate",
    ]

    for col in percent_cols:
        if col in out.columns:
            out[col] = as_percent(
                out[col]
            )

    out.to_csv(
        TABLE
        / "table3_full_defence_outcomes.csv",
        index=False,
    )


# ---------------------------------------------------------------- captions + inventory

def write_captions():
    text = """# Suggested dissertation captions

## Figure 1 — Experimental design

Experimental design of the primary study. Jigsaw toxic examples correctly
classified by the WordPiece victim formed the eligible attack set. Six attack
variants produced 936 attack instances, which were evaluated under seven
defences using a paired clean-versus-attacked design. Security effectiveness,
defence mechanism, and legitimate-input utility were evaluated separately.
The BPE and Unigram branch represents clean-baseline generalisation groundwork
only; no cross-family attack evaluation was executed.

## Figure 2 — Undefended attack effectiveness

Undefended attack success rate on the 156 eligible WordPiece toxic inputs.
TokenBreak and AdvTok each achieved 96.2% ASR, while three visible Unicode
manipulation families achieved high ASR. Invisible Unicode achieved 0% in the
evaluated victim because the perturbations did not alter the model-bound token
IDs.

## Figure 3 — Attack × defence coverage matrix

Coverage of seven defences across six attack variants. Each cell reports the
outcome classification together with defended ASR. "Complete" denotes complete
suppression in this evaluated setting rather than a universal claim that an
attack is solved. "No base attack" denotes an attack whose undefended ASR was
already zero.

## Figure 4a — Detection

Attack detection rate by attack and defence. Detection is separated from attack
suppression so that successful defence outcomes are not automatically
interpreted as explicit attack detection.

## Figure 4b — Blocking

Attack blocking rate by attack and defence. Blocking is treated separately from
repair or re-encoding.

## Figure 4c — Re-encoding

Re-encoding rate by attack and defence. This exposes cases in which apparent
robustness arises because adversarial token IDs are replaced by a canonical
string-to-token encoding rather than because the attack was detected.

## Figure 5 — TokenBreak defence effectiveness

Paired reduction in TokenBreak attack-success rate with 95% confidence
intervals. Tokenizer translation gives the largest reduction but does not
eliminate TokenBreak, illustrating the remaining coverage gap.

## Figure 6 — Complex legitimate-input utility

Operational preservation of seven defences across six complex legitimate-input
categories. The figure exposes security-utility trade-offs that are hidden by
ordinary clean benign controls, particularly for CPT-based defences on URLs,
non-English, mixed-script, and code inputs.

## Figure 7 — Margin-stratified attack success

Attack success stratified by the clean victim's toxic-class probability.
Wilson 95% confidence intervals are shown. High attack success persists among
the 128 high-confidence examples with clean P(toxic)>0.80, indicating that the
headline ASRs are not explained only by borderline classifier decisions.

## Figure 8 — Defence-induced regression

Tokenizer translation introduces successful invisible-Unicode evasions even
though the same attack has zero undefended ASR in the evaluated WordPiece
victim. This demonstrates that defensive preprocessing can itself create a new
failure mode.

## Figure 9 — Cross-tokenizer clean baseline

Clean toxic-example eligibility under WordPiece, BPE, and Unigram classifier
setups, including the shared eligible intersection. This is clean-baseline
generalisation groundwork only; no BPE or Unigram attack-generalisation result
is claimed.

## Figure 10 — Research evolution

Research evolution from the preliminary cross-tokenizer disagreement programme
to the current attack × defence study. The preliminary programme is retained as
evidence of research development, negative-result handling, and reproducibility
discipline; its numerical results are not pooled with the primary study.
"""

    (
        OUT
        / "CAPTIONS.md"
    ).write_text(
        text,
        encoding="utf-8",
    )


def write_inventory():
    text = """# Dissertation figure and table inventory

## Main body — recommended

| item | role |
|---|---|
| Figure 1 | Experimental methodology |
| Figure 2 | Undefended attack baseline |
| Figure 3 | Headline attack × defence result |
| Figure 4a–4c | Defence mechanism: detect/block/reencode |
| Figure 5 | TokenBreak residual coverage gap |
| Figure 6 | Security–utility trade-off |
| Figure 7 | Robustness to clean confidence margin |

## Discussion / limitations

| item | role |
|---|---|
| Figure 8 | Defence-induced vulnerability |
| Figure 9 | Cross-tokenizer clean-baseline groundwork |

## Reflection / appendix

| item | role |
|---|---|
| Figure 10 | Research evolution |
| Table 3 full 42-cell matrix | Complete numerical evidence |

## Tables

| item | role |
|---|---|
| Table 1 | Experimental configuration |
| Table 2 | Undefended attack results |
| Table 3 | Full defence outcomes |
| Table 4 | Complex legitimate-input utility |
"""

    (
        OUT
        / "INVENTORY.md"
    ).write_text(
        text,
        encoding="utf-8",
    )


# ---------------------------------------------------------------- run

if __name__ == "__main__":

    print(
        "building dissertation figures"
    )

    print(
        "python :",
        sys.executable,
    )

    print(
        "note   : no model calls, "
        "no experiments, frozen data only"
    )

    print()

    attack_rows = (
        load_attack_rows()
    )

    coverage = (
        load_coverage()
    )

    margin = (
        load_margin()
    )

    utility = (
        load_complex_utility()
    )

    print(
        "loaded:"
    )

    print(
        " attack rows :",
        len(attack_rows),
    )

    print(
        " coverage    :",
        len(coverage),
    )

    print(
        " margin rows :",
        len(margin),
    )

    print(
        " utility rows:",
        len(utility),
    )

    print()

    # tables first
    table_experimental_configuration()

    attack_summary = (
        figure_undefended_attacks(
            attack_rows
        )
    )

    table_attack_detail(
        attack_rows
    )

    table_defence_outcomes(
        coverage
    )

    # figures
    figure_pipeline()

    figure_coverage_heatmap(
        coverage
    )

    figure_mechanisms(
        coverage
    )

    figure_tokenbreak_defences(
        coverage
    )

    figure_complex_utility(
        utility
    )

    figure_margin(
        margin
    )

    figure_harmful_regression(
        coverage
    )

    figure_cross_tokenizer()

    figure_research_evolution()

    # documentation
    write_captions()
    write_inventory()

    print(
        "done"
    )

    print()

    print(
        "figures:",
        FIG,
    )

    print(
        "tables :",
        TABLE,
    )

    print(
        "captions:",
        OUT / "CAPTIONS.md",
    )

    print(
        "inventory:",
        OUT / "INVENTORY.md",
    )

    print()

    print(
        "generated figure files:"
    )

    for path in sorted(
        FIG.glob("*.png")
    ):
        print(
            " ",
            path
        )

    print()

    print(
        "generated table files:"
    )

    for path in sorted(
        TABLE.glob("*.csv")
    ):
        print(
            " ",
            path
        )

    print()

    print(
        "publication figure build complete"
    )