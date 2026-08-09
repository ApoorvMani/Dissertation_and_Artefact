"""
camera-ready dissertation figures.

no model calls.
no attacks.
no defence reruns.
no threshold tuning.
no new statistics.

this script only re-renders frozen evidence
for publication / print use.

main design goals:
- vector-first pdf + svg output
- grayscale safe
- no meaning encoded by colour alone
- readable at final dissertation size
- thicker lines and markers
- restrained titles / annotations
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.patches import (
    FancyBboxPatch,
    Patch,
    Rectangle,
)


# ---------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------

OUT = Path(
    "results/dissertation/camera_ready"
)

FIG = OUT / "figures"
APP = OUT / "appendix"

FIG.mkdir(
    parents=True,
    exist_ok=True,
)

APP.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------
# publication dimensions
# ---------------------------------------------------------------------

# approximate useful width for the single-column dissertation.
FULL_W = 7.0

# only used for intentionally compact figures.
SMALL_W = 5.2


# ---------------------------------------------------------------------
# global publication style
# ---------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family":
            "serif",

        "font.serif": [
            "Times New Roman",
            "Times",
            "DejaVu Serif",
        ],

        "font.size":
            9.5,

        "axes.titlesize":
            10.5,

        "axes.labelsize":
            9.5,

        "xtick.labelsize":
            8.5,

        "ytick.labelsize":
            8.5,

        "legend.fontsize":
            8,

        "axes.linewidth":
            1.15,

        "lines.linewidth":
            1.5,

        "lines.markersize":
            6,

        "xtick.major.width":
            1.0,

        "ytick.major.width":
            1.0,

        "xtick.major.size":
            4,

        "ytick.major.size":
            4,

        "pdf.fonttype":
            42,

        "ps.fonttype":
            42,

        "svg.fonttype":
            "none",

        "savefig.bbox":
            "tight",

        "savefig.pad_inches":
            0.06,
    }
)


# ---------------------------------------------------------------------
# ordering
# ---------------------------------------------------------------------

ATTACK_ORDER = [
    "TokenBreak",
    "AdvTok",
    "Unicode invisible",
    "Unicode homoglyph",
    "Unicode compatibility",
    "Unicode reorder",
]


ATTACK_SHORT = {
    "TokenBreak":
        "TokenBreak",

    "AdvTok":
        "AdvTok",

    "Unicode invisible":
        "Invisible",

    "Unicode homoglyph":
        "Homoglyph",

    "Unicode compatibility":
        "Compatibility",

    "Unicode reorder":
        "Reorder",
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


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def find_file(*candidates):
    """find the first existing file."""

    for candidate in candidates:

        path = Path(candidate)

        if path.exists():
            return path

    raise FileNotFoundError(
        "could not find:\n"
        + "\n".join(
            f"  {x}"
            for x in candidates
        )
    )


def save_figure(
    fig,
    name,
    appendix=False,
):
    """
    vector first.

    png is retained only as a convenient preview.
    """

    folder = (
        APP
        if appendix
        else FIG
    )

    fig.savefig(
        folder / f"{name}.pdf"
    )

    fig.savefig(
        folder / f"{name}.svg"
    )

    fig.savefig(
        folder / f"{name}.png",
        dpi=400,
    )

    plt.close(
        fig
    )


def as_percent(values):
    """
    accepts either fractional rates or
    already-percent values.
    """

    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    finite = (
        values
        .dropna()
    )

    if len(finite) == 0:
        return values

    if finite.abs().max() <= 1.5:
        return values * 100

    return values


def normalise_attack(value):

    x = (
        str(value)
        .strip()
        .lower()
        .replace(
            "_",
            " ",
        )
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
        .replace(
            "_",
            " ",
        )
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


def clean_axes(ax):
    """small amount of visual cleanup."""

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )


# ---------------------------------------------------------------------
# data
# ---------------------------------------------------------------------

def load_attack_rows():

    tokenbreak = pd.read_csv(
        find_file(
            "data/tokenbreak_results.csv"
        )
    )

    advtok = pd.read_csv(
        find_file(
            "data/advtok_results.csv"
        )
    )

    unicode_df = pd.read_csv(
        find_file(
            "data/unicode_results.csv"
        )
    )

    df = pd.concat(
        [
            tokenbreak,
            advtok,
            unicode_df,
        ],
        ignore_index=True,
        sort=False,
    )

    df["attack"] = (
        df["attack"]
        .map(
            normalise_attack
        )
    )

    return df


def load_coverage():

    df = pd.read_csv(
        find_file(
            "results/final_current/"
            "coverage_matrix.csv"
        )
    )

    df["attack"] = (
        df["attack"]
        .map(
            normalise_attack
        )
    )

    df["defence"] = (
        df["defence"]
        .map(
            normalise_defence
        )
    )

    return df


def load_margin():

    df = pd.read_csv(
        find_file(
            "results/final_current/"
            "margin_stratified_asr.csv"
        )
    )

    df["attack"] = (
        df["attack"]
        .map(
            normalise_attack
        )
    )

    return df


def load_utility():
    """
    use already-verified polished utility matrices.

    these are only plotting intermediates derived
    from frozen step11 results.
    """

    preserve = pd.read_csv(
        find_file(
            "results/dissertation/polished/"
            "utility_operational_preservation.csv"
        ),
        index_col=0,
    )

    changed = pd.read_csv(
        find_file(
            "results/dissertation/polished/"
            "utility_representation_change.csv"
        ),
        index_col=0,
    )

    preserve.index = [
        normalise_defence(x)
        for x in preserve.index
    ]

    changed.index = [
        normalise_defence(x)
        for x in changed.index
    ]

    preserve = preserve.reindex(
        DEFENCE_ORDER
    )

    changed = changed.reindex(
        DEFENCE_ORDER
    )

    return (
        preserve,
        changed,
    )


# =====================================================================
# FIGURE 1
# experimental design
# =====================================================================

def figure_1_design():

    fig, ax = plt.subplots(
        figsize=(
            FULL_W,
            4.7,
        )
    )

    ax.set_xlim(
        0,
        12,
    )

    ax.set_ylim(
        0,
        10,
    )

    ax.axis(
        "off"
    )

    def box(
        x,
        y,
        w,
        h,
        text,
        dashed=False,
        fontsize=8.2,
        linewidth=1.4,
    ):

        patch = FancyBboxPatch(
            (
                x,
                y,
            ),
            w,
            h,
            boxstyle=(
                "round,pad=0.02"
            ),
            facecolor="white",
            edgecolor="black",
            linewidth=linewidth,
            linestyle=(
                "--"
                if dashed
                else "-"
            ),
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
        dashed=False,
    ):

        ax.annotate(
            "",
            xy=(
                x2,
                y2,
            ),
            xytext=(
                x1,
                y1,
            ),
            arrowprops={
                "arrowstyle":
                    "->",

                "linewidth":
                    1.4,

                "linestyle":
                    "--"
                    if dashed
                    else "-",

                "color":
                    "black",
            },
        )

    # source and victim
    box(
        3.7,
        8.65,
        3.2,
        0.75,
        "Jigsaw source data\n"
        "250 toxic + 250 benign",
    )

    box(
        3.7,
        7.25,
        3.2,
        0.75,
        "WordPiece victim\n"
        "156 / 250 toxic eligible",
    )

    arrow(
        5.3,
        8.65,
        5.3,
        8.0,
    )

    # attacks
    attack_boxes = [
        (
            0.35,
            "TokenBreak\n156 cases",
        ),
        (
            4.1,
            "AdvTok\n156 cases",
        ),
        (
            7.85,
            "Unicode\n4 × 156 cases",
        ),
    ]

    for x, text in attack_boxes:

        box(
            x,
            5.55,
            2.5,
            0.85,
            text,
        )

    arrow(
        4.45,
        7.25,
        1.6,
        6.4,
    )

    arrow(
        5.3,
        7.25,
        5.35,
        6.4,
    )

    arrow(
        6.15,
        7.25,
        9.1,
        6.4,
    )

    # pooled attacks
    box(
        3.7,
        4.0,
        3.2,
        0.75,
        "936 attack instances",
    )

    arrow(
        1.6,
        5.55,
        4.45,
        4.75,
    )

    arrow(
        5.35,
        5.55,
        5.35,
        4.75,
    )

    arrow(
        9.1,
        5.55,
        6.15,
        4.75,
    )

    # defence stage
    box(
        3.7,
        2.65,
        3.2,
        0.8,
        "7 defences\n"
        "6,552 paired evaluations",
    )

    arrow(
        5.3,
        4.0,
        5.3,
        3.45,
    )

    # outputs
    bottom = [
        (
            0.15,
            "Security\n"
            "ASR + paired CIs",
        ),
        (
            4.05,
            "Action logging\n"
            "detect / block / re-encode",
        ),
        (
            7.95,
            "Utility\n"
            "clean + complex inputs",
        ),
    ]

    for x, text in bottom:

        box(
            x,
            0.55,
            2.8,
            0.85,
            text,
        )

    arrow(
        4.5,
        2.65,
        1.55,
        1.4,
    )

    arrow(
        5.3,
        2.65,
        5.45,
        1.4,
    )

    arrow(
        6.1,
        2.65,
        9.35,
        1.4,
    )

    # supporting baseline
    ax.plot(
        [
            10.75,
            10.75,
        ],
        [
            4.8,
            9.3,
        ],
        color="black",
        linestyle="--",
        linewidth=1.2,
    )

    ax.text(
        11.35,
        9.05,
        "SUPPORTING",
        ha="center",
        fontweight="bold",
        fontsize=8,
    )

    box(
        10.95,
        7.1,
        0.95,
        1.55,
        "Clean\neligibility\n\n"
        "WP 156\n"
        "BPE 214\n"
        "Uni 242\n"
        "shared 155",
        dashed=True,
        fontsize=6.7,
    )

    arrow(
        6.9,
        9.0,
        10.95,
        7.95,
        dashed=True,
    )

    ax.text(
        11.42,
        6.3,
        "Different model–\n"
        "tokenizer setups\n"
        "Clean baseline only",
        ha="center",
        va="center",
        fontsize=6.7,
    )

    ax.text(
        11.42,
        5.35,
        "No BPE/Unigram\n"
        "attacks executed",
        ha="center",
        va="center",
        fontsize=6.7,
        fontweight="bold",
    )

    save_figure(
        fig,
        "fig01_experimental_design",
    )


# =====================================================================
# FIGURE 2
# undefended attack ASR
# =====================================================================

def figure_2_attacks(
    attacks,
):

    records = []

    for attack in ATTACK_ORDER:

        part = attacks[
            attacks["attack"]
            == attack
        ]

        n = len(
            part
        )

        successes = int(
            part[
                "attack_success"
            ]
            .astype(int)
            .sum()
        )

        records.append(
            {
                "attack":
                    ATTACK_SHORT[
                        attack
                    ],

                "n":
                    n,

                "successes":
                    successes,

                "asr":
                    (
                        100
                        * successes
                        / n
                    ),
            }
        )

    result = pd.DataFrame(
        records
    )

    plot = result.iloc[
        ::-1
    ]

    fig, ax = plt.subplots(
        figsize=(
            FULL_W,
            3.45,
        )
    )

    bars = ax.barh(
        plot["attack"],
        plot["asr"],
        facecolor="0.72",
        edgecolor="black",
        linewidth=1.2,
    )

    ax.set_xlim(
        0,
        105,
    )

    ax.set_xlabel(
        "Attack success rate (%)"
    )

    clean_axes(
        ax
    )

    ax.grid(
        axis="x",
        color="0.85",
        linewidth=0.7,
        zorder=0,
    )

    ax.set_axisbelow(
        True
    )

    for bar, (_, row) in zip(
        bars,
        plot.iterrows(),
    ):

        ax.text(
            min(
                row["asr"] + 1.1,
                101,
            ),
            (
                bar.get_y()
                + bar.get_height()
                / 2
            ),
            (
                f"{row['asr']:.1f}% "
                f"({int(row['successes'])}/"
                f"{int(row['n'])})"
            ),
            va="center",
            fontsize=8.2,
        )

    save_figure(
        fig,
        "fig02_undefended_attack_asr",
    )


# =====================================================================
# FIGURE 3
# grayscale / hatch coverage matrix
# =====================================================================

def figure_3_coverage(
    coverage,
):

    styles = {
        "NO BASE ATTACK": {
            "facecolor":
                "white",

            "hatch":
                "..",

            "short":
                "N/A",
        },

        "NO CLEAR EFFECT": {
            "facecolor":
                "0.88",

            "hatch":
                "",

            "short":
                "NO EFFECT",
        },

        "PARTIAL": {
            "facecolor":
                "white",

            "hatch":
                "////",

            "short":
                "PARTIAL",
        },

        "COMPLETE": {
            "facecolor":
                "0.45",

            "hatch":
                "",

            "short":
                "COMPLETE",
        },

        "HARMFUL": {
            "facecolor":
                "white",

            "hatch":
                "xx",

            "short":
                "HARMFUL",
        },
    }

    fig, ax = plt.subplots(
        figsize=(
            FULL_W,
            4.15,
        )
    )

    ax.set_xlim(
        0,
        len(
            ATTACK_ORDER
        ),
    )

    ax.set_ylim(
        0,
        len(
            DEFENCE_ORDER
        ),
    )

    # cells
    for y, defence in enumerate(
        DEFENCE_ORDER
    ):

        for x, attack in enumerate(
            ATTACK_ORDER
        ):

            row = coverage[
                (
                    coverage[
                        "attack"
                    ]
                    == attack
                )
                &
                (
                    coverage[
                        "defence"
                    ]
                    == defence
                )
            ]

            if len(row) != 1:
                continue

            row = row.iloc[
                0
            ]

            status = row[
                "coverage_status"
            ]

            style = styles[
                status
            ]

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

            rect = Rectangle(
                (
                    x,
                    (
                        len(
                            DEFENCE_ORDER
                        )
                        - y
                        - 1
                    ),
                ),
                1,
                1,
                facecolor=style[
                    "facecolor"
                ],
                edgecolor="black",
                linewidth=0.9,
                hatch=style[
                    "hatch"
                ],
            )

            ax.add_patch(
                rect
            )

            text_color = (
                "white"
                if status
                == "COMPLETE"
                else "black"
            )

            if status == (
                "NO BASE ATTACK"
            ):

                cell_text = (
                    "N/A\n"
                    "base ASR 0%"
                )

            else:

                cell_text = (
                    f"{style['short']}\n"
                    f"{defended:.1f}%"
                )

            ax.text(
                x + 0.5,
                (
                    len(
                        DEFENCE_ORDER
                    )
                    - y
                    - 0.5
                ),
                cell_text,
                ha="center",
                va="center",
                fontsize=6.6,
                color=text_color,
                fontweight=(
                    "bold"
                    if status
                    in {
                        "COMPLETE",
                        "HARMFUL",
                    }
                    else "normal"
                ),
            )

    ax.set_xticks(
        (
            np.arange(
                len(
                    ATTACK_ORDER
                )
            )
            + 0.5
        )
    )

    ax.set_xticklabels(
        [
            ATTACK_SHORT[x]
            for x in ATTACK_ORDER
        ],
        rotation=18,
        ha="right",
    )

    ax.set_yticks(
        (
            np.arange(
                len(
                    DEFENCE_ORDER
                )
            )
            + 0.5
        )
    )

    ax.set_yticklabels(
        DEFENCE_ORDER[
            ::-1
        ]
    )

    ax.set_xlabel(
        "Attack"
    )

    ax.set_ylabel(
        "Defence"
    )

    legend = [
        Patch(
            facecolor="0.45",
            edgecolor="black",
            label="Complete",
        ),

        Patch(
            facecolor="white",
            edgecolor="black",
            hatch="////",
            label="Partial",
        ),

        Patch(
            facecolor="0.88",
            edgecolor="black",
            label="No clear effect",
        ),

        Patch(
            facecolor="white",
            edgecolor="black",
            hatch="xx",
            label="Harmful",
        ),

        Patch(
            facecolor="white",
            edgecolor="black",
            hatch="..",
            label="No base attack",
        ),
    ]

    ax.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            1.13,
        ),
        ncol=5,
        frameon=False,
        handlelength=1.7,
    )

    ax.set_aspect(
        "auto"
    )

    save_figure(
        fig,
        "fig03_attack_defence_outcomes",
    )


# =====================================================================
# FIGURE 4
# TokenBreak forest plot
# =====================================================================

def figure_4_tokenbreak(
    coverage,
):

    df = coverage[
        coverage["attack"]
        == "TokenBreak"
    ].copy()

    df["reduction"] = as_percent(
        df[
            "asr_reduction_pp"
        ]
    )

    df["ci_low"] = as_percent(
        df[
            "asr_reduction_ci_low_pp"
        ]
    )

    df["ci_high"] = as_percent(
        df[
            "asr_reduction_ci_high_pp"
        ]
    )

    df["defended"] = as_percent(
        df[
            "paired_defended_asr"
        ]
    )

    df = (
        df.sort_values(
            "reduction",
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )

    y = np.arange(
        len(df)
    )

    low_err = np.maximum(
        (
            df[
                "reduction"
            ]
            - df[
                "ci_low"
            ]
        ).to_numpy(),
        0,
    )

    high_err = np.maximum(
        (
            df[
                "ci_high"
            ]
            - df[
                "reduction"
            ]
        ).to_numpy(),
        0,
    )

    fig, ax = plt.subplots(
        figsize=(
            FULL_W,
            3.6,
        )
    )

    ax.errorbar(
        df[
            "reduction"
        ],
        y,
        xerr=np.vstack(
            [
                low_err,
                high_err,
            ]
        ),
        fmt="o",
        color="black",
        ecolor="black",
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.3,
        markersize=6,
        capsize=4,
        elinewidth=1.5,
        capthick=1.3,
    )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.1,
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        df[
            "defence"
        ]
    )

    ax.set_xlabel(
        "Paired ASR reduction "
        "(percentage points)"
    )

    ax.grid(
        axis="x",
        color="0.86",
        linewidth=0.7,
    )

    ax.set_axisbelow(
        True
    )

    clean_axes(
        ax
    )

    # reserve annotation space
    x_max = max(
        68,
        float(
            df[
                "ci_high"
            ].max()
        )
        + 16,
    )

    ax.set_xlim(
        -3,
        x_max,
    )

    for i, row in df.iterrows():

        if row[
            "reduction"
        ] > 0.05:

            label = (
                f"{row['reduction']:.1f} pp "
                f"[{row['ci_low']:.1f}, "
                f"{row['ci_high']:.1f}]"
            )

        else:

            label = (
                f"defended ASR "
                f"{row['defended']:.1f}%"
            )

        ax.text(
            row[
                "ci_high"
            ]
            + 0.8,
            i,
            label,
            va="center",
            fontsize=7.5,
        )

    save_figure(
        fig,
        "fig04_tokenbreak_paired_reduction",
    )


# =====================================================================
# FIGURE 5
# utility - two grayscale panels
# =====================================================================

def utility_panel(
    ax,
    matrix,
    title,
):
    """
    grayscale quantitative matrix.
    explicit values preserve meaning in print.
    """

    image = ax.imshow(
        matrix,
        cmap="Greys",
        vmin=0,
        vmax=100,
        aspect="auto",
    )

    n_rows, n_cols = (
        matrix.shape
    )

    # cell borders
    for y in range(
        n_rows + 1
    ):

        ax.axhline(
            y - 0.5,
            color="black",
            linewidth=0.5,
        )

    for x in range(
        n_cols + 1
    ):

        ax.axvline(
            x - 0.5,
            color="black",
            linewidth=0.5,
        )

    for y in range(
        n_rows
    ):

        for x in range(
            n_cols
        ):

            value = matrix[
                y,
                x,
            ]

            if np.isnan(
                value
            ):
                continue

            text_colour = (
                "white"
                if value >= 62
                else "black"
            )

            ax.text(
                x,
                y,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=7,
                color=text_colour,
                fontweight="bold",
            )

    ax.set_title(
        title,
        fontsize=9.3,
    )

    return image


def figure_5_utility(
    preserve,
    changed,
):

    preserve_matrix = (
        preserve
        .to_numpy(
            dtype=float
        )
    )

    changed_matrix = (
        changed
        .to_numpy(
            dtype=float
        )
    )

    categories = list(
        preserve.columns
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            FULL_W,
            3.65,
        ),
        constrained_layout=True,
    )

    image_a = utility_panel(
        axes[0],
        preserve_matrix,
        "A. Operational preservation (%)",
    )

    image_b = utility_panel(
        axes[1],
        changed_matrix,
        "B. Representation changed (%)",
    )

    for index, ax in enumerate(
        axes
    ):

        ax.set_xticks(
            range(
                len(
                    categories
                )
            )
        )

        ax.set_xticklabels(
            categories,
            rotation=30,
            ha="right",
            fontsize=7.3,
        )

        ax.set_yticks(
            range(
                len(
                    DEFENCE_ORDER
                )
            )
        )

        if index == 0:

            ax.set_yticklabels(
                DEFENCE_ORDER,
                fontsize=7.3,
            )

            ax.set_ylabel(
                "Defence"
            )

        else:

            ax.set_yticklabels(
                []
            )

    # one shared grayscale key
    cbar = fig.colorbar(
        image_b,
        ax=axes,
        shrink=0.82,
        fraction=0.035,
        pad=0.025,
    )

    cbar.set_label(
        "Rate (%)",
        fontsize=8,
    )

    cbar.ax.tick_params(
        labelsize=7
    )

    save_figure(
        fig,
        "fig05_complex_legitimate_utility",
    )


# =====================================================================
# FIGURE 6
# margin analysis with hatch
# =====================================================================

def figure_6_margin(
    margin,
):

    fig, ax = plt.subplots(
        figsize=(
            FULL_W,
            3.55,
        )
    )

    x = np.arange(
        len(
            ATTACK_ORDER
        )
    )

    width = 0.24

    faces = [
        "white",
        "0.72",
        "0.25",
    ]

    hatches = [
        "////",
        "..",
        "",
    ]

    for index, label in enumerate(
        MARGIN_ORDER
    ):

        part = (
            margin[
                margin[
                    "clean_probability_bin"
                ]
                == label
            ]
            .set_index(
                "attack"
            )
            .reindex(
                ATTACK_ORDER
            )
        )

        y = as_percent(
            part[
                "asr"
            ]
        ).to_numpy(
            dtype=float
        )

        lo = as_percent(
            part[
                "asr_ci_low"
            ]
        ).to_numpy(
            dtype=float
        )

        hi = as_percent(
            part[
                "asr_ci_high"
            ]
        ).to_numpy(
            dtype=float
        )

        low_error = np.maximum(
            y - lo,
            0,
        )

        high_error = np.maximum(
            hi - y,
            0,
        )

        n_values = (
            part[
                "n"
            ]
            .dropna()
            .astype(int)
            .unique()
        )

        n_text = (
            f", n={n_values[0]}"
            if len(
                n_values
            )
            == 1
            else ""
        )

        bars = ax.bar(
            (
                x
                + (
                    index - 1
                )
                * width
            ),
            y,
            width=width,
            facecolor=faces[
                index
            ],
            edgecolor="black",
            linewidth=1.0,
            hatch=hatches[
                index
            ],
            label=(
                f"{label}{n_text}"
            ),
            zorder=2,
        )

        ax.errorbar(
            (
                x
                + (
                    index - 1
                )
                * width
            ),
            y,
            yerr=np.vstack(
                [
                    low_error,
                    high_error,
                ]
            ),
            fmt="none",
            ecolor="black",
            elinewidth=1.15,
            capsize=3,
            capthick=1.1,
            zorder=3,
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            ATTACK_SHORT[x]
            for x in ATTACK_ORDER
        ],
        rotation=18,
        ha="right",
    )

    ax.set_ylim(
        0,
        106,
    )

    ax.set_ylabel(
        "Attack success rate (%)"
    )

    ax.set_xlabel(
        "Attack"
    )

    ax.grid(
        axis="y",
        color="0.86",
        linewidth=0.7,
    )

    ax.set_axisbelow(
        True
    )

    clean_axes(
        ax
    )

    ax.legend(
        title="Clean P(toxic)",
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            1.16,
        ),
    )

    save_figure(
        fig,
        "fig06_margin_stratified_asr",
    )


# =====================================================================
# FIGURE 7
# defence-induced regression
# =====================================================================

def figure_7_regression(
    coverage,
):

    row = coverage[
        (
            coverage[
                "attack"
            ]
            == "Unicode invisible"
        )
        &
        (
            coverage[
                "defence"
            ]
            == "Tokenizer translation"
        )
    ]

    if len(row) != 1:

        raise RuntimeError(
            "could not locate "
            "Invisible × Tokenizer translation"
        )

    row = row.iloc[
        0
    ]

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

    reduction = float(
        as_percent(
            pd.Series(
                [
                    row[
                        "asr_reduction_pp"
                    ]
                ]
            )
        ).iloc[0]
    )

    low = float(
        as_percent(
            pd.Series(
                [
                    row[
                        "asr_reduction_ci_low_pp"
                    ]
                ]
            )
        ).iloc[0]
    )

    high = float(
        as_percent(
            pd.Series(
                [
                    row[
                        "asr_reduction_ci_high_pp"
                    ]
                ]
            )
        ).iloc[0]
    )

    increase = (
        -reduction
    )

    increase_low = (
        -high
    )

    increase_high = (
        -low
    )

    n = int(
        row[
            "n_clean_retained"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(
            SMALL_W,
            3.15,
        )
    )

    ax.plot(
        [
            0,
            1,
        ],
        [
            undefended,
            defended,
        ],
        color="black",
        linewidth=1.8,
        marker="o",
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.5,
        markersize=7,
    )

    ax.set_xlim(
        -0.25,
        1.25,
    )

    ax.set_ylim(
        0,
        20,
    )

    ax.set_xticks(
        [
            0,
            1,
        ]
    )

    ax.set_xticklabels(
        [
            "No defence",
            "Tokenizer translation",
        ]
    )

    ax.set_ylabel(
        "Attack success rate (%)"
    )

    clean_axes(
        ax
    )

    ax.grid(
        axis="y",
        color="0.88",
        linewidth=0.7,
    )

    ax.set_axisbelow(
        True
    )

    ax.text(
        0,
        undefended + 0.8,
        f"{undefended:.1f}%",
        ha="center",
        fontsize=8.5,
    )

    ax.text(
        1,
        defended + 0.8,
        f"{defended:.1f}%",
        ha="center",
        fontsize=8.5,
    )

    ax.text(
        0.5,
        8.2,
        (
            f"ΔASR = +{increase:.1f} pp\n"
            f"95% CI "
            f"[+{increase_low:.1f}, "
            f"+{increase_high:.1f}] pp\n"
            f"paired n = {n}"
        ),
        ha="center",
        va="center",
        fontsize=8,
    )

    save_figure(
        fig,
        "fig07_defence_induced_regression",
    )


# =====================================================================
# APPENDIX A1
# AdvTok recorded actions
# =====================================================================

def appendix_advtok_actions(
    coverage,
):
    """
    supporting implementation-audit view only.

    no claim that these action distinctions are
    themselves a novel research result.
    """

    df = coverage[
        coverage[
            "attack"
        ]
        == "AdvTok"
    ].copy()

    df["defence"] = pd.Categorical(
        df[
            "defence"
        ],
        categories=DEFENCE_ORDER,
        ordered=True,
    )

    df = (
        df.sort_values(
            "defence"
        )
        .reset_index(
            drop=True
        )
    )

    values = np.column_stack(
        [
            as_percent(
                df[
                    "attack_detection_rate"
                ]
            ),

            as_percent(
                df[
                    "attack_block_rate"
                ]
            ),

            as_percent(
                df[
                    "reencode_rate"
                ]
            ),
        ]
    )

    fig, ax = plt.subplots(
        figsize=(
            FULL_W,
            3.5,
        )
    )

    ax.set_xlim(
        -0.75,
        2.75,
    )

    ax.set_ylim(
        -0.6,
        len(df) - 0.4,
    )

    ax.invert_yaxis()

    headers = [
        "Detection",
        "Blocking",
        "Re-encoding",
    ]

    ax.set_xticks(
        [
            0,
            1,
            2,
        ]
    )

    ax.set_xticklabels(
        headers
    )

    ax.set_yticks(
        range(
            len(df)
        )
    )

    ax.set_yticklabels(
        df[
            "defence"
        ]
    )

    for y in range(
        len(df)
    ):

        for x in range(
            3
        ):

            value = values[
                y,
                x,
            ]

            if value >= 99.5:

                ax.scatter(
                    x,
                    y,
                    s=90,
                    facecolor="black",
                    edgecolor="black",
                    zorder=3,
                )

                text = "100"

                colour = "white"

            elif value <= 0.5:

                ax.scatter(
                    x,
                    y,
                    s=90,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=1.2,
                    zorder=3,
                )

                text = "0"

                colour = "black"

            else:

                ax.scatter(
                    x,
                    y,
                    s=90,
                    facecolor="0.65",
                    edgecolor="black",
                    linewidth=1.1,
                    zorder=3,
                )

                text = (
                    f"{value:.0f}"
                )

                colour = "black"

            ax.text(
                x,
                y,
                text,
                ha="center",
                va="center",
                fontsize=6.5,
                color=colour,
                zorder=4,
            )

    # row/column guides
    for y in (
        np.arange(
            len(df) + 1
        )
        - 0.5
    ):

        ax.axhline(
            y,
            color="0.82",
            linewidth=0.6,
            zorder=0,
        )

    for x in [
        -0.5,
        0.5,
        1.5,
        2.5,
    ]:

        ax.axvline(
            x,
            color="0.82",
            linewidth=0.6,
            zorder=0,
        )

    ax.set_xlabel(
        "Recorded action rate (%)"
    )

    for side in [
        "top",
        "right",
        "bottom",
        "left",
    ]:

        ax.spines[
            side
        ].set_visible(
            False
        )

    save_figure(
        fig,
        "figA1_advtok_recorded_actions",
        appendix=True,
    )


# =====================================================================
# APPENDIX A2
# model-tokenizer eligibility
# =====================================================================

def appendix_model_tokenizer():

    df = pd.read_csv(
        find_file(
            "data/"
            "cross_tokenizer_eligibility.csv"
        )
    )

    total = len(
        df
    )

    counts = [
        int(
            df[
                "eligible_wordpiece"
            ]
            .astype(int)
            .sum()
        ),

        int(
            df[
                "eligible_bpe"
            ]
            .astype(int)
            .sum()
        ),

        int(
            df[
                "eligible_unigram"
            ]
            .astype(int)
            .sum()
        ),

        int(
            df[
                "eligible_all_three"
            ]
            .astype(int)
            .sum()
        ),
    ]

    labels = [
        "DistilBERT\nWordPiece",
        "RoBERTa\nBPE",
        "DeBERTa-v3\nUnigram",
        "Shared\nintersection",
    ]

    percentages = [
        (
            100
            * x
            / total
        )
        for x
        in counts
    ]

    fig, ax = plt.subplots(
        figsize=(
            FULL_W,
            3.25,
        )
    )

    bars = ax.bar(
        labels,
        percentages,
        facecolor="0.65",
        edgecolor="black",
        linewidth=1.1,
    )

    ax.set_ylim(
        0,
        105,
    )

    ax.set_ylabel(
        "Clean toxic examples eligible (%)"
    )

    clean_axes(
        ax
    )

    ax.grid(
        axis="y",
        color="0.88",
        linewidth=0.7,
    )

    ax.set_axisbelow(
        True
    )

    for bar, count, pct in zip(
        bars,
        counts,
        percentages,
    ):

        ax.text(
            (
                bar.get_x()
                + bar.get_width()
                / 2
            ),
            pct + 1.2,
            (
                f"{count}/{total}\n"
                f"{pct:.1f}%"
            ),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    save_figure(
        fig,
        "figA2_model_tokenizer_clean_eligibility",
        appendix=True,
    )


# =====================================================================
# captions / usage guide
# =====================================================================

def write_notes():

    text = """# Camera-ready figure notes

## Main dissertation

### Figure 1
Experimental design of the primary evaluation. The clean BPE/Unigram
comparison is shown as supporting evidence only and is visually separated
from the primary WordPiece attack pipeline.

### Figure 2
Undefended attack success rates on the 156 eligible toxic inputs.

### Figure 3
Attack-by-defence outcome matrix. Cells report defended ASR.
Patterns, text and grayscale shading are redundant so the figure remains
interpretable in monochrome. "Complete" means complete suppression only
in the evaluated setting. "N/A" indicates that the corresponding
undefended attack had zero ASR.

### Figure 4
Paired TokenBreak ASR reduction with 95% confidence intervals.
This figure should be used to support the bounded claim that no tested
defence completely suppressed TokenBreak in this experiment.

### Figure 5
Complex legitimate-input utility. Panel A reports operational
preservation. Panel B reports legitimate inputs whose representation
changed. These are deliberately separated.

### Figure 6
Attack success stratified by clean victim confidence. Bars use both
grayscale and hatching, and error bars are Wilson 95% confidence
intervals.

### Figure 7
Defence-induced regression for invisible Unicode under tokenizer
translation, reporting the paired ASR increase and confidence interval.

## Appendix

### Figure A1
Recorded defence actions for AdvTok. This is an implementation-audit
view, not a claimed novel security result.

### Figure A2
Clean eligibility across different model-tokenizer setups.
Model and tokenizer change together, so this figure must not be used to
attribute the differences to tokenizer family alone.

## File formats

Use PDF in the final dissertation whenever the Word/LaTeX workflow
preserves vector graphics correctly.

SVG is provided as an alternative vector format.

PNG is a preview/fallback only and should not be used when a vector
version can be inserted.

## Print check

Before final submission:
1. print/export one page in grayscale;
2. view it at 100%;
3. verify every axis label, cell value and confidence interval remains
   legible;
4. do not enlarge a figure in the PDF viewer when deciding whether the
   text is readable at dissertation size.
"""

    (
        OUT
        / "CAMERA_READY_NOTES.md"
    ).write_text(
        text,
        encoding="utf-8",
    )


# =====================================================================
# run
# =====================================================================

if __name__ == "__main__":

    print(
        "building camera-ready figures"
    )

    print(
        "python :",
        sys.executable,
    )

    print(
        "note   : frozen evidence only"
    )

    print(
        "note   : no experiments or model calls"
    )

    print()

    attacks = (
        load_attack_rows()
    )

    coverage = (
        load_coverage()
    )

    margin = (
        load_margin()
    )

    (
        utility_preserve,
        utility_changed,
    ) = load_utility()

    print(
        "loaded:"
    )

    print(
        " attack rows :",
        len(attacks),
    )

    print(
        " coverage    :",
        len(coverage),
    )

    print(
        " margin      :",
        len(margin),
    )

    print()

    print(
        "main figure 1..."
    )

    figure_1_design()

    print(
        "main figure 2..."
    )

    figure_2_attacks(
        attacks
    )

    print(
        "main figure 3..."
    )

    figure_3_coverage(
        coverage
    )

    print(
        "main figure 4..."
    )

    figure_4_tokenbreak(
        coverage
    )

    print(
        "main figure 5..."
    )

    figure_5_utility(
        utility_preserve,
        utility_changed,
    )

    print(
        "main figure 6..."
    )

    figure_6_margin(
        margin
    )

    print(
        "main figure 7..."
    )

    figure_7_regression(
        coverage
    )

    print(
        "appendix A1..."
    )

    appendix_advtok_actions(
        coverage
    )

    print(
        "appendix A2..."
    )

    appendix_model_tokenizer()

    write_notes()

    print()

    print(
        "camera-ready build complete"
    )

    print()

    print(
        "main figures:"
    )

    for path in sorted(
        FIG.glob("*.pdf")
    ):

        print(
            " ",
            path
        )

    print()

    print(
        "appendix figures:"
    )

    for path in sorted(
        APP.glob("*.pdf")
    ):

        print(
            " ",
            path
        )

    print()

    print(
        "notes:",
        OUT
        / "CAMERA_READY_NOTES.md",
    )