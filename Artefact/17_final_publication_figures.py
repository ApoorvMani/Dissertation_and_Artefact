"""
final publication figures for the dissertation.

IMPORTANT:
- frozen evidence only
- no model calls
- no attacks
- no defence reruns
- no threshold changes
- no new statistics
- does not overwrite earlier figure directories

outputs:
results/dissertation/final_publication_v2/
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle


# ================================================================
# output
# ================================================================

OUT = Path(
    "results/dissertation/final_publication_v2"
)

MAIN = OUT / "main"
APP = OUT / "appendix"

MAIN.mkdir(
    parents=True,
    exist_ok=True,
)

APP.mkdir(
    parents=True,
    exist_ok=True,
)


# ================================================================
# fixed ordering
# ================================================================

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


UTILITY_CATEGORY_ORDER = [
    "Code",
    "URLs",
    "Emoji",
    "Misspelling",
    "Non-English",
    "Mixed-script",
]


# ================================================================
# restrained publication palette
#
# based around colourblind-safe contrasts.
# text remains the primary encoding.
# ================================================================

COLORS = {
    "blue":
        "#0072B2",

    "sky":
        "#56B4E9",

    "green":
        "#009E73",

    "orange":
        "#E69F00",

    "vermillion":
        "#D55E00",

    "purple":
        "#CC79A7",

    "grey":
        "#B8B8B8",

    "light_grey":
        "#E4E4E4",

    "very_light":
        "#F6F6F6",

    "dark":
        "#333333",
}


# ================================================================
# publication style
# ================================================================

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
            10.5,

        "axes.titlesize":
            11.5,

        "axes.labelsize":
            10.5,

        "xtick.labelsize":
            9.5,

        "ytick.labelsize":
            9.5,

        "legend.fontsize":
            9,

        "axes.linewidth":
            1.25,

        "lines.linewidth":
            1.7,

        "lines.markersize":
            7,

        "xtick.major.width":
            1.1,

        "ytick.major.width":
            1.1,

        "xtick.major.size":
            4.5,

        "ytick.major.size":
            4.5,

        "pdf.fonttype":
            42,

        "ps.fonttype":
            42,

        "svg.fonttype":
            "none",

        "savefig.bbox":
            "tight",

        "savefig.pad_inches":
            0.08,
    }
)


# ================================================================
# helpers
# ================================================================

def find_file(*candidates):
    """return the first existing file."""

    for candidate in candidates:

        path = Path(
            candidate
        )

        if path.exists():
            return path

    raise FileNotFoundError(
        "could not find any of:\n"
        + "\n".join(
            f"  {x}"
            for x in candidates
        )
    )


def save_figure(
    fig,
    filename,
    appendix=False,
):
    """
    save vector-first.

    png is only a preview/fallback.
    """

    folder = (
        APP
        if appendix
        else MAIN
    )

    fig.savefig(
        folder / f"{filename}.pdf"
    )

    fig.savefig(
        folder / f"{filename}.svg"
    )

    fig.savefig(
        folder / f"{filename}.png",
        dpi=400,
    )

    plt.close(
        fig
    )


def as_percent(values):
    """
    convert fractional rates to percentage.

    0.962 -> 96.2
    48.8  -> 48.8
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
    """remove unnecessary borders."""

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


# ================================================================
# load frozen evidence
# ================================================================

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

    unicode_df = pd.read_csv(
        find_file(
            "data/unicode_results.csv",
            "unicode_results.csv",
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
    use the verified intermediate utility matrices.

    these were derived from frozen step11 results.
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
        normalise_defence(
            x
        )
        for x
        in preserve.index
    ]

    changed.index = [
        normalise_defence(
            x
        )
        for x
        in changed.index
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


# ================================================================
# figure 1
# experimental design
# ================================================================

def figure_1_design():

    fig, ax = plt.subplots(
        figsize=(
            11.5,
            7.3,
        )
    )

    ax.set_xlim(
        0,
        13.5,
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
        facecolor="white",
        edgecolor=COLORS["dark"],
        dashed=False,
        fontsize=10,
        linewidth=1.7,
    ):

        patch = FancyBboxPatch(
            (
                x,
                y,
            ),
            w,
            h,
            boxstyle="round,pad=0.025",
            facecolor=facecolor,
            edgecolor=edgecolor,
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
                    1.7,

                "color":
                    COLORS["dark"],

                "linestyle":
                    "--"
                    if dashed
                    else "-",
            },
        )

    # source
    box(
        4.2,
        8.75,
        3.5,
        0.8,
        "Jigsaw source data\n"
        "250 toxic + 250 benign",
        facecolor="#F8F8F8",
    )

    # victim
    box(
        4.2,
        7.25,
        3.5,
        0.8,
        "WordPiece victim\n"
        "156 / 250 toxic eligible",
        facecolor="#F3F7FA",
    )

    arrow(
        5.95,
        8.75,
        5.95,
        8.05,
    )

    # attack family row
    box(
        0.55,
        5.45,
        2.6,
        1.0,
        "TokenBreak\n156 cases",
        facecolor="#EAF3F8",
    )

    box(
        4.65,
        5.45,
        2.6,
        1.0,
        "AdvTok\n156 cases",
        facecolor="#EAF3F8",
    )

    box(
        8.75,
        5.45,
        2.6,
        1.0,
        "Unicode\n4 × 156 cases",
        facecolor="#EAF3F8",
    )

    arrow(
        5.1,
        7.25,
        1.85,
        6.45,
    )

    arrow(
        5.95,
        7.25,
        5.95,
        6.45,
    )

    arrow(
        6.8,
        7.25,
        10.05,
        6.45,
    )

    # pooled attacks
    box(
        4.15,
        3.9,
        3.6,
        0.85,
        "936 attack instances",
        facecolor="#F8F8F8",
    )

    arrow(
        1.85,
        5.45,
        4.95,
        4.75,
    )

    arrow(
        5.95,
        5.45,
        5.95,
        4.75,
    )

    arrow(
        10.05,
        5.45,
        6.95,
        4.75,
    )

    # defence evaluation
    box(
        4.15,
        2.45,
        3.6,
        0.9,
        "7 defences\n"
        "6,552 paired evaluations",
        facecolor="#EDF7F2",
    )

    arrow(
        5.95,
        3.9,
        5.95,
        3.35,
    )

    # outcomes
    box(
        0.3,
        0.35,
        3.1,
        1.0,
        "Security\n"
        "ASR + paired CIs",
        facecolor="#F8F8F8",
    )

    box(
        4.4,
        0.35,
        3.1,
        1.0,
        "Defence audit\n"
        "flag / block / transform",
        facecolor="#F8F8F8",
    )

    box(
        8.5,
        0.35,
        3.1,
        1.0,
        "Utility\n"
        "clean + complex inputs",
        facecolor="#F8F8F8",
    )

    arrow(
        5.0,
        2.45,
        1.85,
        1.35,
    )

    arrow(
        5.95,
        2.45,
        5.95,
        1.35,
    )

    arrow(
        6.9,
        2.45,
        10.05,
        1.35,
    )

    # supporting baseline panel
    ax.plot(
        [
            12.05,
            12.05,
        ],
        [
            4.7,
            9.4,
        ],
        color=COLORS["grey"],
        linestyle="--",
        linewidth=1.6,
    )

    ax.text(
        12.8,
        9.1,
        "SUPPORTING",
        ha="center",
        fontweight="bold",
        fontsize=9.5,
    )

    box(
        12.2,
        7.0,
        1.15,
        1.6,
        "Clean eligibility\n\n"
        "WP 156\n"
        "BPE 214\n"
        "Uni 242\n"
        "shared 155",
        facecolor="#FAFAFA",
        dashed=True,
        fontsize=8,
    )

    arrow(
        7.7,
        9.1,
        12.2,
        7.95,
        dashed=True,
    )

    ax.text(
        12.78,
        6.15,
        "Different model–tokenizer\n"
        "setups; clean baseline only",
        ha="center",
        va="center",
        fontsize=7.7,
    )

    ax.text(
        12.78,
        5.25,
        "No BPE/Unigram\n"
        "attacks executed",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
    )

    save_figure(
        fig,
        "fig01_experimental_design",
    )


# ================================================================
# figure 2
# undefended ASR
# ================================================================

def figure_2_undefended(
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
            9.5,
            5.2,
        )
    )

    bars = ax.barh(
        plot["attack"],
        plot["asr"],
        facecolor=COLORS["sky"],
        edgecolor=COLORS["dark"],
        linewidth=1.35,
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
        color="#DDDDDD",
        linewidth=0.8,
    )

    ax.set_axisbelow(
        True
    )

    for bar, (
        _,
        row,
    ) in zip(
        bars,
        plot.iterrows(),
    ):

        ax.text(
            min(
                row["asr"] + 1.0,
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
            fontsize=9.2,
        )

    save_figure(
        fig,
        "fig02_undefended_attack_asr",
    )


# ================================================================
# figure 3
# headline coverage matrix
# ================================================================

def figure_3_coverage(
    coverage,
):

    style = {
        "NO BASE ATTACK": {
            "color":
                "#F7F7F7",

            "label":
                "NO BASE ATTACK",
        },

        "NO CLEAR EFFECT": {
            "color":
                "#DDDDDD",

            "label":
                "NO CLEAR EFFECT",
        },

        "PARTIAL": {
            "color":
                "#8EC7E8",

            "label":
                "PARTIAL",
        },

        "COMPLETE": {
            "color":
                "#78C6B5",

            "label":
                "COMPLETE",
        },

        "HARMFUL": {
            "color":
                "#E98973",

            "label":
                "HARMFUL",
        },
    }

    fig, ax = plt.subplots(
        figsize=(
            13.5,
            7.2,
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

    for row_index, defence in enumerate(
        DEFENCE_ORDER
    ):

        y = (
            len(
                DEFENCE_ORDER
            )
            - row_index
            - 1
        )

        for x, attack in enumerate(
            ATTACK_ORDER
        ):

            part = coverage[
                (
                    coverage["attack"]
                    == attack
                )
                &
                (
                    coverage["defence"]
                    == defence
                )
            ]

            if len(part) != 1:
                continue

            row = part.iloc[
                0
            ]

            status = row[
                "coverage_status"
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
                    y,
                ),
                1,
                1,
                facecolor=style[
                    status
                ][
                    "color"
                ],
                edgecolor="white",
                linewidth=2.0,
            )

            ax.add_patch(
                rect
            )

            # grayscale redundancy:
            # harmful gets strong border
            if status == "HARMFUL":

                warning = Rectangle(
                    (
                        x + 0.02,
                        y + 0.02,
                    ),
                    0.96,
                    0.96,
                    fill=False,
                    edgecolor=COLORS[
                        "dark"
                    ],
                    linewidth=2.2,
                    linestyle="--",
                )

                ax.add_patch(
                    warning
                )

            if status == (
                "NO BASE ATTACK"
            ):

                cell_text = (
                    "NO BASE ATTACK\n"
                    "undefended ASR 0%"
                )

            else:

                cell_text = (
                    f"{style[status]['label']}\n"
                    f"ASR {defended:.1f}%"
                )

            ax.text(
                x + 0.5,
                y + 0.5,
                cell_text,
                ha="center",
                va="center",
                fontsize=8.1,
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
            for x
            in ATTACK_ORDER
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

    legend_handles = [
        Patch(
            facecolor=style[
                "COMPLETE"
            ][
                "color"
            ],
            label="Complete",
        ),

        Patch(
            facecolor=style[
                "PARTIAL"
            ][
                "color"
            ],
            label="Partial",
        ),

        Patch(
            facecolor=style[
                "NO CLEAR EFFECT"
            ][
                "color"
            ],
            label="No clear effect",
        ),

        Patch(
            facecolor=style[
                "HARMFUL"
            ][
                "color"
            ],
            edgecolor=COLORS[
                "dark"
            ],
            linestyle="--",
            label="Harmful",
        ),

        Patch(
            facecolor=style[
                "NO BASE ATTACK"
            ][
                "color"
            ],
            edgecolor="#AAAAAA",
            label="No base attack",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            1.09,
        ),
        frameon=False,
    )

    for spine in ax.spines.values():
        spine.set_visible(
            False
        )

    save_figure(
        fig,
        "fig03_attack_defence_coverage",
    )


# ================================================================
# figure 4
# TokenBreak forest plot
# ================================================================

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
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    y = np.arange(
        len(df)
    )

    low_error = np.maximum(
        (
            df["reduction"]
            - df["ci_low"]
        ).to_numpy(),
        0,
    )

    high_error = np.maximum(
        (
            df["ci_high"]
            - df["reduction"]
        ).to_numpy(),
        0,
    )

    fig, ax = plt.subplots(
        figsize=(
            10.5,
            5.8,
        )
    )

    ax.errorbar(
        df["reduction"],
        y,
        xerr=np.vstack(
            [
                low_error,
                high_error,
            ]
        ),
        fmt="o",
        color=COLORS["blue"],
        ecolor=COLORS["dark"],
        markerfacecolor="white",
        markeredgecolor=COLORS["blue"],
        markeredgewidth=2.0,
        markersize=8,
        elinewidth=2.0,
        capsize=5,
        capthick=1.8,
    )

    ax.axvline(
        0,
        color=COLORS["dark"],
        linestyle="--",
        linewidth=1.4,
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        df["defence"]
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Paired reduction in attack success rate "
        "(percentage points)"
    )

    ax.grid(
        axis="x",
        color="#DDDDDD",
        linewidth=0.8,
    )

    ax.set_axisbelow(
        True
    )

    clean_axes(
        ax
    )

    ax.set_xlim(
        -4,
        max(
            72,
            float(
                df["ci_high"].max()
            )
            + 15,
        ),
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
            + 1.1,
            i,
            label,
            va="center",
            fontsize=9,
        )

    save_figure(
        fig,
        "fig04_tokenbreak_paired_reduction",
    )


# ================================================================
# figure 5
# utility two-panel
# ================================================================

def draw_utility_panel(
    ax,
    matrix,
    title,
):
    """
    cividis is colourblind-safe and remains ordered
    when printed without colour.
    """

    image = ax.imshow(
        matrix,
        cmap="cividis",
        vmin=0,
        vmax=100,
        aspect="auto",
    )

    rows, cols = (
        matrix.shape
    )

    for y in range(
        rows + 1
    ):

        ax.axhline(
            y - 0.5,
            color="white",
            linewidth=1.5,
        )

    for x in range(
        cols + 1
    ):

        ax.axvline(
            x - 0.5,
            color="white",
            linewidth=1.5,
        )

    for y in range(
        rows
    ):

        for x in range(
            cols
        ):

            value = matrix[
                y,
                x,
            ]

            if np.isnan(
                value
            ):
                continue

            colour = (
                "white"
                if value < 35
                or value > 72
                else "black"
            )

            ax.text(
                x,
                y,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=colour,
            )

    ax.set_title(
        title,
        fontsize=11,
    )

    return image


def figure_5_utility(
    preserve,
    changed,
):

    preserve_matrix = (
        preserve.to_numpy(
            dtype=float
        )
    )

    changed_matrix = (
        changed.to_numpy(
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
            14.0,
            6.2,
        ),
        constrained_layout=True,
    )

    image_a = draw_utility_panel(
        axes[0],
        preserve_matrix,
        "A. Operational preservation (%)",
    )

    image_b = draw_utility_panel(
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
            rotation=24,
            ha="right",
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
                DEFENCE_ORDER
            )

            ax.set_ylabel(
                "Defence"
            )

        else:

            ax.set_yticklabels(
                []
            )

        for spine in ax.spines.values():
            spine.set_visible(
                False
            )

    cbar = fig.colorbar(
        image_b,
        ax=axes,
        shrink=0.82,
        fraction=0.035,
        pad=0.025,
    )

    cbar.set_label(
        "Rate (%)"
    )

    save_figure(
        fig,
        "fig05_complex_utility",
    )


# ================================================================
# figure 6
# margin analysis
# ================================================================

def figure_6_margin(
    margin,
):

    fig, ax = plt.subplots(
        figsize=(
            11.5,
            5.8,
        )
    )

    x = np.arange(
        len(
            ATTACK_ORDER
        )
    )

    width = 0.24

    facecolors = [
        "#D9EDF7",
        "#8EC7E8",
        "#336B8E",
    ]

    hatches = [
        "///",
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

        values = as_percent(
            part["asr"]
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

        lower = np.maximum(
            values - lo,
            0,
        )

        upper = np.maximum(
            hi - values,
            0,
        )

        n_values = (
            part["n"]
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

        positions = (
            x
            + (
                index - 1
            )
            * width
        )

        ax.bar(
            positions,
            values,
            width=width,
            facecolor=facecolors[
                index
            ],
            edgecolor=COLORS[
                "dark"
            ],
            linewidth=1.2,
            hatch=hatches[
                index
            ],
            label=(
                f"{label}{n_text}"
            ),
            zorder=2,
        )

        ax.errorbar(
            positions,
            values,
            yerr=np.vstack(
                [
                    lower,
                    upper,
                ]
            ),
            fmt="none",
            ecolor=COLORS[
                "dark"
            ],
            elinewidth=1.7,
            capsize=4,
            capthick=1.5,
            zorder=3,
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            ATTACK_SHORT[x]
            for x
            in ATTACK_ORDER
        ],
        rotation=16,
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
        color="#DDDDDD",
        linewidth=0.8,
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
            1.17,
        ),
    )

    save_figure(
        fig,
        "fig06_margin_stratified_asr",
    )


# ================================================================
# figure 7
# defence-induced regression
# ================================================================

def figure_7_regression(
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
            7.5,
            5.0,
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
        color=COLORS[
            "blue"
        ],
        linewidth=2.4,
        marker="o",
        markerfacecolor="white",
        markeredgecolor=COLORS[
            "blue"
        ],
        markeredgewidth=2.2,
        markersize=10,
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

    ax.grid(
        axis="y",
        color="#DDDDDD",
        linewidth=0.8,
    )

    ax.set_axisbelow(
        True
    )

    clean_axes(
        ax
    )

    ax.text(
        0,
        undefended + 0.8,
        f"{undefended:.1f}%",
        ha="center",
        fontsize=10,
    )

    ax.text(
        1,
        defended + 0.8,
        f"{defended:.1f}%",
        ha="center",
        fontsize=10,
    )

    ax.text(
        0.5,
        8.3,
        (
            f"ΔASR = +{increase:.1f} pp\n"
            f"95% CI "
            f"[+{increase_low:.1f}, "
            f"+{increase_high:.1f}] pp\n"
            f"paired n = {n}"
        ),
        ha="center",
        va="center",
        fontsize=9.5,
    )

    save_figure(
        fig,
        "fig07_defence_induced_regression",
    )


# ================================================================
# appendix figure A1
# AdvTok action audit
#
# deliberately supporting only.
# ================================================================

def appendix_advtok_audit(
    coverage,
):

    df = coverage[
        coverage["attack"]
        == "AdvTok"
    ].copy()

    df["defence"] = pd.Categorical(
        df["defence"],
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

    detection = as_percent(
        df[
            "attack_detection_rate"
        ]
    ).to_numpy(
        dtype=float
    )

    blocking = as_percent(
        df[
            "attack_block_rate"
        ]
    ).to_numpy(
        dtype=float
    )

    reencode = as_percent(
        df[
            "reencode_rate"
        ]
    ).to_numpy(
        dtype=float
    )

    matrix = np.column_stack(
        [
            detection,
            blocking,
            reencode,
        ]
    )

    fig, ax = plt.subplots(
        figsize=(
            8.5,
            5.3,
        )
    )

    image = ax.imshow(
        matrix,
        cmap="cividis",
        vmin=0,
        vmax=100,
        aspect="auto",
    )

    ax.set_xticks(
        [
            0,
            1,
            2,
        ]
    )

    ax.set_xticklabels(
        [
            "Flagged",
            "Blocked",
            "Re-encoded",
        ]
    )

    ax.set_yticks(
        range(
            len(
                DEFENCE_ORDER
            )
        )
    )

    ax.set_yticklabels(
        DEFENCE_ORDER
    )

    for y in range(
        matrix.shape[0]
    ):

        for x in range(
            matrix.shape[1]
        ):

            value = matrix[
                y,
                x,
            ]

            ax.text(
                x,
                y,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=(
                    "white"
                    if value > 70
                    else "black"
                ),
            )

    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.045,
        pad=0.03,
    )

    cbar.set_label(
        "Recorded action rate (%)"
    )

    save_figure(
        fig,
        "figA1_advtok_action_audit",
        appendix=True,
    )


# ================================================================
# appendix figure A2
# model-tokenizer clean baseline
# ================================================================

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
        100
        * count
        / total
        for count
        in counts
    ]

    fig, ax = plt.subplots(
        figsize=(
            8.5,
            5.1,
        )
    )

    bars = ax.bar(
        labels,
        percentages,
        facecolor=COLORS[
            "sky"
        ],
        edgecolor=COLORS[
            "dark"
        ],
        linewidth=1.3,
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
        color="#DDDDDD",
        linewidth=0.8,
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
            fontsize=9.5,
        )

    save_figure(
        fig,
        "figA2_model_tokenizer_baseline",
        appendix=True,
    )


# ================================================================
# appendix figure A3
# research evolution
# ================================================================

def appendix_research_evolution():

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
            "293,034-row paired\n"
            "attack corpus",
        ),

        (
            "31 Jul",
            "Environment issue\n"
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
            "Evidence freeze",
        ),
    ]

    x = np.arange(
        len(
            labels
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            11.5,
            4.2,
        )
    )

    ax.plot(
        x,
        np.zeros_like(
            x
        ),
        color=COLORS[
            "blue"
        ],
        linewidth=2.0,
        marker="o",
        markersize=7,
    )

    for index, (
        date,
        label,
    ) in enumerate(
        labels
    ):

        offset = (
            0.28
            if index % 2 == 0
            else -0.28
        )

        ax.plot(
            [
                index,
                index,
            ],
            [
                0,
                offset,
            ],
            color=COLORS[
                "grey"
            ],
            linewidth=1.4,
        )

        ax.text(
            index,
            offset,
            (
                f"{date}\n"
                f"{label}"
            ),
            ha="center",
            va=(
                "bottom"
                if offset > 0
                else "top"
            ),
            fontsize=8.3,
        )

    ax.axvline(
        4.5,
        color=COLORS[
            "grey"
        ],
        linestyle="--",
        linewidth=1.4,
    )

    ax.text(
        2.0,
        0.62,
        "Preliminary / historical programme",
        ha="center",
        fontsize=9.5,
        fontweight="bold",
    )

    ax.text(
        5.5,
        0.62,
        "Primary dissertation",
        ha="center",
        fontsize=9.5,
        fontweight="bold",
    )

    ax.set_ylim(
        -0.9,
        0.9,
    )

    ax.set_xlim(
        -0.5,
        len(
            labels
        )
        - 0.5,
    )

    ax.axis(
        "off"
    )

    save_figure(
        fig,
        "figA3_research_evolution",
        appendix=True,
    )


# ================================================================
# notes
# ================================================================

def write_notes():

    text = """# Final publication figure notes

These figures are generated from frozen result artefacts only.

No attack, defence, model inference, calibration or statistical analysis
is rerun by this script.

## Main-body recommendation

1. fig01_experimental_design
2. fig02_undefended_attack_asr
3. fig03_attack_defence_coverage
4. fig04_tokenbreak_paired_reduction
5. fig05_complex_utility
6. fig06_margin_stratified_asr
7. fig07_defence_induced_regression

## Appendix recommendation

- figA1_advtok_action_audit
- figA2_model_tokenizer_baseline
- figA3_research_evolution

## Important interpretation rules

Figure 3:
"COMPLETE" means defended ASR reached zero in the evaluated setting.
It must not be written as "the attack is solved".

Figure A1:
This is an implementation/audit figure only. The recorded actions are
not presented as a novel security mechanism or contribution.

Figure A2:
Model and tokenizer both differ. The eligibility differences therefore
cannot be attributed to tokenizer family alone.

Figure A3:
Historical results are evidence of research development and are not
pooled with the primary attack-defence experiment.

## File format

Use PDF or SVG in the final dissertation whenever possible.

PNG is included only as a preview or fallback.

## Print check

Before submission:

- insert figures at their real dissertation width;
- export the dissertation to PDF;
- inspect at 100% zoom;
- print/export one page in grayscale;
- verify all labels and numbers remain readable;
- confirm that no conclusion depends on colour alone.
"""

    (
        OUT
        / "FINAL_FIGURE_NOTES.md"
    ).write_text(
        text,
        encoding="utf-8",
    )


# ================================================================
# run
# ================================================================

if __name__ == "__main__":

    print(
        "building final publication figures"
    )

    print(
        "python :",
        sys.executable,
    )

    print(
        "output :",
        OUT,
    )

    print(
        "note   : new directory only"
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

    figure_2_undefended(
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

    appendix_advtok_audit(
        coverage
    )

    print(
        "appendix A2..."
    )

    appendix_model_tokenizer()

    print(
        "appendix A3..."
    )

    appendix_research_evolution()

    write_notes()

    print()

    print(
        "final publication build complete"
    )

    print()

    print(
        "main:"
    )

    for path in sorted(
        MAIN.glob(
            "*.pdf"
        )
    ):

        print(
            " ",
            path
        )

    print()

    print(
        "appendix:"
    )

    for path in sorted(
        APP.glob(
            "*.pdf"
        )
    ):

        print(
            " ",
            path
        )

    print()

    print(
        "notes:",
        OUT
        / "FINAL_FIGURE_NOTES.md",
    )