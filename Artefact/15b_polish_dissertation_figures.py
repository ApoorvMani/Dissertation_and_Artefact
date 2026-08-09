"""
final visual polish for dissertation figures.

no model calls.
no attacks.
no defence reruns.
no threshold changes.

reads frozen results only.

outputs:
results/dissertation/polished/
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch


# --------------------------------------------------------------------
# paths
# --------------------------------------------------------------------

OUT = Path("results/dissertation/polished")
FIG = OUT / "figures"

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

FIG.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------------------------
# fixed ordering
# --------------------------------------------------------------------

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


STATUS_ORDER = [
    "NO BASE ATTACK",
    "NO CLEAR EFFECT",
    "PARTIAL",
    "COMPLETE",
    "HARMFUL",
]


# --------------------------------------------------------------------
# plot defaults
# --------------------------------------------------------------------

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


# --------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------

def find_file(*candidates):
    """return the first existing path."""

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


def save_figure(fig, filename):
    """save each figure as png and pdf."""

    fig.savefig(
        FIG / f"{filename}.png",
        dpi=300,
    )

    fig.savefig(
        FIG / f"{filename}.pdf",
    )

    plt.close(fig)


def as_percent(values):
    """
    convert fractions to percentages.

    0.96 -> 96
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


def choose_column(df, candidates):
    """find the first matching column."""

    for col in candidates:

        if col in df.columns:
            return col

    return None


def fuzzy_column(
    df,
    groups,
):
    """
    find a column containing one term
    from every supplied group.
    """

    for col in df.columns:

        clean = (
            col.lower()
            .replace("_", " ")
            .replace("-", " ")
        )

        matched = True

        for group in groups:

            if not any(
                term in clean
                for term in group
            ):
                matched = False
                break

        if matched:
            return col

    return None


def mechanism_label(row):
    """short mechanism label for coverage cells."""

    note = row.get(
        "mechanism_note",
        "",
    )

    if pd.isna(note):
        note = ""

    note = str(note).strip()

    if note:
        mapping = {
            "detect+block":
                "DETECT+BLOCK",

            "detect/repair":
                "DETECT+REPAIR",

            "reencode":
                "RE-ENCODE",

            "block":
                "BLOCK",
        }

        return mapping.get(
            note,
            note.upper(),
        )

    if (
        row.get(
            "coverage_status"
        )
        == "HARMFUL"
    ):
        return "DEFENCE-INDUCED"

    return ""


def annotate_matrix(
    ax,
    matrix,
    decimals=0,
):
    """write values into heatmap cells."""

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
                f"{value:.{decimals}f}",
                ha="center",
                va="center",
                fontsize=8,
            )


# --------------------------------------------------------------------
# load frozen data
# --------------------------------------------------------------------

def load_coverage():

    df = pd.read_csv(
        find_file(
            "results/final_current/coverage_matrix.csv",
            "results/dissertation/tables/"
            "table3_full_defence_outcomes.csv",
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
            "results/final_current/"
            "margin_stratified_asr.csv",
        )
    )

    df = df.copy()

    df["attack"] = (
        df["attack"]
        .map(normalise_attack)
    )

    return df


# --------------------------------------------------------------------
# utility loading
# --------------------------------------------------------------------

def load_complex_utility():
    """
    return two dataframes:

    operational preservation
    representation change

    tries summary first, then row-level results.
    """

    summary = pd.read_csv(
        find_file(
            "results/step11/"
            "complex_utility_summary.csv",
        )
    )

    rows = pd.read_csv(
        find_file(
            "results/step11/"
            "complex_utility_rows.csv",
        )
    )

    # ------------------------------------------------------------
    # shared identifiers
    # ------------------------------------------------------------

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

    if defence_col is None:
        defence_col = fuzzy_column(
            summary,
            [
                ["defence", "defense"],
            ],
        )

    if category_col is None:
        category_col = fuzzy_column(
            summary,
            [
                ["category", "type"],
            ],
        )

    if (
        defence_col is None
        or category_col is None
    ):
        raise RuntimeError(
            "could not locate defence/category "
            "columns in complex utility summary.\n"
            f"columns = {list(summary.columns)}"
        )

    # ------------------------------------------------------------
    # operational preservation
    # ------------------------------------------------------------

    preserve_col = choose_column(
        summary,
        [
            "operational_preservation",
            "operational_preservation_rate",
            "operational_preserve_rate",
            "operational_retention",
            "preservation_rate",
        ],
    )

    if preserve_col is None:

        preserve_col = fuzzy_column(
            summary,
            [
                ["operational"],
                [
                    "preserv",
                    "retain",
                    "retention",
                ],
            ],
        )

    # ------------------------------------------------------------
    # representation change
    # ------------------------------------------------------------

    rep_col = choose_column(
        summary,
        [
            "representation_change_rate",
            "representation_changed_rate",
            "representation_change",
            "representation_changed",
            "rep_change_rate",
            "repr_change_rate",
        ],
    )

    if rep_col is None:

        rep_col = fuzzy_column(
            summary,
            [
                [
                    "representation",
                    "repr",
                    "rep ",
                ],
                [
                    "change",
                    "changed",
                ],
            ],
        )

    # ------------------------------------------------------------
    # summary route
    # ------------------------------------------------------------

    if (
        preserve_col is not None
        and rep_col is not None
    ):

        preserve = summary[
            [
                defence_col,
                category_col,
                preserve_col,
            ]
        ].copy()

        preserve.columns = [
            "defence",
            "category",
            "value",
        ]

        representation = summary[
            [
                defence_col,
                category_col,
                rep_col,
            ]
        ].copy()

        representation.columns = [
            "defence",
            "category",
            "value",
        ]

    else:
        # --------------------------------------------------------
        # row-level fallback
        # --------------------------------------------------------

        row_defence = choose_column(
            rows,
            [
                "defence",
                "defense",
            ],
        )

        row_category = choose_column(
            rows,
            [
                "category",
                "input_category",
                "input_type",
                "type",
            ],
        )

        row_preserve = choose_column(
            rows,
            [
                "operational_preserved",
                "operational_preservation",
                "operational_keep",
                "preserved",
            ],
        )

        row_rep = choose_column(
            rows,
            [
                "representation_changed",
                "representation_change",
                "rep_changed",
                "repr_changed",
            ],
        )

        if row_preserve is None:

            row_preserve = fuzzy_column(
                rows,
                [
                    ["operational"],
                    [
                        "preserv",
                        "retain",
                        "keep",
                    ],
                ],
            )

        if row_rep is None:

            row_rep = fuzzy_column(
                rows,
                [
                    [
                        "representation",
                        "repr",
                        "rep ",
                    ],
                    [
                        "change",
                        "changed",
                    ],
                ],
            )

        missing = []

        if row_defence is None:
            missing.append(
                "defence"
            )

        if row_category is None:
            missing.append(
                "category"
            )

        if row_preserve is None:
            missing.append(
                "operational preservation"
            )

        if row_rep is None:
            missing.append(
                "representation change"
            )

        if missing:
            raise RuntimeError(
                "could not locate these complex "
                "utility metrics:\n"
                f"{missing}\n\n"
                "summary columns:\n"
                f"{list(summary.columns)}\n\n"
                "row columns:\n"
                f"{list(rows.columns)}"
            )

        temp = rows.copy()

        temp[row_preserve] = pd.to_numeric(
            temp[row_preserve],
            errors="coerce",
        )

        temp[row_rep] = pd.to_numeric(
            temp[row_rep],
            errors="coerce",
        )

        preserve = (
            temp.groupby(
                [
                    row_defence,
                    row_category,
                ],
                as_index=False,
            )[row_preserve]
            .mean()
        )

        preserve.columns = [
            "defence",
            "category",
            "value",
        ]

        representation = (
            temp.groupby(
                [
                    row_defence,
                    row_category,
                ],
                as_index=False,
            )[row_rep]
            .mean()
        )

        representation.columns = [
            "defence",
            "category",
            "value",
        ]

    for df in [
        preserve,
        representation,
    ]:

        df["defence"] = (
            df["defence"]
            .map(normalise_defence)
        )

        df["value"] = (
            as_percent(
                df["value"]
            )
        )

    return (
        preserve,
        representation,
    )


# --------------------------------------------------------------------
# category cleanup
# --------------------------------------------------------------------

def clean_category(value):

    x = (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
    )

    mapping = {
        "code":
            "Code",

        "url":
            "URLs",

        "urls":
            "URLs",

        "emoji":
            "Emoji",

        "misspelling":
            "Misspelling",

        "misspellings":
            "Misspelling",

        "non english":
            "Non-English",

        "non-english":
            "Non-English",

        "mixed script":
            "Mixed-script",

        "mixed-script":
            "Mixed-script",
    }

    return mapping.get(
        x,
        str(value),
    )


UTILITY_CATEGORY_ORDER = [
    "Code",
    "URLs",
    "Emoji",
    "Misspelling",
    "Non-English",
    "Mixed-script",
]


# --------------------------------------------------------------------
# figure 1
# --------------------------------------------------------------------

def figure_1_design():
    """
    cleaner experimental-design diagram.

    supporting cross-model baseline is visually
    separated from the primary pipeline.
    """

    fig, ax = plt.subplots(
        figsize=(13, 8)
    )

    ax.set_xlim(
        0,
        13,
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
        fontsize=10,
    ):

        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03",
            linewidth=1.4,
            linestyle=(
                "--"
                if dashed
                else "-"
            ),
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
                    1.3,

                "linestyle":
                    "--"
                    if dashed
                    else "-",
            },
        )

    # primary pipeline
    box(
        4.0,
        8.7,
        3.4,
        0.8,
        "Jigsaw source data\n"
        "250 toxic + 250 benign",
    )

    box(
        4.0,
        7.25,
        3.4,
        0.8,
        "WordPiece victim\n"
        "156/250 toxic eligible",
    )

    arrow(
        5.7,
        8.7,
        5.7,
        8.05,
    )

    box(
        0.5,
        5.5,
        2.5,
        1.0,
        "TokenBreak\n156 cases",
    )

    box(
        4.45,
        5.5,
        2.5,
        1.0,
        "AdvTok\n156 cases",
    )

    box(
        8.4,
        5.5,
        2.5,
        1.0,
        "Unicode\n4 × 156 cases",
    )

    arrow(
        4.8,
        7.25,
        1.75,
        6.5,
    )

    arrow(
        5.7,
        7.25,
        5.7,
        6.5,
    )

    arrow(
        6.6,
        7.25,
        9.65,
        6.5,
    )

    box(
        4.0,
        3.95,
        3.4,
        0.85,
        "936 attack instances",
    )

    arrow(
        1.75,
        5.5,
        4.8,
        4.8,
    )

    arrow(
        5.7,
        5.5,
        5.7,
        4.8,
    )

    arrow(
        9.65,
        5.5,
        6.6,
        4.8,
    )

    box(
        4.0,
        2.45,
        3.4,
        0.9,
        "7 defences\n"
        "6,552 paired evaluations",
    )

    arrow(
        5.7,
        3.95,
        5.7,
        3.35,
    )

    box(
        0.3,
        0.45,
        3.0,
        1.0,
        "Security\n"
        "ASR + paired CIs",
    )

    box(
        4.2,
        0.45,
        3.0,
        1.0,
        "Mechanism\n"
        "Detect / block / re-encode",
    )

    box(
        8.1,
        0.45,
        3.0,
        1.0,
        "Utility\n"
        "Clean + complex inputs",
    )

    arrow(
        4.8,
        2.45,
        1.8,
        1.45,
    )

    arrow(
        5.7,
        2.45,
        5.7,
        1.45,
    )

    arrow(
        6.6,
        2.45,
        9.6,
        1.45,
    )

    # supporting side panel
    ax.plot(
        [
            11.55,
            11.55,
        ],
        [
            4.8,
            9.5,
        ],
        linestyle="--",
        linewidth=1,
    )

    ax.text(
        12.1,
        9.25,
        "SUPPORTING",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
    )

    box(
        11.7,
        7.1,
        1.2,
        1.75,
        "Clean eligibility\n\n"
        "WP 156\n"
        "BPE 214\n"
        "Unigram 242\n"
        "Shared 155",
        dashed=True,
        fontsize=8,
    )

    arrow(
        7.4,
        9.1,
        11.7,
        8.1,
        dashed=True,
    )

    ax.text(
        12.3,
        6.55,
        "Different model–tokenizer\n"
        "setups; clean baseline only",
        ha="center",
        va="center",
        fontsize=8,
    )

    ax.text(
        12.3,
        5.85,
        "No BPE/Unigram\n"
        "attacks executed",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
    )

    ax.set_title(
        "Experimental design",
        pad=15,
    )

    save_figure(
        fig,
        "fig01_experimental_design_polished",
    )


# --------------------------------------------------------------------
# figure 3
# --------------------------------------------------------------------

def figure_3_coverage(
    coverage,
):
    """
    headline attack x defence matrix.

    adds the mechanism where available so that
    all zero-asr cells are not presented equally.
    """

    status_code = {
        status: index
        for index, status
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

    text = [
        [
            ""
            for _ in ATTACK_ORDER
        ]
        for _ in DEFENCE_ORDER
    ]

    for _, row in coverage.iterrows():

        attack = row["attack"]
        defence = row["defence"]

        if (
            attack not in ATTACK_ORDER
            or defence not in DEFENCE_ORDER
        ):
            continue

        x = ATTACK_ORDER.index(
            attack
        )

        y = DEFENCE_ORDER.index(
            defence
        )

        status = row[
            "coverage_status"
        ]

        matrix[
            y,
            x,
        ] = status_code[
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

        mechanism = (
            mechanism_label(
                row
            )
        )

        cell = (
            f"{status}\n"
            f"ASR {defended:.1f}%"
        )

        if mechanism:
            cell += (
                f"\n{mechanism}"
            )

        text[
            y
        ][
            x
        ] = cell

    # discrete status palette
    cmap = ListedColormap(
        [
            "#d9d9d9",
            "#bdbdbd",
            "#80b1d3",
            "#8dd3c7",
            "#fb8072",
        ]
    )

    fig, ax = plt.subplots(
        figsize=(15.2, 8.2)
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        vmin=-0.5,
        vmax=4.5,
    )

    ax.set_xticks(
        range(
            len(ATTACK_ORDER)
        )
    )

    ax.set_xticklabels(
        [
            ATTACK_SHORT[x]
            for x
            in ATTACK_ORDER
        ],
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

    ax.set_xlabel(
        "Attack"
    )

    ax.set_ylabel(
        "Defence"
    )

    ax.set_title(
        "Attack × defence coverage and mechanism"
    )

    for y in range(
        len(DEFENCE_ORDER)
    ):

        for x in range(
            len(ATTACK_ORDER)
        ):

            if text[y][x]:

                ax.text(
                    x,
                    y,
                    text[y][x],
                    ha="center",
                    va="center",
                    fontsize=7.2,
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

    ax.text(
        0,
        -0.18,
        "Cell percentage = defended ASR. "
        "Mechanism labels distinguish detection/blocking "
        "from repair or interface re-encoding.",
        transform=ax.transAxes,
        fontsize=8.5,
    )

    save_figure(
        fig,
        "fig03_attack_defence_coverage_polished",
    )


# --------------------------------------------------------------------
# figure 4
# --------------------------------------------------------------------

def figure_4_advtok_mechanisms(
    coverage,
):
    """
    advtok outcome + mechanism.

    left:
        actual defended asr.

    right:
        how each defence acted.

    this avoids treating 100 - defended_asr
    as a paired suppression statistic.
    """

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

    defended = as_percent(
        df[
            "paired_defended_asr"
        ]
    ).to_numpy(
        dtype=float
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

    mechanism_matrix = np.column_stack(
        [
            detection,
            blocking,
            reencode,
        ]
    )

    y = np.arange(
        len(df)
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 6.8),
        gridspec_kw={
            "width_ratios": [
                1.1,
                1.8,
            ]
        },
        constrained_layout=True,
    )

    # --------------------------------------------------------
    # panel a: defended outcome
    # --------------------------------------------------------

    ax = axes[0]

    bars = ax.barh(
        y,
        defended,
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        df["defence"]
    )

    ax.invert_yaxis()

    ax.set_xlim(
        0,
        100,
    )

    ax.set_xlabel(
        "Defended ASR (%)"
    )

    ax.set_title(
        "A. Outcome"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    for bar, value in zip(
        bars,
        defended,
    ):

        ax.text(
            min(
                value + 1.5,
                96,
            ),
            bar.get_y()
            + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=8.5,
        )

    # --------------------------------------------------------
    # panel b: defence mechanism
    # --------------------------------------------------------

    ax = axes[1]

    image = ax.imshow(
        mechanism_matrix,
        aspect="auto",
        vmin=0,
        vmax=100,
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
            "Detection",
            "Blocking",
            "Re-encoding",
        ]
    )

    ax.set_yticks(
        y
    )

    # left panel already gives defence names
    ax.set_yticklabels(
        []
    )

    ax.set_title(
        "B. Mechanism"
    )

    annotate_matrix(
        ax,
        mechanism_matrix,
        decimals=0,
    )

    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.05,
        pad=0.03,
    )

    cbar.set_label(
        "Rate (%)"
    )

    fig.suptitle(
        "AdvTok: identical outcomes can arise through different mechanisms",
        fontsize=14,
    )

    fig.text(
        0.5,
        -0.02,
        "Defended ASR reports the outcome. "
        "Detection, blocking and re-encoding report "
        "how the defence acted.",
        ha="center",
        fontsize=8.5,
    )

    save_figure(
        fig,
        "fig04_advtok_mechanism_decomposition",
    )


# --------------------------------------------------------------------
# figure 5
# --------------------------------------------------------------------

def figure_5_tokenbreak(
    coverage,
):
    """cleaner TokenBreak forest plot."""

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

    # strongest defence first
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

    lower = np.maximum(
        df["reduction"].to_numpy()
        - df["ci_low"].to_numpy(),
        0,
    )

    upper = np.maximum(
        df["ci_high"].to_numpy()
        - df["reduction"].to_numpy(),
        0,
    )

    fig, ax = plt.subplots(
        figsize=(11, 6.3)
    )

    ax.errorbar(
        df["reduction"],
        y,
        xerr=np.vstack(
            [
                lower,
                upper,
            ]
        ),
        fmt="o",
        capsize=4,
        markersize=6,
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
        df["defence"]
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Paired reduction in attack success rate "
        "(percentage points)"
    )

    ax.set_title(
        "TokenBreak remains incompletely mitigated"
    )

    max_ci = float(
        df["ci_high"].max()
    )

    ax.set_xlim(
        min(
            -4,
            float(
                df["ci_low"].min()
            )
            - 3,
        ),
        max_ci + 28,
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    for i, row in df.iterrows():

        ax.text(
            row["ci_high"] + 1.2,
            i,
            (
                f"Δ {row['reduction']:.1f} pp  "
                f"[{row['ci_low']:.1f}, "
                f"{row['ci_high']:.1f}]  |  "
                f"defended ASR "
                f"{row['defended']:.1f}%"
            ),
            va="center",
            fontsize=8.3,
        )

    save_figure(
        fig,
        "fig05_tokenbreak_effectiveness_polished",
    )


# --------------------------------------------------------------------
# figure 6
# --------------------------------------------------------------------

def utility_matrix(
    df,
):
    """convert utility dataframe to ordered matrix."""

    df = df.copy()

    df["category"] = (
        df["category"]
        .map(clean_category)
    )

    pivot = df.pivot_table(
        index="defence",
        columns="category",
        values="value",
        aggfunc="mean",
    )

    pivot = pivot.reindex(
        DEFENCE_ORDER
    )

    pivot = pivot.reindex(
        columns=UTILITY_CATEGORY_ORDER
    )

    return pivot


def figure_6_utility(
    preserve,
    representation,
):
    """
    two-panel utility figure.

    A: classifier/operational preservation
    B: representation changed
    """

    keep = utility_matrix(
        preserve
    )

    changed = utility_matrix(
        representation
    )

    keep.to_csv(
        OUT
        / "utility_operational_preservation.csv"
    )

    changed.to_csv(
        OUT
        / "utility_representation_change.csv"
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6.8),
        constrained_layout=True,
    )

    matrices = [
        keep.to_numpy(
            dtype=float
        ),
        changed.to_numpy(
            dtype=float
        ),
    ]

    titles = [
        "A. Operational preservation",
        "B. Legitimate inputs whose representation changed",
    ]

    colorbar_labels = [
        "Operational preservation (%)",
        "Representation changed (%)",
    ]

    for index, (
        ax,
        matrix,
        title,
        bar_label,
    ) in enumerate(
        zip(
            axes,
            matrices,
            titles,
            colorbar_labels,
        )
    ):

        image = ax.imshow(
            matrix,
            aspect="auto",
            vmin=0,
            vmax=100,
        )

        ax.set_xticks(
            range(
                len(
                    UTILITY_CATEGORY_ORDER
                )
            )
        )

        ax.set_xticklabels(
            UTILITY_CATEGORY_ORDER,
            rotation=22,
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

        ax.set_title(
            title
        )

        annotate_matrix(
            ax,
            matrix,
            decimals=0,
        )

        cbar = fig.colorbar(
            image,
            ax=ax,
            fraction=0.045,
            pad=0.025,
        )

        cbar.set_label(
            bar_label
        )

    fig.suptitle(
        "Utility costs on complex legitimate inputs",
        fontsize=14,
    )

    fig.text(
        0.5,
        -0.015,
        "Decision preservation and input preservation are "
        "reported separately: a defence may retain the "
        "classifier decision while still altering legitimate text.",
        ha="center",
        fontsize=8.5,
    )

    save_figure(
        fig,
        "fig06_complex_utility_two_panel",
    )


# --------------------------------------------------------------------
# figure 7
# --------------------------------------------------------------------

def figure_7_margin(
    margin,
):
    """confidence-stratified attack success with n."""

    fig, ax = plt.subplots(
        figsize=(12.2, 6.7)
    )

    x = np.arange(
        len(
            ATTACK_ORDER
        ),
        dtype=float,
    )

    width = 0.24

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
            y - lo,
            0,
        )

        upper = np.maximum(
            hi - y,
            0,
        )

        n_values = (
            part["n"]
            .dropna()
            .astype(int)
            .unique()
        )

        if len(n_values) == 1:
            legend_label = (
                f"{label}  "
                f"(n={n_values[0]})"
            )

        else:
            legend_label = label

        ax.bar(
            x
            + (
                index - 1
            )
            * width,
            y,
            width=width,
            label=legend_label,
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
            ATTACK_SHORT[x]
            for x
            in ATTACK_ORDER
        ],
        rotation=18,
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
        "Attack success persists on high-confidence clean inputs"
    )

    ax.legend(
        title="Clean P(toxic)",
        loc="lower left",
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.text(
        0,
        -0.18,
        "Error bars are Wilson 95% confidence intervals.",
        transform=ax.transAxes,
        fontsize=8.5,
    )

    save_figure(
        fig,
        "fig07_margin_stratified_asr_polished",
    )


# --------------------------------------------------------------------
# figure 8
# --------------------------------------------------------------------

def figure_8_harmful(
    coverage,
):
    """
    defence-induced invisible unicode regression
    with paired difference and ci.
    """

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
            "Invisible × Tokenizer translation"
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

    red_lo = float(
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

    red_hi = float(
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

    # reduction = undefended - defended.
    # for harmful result, report increase instead.
    increase = -reduction

    increase_lo = -red_hi
    increase_hi = -red_lo

    n = int(
        row[
            "n_clean_retained"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(8.5, 5.7)
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
        marker="o",
        linewidth=2,
        markersize=8,
    )

    ax.set_xlim(
        -0.25,
        1.25,
    )

    ax.set_ylim(
        0,
        max(
            21,
            defended + 7,
        ),
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

    ax.set_title(
        "Defensive preprocessing introduced a new failure mode"
    )

    ax.text(
        0,
        undefended + 0.8,
        f"{undefended:.1f}%",
        ha="center",
        fontsize=11,
    )

    ax.text(
        1,
        defended + 0.8,
        f"{defended:.1f}%",
        ha="center",
        fontsize=11,
    )

    ax.text(
        0.5,
        defended / 2 + 3.0,
        (
            f"ΔASR = +{increase:.1f} pp\n"
            f"95% CI "
            f"[+{increase_lo:.1f}, "
            f"+{increase_hi:.1f}] pp\n"
            f"paired n = {n}"
        ),
        ha="center",
        va="center",
        fontsize=10,
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    save_figure(
        fig,
        "fig08_defence_induced_regression_polished",
    )


# --------------------------------------------------------------------
# figure 9
# --------------------------------------------------------------------

def figure_9_model_tokenizer():

    df = pd.read_csv(
        find_file(
            "data/"
            "cross_tokenizer_eligibility.csv",
        )
    )

    columns = [
        "eligible_wordpiece",
        "eligible_bpe",
        "eligible_unigram",
        "eligible_all_three",
    ]

    for column in columns:

        if column not in df.columns:

            raise RuntimeError(
                f"missing column: {column}"
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

    percentages = [
        100 * x / total
        for x in counts
    ]

    labels = [
        "DistilBERT\nWordPiece",
        "RoBERTa\nBPE",
        "DeBERTa-v3\nUnigram",
        "Shared\nintersection",
    ]

    fig, ax = plt.subplots(
        figsize=(9, 5.8)
    )

    bars = ax.bar(
        labels,
        percentages,
    )

    ax.set_ylim(
        0,
        108,
    )

    ax.set_ylabel(
        "Clean toxic examples eligible (%)"
    )

    ax.set_title(
        "Clean eligibility across model–tokenizer setups"
    )

    for bar, count, pct in zip(
        bars,
        counts,
        percentages,
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            pct + 1.2,
            (
                f"{count}/{total}\n"
                f"{pct:.1f}%"
            ),
            ha="center",
            va="bottom",
            fontsize=9.5,
        )

    ax.text(
        0.5,
        -0.18,
        "Supporting clean-baseline comparison only. "
        "No BPE or Unigram attack-generalisation experiment "
        "was executed.",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.5,
        fontweight="bold",
    )

    ax.text(
        0.5,
        -0.25,
        "Differences cannot be attributed to tokenizer family "
        "alone because the classifier architectures also differ.",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.2,
    )

    save_figure(
        fig,
        "fig09_model_tokenizer_clean_eligibility",
    )


# --------------------------------------------------------------------
# captions
# --------------------------------------------------------------------

def write_captions():

    text = """# Polished dissertation figure captions

## Figure 1 — Experimental design

Experimental design of the primary WordPiece study. Of 250 toxic Jigsaw
examples, 156 were correctly classified by the victim and formed the common
eligible attack set. Six attack variants generated 936 attack instances, which
were evaluated under seven defences in 6,552 paired evaluations. Security
effectiveness, defence mechanism, and legitimate-input utility were evaluated
separately. The side branch is a supporting clean-baseline comparison across
different model–tokenizer setups; no BPE or Unigram attacks were executed.

## Figure 3 — Attack × defence coverage and mechanism

Coverage of seven defences across six attack variants. Each cell reports
defended attack success rate together with a qualitative coverage classification.
Where relevant, the cell additionally identifies whether suppression resulted
from explicit detection/blocking, repair, or re-encoding. "Complete" therefore
means complete suppression in this evaluated setting and does not imply that
all complete cells operate through equivalent mechanisms.

## Figure 4 — AdvTok mechanism decomposition

AdvTok suppression and defence mechanism. Several defences reduce defended
AdvTok ASR to zero, but they do so through different mechanisms. Canonical
rejection explicitly detects and blocks non-canonical token IDs, whereas
tokenizer translation and several string transformations suppress the attack
through re-encoding. Outcome-level robustness should therefore not be
interpreted automatically as attack detection.

## Figure 5 — TokenBreak residual coverage gap

Paired reduction in TokenBreak attack-success rate with 95% confidence
intervals. Tokenizer translation produces the largest measured reduction, but
46.5% defended ASR remains. None of the evaluated defences completely suppresses
TokenBreak in this setting.

## Figure 6 — Utility costs

Utility of the seven defences on six complex legitimate-input categories.
Panel A reports operational preservation, while Panel B reports the proportion
of legitimate inputs whose representation was changed. Reporting both
distinguishes retained classifier behaviour from preservation of the original
legitimate input representation.

## Figure 7 — Confidence-stratified attack success

Attack success stratified by the clean victim toxic-class probability, with
Wilson 95% confidence intervals. The highest-confidence bin contains 128 of
the 156 eligible examples. High ASR persists in this bin for TokenBreak,
AdvTok, homoglyph, compatibility and reorder attacks, showing that the headline
attack rates are not explained only by borderline clean classifications.

## Figure 8 — Defence-induced regression

Tokenizer translation increases invisible-Unicode ASR from 0% to 13.4% among
the clean-retained paired evaluation set. The paired difference and confidence
interval demonstrate that defensive preprocessing can introduce a failure mode
that was absent in the undefended victim.

## Figure 9 — Model–tokenizer clean eligibility

Clean toxic-example eligibility across the three evaluated model–tokenizer
setups and their shared intersection. The comparison is supporting baseline
evidence only. Because model architecture and tokenizer family change together,
differences in eligibility cannot be attributed to tokenizer family alone, and
no cross-family attack-generalisation result is claimed.
"""

    (
        OUT
        / "POLISHED_CAPTIONS.md"
    ).write_text(
        text,
        encoding="utf-8",
    )


# --------------------------------------------------------------------
# run
# --------------------------------------------------------------------

if __name__ == "__main__":

    print(
        "polishing dissertation figures"
    )

    print(
        "python :",
        sys.executable,
    )

    print(
        "note   : frozen result artefacts only"
    )

    print(
        "note   : no model calls or experiments"
    )

    print()

    coverage = load_coverage()

    margin = load_margin()

    (
        preservation,
        representation,
    ) = load_complex_utility()

    print(
        "loaded:"
    )

    print(
        " coverage rows       :",
        len(coverage),
    )

    print(
        " margin rows         :",
        len(margin),
    )

    print(
        " utility preservation:",
        len(preservation),
    )

    print(
        " representation rows :",
        len(representation),
    )

    print()

    print(
        "building figure 1..."
    )

    figure_1_design()

    print(
        "building figure 3..."
    )

    figure_3_coverage(
        coverage
    )

    print(
        "building figure 4..."
    )

    figure_4_advtok_mechanisms(
        coverage
    )

    print(
        "building figure 5..."
    )

    figure_5_tokenbreak(
        coverage
    )

    print(
        "building figure 6..."
    )

    figure_6_utility(
        preservation,
        representation,
    )

    print(
        "building figure 7..."
    )

    figure_7_margin(
        margin
    )

    print(
        "building figure 8..."
    )

    figure_8_harmful(
        coverage
    )

    print(
        "building figure 9..."
    )

    figure_9_model_tokenizer()

    write_captions()

    print()

    print(
        "polished figure build complete"
    )

    print()

    print(
        "output:",
        OUT,
    )

    print()

    print(
        "generated:"
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
        "captions:",
        OUT
        / "POLISHED_CAPTIONS.md",
    )