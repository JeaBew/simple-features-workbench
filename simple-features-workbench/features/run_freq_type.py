import pandas as pd
import os
from pathlib import Path
from statistics import mean
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Mapping, Optional, List
from py_lift.utils.core import load_cas_from_xmi_with_lift_ts
# Token type constant used to iterate over token annotations.
from py_lift.dkpro import T_TOKEN

from utils import process_folder as _original_process_folder  # noqa: F401
from constants import (
    TIGER_ORIG,
    ASGC_ORIG,
    ASGC_PLAIN,
    ASGC_EASY,
    G4A_ORIG,
    G4A_EASY,
    DEplain_ORIG,
    DEplain_PLAIN,
    Leiko_PLAIN,
    Leiko_EASY
)
from py_lift.annotators.frequency import SE_TokenZipfFrequency
from py_lift.extractors_specific import FE_FreqBandRatios

BAND_ORDER = ["f7", "f6", "f5", "f4", "f3", "f2", "f1", "oov"]

# Maps the short band keys returned by _local_process_folder to the long
# feature names expected by plot_stacked_bars and write_latex_table.
BAND_TO_FEATURE: dict[str, str] = {
    "f7": "Freq_Ratio_F7_PER_Token",
    "f6": "Freq_Ratio_F6_PER_Token",
    "f5": "Freq_Ratio_F5_PER_Token",
    "f4": "Freq_Ratio_F4_PER_Token",
    "f3": "Freq_Ratio_F3_PER_Token",
    "f2": "Freq_Ratio_F2_PER_Token",
    "f1": "Freq_Ratio_F1_PER_Token",
    "oov": "Freq_Ratio_OOV_PER_Token",
}

def _local_process_folder(corpus_dir: Path) -> dict[str, List[float]]:
    """Return a mapping ``band_name → list of proportions per file``.

    For each file in *corpus_dir* a **file‑local** token→band dictionary is built
    (recording each token only once).  The band counts are converted to a
    proportion of the unique tokens in that file.  These per‑file proportions are
    collected into a list for each band, preserving the order of files as they
    are processed.
    """
    # Initialise result container with empty lists for each known band.
    band_to_proportions: dict[str, List[float]] = {band: [] for band in BAND_ORDER}

    # Gather both XMI and plain‑text files.
    files = sorted(list(corpus_dir.glob("*.xmi")) + list(corpus_dir.glob("*.txt")))
    for file_path in tqdm(files, desc=f"Processing {corpus_dir}"):
        if file_path.suffix.lower() == ".xmi":
            cas = load_cas_from_xmi_with_lift_ts(file_path)
        else:
            # Plain text – run the shared preprocessing pipeline.
            from utils import prep  # lazy import to avoid circular deps
            text = file_path.read_text(encoding="utf-8")
            cas = prep.run(text)

        # Ensure frequency annotations are present.
        SE_TokenZipfFrequency("de").process(cas)

        token_to_band: dict[str, str] = {}
        for tok in cas.select(T_TOKEN):
            word = tok.get_covered_text()
            if not word:
                continue
            word = word.strip().lower()
            if not word:
                continue
            if word in token_to_band:
                continue
            for freq in cas.select_covered("org.lift.type.Frequency", tok):
                band = freq.get("frequencyBand")
                if band is None:
                    continue
                token_to_band[word] = str(band).strip().lower()
                break

        # Count bands for this file.
        band_counts: dict[str, int] = {}
        for band in token_to_band.values():
            band_counts[band] = band_counts.get(band, 0) + 1

        total_tokens = len(token_to_band)
        # Avoid division by zero – if a file yields no tokens, record 0.0 for all bands.
        if total_tokens == 0:
            for band in BAND_ORDER:
                band_to_proportions[band].append(0.0)
            continue

        for band in BAND_ORDER:
            count = band_counts.get(band, 0)
            band_to_proportions[band].append(count / total_tokens)

    return band_to_proportions

def plot_stacked_bars(avg_scores_by_corpus, output_path):
    # Preserve the original order of features (F7 → OOV) but reverse it for
    # plotting so that the highest frequency band appears on top of the stacked
    # bar. ``feature_order`` now starts with the lowest band (OOV) and ends with
    # the highest (F7).
    first_corpus_scores = next(iter(avg_scores_by_corpus.values()))
    feature_order = list(first_corpus_scores.keys())
    feature_order = list(reversed(feature_order))

    corpus_labels = list(avg_scores_by_corpus.keys())

    # ↓ Make overall chart less wide (reduce first number)
    fig, ax = plt.subplots(figsize=(6, 6))  # e.g. (6,6) instead of (8,6)

    colors = sns.color_palette("colorblind", len(feature_order))
    cumulative = {label: 0.0 for label in corpus_labels}

    for idx, feature in enumerate(feature_order):
        heights, bottoms = [], []
        for label in corpus_labels:
            avg = avg_scores_by_corpus[label].get(feature, np.nan)
            height = 0.0 if np.isnan(avg) else float(avg)
            heights.append(height)
            bottoms.append(cumulative[label])

        ax.bar(
            corpus_labels,
            heights,
            bottom=bottoms,
            label=feature,
            color=colors[idx % len(colors)],
            width=0.35,
        )

        for label, h in zip(corpus_labels, heights):
            cumulative[label] += h

    ax.set_title("Average Frequency Ratio Features by Corpus")
    # Hide y‑axis ticks and labels for a cleaner appearance.
    ax.set_yticks([])
    ax.set_yticklabels([])
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    fig.subplots_adjust(bottom=0.35)
    
    # ↓ Combine custom legend names with a legend placed outside the axes.
    # First retrieve the automatically generated handles (the colored patches).
    handles, _ = ax.get_legend_handles_labels()
    # Map the original feature names to the desired short names.
    feature_map = {
        "Freq_Ratio_F7_PER_Token": "F7",
        "Freq_Ratio_F6_PER_Token": "F6",
        "Freq_Ratio_F5_PER_Token": "F5",
        "Freq_Ratio_F4_PER_Token": "F4",
        "Freq_Ratio_F3_PER_Token": "F3",
        "Freq_Ratio_F2_PER_Token": "F2",
        "Freq_Ratio_F1_PER_Token": "F1",
        "Freq_Ratio_OOV_PER_Token": "OOV",
    }
    custom_labels = [feature_map.get(f, f) for f in feature_order]
    
    # The handles and custom_labels are already in the correct order because
    # we plotted the features in the reversed order above. No further inversion
    # is required.

    # Use the figure's legend method so we can position it outside the axes.
    fig.legend(
        handles,
        custom_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=4,  # display legend entries in three columns
        frameon=False,
    )
        
    # Reserve room at top for the legend
    fig.subplots_adjust(top=0.80)  # smaller => more space for legend

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    #plt.show()

def plot_stacked_bars_old(avg_scores_by_corpus, output_path):
    # Feature order: F7 → F1 → OOV
    first_corpus_scores = next(iter(avg_scores_by_corpus.values()))
    feature_order = list(first_corpus_scores.keys())

    corpus_labels = list(avg_scores_by_corpus.keys())

    # Horizontal plot: make figure taller rather than wider
    fig, ax = plt.subplots(figsize=(7, 5.5))

    colors = sns.color_palette("colorblind", len(feature_order))
    cumulative = {label: 0.0 for label in corpus_labels}

    for idx, feature in enumerate(feature_order):
        widths, lefts = [], []

        for label in corpus_labels:
            avg = avg_scores_by_corpus[label].get(feature, np.nan)
            width = 0.0 if np.isnan(avg) else float(avg)

            widths.append(width)
            lefts.append(cumulative[label])

        ax.barh(
            corpus_labels,
            widths,
            left=lefts,
            label=feature,
            color=colors[idx % len(colors)],
            height=0.55,
        )

        for label, w in zip(corpus_labels, widths):
            cumulative[label] += w

    ax.set_title("Average Frequency Ratio Features by Corpus")

    # Optional: since ratios sum to roughly 1
    ax.set_xlim(0, 1)

    # Keep corpus labels readable
    ax.set_ylabel("")
    ax.set_xlabel("")

    # Hide x-axis ticks/labels for a cleaner stacked-ratio plot
    ax.set_xticks([])
    ax.set_xticklabels([])

    # Put first corpus at the top
    ax.invert_yaxis()

    feature_map = {
        "Freq_Ratio_F7_PER_Token": "F7",
        "Freq_Ratio_F6_PER_Token": "F6",
        "Freq_Ratio_F5_PER_Token": "F5",
        "Freq_Ratio_F4_PER_Token": "F4",
        "Freq_Ratio_F3_PER_Token": "F3",
        "Freq_Ratio_F2_PER_Token": "F2",
        "Freq_Ratio_F1_PER_Token": "F1",
        "Freq_Ratio_OOV_PER_Token": "OOV",
    }

    handles, labels = ax.get_legend_handles_labels()
    custom_labels = [feature_map.get(f, f) for f in feature_order]

    fig.legend(
        handles,
        custom_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        frameon=False,
    )

    fig.subplots_adjust(top=0.82, left=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

def _latex_escape(text: str) -> str:
    """Escape a few characters that have special meaning in LaTeX tables.

    This is a minimal implementation sufficient for the corpus identifiers
    used in this project (they contain only alphanumerics and underscores).
    """
    return text.replace("_", "\\_")

def write_latex_table(avg_by_corpus: Mapping[str, Mapping[str, float]], output_path: Path) -> None:
    """Write a LaTeX tabular representation of ``avg_by_corpus``.

    The table rows correspond to corpora and the columns correspond to the
    frequency‑ratio bands ordered ``F7`` … ``F1`` then ``OOV``.  Values are
    formatted with three decimal places.
    """
    # Mapping from the long feature names used in ``avg_by_corpus`` to the short
    # column headings required for the LaTeX table.
    feature_map = {
        "Freq_Ratio_F7_PER_Token": "F7",
        "Freq_Ratio_F6_PER_Token": "F6",
        "Freq_Ratio_F5_PER_Token": "F5",
        "Freq_Ratio_F4_PER_Token": "F4",
        "Freq_Ratio_F3_PER_Token": "F3",
        "Freq_Ratio_F2_PER_Token": "F2",
        "Freq_Ratio_F1_PER_Token": "F1",
        "Freq_Ratio_OOV_PER_Token": "OOV",
    }
    # Desired column order.
    column_order = ["F7", "F6", "F5", "F4", "F3", "F2", "F1", "OOV"]

    # Build header line.
    header_cells = ["Corpus"] + column_order
    # Escape backslashes correctly for LaTeX commands.
    header = " & ".join(header_cells) + " \\\\ \\hline\n"

    # Build rows.
    rows = []
    for corpus, scores in avg_by_corpus.items():
        # Ensure deterministic order of rows (alphabetical by corpus label).
        escaped_corpus = _latex_escape(corpus)
        cell_values = []
        # Reverse lookup from short name to long feature name.
        reverse_map = {v: k for k, v in feature_map.items()}
        for col in column_order:
            long_name = reverse_map.get(col)
            val = scores.get(long_name, float("nan"))
            if isinstance(val, float) and not (val != val):  # not NaN
                # Round to two decimal places, strip trailing zeros, and drop leading zero.
                formatted = f"{val:.2f}".rstrip('0').rstrip('.')
                if formatted.startswith('0.'):
                    formatted = formatted[1:]  # remove leading zero
                cell_values.append(formatted)
            else:
                cell_values.append("-")
        row = f"{escaped_corpus} & " + " & ".join(cell_values) + " \\\\"
        rows.append(row)

    # Assemble full LaTeX table.
    table_body = "\n".join(rows)
    latex_content = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\begin{tabular}{l" + "r" * len(column_order) + "}\n"
        "\\toprule\n"
        + header +
        table_body + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Average frequency‑ratio features per corpus.}\n"
        "\\label{tab:freq_ratios}\n"
        "\\end{table}\n"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_content, encoding="utf-8")

def main() -> None:
    scores_tiger_orig = _local_process_folder(TIGER_ORIG)

    scores_asgc_orig = _local_process_folder(ASGC_ORIG)
    scores_asgc_plain = _local_process_folder(ASGC_PLAIN)
    scores_asgc_easy = _local_process_folder(ASGC_EASY)

    scores_g4a_orig = _local_process_folder(G4A_ORIG)
    scores_g4a_easy = _local_process_folder(G4A_EASY)

    scores_deplain_orig = _local_process_folder(DEplain_ORIG)
    scores_deplain_plain = _local_process_folder(DEplain_PLAIN)
    
    scores_leiko_plain = _local_process_folder(Leiko_PLAIN)
    scores_leiko_easy = _local_process_folder(Leiko_EASY)
    
    # ---------------------------------------------------------------------
    # Compute the average score for each feature across all documents.
    # ---------------------------------------------------------------------
    # Compute average scores per feature for each corpus.
    # scores_dict keys are short band names ("f7" … "oov"); map them to the
    # long feature names expected by plot_stacked_bars and write_latex_table.
    def compute_avg(scores_dict: dict[str, list[float]]) -> dict[str, float]:
        return {
            BAND_TO_FEATURE[band]: (mean(vals) if vals else float('nan'))
            for band, vals in scores_dict.items()
            if band in BAND_TO_FEATURE
        }

    avg_by_corpus: dict[str, dict[str, float]] = {
        "TIGER_ORIG": compute_avg(scores_tiger_orig),
        "ASGC_ORIG": compute_avg(scores_asgc_orig),
        "DEplain_ORIG": compute_avg(scores_deplain_orig),
        "G4A_ORIG": compute_avg(scores_g4a_orig),
        "ASGC_PLAIN": compute_avg(scores_asgc_plain),
        "DEplain_PLAIN": compute_avg(scores_deplain_plain),
        "Leiko_PLAIN": compute_avg(scores_leiko_plain),
        "ASGC_EASY": compute_avg(scores_asgc_easy),
        "G4A_EASY": compute_avg(scores_g4a_easy),
        "Leiko_EASY": compute_avg(scores_leiko_easy),
    }

    # Plot stacked bars for all corpora.
    plot_stacked_bars(avg_by_corpus, output_path=Path("../output") / "freq_stacked_types_horizontal.png")
    
    # Plot grouped bars (features on x‑axis) for all corpora.
    #plot_grouped_bars(avg_by_corpus, output_path=Path("output") / "freq_grouped.png")

    # ---------------------------------------------------------------------
    # Write LaTeX table summarizing the average frequency‑ratio features.
    # ---------------------------------------------------------------------
    write_latex_table(avg_by_corpus, output_path=Path("../output") / "freq_table_types.tex")

if __name__ == "__main__":
    main()
