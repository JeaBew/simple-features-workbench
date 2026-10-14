import pandas as pd
import os
from pathlib import Path
from statistics import mean
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Mapping, Optional
from py_lift.utils.core import load_cas_from_xmi_with_lift_ts
from utils import process_folder
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

def plot_stacked_bars(avg_scores_by_corpus, output_path):
    first_corpus_scores = next(iter(avg_scores_by_corpus.values()))

    # Deine feature_names sind aktuell OOV → F7.
    # Für die horizontale Darstellung drehen wir das um: F7 → OOV.
    feature_order = list(first_corpus_scores.keys())[::-1]

    corpus_labels = list(avg_scores_by_corpus.keys())

    # Horizontal: eher breiter als hoch
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

    # Bei Ratio-Werten sinnvoll
    ax.set_xlim(0, 1)

    # Achsen optisch reduzieren
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_xticklabels([])

    # Erstes Korpus oben anzeigen
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

    handles, _ = ax.get_legend_handles_labels()
    custom_labels = [feature_map.get(f, f) for f in feature_order]

    fig.legend(
        handles,
        custom_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        frameon=False,
    )

    # Platz für Korpuslabels links und Legende oben
    fig.subplots_adjust(top=0.82, left=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

def plot_grouped_bars(avg_scores_by_corpus, output_path):
    """Create a grouped bar chart where *features* are on the x‑axis.

    The x‑labels are the frequency‑ratio features (``F7`` … ``F1`` and ``OOV``).
    For each feature a set of bars – one per corpus – shows the average value
    for that corpus.  This is the inverse of the previous grouping where the
    corpora were on the x‑axis.
    """
    # Determine ordering based on the first corpus (consistent with the stacked chart).
    first_corpus_scores = next(iter(avg_scores_by_corpus.values()))
    feature_order = list(first_corpus_scores.keys())
    corpus_labels = list(avg_scores_by_corpus.keys())

    # Figure size similar to the stacked version.
    fig, ax = plt.subplots(figsize=(6, 6))

    colors = sns.color_palette("colorblind", len(corpus_labels))
    n_corpora = len(corpus_labels)
    total_width = 0.8
    bar_width = total_width / n_corpora
    # Base positions for each feature group.
    indices = np.arange(len(feature_order))

    for idx, corpus in enumerate(corpus_labels):
        heights = []
        for feature in feature_order:
            avg = avg_scores_by_corpus[corpus].get(feature, np.nan)
            height = 0.0 if np.isnan(avg) else float(avg)
            heights.append(height)
        # Compute offset so bars are centered on the tick.
        offset = (idx - n_corpora / 2) * bar_width + bar_width / 2
        ax.bar(
            indices + offset,
            heights,
            width=bar_width,
            label=corpus,
            color=colors[idx % len(colors)],
        )

    ax.set_title("Average Frequency Ratio Features by Corpus (Grouped)")
    # Map feature keys to short names for the x‑axis.
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
    short_features = [feature_map.get(f, f) for f in feature_order]
    ax.set_xticks(indices)
    ax.set_xticklabels(short_features, rotation=0, ha="center")
    ax.set_ylabel("Average Ratio")
    # Hide y‑axis ticks for a cleaner look, similar to stacked version.
    ax.set_yticks([])
    ax.set_yticklabels([])

    # Legend now shows corpus names.
    handles, _ = ax.get_legend_handles_labels()
    # Invert order so legend matches the bar order (optional, keep consistent with stacked).
    handles = handles[::-1]
    corpus_labels_rev = corpus_labels[::-1]
    fig.legend(
        handles,
        corpus_labels_rev,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=4,
        frameon=False,
    )
    fig.subplots_adjust(top=0.80, bottom=0.35)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    #plt.show()

def plot_token_proportions(token_counts_by_corpus: Mapping[str, pd.DataFrame], output_path: Path) -> None:
    """Plot stacked bars of token‑band **proportions** for each corpus.

    ``token_counts_by_corpus`` maps a corpus label to a ``DataFrame`` produced by
    :func:`get_freq_token`.  The DataFrame contains a ``band`` column with the
    frequency‑band label for each token (already categorical with the order
    defined in ``BAND_ORDER``).  This function aggregates the token counts per
    band, converts them to proportions of the total token count for the
    corpus, and visualises them as a stacked bar chart – analogous to the
    ``plot_stacked_bars`` function but based on *raw token frequencies* rather
    than the averaged extractor ratios.
    """
    # Determine the ordering of bands (consistent with the extractor).
    band_order = list(BAND_ORDER)

    corpus_labels = list(token_counts_by_corpus.keys())
    # Figure size similar to other plots.
    fig, ax = plt.subplots(figsize=(6, 6))

    colors = sns.color_palette("colorblind", len(band_order))
    cumulative = {label: 0.0 for label in corpus_labels}

    for idx, band in enumerate(band_order):
        heights, bottoms = [], []
        for label in corpus_labels:
            df = token_counts_by_corpus[label]
            total_tokens = len(df)
            if total_tokens == 0:
                proportion = 0.0
            else:
                count = (df["band"] == band).sum()
                proportion = count / total_tokens
            heights.append(proportion)
            bottoms.append(cumulative[label])
        ax.bar(
            corpus_labels,
            heights,
            bottom=bottoms,
            label=band,
            color=colors[idx % len(colors)],
            width=0.35,
        )
        for label, h in zip(corpus_labels, heights):
            cumulative[label] += h

    ax.set_title("Token‑Band Proportions by Corpus")
    ax.set_yticks([])
    ax.set_yticklabels([])
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

    # Legend with short band names.
    handles, _ = ax.get_legend_handles_labels()
    # Reverse order to match stacked appearance.
    handles = handles[::-1]
    band_labels = [band.upper() for band in band_order][::-1]
    fig.legend(
        handles,
        band_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=4,
        frameon=False,
    )
    fig.subplots_adjust(top=0.80, bottom=0.35)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    #plt.show()

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

def get_freq_token(
    corpus_dir: Path,
    out_csv: Optional[Path] = None,
    lang: str = "de",
) -> pd.DataFrame:
    """Aggregate token frequencies for a corpus.

    The original implementation only processed ``*.xmi`` files, which meant that
    corpora containing plain ``.txt`` files produced empty CSVs.  This function now
    handles both ``.xmi`` and ``.txt`` inputs:

    * ``.xmi`` – loaded directly via :func:`load_cas_from_xmi_with_lift_ts`.
    * ``.txt`` – read, tokenised with the shared ``prep`` pipeline from
      :pymod:`utils`, and then annotated with ``SE_TokenZipfFrequency``.

    The rest of the logic (majority‑vote band selection and DataFrame creation)
    remains unchanged.
    """

    # Import the preprocessing pipeline used elsewhere in the project.
    from utils import prep  # type: ignore  # noqa: F401

    # word -> total token count
    word_counts: dict[str, int] = {}
    # word -> band -> count (to decide the majority band)
    word_band_votes: dict[str, dict[str, int]] = {}

    # Gather both XMI and plain‑text files.
    files = sorted(list(corpus_dir.glob("*.xmi")) + list(corpus_dir.glob("*.txt")))
    for file_path in tqdm(files, desc=f"Processing {corpus_dir}"):
        # Load or create a CAS depending on the file type.
        if file_path.suffix.lower() == ".xmi":
            cas = load_cas_from_xmi_with_lift_ts(file_path)
        else:
            # Plain text – run the preprocessing pipeline to obtain a CAS.
            text = file_path.read_text(encoding="utf-8")
            cas = prep.run(text)

        # Ensure frequency annotations are present.
        SE_TokenZipfFrequency(lang).process(cas)

        # Build a map from token span to its frequency band.
        freq_by_span: dict[tuple[int, int], str] = {}
        for freq in cas.select("org.lift.type.Frequency"):
            band = freq.get("frequencyBand")
            if band is None:
                continue
            freq_by_span[(int(freq.begin), int(freq.end))] = str(band).strip().lower()

        for tok in cas.select("Token"):
            word = tok.get_covered_text()
            if not word:
                continue
            word = word.strip().lower()
            if not word:
                continue

            span = (int(tok.begin), int(tok.end))
            band = freq_by_span.get(span, "oov")

            word_counts[word] = word_counts.get(word, 0) + 1
            word_band_votes.setdefault(word, {})
            word_band_votes[word][band] = word_band_votes[word].get(band, 0) + 1

    # Build the result table.
    rows = []
    for word, cnt in word_counts.items():
        votes = word_band_votes.get(word, {})
        # Choose the band with the most votes; break ties using ``BAND_ORDER``.
        def rank(b: str) -> int:
            return BAND_ORDER.index(b) if b in BAND_ORDER else 999

        best_band = (
            sorted(votes.items(), key=lambda kv: (-kv[1], rank(kv[0])))[0][0]
            if votes
            else "oov"
        )
        rows.append({"word": word, "count": cnt, "band": best_band})

    df = pd.DataFrame(rows)
    # Guard against an empty DataFrame lacking the ``band`` column.
    if "band" not in df.columns:
        df = pd.DataFrame(columns=["word", "count", "band"])
    df["band"] = pd.Categorical(df["band"], categories=BAND_ORDER, ordered=True)
    if not df.empty:
        df = df.sort_values(by=["band", "count", "word"], ascending=[True, False, True]).reset_index(drop=True)

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

    return df


def main() -> None:
    extractor = lambda cas: (
        SE_TokenZipfFrequency("de").process(cas),
        FE_FreqBandRatios().extract(cas),
    )
    feature_names = [
            'Freq_Ratio_OOV_PER_Token',
            'Freq_Ratio_F1_PER_Token',
            'Freq_Ratio_F2_PER_Token',
            'Freq_Ratio_F3_PER_Token',
            'Freq_Ratio_F4_PER_Token',
            'Freq_Ratio_F5_PER_Token',
            'Freq_Ratio_F6_PER_Token',
            'Freq_Ratio_F7_PER_Token'
    ]

    scores_tiger_orig = process_folder(TIGER_ORIG, feature_extractor=extractor, feature_names=feature_names)

    scores_asgc_orig = process_folder(ASGC_ORIG, feature_extractor=extractor, feature_names=feature_names)
    scores_asgc_plain = process_folder(ASGC_PLAIN, feature_extractor=extractor, feature_names=feature_names)
    scores_asgc_easy = process_folder(ASGC_EASY, feature_extractor=extractor, feature_names=feature_names)

    scores_g4a_orig = process_folder(G4A_ORIG, feature_extractor=extractor, feature_names=feature_names)
    scores_g4a_easy = process_folder(G4A_EASY, feature_extractor=extractor, feature_names=feature_names)

    scores_deplain_orig = process_folder(DEplain_ORIG, feature_extractor=extractor, feature_names=feature_names)
    scores_deplain_plain = process_folder(DEplain_PLAIN, feature_extractor=extractor, feature_names=feature_names)

    scores_leiko_plain = process_folder(Leiko_PLAIN, feature_extractor=extractor, feature_names=feature_names)
    scores_leiko_easy = process_folder(Leiko_EASY, feature_extractor=extractor, feature_names=feature_names)
    
    # ---------------------------------------------------------------------
    # Compute the average score for each feature across all documents.
    # ---------------------------------------------------------------------
    # Compute average scores per feature for each corpus.
    def compute_avg(scores_dict: dict[str, list[float]]) -> dict[str, float]:
        return {name: (mean(scores_dict.get(name, [])) if scores_dict.get(name) else float('nan')) for name in feature_names}

    avg_by_corpus: dict[str, dict[str, float]] = {
        "TIGER_ORIG": compute_avg(scores_tiger_orig),
        "ASGC_ORIG": compute_avg(scores_asgc_orig),
        "G4A_ORIG": compute_avg(scores_g4a_orig),
        "DEplain_ORIG": compute_avg(scores_deplain_orig),
        "ASGC_PLAIN": compute_avg(scores_asgc_plain),
        "DEplain_PLAIN": compute_avg(scores_deplain_plain),
        "Leiko_PLAIN": compute_avg(scores_leiko_plain),
        "ASGC_EASY": compute_avg(scores_asgc_easy),
        "G4A_EASY": compute_avg(scores_g4a_easy),
        "Leiko_EASY": compute_avg(scores_leiko_easy),
    }

    # Plot stacked bars for all corpora.
    plot_stacked_bars(avg_by_corpus, output_path=Path("../output") / "freq_stacked.png")
    
    # Plot grouped bars (features on x‑axis) for all corpora.
    #plot_grouped_bars(avg_by_corpus, output_path=Path("output") / "freq_grouped.png")

    # ---------------------------------------------------------------------
    # Write LaTeX table summarizing the average frequency‑ratio features.
    # ---------------------------------------------------------------------
    write_latex_table(avg_by_corpus, output_path=Path("../output") / "freq_table.tex")

    # ---------------------------------------------------------------------
    # Save token frequency tables for each corpus individually.
    # ---------------------------------------------------------------------
    freq_output_dir = Path("../output") / "freq_token"
    freq_output_dir.mkdir(parents=True, exist_ok=True)
    corpus_map = {
        "TIGER_ORIG": TIGER_ORIG,
        "ASGC_ORIG": ASGC_ORIG,
        "ASGC_PLAIN": ASGC_PLAIN,
        "ASGC_EASY": ASGC_EASY,
        "G4A_ORIG": G4A_ORIG,
        "G4A_EASY": G4A_EASY,
        "DEplain_ORIG": DEplain_ORIG,
        "DEplain_PLAIN": DEplain_PLAIN,
        "Leiko_PLAIN": Leiko_PLAIN,
        "Leiko_EASY": Leiko_EASY,
    }
    # Collect token frequency DataFrames for plotting token‑band proportions.
    token_counts_by_corpus: dict[str, pd.DataFrame] = {}
    for label, path in corpus_map.items():
        out_csv = freq_output_dir / f"{label}_freq.csv"
        df = get_freq_token(path, out_csv=out_csv)
        token_counts_by_corpus[label] = df

    # Plot token‑band proportion stacked bars.
    plot_token_proportions(
        token_counts_by_corpus,
        output_path=Path("../output") / "freq_token_proportions.png",
    )


if __name__ == "__main__":
    main()
