import pandas as pd
import os
from pathlib import Path
from statistics import mean

import seaborn as sns
import matplotlib.pyplot as plt

from utils import process_folder, _category
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
from py_lift.readability import FE_TextstatWienerSachtextformel_1  # type: ignore

def _plot_scores(corpus_scores: dict, title: str, output_path: Path) -> None:
    """Create a violin + swarm plot for the given corpus scores.

    The colour scheme groups corpora by *category* (orig, plain, easy) rather than
    by the exact corpus name. All corpora that belong to the same category share
    the same colour, while the three categories receive distinct colours.

    Args:
        corpus_scores: Mapping from corpus name (e.g., "CRAWLED_ORIG") to a list
            of readability scores.
        title: Title for the plot.
        output_path: File path where the PNG image will be saved.
    """
    # Helper to map a corpus identifier to its high‑level category.
    # Use shared category helper from utils.
    df = pd.DataFrame({
        "corpus": sum([[name] * len(scores) for name, scores in corpus_scores.items()], []),
        "score": sum([list(scores) for scores in corpus_scores.values()], []),
    })
    df["category"] = df["corpus"].apply(_category)

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    # Use ``category`` as hue so that all corpora of the same category share a colour.
    palette = {"orig": "tab:blue", "plain": "tab:orange", "easy": "tab:green"}
    sns.violinplot(
        x="corpus",
        y="score",
        hue="category",
        data=df,
        inner="quartile",
        palette=palette,
        legend=False,
    )
    # Swarm overlay – keep points black for contrast.
    sns.swarmplot(x="corpus", y="score", data=df, color="k", alpha=0.5, size=3)
    plt.title(title)
    plt.ylabel("Readability Score")
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="center")
    # Adjust bottom margin so rotated labels are not cut off (similar to run_freq.py).
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.35)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def _plot_scores_horizontal(corpus_scores: dict, title: str, output_path: Path) -> None:
    """Create a horizontal violin + swarm plot for readability scores.

    Mirrors :func:`_plot_scores` but swaps the axes so that corpus names are on
    the y‑axis and the readability score is on the x‑axis. This makes it easier
    to compare values when there are many corpora.
    """
    df = pd.DataFrame({
        "corpus": sum([[name] * len(scores) for name, scores in corpus_scores.items()], []),
        "score": sum([list(scores) for scores in corpus_scores.values()], []),
    })
    df["category"] = df["corpus"].apply(_category)

    sns.set(style="whitegrid")
    # Use a slightly narrower figure width for a compact horizontal layout.
    plt.figure(figsize=(8, 6))
    palette = {"orig": "tab:blue", "plain": "tab:orange", "easy": "tab:green"}
    sns.violinplot(
        x="score",
        y="corpus",
        hue="category",
        data=df,
        inner="quartile",
        palette=palette,
        legend=False,
    )
    # Horizontal swarm overlay.
    sns.swarmplot(x="score", y="corpus", data=df, color="k", alpha=0.5, size=3)
    plt.title(title)
    plt.xlabel("Readability Score")
    plt.ylabel("Corpus")
    ax = plt.gca()
    # Ensure corpus labels are fully visible.
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right")
    fig = plt.gcf()
    fig.subplots_adjust(left=0.30, right=0.95)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_box(corpus_scores: dict, title: str, output_path: Path) -> None:
    """Create a box plot for the given scores using the same colour scheme as
    :func:`_plot_scores`.

    Args:
        corpus_scores: Mapping from corpus name to list of scores.
        title: Plot title.
        output_path: Destination PNG file.
    """
    df = pd.DataFrame({
        "corpus": sum([[name] * len(scores) for name, scores in corpus_scores.items()], []),
        "score": sum([list(scores) for scores in corpus_scores.values()], []),
    })
    df["category"] = df["corpus"].apply(_category)

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    palette = {"orig": "tab:blue", "plain": "tab:orange", "easy": "tab:green"}
    sns.boxplot(x="corpus", y="score", data=df, hue="category", palette=palette)
    plt.title(title)
    plt.ylabel("Readability Score")
    # Rotate corpus labels for readability (90°) and ensure they are fully visible.
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    # Adjust bottom margin similar to run_freq.py.
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.35)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    # The extractor now receives the CAS directly. We wrap the original class
    # method to match the new ``feature_extractor`` signature.
    extractor = lambda cas: FE_TextstatWienerSachtextformel_1().extract(cas)
    feature_names = ["Readability_Score_WienerSachtextformel-1_de"]
    
    scores_tiger_orig = process_folder(TIGER_ORIG, feature_extractor=extractor, feature_names=feature_names)

    # Gather all readability scores from the crawled corpora.
    scores_crawled_orig = process_folder(ASGC_ORIG, feature_extractor=extractor, feature_names=feature_names)
    scores_crawled_plain = process_folder(ASGC_PLAIN, feature_extractor=extractor, feature_names=feature_names)
    scores_crawled_easy = process_folder(ASGC_EASY, feature_extractor=extractor, feature_names=feature_names)

    # Compute and output the mean readability score for each crawled corpus.
    # Extract the list for the single feature we are interested in.
    feat = feature_names[0]
    mean_orig = mean(scores_crawled_orig.get(feat, [])) if scores_crawled_orig else float('nan')
    mean_plain = mean(scores_crawled_plain.get(feat, [])) if scores_crawled_plain else float('nan')
    mean_easy = mean(scores_crawled_easy.get(feat, [])) if scores_crawled_easy else float('nan')
    print(f"Mean readability (crawled orig): {mean_orig}")
    print(f"Mean readability (crawled plain): {mean_plain}")
    print(f"Mean readability (crawled easy): {mean_easy}")

    # Plot for crawled dataset
    _plot_scores(
        {"orig": scores_crawled_orig.get(feat, []), "plain": scores_crawled_plain.get(feat, []), "easy": scores_crawled_easy.get(feat, [])},
        "Crawled Corpus Readability Distribution",
        Path("../output") / "crawled_readability.png",
    )
    # Additional box plot for crawled dataset
    _plot_box(
        {"orig": scores_crawled_orig.get(feat, []), "plain": scores_crawled_plain.get(feat, []), "easy": scores_crawled_easy.get(feat, [])},
        "Crawled Corpus Readability Box Plot",
        Path("../output") / "crawled_box.png",
    )

    # German4All dataset
    scores_g4a_orig = process_folder(G4A_ORIG, feature_extractor=extractor, feature_names=feature_names)
    scores_g4a_easy = process_folder(G4A_EASY, feature_extractor=extractor, feature_names=feature_names)
    mean_g4a_orig = mean(scores_g4a_orig.get(feat, [])) if scores_g4a_orig else float('nan')
    mean_g4a_easy = mean(scores_g4a_easy.get(feat, [])) if scores_g4a_easy else float('nan')
    print(f"Mean readability (g4a orig): {mean_g4a_orig}")
    print(f"Mean readability (g4a easy): {mean_g4a_easy}")
    _plot_scores(
        {"orig": scores_g4a_orig.get(feat, []), "easy": scores_g4a_easy.get(feat, [])},
        "German4All Readability Distribution",
        Path("../output") / "g4a_readability.png",
    )
    # Additional box plot for German4All dataset
    _plot_box(
        {"orig": scores_g4a_orig.get(feat, []), "easy": scores_g4a_easy.get(feat, [])},
        "German4All Readability Box Plot",
        Path("../output") / "g4a_box.png",
    )

    # Deplain dataset
    scores_deplain_orig = process_folder(DEplain_ORIG, feature_extractor=extractor, feature_names=feature_names)
    scores_deplain_plain = process_folder(DEplain_PLAIN, feature_extractor=extractor, feature_names=feature_names)
    mean_deplain_orig = mean(scores_deplain_orig.get(feat, [])) if scores_deplain_orig else float('nan')
    mean_deplain_plain = mean(scores_deplain_plain.get(feat, [])) if scores_deplain_plain else float('nan')
    print(f"Mean readability (deplain orig): {mean_deplain_orig}")
    print(f"Mean readability (deplain plain): {mean_deplain_plain}")
    _plot_scores(
        {"orig": scores_deplain_orig.get(feat, []), "plain": scores_deplain_plain.get(feat, [])},
        "Deplain Readability Distribution",
        Path("../output") / "deplain_readability.png",
    )
    # Additional box plot for Deplain dataset
    _plot_box(
        {"orig": scores_deplain_orig.get(feat, []), "plain": scores_deplain_plain.get(feat, [])},
        "Deplain Readability Box Plot",
        Path("../output") / "deplain_box.png",
    )
    
    scores_leiko_plain = process_folder(Leiko_PLAIN, feature_extractor=extractor, feature_names=feature_names)
    scores_leiko_easy = process_folder(Leiko_EASY, feature_extractor=extractor, feature_names=feature_names)
    # ---------------------------------------------------------------------
    # Combined plot for all corpora
    # ---------------------------------------------------------------------
    combined_scores = {
        "TIGER_ORIG": scores_tiger_orig.get(feat, []),
        "ASGC_ORIG": scores_crawled_orig.get(feat, []),
        "DEplain_ORIG": scores_deplain_orig.get(feat, []),
        "G4A_ORIG": scores_g4a_orig.get(feat, []),
        "ASGC_PLAIN": scores_crawled_plain.get(feat, []),
        "DEplain_PLAIN": scores_deplain_plain.get(feat, []),
        "Leiko_PLAIN": scores_leiko_plain.get(feat, []),
        "ASGC_EASY": scores_crawled_easy.get(feat, []),
        "G4A_EASY": scores_g4a_easy.get(feat, []),
        "Leiko_EASY": scores_leiko_easy.get(feat, []),
    }
    _plot_scores(
        combined_scores,
        "Readability Distribution Across All Corpora",
        Path("../output") / "all_readability.png",
    )
    # Horizontal version of the combined readability plot.
    _plot_scores_horizontal(
        combined_scores,
        "Readability Distribution Across All Corpora",
        Path("../output") / "all_readability_horizontal.png",
    )

if __name__ == "__main__":
    main()
