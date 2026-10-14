import pandas as pd
import os
from pathlib import Path
from statistics import mean

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Mapping
from utils import _category, process_folder
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
from py_lift.extractors import FE_TokensPerSentence


def _plot_sentence_lengths(corpus_scores: dict, title: str, output_path: Path) -> None:
    """Create a violin plot with a swarm overlay for sentence length scores.

    Args:
        corpus_scores: Mapping from corpus label to a list of token‑per‑sentence scores.
        title: Plot title.
        output_path: Destination file path for the PNG image.
    """
    # Build a DataFrame suitable for seaborn.
    df = pd.DataFrame({
        "corpus": sum([[name] * len(scores) for name, scores in corpus_scores.items()], []),
        "score": sum([list(scores) for scores in corpus_scores.values()], []),
    })

    # Helper to map a corpus identifier to its high‑level category (orig, plain, easy).
    def _category(name: str) -> str:
        upper = name.upper()
        if "ORIG" in upper:
            return "orig"
        if "EASY" in upper:
            return "easy"
        return "plain"

    # Add a ``category`` column so that all corpora of the same category share a colour.
    df["category"] = df["corpus"].apply(_category)

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    # Use the same colour scheme as in run_readability.
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
    plt.title(title)
    plt.ylabel("Tokens per Sentence")
    plt.xlabel("Corpus")
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="center")
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.35)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def _plot_sentence_lengths_horizontal(corpus_scores: dict, title: str, output_path: Path) -> None:
    """Create a horizontal violin plot with a swarm overlay for sentence length.

    This mirrors :func:`_plot_sentence_lengths` but places the corpus labels on
    the y‑axis and the score on the x‑axis, resulting in a horizontal layout.
    """
    # Build the same DataFrame as the vertical version.
    df = pd.DataFrame({
        "corpus": sum([[name] * len(scores) for name, scores in corpus_scores.items()], []),
        "score": sum([list(scores) for scores in corpus_scores.values()], []),
    })

    def _category(name: str) -> str:
        upper = name.upper()
        if "ORIG" in upper:
            return "orig"
        if "EASY" in upper:
            return "easy"
        return "plain"

    df["category"] = df["corpus"].apply(_category)

    sns.set(style="whitegrid")
    # ↓ Make the chart narrower.
    plt.figure(figsize=(6, 6))

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
    plt.title(title)
    plt.xlabel("Tokens per Sentence")
    plt.ylabel("Corpus")
    ax = plt.gca()
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right")

    # Tighten the margins.
    fig = plt.gcf()
    fig.subplots_adjust(left=0.30, right=0.95)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    extractor = lambda cas: (
        FE_TokensPerSentence().extract(cas),
    )
    feature_names = [
            'Token_COUNT_PER_Sentence_COUNT'
    ]

    scores_tiger_orig = process_folder(TIGER_ORIG, feature_extractor=extractor, feature_names=feature_names)
  
    scores_crawled_orig = process_folder(ASGC_ORIG, feature_extractor=extractor, feature_names=feature_names)
    scores_crawled_plain = process_folder(ASGC_PLAIN, feature_extractor=extractor, feature_names=feature_names)
    scores_crawled_easy = process_folder(ASGC_EASY, feature_extractor=extractor, feature_names=feature_names)

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
        "ASGC_ORIG": compute_avg(scores_crawled_orig),
        "DEplain_ORIG": compute_avg(scores_deplain_orig),
        "G4A_ORIG": compute_avg(scores_g4a_orig),
        "ASGC_PLAIN": compute_avg(scores_crawled_plain),
        "DEplain_PLAIN": compute_avg(scores_deplain_plain),
        "Leiko_PLAIN": compute_avg(scores_leiko_plain),
        "G4A_EASY": compute_avg(scores_g4a_easy),
        "ASGC_EASY": compute_avg(scores_crawled_easy),
        "Leiko_EASY": compute_avg(scores_leiko_easy),
    }

    # Plot violin + swarm for sentence length distribution across corpora.
    _plot_sentence_lengths(
        {
            "TIGER_ORIG": scores_tiger_orig.get(feature_names[0], []),
            "ASGC_ORIG": scores_crawled_orig.get(feature_names[0], []),
            "DEplain_ORIG": scores_deplain_orig.get(feature_names[0], []),
            "G4A_ORIG": scores_g4a_orig.get(feature_names[0], []),
            "ASGC_PLAIN": scores_crawled_plain.get(feature_names[0], []),
            "DEplain_PLAIN": scores_deplain_plain.get(feature_names[0], []),
            "Leiko_PLAIN": scores_leiko_plain.get(feature_names[0], []),
            "ASGC_EASY": scores_crawled_easy.get(feature_names[0], []),
            "G4A_EASY": scores_g4a_easy.get(feature_names[0], []),
            "Leiko_EASY": scores_leiko_easy.get(feature_names[0], []),
        },
        "Sentence Length Distribution (Tokens per Sentence)",
        Path("../output") / "sentence_length.png",
    )

    # Also generate a horizontal version of the violin plot.
    _plot_sentence_lengths_horizontal(
        {
            "TIGER_ORIG": scores_tiger_orig.get(feature_names[0], []),
            "ASGC_ORIG": scores_crawled_orig.get(feature_names[0], []),
            "DEplain_ORIG": scores_deplain_orig.get(feature_names[0], []),
            "G4A_ORIG": scores_g4a_orig.get(feature_names[0], []),
            "ASGC_PLAIN": scores_crawled_plain.get(feature_names[0], []),
            "DEplain_PLAIN": scores_deplain_plain.get(feature_names[0], []),
            "Leiko_PLAIN": scores_leiko_plain.get(feature_names[0], []),
            "ASGC_EASY": scores_crawled_easy.get(feature_names[0], []),
            "G4A_EASY": scores_g4a_easy.get(feature_names[0], []),
            "Leiko_EASY": scores_leiko_easy.get(feature_names[0], []),
        },
        "Sentence Length Distribution (Tokens per Sentence)",
        Path("../output") / "sentence_length_horizontal.png",
    )

    # # ---------------------------------------------------------------------
    # # Collect all sentence lengths together with corpus and document name,
    # # then plot the top N longest sentences (default 20).
    # # ---------------------------------------------------------------------
    # def _collect_lengths(corpus_path: Path, label: str) -> list[tuple[float, str]]:
    #     """Return a list of (length, identifier) for every sentence in the corpus.

    #     ``identifier`` combines the corpus label and the file name for easy
    #     reference in the plot (e.g., "CRAWLED_ORIG/file1.txt").
    #     """
    #     cache_dir = Path(__file__).parent / "cache" / utils_mod.get_corpus_slug(corpus_path)
    #     cache_dir.mkdir(parents=True, exist_ok=True)
    #     entries: list[tuple[float, str]] = []
    #     for file in corpus_path.iterdir():
    #         if not file.is_file():
    #             continue
    #         result = utils_mod._process_file(
    #             file,
    #             use_cache=True,
    #             cache_dir=cache_dir,
    #             feature_extractor=extractor,
    #             feature_names=feature_names,
    #         )
    #         lengths = result.get(feature_names[0], [])
    #         for length in lengths:
    #             identifier = f"{label}/{file.name}"
    #             entries.append((float(length), identifier))
    #     return entries

    # all_entries: list[tuple[float, str]] = []
    # for label, path in {
    #     "TIGER_ORIG": TIGER_ORIG,
    #     "ASGC_ORIG": ASGC_ORIG,
    #     "ASGC_PLAIN": ASGC_PLAIN,
    #     "ASGC_EASY": ASGC_EASY,
    #     "G4A_ORIG": G4A_ORIG,
    #     "G4A_EASY": G4A_EASY,
    #     "DEplain_ORIG": DEplain_ORIG,
    #     "DEplain_PLAIN": DEplain_PLAIN,
    #     "Leiko_EASY": Leiko_EASY,
    # }.items():
    #     all_entries.extend(_collect_lengths(path, label))

    # # Sort by length descending and keep top 20.
    # top_n = 20
    # top_entries = sorted(all_entries, key=lambda x: x[0], reverse=True)[:top_n]

    # # Print the top lengths to the console.
    # print(f"Top {top_n} longest sentences across all corpora (tokens per sentence):")
    # for length, identifier in top_entries:
    #     print(f"{identifier}: {length:.2f}")

if __name__ == "__main__":
    main()
