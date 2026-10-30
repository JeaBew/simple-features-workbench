import argparse
import seaborn as sns
import spacy
from tqdm import tqdm
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional
import pandas as pd
from utils import process_folder, _category, get_corpus_slug, _process_file
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
    Leiko_EASY,
)

# de_core_news_sm/md/lg oder de_dep_news_trf etc.
nlp = spacy.load("de_core_news_lg")

def is_subjunctive(tok):
    return classify_subjunctive(tok) is not None

def classify_subjunctive(tok):
    """
    Returns: "KI", "KII", "CND", or None
    Rule of thumb:
      - KI  ~ Mood=Sub & Tense=Pres
      - KII ~ Mood=Sub & Tense=Past
      - würde-Konstruktion oft Mood=Cnd (optional)
    """
    if tok.pos_ not in ("AUX", "VERB"):
        return None
    if tok.morph.get("VerbForm") and "Fin" not in tok.morph.get("VerbForm"):
        return None

    mood = tok.morph.get("Mood")
    tense = tok.morph.get("Tense")

    if "Sub" in mood:
        if "Pres" in tense:
            return "KI"
        if "Past" in tense:
            return "KII"
        # Konjunktiv, aber Tense nicht vorhanden/unklar
        return "KONJ_SUB_UNKLAR"

    # Optional: würde-Formen als Konditional markieren
    if "Cnd" in mood:
        return "CND"

    return None


def process_folder(
    dir: Path,
) -> float:
    """Process a corpus directory.

    Returns the proportion of documents that contain at least one subjunctive
    token **and** writes every sentence that contains a subjunctive form to a
    result file. Each line in the result file contains the corpus identifier,
    the source file name, and the matching sentence, separated by tabs.
    """

    # Prepare the result file for this corpus.
    result_path = Path("../output") / "subjunctive_sentences.txt"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    # Open in append mode; the caller is responsible for clearing the file
    # before the first call (e.g. in __main__).
    result_file = result_path.open("a", encoding="utf-8")

    # Count how many documents contain at least one subjunctive token.
    file_count = 0
    docs_with_subjunctive = 0
    files = list(dir.iterdir())
    for file in tqdm(files, desc=f"Processing {dir.name}"):
        if not file.is_file() or file.suffix != ".txt":
            continue

        text = file.read_text(encoding="utf-8")
        doc = nlp(text)
        file_count += 1

        # Determine if the document contains a subjunctive token.
        has_subjunctive = any(is_subjunctive(tok) for tok in doc)
        if has_subjunctive:
            docs_with_subjunctive += 1

        # Write each sentence that contains a subjunctive token.
        if has_subjunctive:
            corpus_name = get_corpus_slug(dir)
            for sent in doc.sents:
                if any(is_subjunctive(tok) for tok in sent):
                    # Strip newline characters to keep a single‑line record.
                    sentence_text = sent.text.replace("\n", " ").strip()
                    result_file.write(f"{corpus_name}\t{file.name}\t{sentence_text}\n")

    result_file.close()

    # Proportion of documents that contain at least one subjunctive form.
    return docs_with_subjunctive / file_count if file_count > 0 else float("nan")

def plot_subjunctive_proportions(results: Dict[str, float]) -> None:
    """Create a horizontal bar chart using pre-computed *results*.

    Bars are ordered by corpus category:
    first ORIG, then PLAIN, then EASY.
    """

    import matplotlib.pyplot as plt

    df = pd.DataFrame({
        "corpus": list(results.keys()),
        "proportion": list(results.values()),
    })

    df["category"] = df["corpus"].apply(_category)

    # Gewünschte Reihenfolge der Korpus-Kategorien
    category_order = {
        "orig": 0,
        "plain": 1,
        "easy": 2,
    }

    # Ursprüngliche Reihenfolge innerhalb einer Kategorie beibehalten
    df["_original_order"] = range(len(df))
    df["_category_order"] = df["category"].map(category_order)

    df = df.sort_values(
        by=["_category_order", "_original_order"],
        ascending=[True, True],
    ).reset_index(drop=True)

    # Explizite y-Achsen-Reihenfolge für seaborn
    corpus_order = df["corpus"].tolist()

    sns.set(style="whitegrid")
    plt.figure(figsize=(6, max(2, len(df) * 0.4)))

    palette = {
        "orig": "tab:blue",
        "plain": "tab:orange",
        "easy": "tab:green",
    }

    sns.barplot(
        x="proportion",
        y="corpus",
        hue="category",
        data=df,
        order=corpus_order,
        hue_order=["orig", "plain", "easy"],
        palette=palette,
        dodge=False,
        legend=False,
    )

    plt.xlabel("Subjunctive proportion")
    plt.ylabel("Corpus")
    plt.title("Subjunctive proportion per corpus")

    ax = plt.gca()
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right")

    fig = plt.gcf()
    fig.subplots_adjust(left=0.30, right=0.95)
    plt.tight_layout()

    output_path = Path("../output") / "subjunctive_proportions.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

# Map corpus names to their paths for CLI selection.
CORPUS_MAP: Dict[str, Path] = {
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

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute subjunctive proportions and write matching sentences to a file."
    )
    parser.add_argument(
        "--corpus",
        choices=list(CORPUS_MAP.keys()),
        default=None,
        help="Process only this corpus. Omit to process all corpora and create the plot.",
    )
    args = parser.parse_args()

    # Clear the output file once before any corpus is processed.
    result_path = Path("../output") / "subjunctive_sentences.txt"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text("", encoding="utf-8")

    if args.corpus:
        # Process a single corpus and write its sentences; skip the plot.
        proportion = process_folder(CORPUS_MAP[args.corpus])
        print(f"{args.corpus} subjunctive proportion: {proportion:.4f}")
        print(f"Sentences written to {result_path}")
    else:
        # Process all corpora, then plot.
        results = {name: process_folder(path) for name, path in CORPUS_MAP.items()}
        plot_subjunctive_proportions(results)


if __name__ == "__main__":
    main()