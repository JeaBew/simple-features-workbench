import argparse
import re
import pandas as pd
import os
from pathlib import Path
from cassis import Cas
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import spacy
from typing import Dict, List, Mapping
from utils import get_corpus_slug, _category, get_cache_dir_for_folder
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
from py_lift.decorators import supported_languages
from py_lift.dkpro import T_LEMMA
from py_lift.annotators.api import SEL_BaseAnnotator
from py_lift.annotators.lists import SEL_ListReader
from typing import Optional
from py_lift.utils.core import load_cas_from_xmi_with_lift_ts  # type: ignore
from py_lift.preprocessing import Spacy_Preprocessor  # type: ignore
from py_lift.utils.core import load_lift_typesystem  # type: ignore

# Initialise the preprocessing pipeline once – it is safe to reuse across calls.
prep = Spacy_Preprocessor("de", auto_install_models=True)
ts = load_lift_typesystem()

# Load spaCy model for sentence segmentation and token inspection.
nlp = spacy.load("de_core_news_lg")
# Load negation word list once for fast lookup.
NEGATION_WORDS = set(
    (Path(__file__).parent / "negation_words.txt").read_text(encoding="utf-8").splitlines()
)

# ad hoc type
NEG = ts.create_type(name='org.lift.type.Negation')

@supported_languages('de')
class SE_NegationAnnotator(SEL_BaseAnnotator, SEL_ListReader):

    def __init__(self, language, ts=None):
        filename = Path(__file__).parent / "negation_words.txt"
        SEL_ListReader.__init__(self, filename)
        SEL_BaseAnnotator.__init__(self, language)

    def _process(self, cas: Cas) -> bool:
        negations = self.read_list()
        for lemma in cas.select(T_LEMMA):
            l_str = lemma.value
            if l_str in negations:
                neg_anno = NEG(begin=lemma.get('begin'), end=lemma.get('end'))
                cas.add(neg_anno)
        return True

def run_folder(
    dir: Path,
    use_cache: bool = True,
) -> float:
    """Process a corpus directory.

    Returns the proportion of documents that contain at least one negation
    token **and** writes every sentence that contains a negation to a result
    file. Each line in the result file contains the corpus identifier, the
    source file name, and the matching sentence, separated by tabs.
    """

    if use_cache:
        cache_dir = get_cache_dir_for_folder(dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    # Prepare the result file for this corpus; caller clears it before the first call.
    result_path = Path("../output") / "negation_sentences.txt"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_file = result_path.open("a", encoding="utf-8")

    file_count = 0
    docs_with_negation = 0

    for file in tqdm(dir.iterdir(), desc=f"Processing {dir.name}"):
        if not file.is_file() or file.suffix != ".txt":
            continue

        # Read text here so it is always available for sentence extraction.
        text = file.read_text(encoding="utf-8")
        cas = get_cas_for_file(file, use_cache, cache_dir)

        file_count += 1

        has_neg = has_negation(cas)
        if has_neg:
            docs_with_negation += 1

        # Write each sentence that contains a negation token.
        if has_neg:
            corpus_name = get_corpus_slug(dir)
            doc_spacy = nlp(text)
            for sent in doc_spacy.sents:
                if any(tok.lemma_.lower() in NEGATION_WORDS for tok in sent):
                    sentence_text = sent.text.replace("\n", " ").strip()
                    result_file.write(f"{corpus_name}\t{file.name}\t{sentence_text}\n")

    result_file.close()

    # Proportion of documents that contain at least one negation.
    return docs_with_negation / file_count if file_count > 0 else float("nan")

def get_cas_for_file(
    file: Path,
    use_cache: bool,
    cache_dir: Path | None,
) -> Cas:

    try:
        if use_cache and cache_dir is not None:
            cache_file = cache_dir / f"{file.stem}.xmi"
            # Load from cache if present, otherwise process and cache.
            if cache_file.is_file():
                cas = load_cas_from_xmi_with_lift_ts(cache_file)
            else:
                text = file.read_text(encoding="utf-8")
                cas = prep.run(text)
                # Cache the XMI for future runs.
                cas.to_xmi(str(cache_file))
        else:
            # Direct processing without caching.
            text = file.read_text(encoding="utf-8")
            cas = prep.run(text)

        return cas
    except Exception as e:
        raise RuntimeError(f"Error processing file '{file}': {e}") from e

def has_negation(cas: Cas) -> bool:
    negation_annotator = SE_NegationAnnotator(language="de", ts=ts)
    negation_annotator.process(cas)
    return any(cas.select("org.lift.type.Negation"))

def plot_negation_proportions(results: Dict[str, float]) -> None:
    """Create a horizontal bar chart using pre-computed *results*.

    Bars are ordered by corpus category:
    first ORIG, then PLAIN, then EASY.
    """

    # DataFrame aus results bauen
    df = pd.DataFrame({
        "corpus": list(results.keys()),
        "proportion": list(results.values()),
    })

    df["category"] = df["corpus"].apply(_category)

    # Gewünschte Reihenfolge der Gruppen
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

    # Diese Reihenfolge explizit an seaborn übergeben
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

    plt.xlabel("Negation proportion")
    plt.ylabel("Corpus")
    plt.title("Negation proportion per corpus")

    ax = plt.gca()
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right")

    fig = plt.gcf()
    fig.subplots_adjust(left=0.30, right=0.95)
    plt.tight_layout()

    output_path = Path("../output") / "negation_proportions.png"
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
        description="Compute negation proportions and write matching sentences to a file."
    )
    parser.add_argument(
        "--corpus",
        choices=list(CORPUS_MAP.keys()),
        default=None,
        help="Process only this corpus. Omit to process all corpora and create the plot.",
    )
    args = parser.parse_args()

    # Clear the output file once before any corpus is processed.
    result_path = Path("../output") / "negation_sentences.txt"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text("", encoding="utf-8")

    if args.corpus:
        proportion = run_folder(CORPUS_MAP[args.corpus])
        print(f"{args.corpus} negation proportion: {proportion:.4f}")
        print(f"Sentences written to {result_path}")
    else:
        results = {name: run_folder(path) for name, path in CORPUS_MAP.items()}
        plot_negation_proportions(results)


if __name__ == "__main__":
    main()