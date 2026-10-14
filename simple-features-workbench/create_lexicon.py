from pathlib import Path

import spacy
from spacy.tokens import Doc
from tqdm import tqdm

from constants import (
#    TIGER_ORIG,
#    ASGC_ORIG,
#    ASGC_PLAIN,
    ASGC_EASY,
#    G4A_ORIG,
    G4A_EASY,
#    DEplain_ORIG,
#    DEplain_PLAIN,
#    Leiko_PLAIN,
    Leiko_EASY
)

from utils import (
    get_cache_dir_for_folder,
    load_or_create_cas,
)


TOKEN_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"

nlp = spacy.load("de_core_news_lg")


def extract_token_texts_from_cas(cas) -> list[str]:
    """Extract token texts from a CAS."""
    token_texts: list[str] = []

    for token in cas.select(TOKEN_TYPE):
        text = token.get_covered_text()

        if text is None:
            continue

        text = text.strip()

        if not text:
            continue

        token_texts.append(text)

    return token_texts


def lemmatize_cas_tokens_with_spacy(token_texts: list[str]) -> Doc:
    """Lemmatize CAS tokens with spaCy while preserving CAS tokenization."""
    doc = Doc(nlp.vocab, words=token_texts)

    disable_components = [
        name for name in ["parser", "ner"] if name in nlp.pipe_names
    ]

    with nlp.select_pipes(disable=disable_components):
        for _, component in nlp.pipeline:
            doc = component(doc)

    return doc


def build_lemma_lexicon_from_folder(
    corpus_path: Path,
    corpus_label: str,
    *,
    use_cache: bool = True,
    lowercase: bool = True,
    keep_punctuation: bool = False,
    keep_numbers: bool = False,
) -> set[str]:
    """Build a lemma lexicon from one corpus folder."""
    corpus_path = Path(corpus_path)

    cache_dir = get_cache_dir_for_folder(corpus_path) if use_cache else None

    lemmas: set[str] = set()

    files = sorted([file for file in corpus_path.iterdir() if file.is_file()])

    for file in tqdm(files, desc=f"Lexicon {corpus_label}"):
        cas = load_or_create_cas(
            file=file,
            use_cache=use_cache,
            cache_dir=cache_dir,
        )

        token_texts = extract_token_texts_from_cas(cas)

        if not token_texts:
            print(f"Warning: No CAS tokens found in {corpus_label}/{file.name}")
            continue

        doc = lemmatize_cas_tokens_with_spacy(token_texts)

        for tok in doc:
            if tok.is_space:
                continue

            if not keep_punctuation and tok.is_punct:
                continue

            if not keep_numbers and tok.like_num:
                continue

            lemma = tok.lemma_.strip()

            if not lemma:
                lemma = tok.text.strip()

            if not lemma:
                continue

            if lowercase:
                lemma = lemma.lower()

            lemmas.add(lemma)

    return lemmas


def save_lexicon(lemmas: set[str], output_path: Path) -> None:
    """Save one lemma per line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for lemma in sorted(lemmas, key=str.casefold):
            f.write(lemma + "\n")


def main() -> None:
    corpora: dict[str, Path] = {
        #"TIGER_ORIG": TIGER_ORIG,
        #"ASGC_ORIG": ASGC_ORIG,
        #"DEplain_ORIG": DEplain_ORIG,
        #"G4A_ORIG": G4A_ORIG,
        #"ASGC_PLAIN": ASGC_PLAIN,
        #"DEplain_PLAIN": DEplain_PLAIN,
        #"Leiko_PLAIN": Leiko_PLAIN,
        "ASGC_EASY": ASGC_EASY,
        "G4A_EASY": G4A_EASY,
        "Leiko_EASY": Leiko_EASY,
    }

    complete_lexicon: set[str] = set()

    for label, path in corpora.items():
        corpus_lexicon = build_lemma_lexicon_from_folder(
            corpus_path=Path(path),
            corpus_label=label,
            use_cache=True,
            lowercase=True,
            keep_punctuation=False,
            keep_numbers=False,
        )

        print(f"{label}: {len(corpus_lexicon)} unique lemmas")

        save_lexicon(
            corpus_lexicon,
            Path("output") / "lexicons" / f"{label}_lemma_lexicon.txt",
        )

        complete_lexicon.update(corpus_lexicon)

    print(f"Complete lexicon: {len(complete_lexicon)} unique lemmas")

    save_lexicon(
        complete_lexicon,
        Path("output") / "lemma_lexicon_complete.txt",
    )


if __name__ == "__main__":
    main()