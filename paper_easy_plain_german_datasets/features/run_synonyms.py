"""
Nomen-Konsistenzprüfung für Leichte-Sprache-Texte
====================================================

Prüft, ob im Text mehrere unterschiedliche Nomen für dasselbe Konzept
verwendet werden.

Kombiniert drei Signale:
  1. OpenThesaurus-Abgleich
  2. spaCy-Word-Embeddings
  3. BERT-Kontextvektoren

Zwei Wege, um an die Nomen zu kommen:
  - extract_nouns(text)          -> verarbeitet Rohtext neu mit spaCy
                                     (v.a. für Standalone-Tests ohne CAS)
  - extract_nouns_from_cas(cas)  -> liest POS/Lemma direkt aus bereits
                                     vorhandenen CAS-Annotationen
                                     (empfohlen, wenn die Pipeline das
                                     schon berechnet hat)

check_text() und check_cas() sind die beiden passenden Einstiegspunkte,
die beide auf derselben find_synonym_candidates()-Logik aufsetzen.

Voraussetzungen:
    pip install spacy requests torch transformers
    python -m spacy download de_core_news_lg
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import requests
import spacy

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


OPENTHESAURUS_URL = "https://www.openthesaurus.de/synonyme/search"

SIMILARITY_THRESHOLD = 0.75
BERT_SIMILARITY_THRESHOLD = 0.82

REQUEST_DELAY = 0.3

DEFAULT_BERT_MODEL = "bert-base-german-cased"
# Alternativen:
# DEFAULT_BERT_MODEL = "deepset/gbert-base"
# DEFAULT_BERT_MODEL = "dbmdz/bert-base-german-cased"

# ---------------------------------------------------------------------------
# CAS-Konfiguration
#
# ACHTUNG: Diese Typ-/Feature-Namen sind Standard-DKPro-Core-Konventionen,
# aber ich kenne euer tatsächliches Typsystem nicht sicher. Bitte vor dem
# ersten produktiven Lauf gegenchecken und ggf. anpassen.
# ---------------------------------------------------------------------------

DEFAULT_POS_TYPE_NAME = "de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS"
DEFAULT_LEMMA_TYPE_NAME = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma"
DEFAULT_SENTENCE_TYPE_NAME = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"

# Welche POS-Werte als "Nomen" zählen - ggf. "NE" (Eigennamen) rausnehmen,
# falls Eigennamen hier nicht mitgezählt werden sollen.
NOUN_POS_VALUES = {"NN", "NOUN", "NE"}

ABBREVIATION_MAP: dict[str, set[str]] = {
    "EU": {"Europäische Union"},
    "NRW": {"Nordrhein-Westfalen"},
    "z. B.": {"zum Beispiel"},
    "bzw.": {"beziehungsweise"},
    "Uni": {"Universität", "Hochschule"},
    "Info": {"Information"},
    "Kita": {"Kindertagesstätte"},
    "FernUni": {"FernUniversität in Hagen", "FernUniversität"},
}

# WICHTIG: Hier stand vorher "$" statt "\(" / "\)" - "$" ist der Regex-Anker
# für Stringende, keine escapte Klammer. Dadurch konnte das Pattern nie
# matchen (ein String hat nur ein Ende), egal welcher Text reinkam. Fix:
# echte, escapte Klammern verwenden.
LONG_SHORT_PATTERN = re.compile(
    r"(?P<long>\b[A-ZÄÖÜ][^().]{3,80}?)\s*\((?P<short>[A-Za-zÄÖÜäöüß0-9.\-]{2,20})\)"
)

SHORT_LONG_PATTERN = re.compile(
    r"\b(?P<short>[A-ZÄÖÜ]{2,15})\s*\((?P<long>[A-ZÄÖÜ][^().]{3,80}?)\)"
)

# Fall 4: alleinstehende Großbuchstaben-Akronyme ohne Klammer-Definition im Text
# (z.B. Parteikürzel wie "SPD", "CDU"). Liefert keine Langform - dient nur als
# Kandidat zur weiteren Prüfung/Evaluation. Bewusst grob (reine Großschreibungs-
# Heuristik) - erwartbare Fehlalarme: Ländercodes, römische Zahlen (I, IV, ...),
# sonstige komplett großgeschriebene Wörter (z.B. Überschriften/Hervorhebungen).
STANDALONE_ACRONYM_PATTERN = re.compile(r"\b[A-ZÄÖÜ]{2,5}\b")

# Fall 5: einzelne Buchstaben als Einheiten-Abkürzung (z.B. "s" für Sekunde,
# "m" für Meter). Im Deutschen gibt es kaum echte einzelne Buchstaben als
# eigenständige Wörter, daher relativ präzise - aber Aufzählungsmarker wie
# "a)"/"b)" können mit reinrutschen.
SINGLE_LETTER_PATTERN = re.compile(r"\b[a-zA-ZÄÖÜäöüß]\b")

# Fall 6: gemischte Groß-/Kleinschreibung wie "kW" (Kilowatt), "kWh", "mAh".
# Erkennt mindestens einen Kleinbuchstaben, gefolgt von einem Großbuchstaben,
# irgendwo im selben Wort. Achtung: erkennt auch CamelCase-Markennamen wie
# "iPhone" als Fehlalarm.
MIXED_CASE_PATTERN = re.compile(r"\b[a-zäöüß]+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß]*\b")


@dataclass
class NounContext:
    sentence_text: str
    start_char: int
    end_char: int
    surface: str


@dataclass
class NounOccurrence:
    lemma: str
    surface_forms: set[str] = field(default_factory=set)
    count: int = 0
    sentence_indices: list[int] = field(default_factory=list)
    contexts: list[NounContext] = field(default_factory=list)


@dataclass
class SynonymCandidate:
    lemma_a: str
    lemma_b: str
    source: str
    similarity: float | None = None
    spacy_similarity: float | None = None
    bert_similarity: float | None = None

    def __str__(self) -> str:
        parts = [self.source]

        if self.spacy_similarity is not None:
            parts.append(f"spaCy={self.spacy_similarity:.2f}")

        if self.bert_similarity is not None:
            parts.append(f"BERT={self.bert_similarity:.2f}")

        if (
            self.spacy_similarity is None
            and self.bert_similarity is None
            and self.similarity is not None
        ):
            parts.append(f"Ähnlichkeit={self.similarity:.2f}")

        info = ", ".join(parts)
        return f"'{self.lemma_a}' <-> '{self.lemma_b}'  [{info}]"


@dataclass
class AbbreviationCandidate:
    short_form: str
    long_form: str | None
    source: str
    confidence: float = 0.95
    start_char: int | None = None
    end_char: int | None = None

    def __str__(self) -> str:
        if self.long_form:
            return (
                f"'{self.short_form}' <-> '{self.long_form}' "
                f"[{self.source}, Konfidenz={self.confidence:.2f}]"
            )
        return (
            f"'{self.short_form}' (keine Auflösung im Text gefunden) "
            f"[{self.source}, Konfidenz={self.confidence:.2f}]"
        )


_nlp = None
_bert_tokenizer = None
_bert_model = None
_bert_model_name = None


def get_nlp():
    """Lazy-Loading des spaCy-Modells."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("de_core_news_lg")
    return _nlp


def get_bert(model_name: str = DEFAULT_BERT_MODEL):
    """Lazy-Loading von BERT-Modell und Tokenizer."""
    global _bert_tokenizer, _bert_model, _bert_model_name

    if _bert_model is None or _bert_model_name != model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        if not tokenizer.is_fast:
            raise RuntimeError(
                "Für die BERT-Kontextvektoren wird ein FastTokenizer benötigt, "
                "weil Offset-Mapping verwendet wird."
            )

        model = AutoModel.from_pretrained(model_name)
        model.eval()

        _bert_tokenizer = tokenizer
        _bert_model = model
        _bert_model_name = model_name

    return _bert_tokenizer, _bert_model


def extract_nouns(text: str) -> dict[str, NounOccurrence]:
    """Extrahiert alle Nomen mit Lemma, Formen und Kontextinformationen.

    Verarbeitet den Rohtext neu mit spaCy - gedacht für Standalone-Tests
    ohne CAS. Für den produktiven Einsatz mit eurer CAS-Pipeline lieber
    extract_nouns_from_cas nutzen.
    """
    nlp = get_nlp()
    doc = nlp(text)
    occurrences: dict[str, NounOccurrence] = {}

    for sent_idx, sent in enumerate(doc.sents):
        sentence_text = sent.text
        sent_start = sent.start_char

        for tok in sent:
            if tok.pos_ != "NOUN":
                continue

            lemma = tok.lemma_
            occ = occurrences.setdefault(lemma, NounOccurrence(lemma=lemma))

            local_start = tok.idx - sent_start
            local_end = local_start + len(tok.text)

            occ.surface_forms.add(tok.text)
            occ.count += 1
            occ.sentence_indices.append(sent_idx)
            occ.contexts.append(
                NounContext(
                    sentence_text=sentence_text,
                    start_char=local_start,
                    end_char=local_end,
                    surface=tok.text,
                )
            )

    return occurrences


def extract_nouns_from_cas(
    cas: Any,
    view_name: str = "_InitialView",
    pos_type_name: str = DEFAULT_POS_TYPE_NAME,
    lemma_type_name: str = DEFAULT_LEMMA_TYPE_NAME,
    sentence_type_name: str = DEFAULT_SENTENCE_TYPE_NAME,
    noun_pos_values: set[str] = NOUN_POS_VALUES,
) -> dict[str, NounOccurrence]:
    """Extrahiert Nomen direkt aus bereits vorhandenen CAS-Annotationen.

    POS- und Lemma-Annotationen liegen in eurer CAS als eigene Ebenen auf
    demselben Span (begin/end) - nicht als Features direkt am Token. Diese
    Funktion sammelt daher zuerst alle Lemma-Annotationen in einem
    Span->Lemma-Lookup und matcht dann jede POS-Annotation mit
    PosValue == Nomen über exakt denselben Span.

    Für die BERT-Kontextvektoren werden zusätzlich Satzgrenzen
    (Sentence-Annotation) benötigt, um jedem Nomen-Vorkommen seinen
    umgebenden Satz zuzuordnen.
    """
    try:
        view = cas.get_view(view_name)
    except Exception:
        view = cas

    sofa_string = getattr(view, "sofa_string", None) or ""

    pos_annotations = sorted(view.select(pos_type_name), key=lambda a: (a.begin, a.end))
    lemma_annotations = list(view.select(lemma_type_name))
    sentences = sorted(view.select(sentence_type_name), key=lambda s: (s.begin, s.end))

    lemma_by_span: dict[tuple[int, int], str] = {
        (int(lemma_ann.begin), int(lemma_ann.end)): lemma_ann.value
        for lemma_ann in lemma_annotations
    }

    def sentence_span_for(begin: int, end: int) -> tuple[str, int]:
        for sent in sentences:
            sent_begin, sent_end = int(sent.begin), int(sent.end)
            if sent_begin <= begin and end <= sent_end:
                return sofa_string[sent_begin:sent_end], sent_begin
        return sofa_string[begin:end], begin

    occurrences: dict[str, NounOccurrence] = {}

    for pos_ann in pos_annotations:
        pos_value = getattr(pos_ann, "PosValue", None)
        if pos_value not in noun_pos_values:
            continue

        begin, end = int(pos_ann.begin), int(pos_ann.end)
        surface = sofa_string[begin:end]
        lemma = lemma_by_span.get((begin, end), surface)

        sentence_text, sent_start = sentence_span_for(begin, end)
        local_start = begin - sent_start
        local_end = local_start + len(surface)

        occ = occurrences.setdefault(lemma, NounOccurrence(lemma=lemma))
        occ.surface_forms.add(surface)
        occ.count += 1
        occ.contexts.append(
            NounContext(
                sentence_text=sentence_text,
                start_char=local_start,
                end_char=local_end,
                surface=surface,
            )
        )

    return occurrences


def detect_abbreviations(
    text: str,
    abbreviation_map: dict[str, set[str]] = ABBREVIATION_MAP,
) -> list[AbbreviationCandidate]:
    """Erkennt einfache Abkürzungsrelationen im Text.

    Erkennt:
    - Langform (Kurzform), z.B. Europäische Union (EU)
    - Kurzform (Langform), z.B. EU (Europäische Union)
    - bekannte Paare aus ABBREVIATION_MAP
    """
    candidates: list[AbbreviationCandidate] = []
    seen: set[tuple[str, str, str]] = set()

    def add_candidate(
        short_form: str,
        long_form: str | None,
        source: str,
        confidence: float,
        start_char: int | None = None,
        end_char: int | None = None,
    ) -> None:
        short_form_clean = short_form.strip()
        long_form_clean = long_form.strip() if long_form else None

        if not short_form_clean:
            return
        # Für alle Quellen außer der Standalone-Erkennung ist eine leere
        # Langform ein Fehlerfall (die anderen Muster sollen ja gerade Paare
        # liefern) - beim Standalone-Fall ist "keine Langform gefunden" der
        # Punkt der Übung.
        if source != "standalone_acronym" and not long_form_clean:
            return

        key = (short_form_clean, long_form_clean or "", source)

        if key in seen:
            return

        seen.add(key)

        candidates.append(
            AbbreviationCandidate(
                short_form=short_form_clean,
                long_form=long_form_clean,
                source=source,
                confidence=confidence,
                start_char=start_char,
                end_char=end_char,
            )
        )

    # Fall 1: Langform (Kurzform), z.B. Europäische Union (EU)
    for match in LONG_SHORT_PATTERN.finditer(text):
        long_form = match.group("long")
        short_form = match.group("short")

        add_candidate(
            short_form=short_form,
            long_form=long_form,
            source="regex:long(short)",
            confidence=0.98,
            start_char=match.start(),
            end_char=match.end(),
        )

    # Fall 2: Kurzform (Langform), z.B. EU (Europäische Union)
    for match in SHORT_LONG_PATTERN.finditer(text):
        short_form = match.group("short")
        long_form = match.group("long")

        add_candidate(
            short_form=short_form,
            long_form=long_form,
            source="regex:short(long)",
            confidence=0.98,
            start_char=match.start(),
            end_char=match.end(),
        )

    # Fall 3: bekannte Abkürzungen aus Liste
    for short_form, long_forms in abbreviation_map.items():
        if short_form not in text:
            continue

        for long_form in long_forms:
            if long_form in text:
                add_candidate(
                    short_form=short_form,
                    long_form=long_form,
                    source="abbreviation_map",
                    confidence=0.90,
                )

    # Fall 4: alleinstehende Akronyme ohne erkennbare Auflösung im Text
    # (z.B. "SPD", "CDU"). Nur melden, wenn dieselbe Kurzform nicht schon
    # über Fall 1-3 mit einer Langform gefunden wurde - sonst gäbe es
    # doppelte Einträge für dieselbe Abkürzung.
    already_resolved = {c.short_form for c in candidates}

    for match in STANDALONE_ACRONYM_PATTERN.finditer(text):
        short_form = match.group(0)

        if short_form in already_resolved:
            continue

        add_candidate(
            short_form=short_form,
            long_form=None,
            source="standalone_acronym",
            confidence=0.5,
            start_char=match.start(),
            end_char=match.end(),
        )
        already_resolved.add(short_form)

    # Fall 5: einzelne Buchstaben (z.B. Einheiten wie "s", "m", "g")
    for match in SINGLE_LETTER_PATTERN.finditer(text):
        short_form = match.group(0)

        if short_form in already_resolved:
            continue

        add_candidate(
            short_form=short_form,
            long_form=None,
            source="single_letter",
            confidence=0.3,
            start_char=match.start(),
            end_char=match.end(),
        )
        already_resolved.add(short_form)

    # Fall 6: gemischte Groß-/Kleinschreibung (z.B. "kW", "kWh", "mAh")
    for match in MIXED_CASE_PATTERN.finditer(text):
        short_form = match.group(0)

        if short_form in already_resolved:
            continue

        add_candidate(
            short_form=short_form,
            long_form=None,
            source="mixed_case",
            confidence=0.4,
            start_char=match.start(),
            end_char=match.end(),
        )
        already_resolved.add(short_form)

    return candidates


def get_synonyms_openthesaurus(word: str, timeout: float = 5.0) -> set[str]:
    """Fragt OpenThesaurus nach Synonymen eines Wortes ab."""
    try:
        r = requests.get(
            OPENTHESAURUS_URL,
            params={"q": word, "format": "application/json"},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError) as e:
        print(
            f"  [Warnung] OpenThesaurus-Abfrage für '{word}' fehlgeschlagen: {e}",
            file=sys.stderr,
        )
        return set()

    synonyms = set()

    for synset in data.get("synsets", []):
        for term in synset.get("terms", []):
            synonyms.add(term["term"] if "term" in term else term.get("word", ""))

    synonyms.discard(word)
    synonyms.discard("")

    return synonyms


@torch.inference_mode()
def bert_context_vector(
    sentence: str,
    start_char: int,
    end_char: int,
    model_name: str = DEFAULT_BERT_MODEL,
) -> torch.Tensor | None:
    """Berechnet einen BERT-Kontextvektor für genau ein Nomen im Satz."""
    tokenizer, model = get_bert(model_name)

    encoded = tokenizer(
        sentence,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=256,
    )

    offsets = encoded.pop("offset_mapping")[0]

    outputs = model(**encoded)
    hidden = outputs.last_hidden_state[0]

    token_indices = []

    for idx, offset in enumerate(offsets):
        token_start = int(offset[0])
        token_end = int(offset[1])

        if token_start == token_end == 0:
            continue

        overlaps = token_end > start_char and token_start < end_char

        if overlaps:
            token_indices.append(idx)

    if not token_indices:
        return None

    vec = hidden[token_indices].mean(dim=0)
    vec = F.normalize(vec, p=2, dim=0)

    return vec


def compute_bert_lemma_vectors(
    nouns: dict[str, NounOccurrence],
    model_name: str = DEFAULT_BERT_MODEL,
    max_contexts_per_lemma: int = 8,
) -> dict[str, torch.Tensor]:
    """Berechnet pro Lemma einen durchschnittlichen BERT-Kontextvektor."""
    lemma_vectors: dict[str, torch.Tensor] = {}

    for lemma, occ in nouns.items():
        vectors = []

        for ctx in occ.contexts[:max_contexts_per_lemma]:
            vec = bert_context_vector(
                ctx.sentence_text,
                ctx.start_char,
                ctx.end_char,
                model_name=model_name,
            )

            if vec is not None:
                vectors.append(vec)

        if vectors:
            lemma_vec = torch.stack(vectors).mean(dim=0)
            lemma_vec = F.normalize(lemma_vec, p=2, dim=0)
            lemma_vectors[lemma] = lemma_vec

    return lemma_vectors


def cosine_similarity_torch(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """Cosine Similarity zweier bereits normierter Vektoren."""
    return float(torch.dot(vec_a, vec_b).item())


def find_synonym_candidates(
    nouns: dict[str, NounOccurrence],
    use_thesaurus: bool = True,
    use_embeddings: bool = True,
    use_bert: bool = False,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    bert_similarity_threshold: float = BERT_SIMILARITY_THRESHOLD,
    bert_model_name: str = DEFAULT_BERT_MODEL,
) -> list[SynonymCandidate]:
    """Prüft alle Nomen-Paare auf mögliche Synonymie."""
    nlp = get_nlp()
    lemmas = list(nouns.keys())
    candidates: list[SynonymCandidate] = []

    thesaurus_cache: dict[str, set[str]] = {}

    if use_thesaurus:
        for lemma in lemmas:
            thesaurus_cache[lemma] = get_synonyms_openthesaurus(lemma)
            time.sleep(REQUEST_DELAY)

    bert_vectors: dict[str, torch.Tensor] = {}

    if use_bert:
        bert_vectors = compute_bert_lemma_vectors(
            nouns,
            model_name=bert_model_name,
        )

    for i, lemma_a in enumerate(lemmas):
        for lemma_b in lemmas[i + 1:]:
            sources = []

            is_thesaurus_match = use_thesaurus and (
                lemma_b in thesaurus_cache.get(lemma_a, set())
                or lemma_a in thesaurus_cache.get(lemma_b, set())
            )

            if is_thesaurus_match:
                sources.append("thesaurus")

            spacy_sim = None
            is_embedding_match = False

            if use_embeddings:
                doc_a, doc_b = nlp(lemma_a), nlp(lemma_b)

                if doc_a.has_vector and doc_b.has_vector:
                    spacy_sim = doc_a.similarity(doc_b)
                    is_embedding_match = spacy_sim >= similarity_threshold

            if is_embedding_match:
                sources.append("embedding")

            bert_sim = None
            is_bert_match = False

            if use_bert:
                vec_a = bert_vectors.get(lemma_a)
                vec_b = bert_vectors.get(lemma_b)

                if vec_a is not None and vec_b is not None:
                    bert_sim = cosine_similarity_torch(vec_a, vec_b)
                    is_bert_match = bert_sim >= bert_similarity_threshold

            if is_bert_match:
                sources.append("bert")

            if sources:
                source = "+".join(sources)
                general_similarity = spacy_sim if spacy_sim is not None else bert_sim

                candidates.append(
                    SynonymCandidate(
                        lemma_a=lemma_a,
                        lemma_b=lemma_b,
                        source=source,
                        similarity=general_similarity,
                        spacy_similarity=spacy_sim,
                        bert_similarity=bert_sim,
                    )
                )

    return candidates


def check_text(
    text: str,
    use_thesaurus: bool = True,
    use_embeddings: bool = True,
    use_bert: bool = False,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    bert_similarity_threshold: float = BERT_SIMILARITY_THRESHOLD,
    bert_model_name: str = DEFAULT_BERT_MODEL,
) -> dict:
    """Führt die vollständige Prüfung auf einem rohen Text-String durch."""
    nouns = extract_nouns(text)

    candidates = find_synonym_candidates(
        nouns,
        use_thesaurus=use_thesaurus,
        use_embeddings=use_embeddings,
        use_bert=use_bert,
        similarity_threshold=similarity_threshold,
        bert_similarity_threshold=bert_similarity_threshold,
        bert_model_name=bert_model_name,
    )

    abbreviations = detect_abbreviations(text)

    return {
        "nouns": nouns,
        "candidates": candidates,
        "abbreviations": abbreviations,
        "num_nouns": len(nouns),
        "num_candidates": len(candidates),
        "num_abbreviations": len(abbreviations),
    }


def check_cas(
    cas: Any,
    view_name: str = "_InitialView",
    pos_type_name: str = DEFAULT_POS_TYPE_NAME,
    lemma_type_name: str = DEFAULT_LEMMA_TYPE_NAME,
    sentence_type_name: str = DEFAULT_SENTENCE_TYPE_NAME,
    use_thesaurus: bool = True,
    use_embeddings: bool = True,
    use_bert: bool = False,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    bert_similarity_threshold: float = BERT_SIMILARITY_THRESHOLD,
    bert_model_name: str = DEFAULT_BERT_MODEL,
) -> dict:
    """Führt die vollständige Prüfung direkt auf einer CAS durch."""
    try:
        view = cas.get_view(view_name)
    except Exception:
        view = cas

    sofa_string = getattr(view, "sofa_string", None) or ""

    nouns = extract_nouns_from_cas(
        cas,
        view_name=view_name,
        pos_type_name=pos_type_name,
        lemma_type_name=lemma_type_name,
        sentence_type_name=sentence_type_name,
    )

    candidates = find_synonym_candidates(
        nouns,
        use_thesaurus=use_thesaurus,
        use_embeddings=use_embeddings,
        use_bert=use_bert,
        similarity_threshold=similarity_threshold,
        bert_similarity_threshold=bert_similarity_threshold,
        bert_model_name=bert_model_name,
    )

    abbreviations = detect_abbreviations(sofa_string)

    return {
        "nouns": nouns,
        "candidates": candidates,
        "abbreviations": abbreviations,
        "num_nouns": len(nouns),
        "num_candidates": len(candidates),
        "num_abbreviations": len(abbreviations),
    }


def print_report(report: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"Gefundene Nomen (Lemmata): {report['num_nouns']}")
    print(f"Mögliche Synonym-Inkonsistenzen: {report['num_candidates']}")
    print(f"{'=' * 60}\n")

    if not report["candidates"]:
        print("Keine Kandidaten gefunden - Benennung scheint konsistent.")
    else:
        for cand in report["candidates"]:
            occ_a = report["nouns"][cand.lemma_a]
            occ_b = report["nouns"][cand.lemma_b]

            print(f"⚠  {cand}")
            print(f"   '{cand.lemma_a}': {occ_a.count}x, Formen: {occ_a.surface_forms}")
            print(f"   '{cand.lemma_b}': {occ_b.count}x, Formen: {occ_b.surface_forms}")
            print()

    abbreviations = report.get("abbreviations", [])

    if abbreviations:
        print("\nGefundene Abkürzungen:")
        print("-" * 60)

        for abbr in abbreviations:
            print(f"ℹ  {abbr}")


if __name__ == "__main__":
    text = (
        "Die Christlich Demokratische Union (CDU) ist eine Partei. "
        "Die CDU wurde 1945 gegründet. "
        "Die SPD ist eine andere Partei. "
        "Die Europäische Union (EU) macht Regeln. "
        "Die EU hat viele Mitgliedstaaten. "
        "Maria studiert an der FernUniversität in Hagen. "
        "Viele Menschen sagen auch FernUni."
    )

    report = check_text(
        text,
        use_thesaurus=False,
        use_embeddings=False,
        use_bert=False,
    )

    print_report(report)