from pathlib import Path
from typing import Any

import streamlit as st
from py_lift.utils.core import load_cas_from_xmi_with_lift_ts
from itertools import zip_longest

import pandas as pd
import html
import textwrap

from py_lift.extractors import FE_AverageTokenLength, FE_TokenCount

from features.run_synonyms import check_text as check_synonym_consistency

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_NORMALIZED_DIR = Path("cache") / "normalized"

AVAILABLE_VIEWS = ["NormalizedView", "_InitialView"]
DEFAULT_VIEW_NAME = "_InitialView"

TOKEN_TYPE_NAME = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"

ALIGNMENT_VIEWS = {
    "initial": "_InitialView",
    "normalized": "NormalizedView",
}

# TODO: Add the feature names to display later.

FEATURE_NAMES_TO_SHOW = [
    "Token_length_mean",
    "Token_COUNT",
]

DEFAULT_FEATURE_VIEW = "_InitialView"

# Only features listed here are computed from another view.
# All other features use DEFAULT_FEATURE_VIEW.
FEATURE_VIEW_OVERRIDES = {
    # "Token_length_mean": "NormalizedView",
    # "Token_COUNT": "NormalizedView",
}

# Ordnet jedem anzuzeigenden Feature den passenden Extractor zu.
FEATURE_EXTRACTORS: dict[str, Any] = {
    "Token_length_mean": FE_AverageTokenLength(),
    "Token_COUNT": FE_TokenCount(),
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_view_name_for_feature(feature_name: str) -> str:
    """Return the view that should be used for a given feature."""
    return FEATURE_VIEW_OVERRIDES.get(feature_name, DEFAULT_FEATURE_VIEW)


def render_token_alignment_html(
        alignment_rows: list[dict[str, Any]],
        *,
        show_only_changed: bool = False,
) -> str:
    """Render token alignment as horizontally readable HTML.

    Each token pair is displayed as a small card:
    - top: initial token
    - bottom: normalized token

    Changed tokens are highlighted.
    """
    token_cards: list[str] = []

    for row in alignment_rows:
        initial_token = str(row.get("Initial token", ""))
        normalized_token = str(row.get("Normalized token", ""))
        changed = bool(row.get("Changed", False))

        if show_only_changed and not changed:
            continue

        initial_token_escaped = html.escape(initial_token)
        normalized_token_escaped = html.escape(normalized_token)

        changed_class = "changed" if changed else "unchanged"

        card = (
            f'<div class="token-card {changed_class}">'
            f'<div class="initial-token">{initial_token_escaped}</div>'
            f'<div class="normalized-token">{normalized_token_escaped}</div>'
            f'</div>'
        )

        token_cards.append(card)

    if not token_cards:
        return "<div class='alignment-empty'>No tokens to display.</div>"

    cards_html = "".join(token_cards)

    style = """<style>
.alignment-container {
    display: flex;
    flex-wrap: wrap;
    column-gap: 0.4rem;
    row-gap: 1rem;
    align-items: flex-start;
    line-height: 1.2;
    margin-top: 0.5rem;
}

.token-card {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-width: 3rem;
    padding: 0.35rem 0.5rem;

    border: 1px solid #dddddd;
    border-radius: 0.5rem;
    background-color: #f8f9fa;
    font-family: monospace;
    overflow-wrap: anywhere;
}

.token-card.changed {
    background-color: #fff3cd;
    border-color: #ffca2c;
}

.token-card.unchanged {
    opacity: 0.85;
}

.initial-token {
    color: #333333;
    font-weight: 600;
    border-bottom: 1px solid #cccccc;
    padding-bottom: 0.15rem;
    margin-bottom: 0.15rem;
    text-align: center;
}

.normalized-token {
    color: #0b5ed7;
    font-weight: 600;
    text-align: center;
}

.alignment-empty {
    color: #666666;
    font-style: italic;
    margin-top: 0.5rem;
}
</style>"""

    full_html = style + '<div class="alignment-container">' + cards_html + '</div>'

    return full_html


def get_view_or_default(cas: Any, view_name: str) -> Any:
    """Return a CAS view. Fall back to the CAS itself for _InitialView."""
    try:
        return cas.get_view(view_name)
    except Exception:
        if view_name == "_InitialView":
            return cas
        raise


def extract_tokens_from_view(
        cas: Any,
        view_name: str,
        token_type_name: str = TOKEN_TYPE_NAME,
) -> list[dict[str, Any]]:
    """Extract token texts and offsets from a CAS view."""
    view = get_view_or_default(cas, view_name)

    sofa_string = getattr(view, "sofa_string", None) or ""

    tokens = list(view.select(token_type_name))
    tokens = sorted(tokens, key=lambda t: (t.begin, t.end))

    result: list[dict[str, Any]] = []

    for token in tokens:
        begin = int(token.begin)
        end = int(token.end)

        result.append(
            {
                "text": sofa_string[begin:end],
                "begin": begin,
                "end": end,
            }
        )

    return result


@st.cache_data(show_spinner=False)
def load_token_alignment_data(xmi_path_str: str) -> list[dict[str, Any]]:
    """Load initial and normalized tokens and align them by token index."""
    xmi_path = Path(xmi_path_str)
    cas = load_cas_from_xmi_with_lift_ts(xmi_path)

    initial_tokens = extract_tokens_from_view(
        cas,
        ALIGNMENT_VIEWS["initial"],
    )

    normalized_tokens = extract_tokens_from_view(
        cas,
        ALIGNMENT_VIEWS["normalized"],
    )

    rows: list[dict[str, Any]] = []

    for index, (initial_token, normalized_token) in enumerate(
            zip_longest(initial_tokens, normalized_tokens),
            start=1,
    ):
        initial_text = initial_token["text"] if initial_token is not None else ""
        normalized_text = normalized_token["text"] if normalized_token is not None else ""

        rows.append(
            {
                "#": index,
                "Initial token": initial_text,
                "Normalized token": normalized_text,
                "Changed": initial_text != normalized_text,
                "Initial begin": initial_token["begin"] if initial_token is not None else None,
                "Initial end": initial_token["end"] if initial_token is not None else None,
                "Normalized begin": normalized_token["begin"] if normalized_token is not None else None,
                "Normalized end": normalized_token["end"] if normalized_token is not None else None,
            }
        )

    return rows


def highlight_changed_tokens(row: pd.Series) -> list[str]:
    """Highlight rows where the normalized token differs from the initial token."""
    if bool(row["Changed"]):
        return [
            "background-color: #fff3cd; font-weight: bold;"
            for _ in row
        ]

    return ["" for _ in row]


def get_corpus_dirs(base_dir: Path) -> list[Path]:
    """Return all corpus directories inside cache/normalized."""
    if not base_dir.exists():
        return []

    return sorted([p for p in base_dir.iterdir() if p.is_dir()])


def get_cas_files(corpus_dir: Path) -> list[Path]:
    """Return all CAS/XMI files in a corpus directory."""
    return sorted([
        p for p in corpus_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".xmi"
    ])


def compute_selected_features_from_cas(cas: Any) -> dict[str, str]:
    """Berechnet die gewünschten Features live auf der CAS.

    Statt sich auf bereits in der XMI vorhandene Feature-Annotationen zu
    verlassen (die im aktuellen Cache nicht persistiert werden), wird der
    passende Extractor pro Feature direkt auf der jeweils konfigurierten
    View ausgeführt und der Wert anschließend ausgelesen.
    """
    features: dict[str, str] = {}

    if not FEATURE_NAMES_TO_SHOW:
        return features

    for feature_name in FEATURE_NAMES_TO_SHOW:
        extractor = FEATURE_EXTRACTORS.get(feature_name)
        if extractor is None:
            continue

        view_name = get_view_name_for_feature(feature_name)
        view = get_view_or_default(cas, view_name)

        try:
            extractor.extract(view)
        except Exception:
            continue

        value = None
        try:
            for feature in view.select("org.lift.type.FeatureAnnotationNumeric"):
                if feature.get("name") == feature_name:
                    value = feature.get("value")
                    break
        except Exception:
            pass

        if value is not None:
            features[feature_name] = str(value)

    return features


@st.cache_data(show_spinner=False)
def load_cas_display_data(
        xmi_path_str: str,
        view_name: str = DEFAULT_VIEW_NAME,
) -> dict[str, Any]:
    """Load a CAS from XMI and return display data.

    Returns:
        Dictionary with:
        - ``sofa_string``
        - ``features``
    """
    xmi_path = Path(xmi_path_str)
    cas = load_cas_from_xmi_with_lift_ts(xmi_path)

    try:
        view = cas.get_view(view_name)
    except Exception:
        view = cas

    sofa_string = getattr(view, "sofa_string", None) or ""

    features = compute_selected_features_from_cas(cas)

    return {
        "sofa_string": sofa_string,
        "features": features,
    }


def reset_index_if_corpus_changed(selected_corpus_name: str) -> None:
    """Reset file index when the selected corpus changes."""
    previous_corpus = st.session_state.get("previous_corpus_name")

    if previous_corpus != selected_corpus_name:
        st.session_state["file_index"] = 0
        st.session_state["previous_corpus_name"] = selected_corpus_name


def go_previous(num_files: int, file_names: list[str]) -> None:
    """Move to previous file and keep the file dropdown in sync."""
    if num_files == 0:
        return

    current_index = st.session_state.get("file_index", 0)
    new_index = max(0, current_index - 1)

    st.session_state["file_index"] = new_index
    st.session_state["selected_file_name"] = file_names[new_index]


def go_next(num_files: int, file_names: list[str]) -> None:
    """Move to next file and keep the file dropdown in sync."""
    if num_files == 0:
        return

    current_index = st.session_state.get("file_index", 0)
    new_index = min(num_files - 1, current_index + 1)

    st.session_state["file_index"] = new_index
    st.session_state["selected_file_name"] = file_names[new_index]


def _round2(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


@st.cache_data(show_spinner=False)
def load_synonym_report(
    sofa_string: str,
    use_thesaurus: bool = True,
    use_embeddings: bool = True,
    use_bert: bool = True,
    similarity_threshold: float = 0.75,
    bert_similarity_threshold: float = 0.82,
) -> dict:
    if not sofa_string:
        return {"candidates": []}

    return check_synonym_consistency(
        sofa_string,
        use_thesaurus=use_thesaurus,
        use_embeddings=use_embeddings,
        use_bert=use_bert,
        similarity_threshold=similarity_threshold,
        bert_similarity_threshold=bert_similarity_threshold,
    )


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="CAS Browser",
        layout="wide",
    )

    st.title("CAS Browser")
    st.write("Browse normalized CAS files from `cache/normalized/`.")

    if "file_index" not in st.session_state:
        st.session_state["file_index"] = 0

    base_dir = CACHE_NORMALIZED_DIR

    if not base_dir.exists():
        st.error(f"Directory does not exist: `{base_dir.resolve()}`")
        return

    corpus_dirs = get_corpus_dirs(base_dir)

    if not corpus_dirs:
        st.warning(f"No corpus directories found in `{base_dir.resolve()}`.")
        return

    corpus_names = [p.name for p in corpus_dirs]

    # -----------------------------------------------------------------------
    # Sidebar: corpus selection
    # -----------------------------------------------------------------------

    with st.sidebar:
        selected_corpus_name = st.selectbox(
            "Corpus",
            options=corpus_names,
            key="selected_corpus_name",
        )

    # Reset file selection when the corpus changes.
    previous_corpus = st.session_state.get("previous_corpus_name")

    if previous_corpus != selected_corpus_name:
        st.session_state["file_index"] = 0
        st.session_state["previous_corpus_name"] = selected_corpus_name

        # Remove old file selection because the available files changed.
        if "selected_file_name" in st.session_state:
            del st.session_state["selected_file_name"]

    selected_corpus_dir = base_dir / selected_corpus_name
    cas_files = get_cas_files(selected_corpus_dir)

    if not cas_files:
        st.warning(f"No `.xmi` files found in `{selected_corpus_dir.resolve()}`.")
        return

    file_names = [p.name for p in cas_files]
    num_files = len(cas_files)

    # Keep the file index valid.
    st.session_state["file_index"] = min(
        max(st.session_state["file_index"], 0),
        num_files - 1,
    )

    # Initialize selected_file_name if necessary.
    if (
            "selected_file_name" not in st.session_state
            or st.session_state["selected_file_name"] not in file_names
    ):
        st.session_state["selected_file_name"] = file_names[st.session_state["file_index"]]

    # Initialize view selection if necessary.
    if (
            "view_name" not in st.session_state
            or st.session_state["view_name"] not in AVAILABLE_VIEWS
    ):
        st.session_state["view_name"] = DEFAULT_VIEW_NAME

    # -----------------------------------------------------------------------
    # Sidebar: file and view selection
    # -----------------------------------------------------------------------

    with st.sidebar:
        selected_file_name = st.selectbox(
            "File",
            options=file_names,
            key="selected_file_name",
        )

        view_name = st.selectbox(
            "CAS view",
            options=AVAILABLE_VIEWS,
            key="view_name",
            help="Select which CAS view should be displayed.",
        )

    # Update current index based on file dropdown.
    current_index = file_names.index(selected_file_name)
    st.session_state["file_index"] = current_index
    current_file = cas_files[current_index]

    # -----------------------------------------------------------------------
    # Navigation controls
    # -----------------------------------------------------------------------

    col_prev, col_info, col_next = st.columns([1, 2, 1])

    with col_prev:
        st.button(
            "← Previous",
            on_click=go_previous,
            args=(num_files, file_names),
            disabled=current_index == 0,
            use_container_width=True,
        )

    with col_info:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <strong>File {current_index + 1} of {num_files}</strong><br>
                <code>{current_file.name}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_next:
        st.button(
            "Next →",
            on_click=go_next,
            args=(num_files, file_names),
            disabled=current_index == num_files - 1,
            use_container_width=True,
        )

    st.divider()

    # -----------------------------------------------------------------------
    # Load CAS display data
    # -----------------------------------------------------------------------

    with st.spinner("Loading CAS..."):
        current_file_str = str(current_file.resolve())

        display_data = load_cas_display_data(
            current_file_str,
            view_name=view_name,
        )

        alignment_rows = load_token_alignment_data(current_file_str)

    sofa_string = display_data["sofa_string"]
    features = display_data["features"]

    with st.spinner("Checking noun consistency including BERT..."):
        synonym_report = load_synonym_report(
            sofa_string,
            use_thesaurus=True,
            use_embeddings=True,
            use_bert=True,
            similarity_threshold=0.75,
            bert_similarity_threshold=0.82,
        )

    # -----------------------------------------------------------------------
    # Main content: left column = text, right column = features
    # -----------------------------------------------------------------------

    left_col, right_col = st.columns([2, 1])

    with left_col:
        text_tab, alignment_tab = st.tabs(["Sofa string", "Token alignment"])

        with text_tab:
            st.subheader("Sofa string")

            if sofa_string:
                st.text_area(
                    label="",
                    value=sofa_string,
                    height=600,
                )
            else:
                st.info("The selected view does not contain a sofa string.")

        with alignment_tab:
            st.subheader("Initial vs. Normalized")

            if not alignment_rows:
                st.info("No token alignment available.")
            else:
                changed_count = sum(1 for row in alignment_rows if row.get("Changed"))
                total_count = len(alignment_rows)

                st.caption(f"Changed tokens: {changed_count} / {total_count}")

                show_only_changed = st.checkbox(
                    "Show only changed tokens",
                    value=False,
                )

                alignment_html = render_token_alignment_html(
                    alignment_rows,
                    show_only_changed=show_only_changed,
                )

                st.markdown(
                    alignment_html,
                    unsafe_allow_html=True,
                )

    with right_col:
        st.subheader("Features")

        word_tab, sentence_tab = st.tabs(["Word", "Sentence"])

        if not FEATURE_NAMES_TO_SHOW:
            with word_tab:
                st.info("No features selected yet.")
            with sentence_tab:
                st.info("No features selected yet.")
        elif not features:
            with word_tab:
                st.warning("No selected features found in this CAS.")
            with sentence_tab:
                st.warning("No selected features found in this CAS.")
        else:
            with word_tab:
                if "Token_length_mean" in features:
                    with st.expander("Token length (mean)"):
                        value = features["Token_length_mean"]
                        st.table(
                            pd.DataFrame(
                                {
                                    "Feature": ["Token_length_mean"],
                                    "Value": [_round2(value)],
                                    "Expected": [round(0, 2)],
                                }
                            )
                        )

                if "Token_COUNT" in features:
                    with st.expander("Token count"):
                        value = features["Token_COUNT"]
                        st.table(
                            pd.DataFrame(
                                {
                                    "Feature": ["Token_COUNT"],
                                    "Value": [_round2(value)],
                                    "Expected": [round(0, 2)],
                                }
                            )
                        )

                with st.expander("Synonyme"):
                    candidates = synonym_report["candidates"]

                    if not candidates:
                        st.write("Keine möglichen Synonym-Inkonsistenzen gefunden.")
                    else:
                        candidates_df = pd.DataFrame(
                            {
                                "Nomen A": [c.lemma_a for c in candidates],
                                "Nomen B": [c.lemma_b for c in candidates],
                                "Quelle": [c.source for c in candidates],
                                "Gesamt-Ähnlichkeit": [
                                    round(c.similarity, 2) if getattr(c, "similarity", None) is not None else None
                                    for c in candidates
                                ],
                                "spaCy-Ähnlichkeit": [
                                    round(getattr(c, "spacy_similarity", None), 2)
                                    if getattr(c, "spacy_similarity", None) is not None
                                    else None
                                    for c in candidates
                                ],
                                "BERT-Ähnlichkeit": [
                                    round(getattr(c, "bert_similarity", None), 2)
                                    if getattr(c, "bert_similarity", None) is not None
                                    else None
                                    for c in candidates
                                ],
                            }
                        )

                        st.table(candidates_df)
                with st.expander("Abkürzungen"):
                    abbreviations = synonym_report.get("abbreviations", [])

                    if not abbreviations:
                        st.write("Keine Abkürzungen gefunden.")
                    else:
                        abbreviations_df = pd.DataFrame(
                            {
                                "Kurzform": [a.short_form for a in abbreviations],
                                "Langform": [a.long_form for a in abbreviations],
                                "Quelle": [a.source for a in abbreviations],
                                "Konfidenz": [round(a.confidence, 2) for a in abbreviations],
                            }
                        )

                        st.table(abbreviations_df)

            with sentence_tab:
                # Hier später deine Sentence-Features analog ergänzen, z.B.:
                # if "Some_Sentence_Feature" in features:
                #     with st.expander("Beschreibung"):
                #         value = features["Some_Sentence_Feature"]
                #         st.write(f"Wert: {_round2(value)}")
                st.info("No sentence-level features found in this CAS.")


if __name__ == "__main__":
    main()