from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Any

from py_lift.utils.core import load_cas_from_xmi_with_lift_ts

import utils
from utils import (
    load_or_create_cas,
    get_cache_dir_for_folder,
)


class CasNormalizer:
    """Create CAS files with an additional normalized view.

    The normalized CAS files are written to

        cache/normalized/<corpus_name>/<filename>.xmi

    The existing preprocessing cache from ``utils.py`` is reused.
    """

    def __init__(
        self,
        *,
        source_view_name: str = "_InitialView",
        target_view_name: str = "NormalizedView",
        token_function: Optional[Callable[[str], str]] = None,
        token_type_name: str = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token",
        sentence_type_name: str = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence",
        normalized_cache_root: Optional[Path] = None,
        overwrite: bool = False,
        copy_tokens: bool = True,
        copy_sentences: bool = True,
        verbose: bool = True,
    ) -> None:
        self.source_view_name = source_view_name
        self.target_view_name = target_view_name

        # If no function is provided, tokens remain unchanged.
        self.token_function = token_function or self.default_token_function

        self.token_type_name = token_type_name
        self.sentence_type_name = sentence_type_name

        self.copy_tokens = copy_tokens
        self.copy_sentences = copy_sentences

        self.overwrite = overwrite
        self.verbose = verbose

        self.normalized_cache_root = (
            Path(normalized_cache_root)
            if normalized_cache_root is not None
            else Path(utils.__file__).parent / "cache" / "normalized"
        )

    # ------------------------------------------------------------------
    # Small helper methods
    # ------------------------------------------------------------------

    def default_token_function(self, token_text: str) -> str:
        """Default normalization used when no token_function is provided."""
        return token_text

    def _log(self, message: str) -> None:
        """Print a message if verbose=True."""
        if self.verbose:
            print(message)

    def _write_cas(self, cas: Any, path: Path) -> None:
        """Write a CAS object as an XMI file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        cas.to_xmi(str(path))

    def _iter_files(
        self,
        corpus_dir: Path,
        suffixes: Optional[tuple[str, ...]] = None,
    ) -> list[Path]:
        """Return all files in a corpus directory that should be processed."""
        files: list[Path] = []

        normalized_suffixes = None
        if suffixes is not None:
            normalized_suffixes = tuple(s.lower() for s in suffixes)

        for file in sorted(corpus_dir.iterdir()):
            if not file.is_file():
                continue

            if normalized_suffixes is not None and file.suffix.lower() not in normalized_suffixes:
                self._log(f"[SKIP] {file.name} because of suffix {file.suffix}")
                continue

            files.append(file)

        return files

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def normalize_corpora(
        self,
        corpora: dict[str, Path],
        *,
        use_input_cache: bool = True,
        suffixes: Optional[tuple[str, ...]] = None,
    ) -> dict[str, list[Path]]:
        """Normalize multiple corpora.

        Args:
            corpora:
                Mapping from corpus name to corpus path, e.g.
                ``{"TIGER_ORIG": TIGER_ORIG, ...}``.
            use_input_cache:
                If ``True``, reuse the existing preprocessing cache from ``utils.py``.
            suffixes:
                Optional file suffix filter, e.g. ``(".txt",)``.
                If ``None``, all files in the corpus directory are processed.

        Returns:
            Mapping from corpus name to generated XMI files.
        """
        result: dict[str, list[Path]] = {}

        for corpus_name, corpus_dir in corpora.items():
            result[corpus_name] = self.normalize_folder(
                corpus_name=corpus_name,
                corpus_dir=corpus_dir,
                use_input_cache=use_input_cache,
                suffixes=suffixes,
            )

        return result

    def normalize_folder(
        self,
        *,
        corpus_name: str,
        corpus_dir: Path,
        use_input_cache: bool = True,
        suffixes: Optional[tuple[str, ...]] = None,
    ) -> list[Path]:
        """Normalize all files in a single corpus directory.

        The output is written to:

            cache/normalized/<corpus_name>/<filename>.xmi
        """
        corpus_dir = Path(corpus_dir)

        normalized_dir = self.get_normalized_corpus_dir(corpus_name)
        normalized_dir.mkdir(parents=True, exist_ok=True)

        self._log(f"[INFO] Corpus: {corpus_name}")
        self._log(f"[INFO] Corpus path: {corpus_dir.resolve()}")
        self._log(f"[INFO] Normalized cache: {normalized_dir.resolve()}")

        input_cache_dir = None
        if use_input_cache:
            input_cache_dir = get_cache_dir_for_folder(corpus_dir)
            input_cache_dir.mkdir(parents=True, exist_ok=True)
            self._log(f"[INFO] Input cache: {input_cache_dir.resolve()}")

        result_paths: list[Path] = []

        files = self._iter_files(corpus_dir, suffixes=suffixes)
        self._log(f"[INFO] Found files: {len(files)}")

        for file in files:
            normalized_xmi_path = self.get_normalized_file_path(
                corpus_name=corpus_name,
                source_file=file,
            )

            if normalized_xmi_path.is_file() and not self.overwrite:
                self._log(f"[CACHE] Reusing existing file: {normalized_xmi_path.name}")
                result_paths.append(normalized_xmi_path)
                continue

            self._log(f"[RUN] Normalizing: {file.name}")

            cas = load_or_create_cas(
                file=file,
                use_cache=use_input_cache,
                cache_dir=input_cache_dir,
            )

            cas = self.normalize_cas(cas)

            self._write_cas(cas, normalized_xmi_path)

            self._log(f"[WRITE] {normalized_xmi_path.resolve()}")

            result_paths.append(normalized_xmi_path)

        self._log(f"[DONE] {corpus_name}: {len(result_paths)} files")

        return result_paths

    def load_normalized_cas(
        self,
        *,
        corpus_name: str,
        source_file: Path,
    ) -> Any:
        """Load an already normalized CAS from cache/normalized."""
        normalized_xmi_path = self.get_normalized_file_path(
            corpus_name=corpus_name,
            source_file=source_file,
        )

        if not normalized_xmi_path.is_file():
            raise FileNotFoundError(
                f"Normalized CAS not found: {normalized_xmi_path}"
            )

        return load_cas_from_xmi_with_lift_ts(normalized_xmi_path)

    # ------------------------------------------------------------------
    # CAS normalization
    # ------------------------------------------------------------------

    def normalize_cas(self, cas: Any) -> Any:
        """Add a new view with normalized token text to the CAS."""
        if self._view_exists(cas, self.target_view_name):
            raise ValueError(
                f"The CAS already contains a view named {self.target_view_name!r}. "
                "Set overwrite=True and make sure that the input is not already "
                "a normalized CAS, or use a different target_view_name."
            )

        source_view = self._get_source_view(cas)

        original_text = source_view.sofa_string
        if original_text is None:
            raise ValueError(
                f"The source view {self.source_view_name!r} does not contain a sofa_string."
            )

        tokens = list(source_view.select(self.token_type_name))
        tokens = sorted(tokens, key=lambda t: (t.begin, t.end))

        if not tokens:
            raise ValueError(
                f"No token annotations of type {self.token_type_name!r} found."
            )

        transformed_parts: list[str] = []
        token_span_mapping: list[dict[str, int]] = []

        original_cursor = 0
        new_cursor = 0

        for token in tokens:
            # Keep everything between tokens, e.g. whitespace, line breaks,
            # or characters that are not tokenized.
            gap = original_text[original_cursor:token.begin]
            transformed_parts.append(gap)
            new_cursor += len(gap)

            original_token_text = original_text[token.begin:token.end]
            normalized_token_text = self.token_function(original_token_text)

            if not isinstance(normalized_token_text, str):
                raise TypeError(
                    "token_function must return a string. "
                    f"Received: {type(normalized_token_text)!r}"
                )

            new_begin = new_cursor
            transformed_parts.append(normalized_token_text)
            new_cursor += len(normalized_token_text)
            new_end = new_cursor

            token_span_mapping.append(
                {
                    "old_begin": int(token.begin),
                    "old_end": int(token.end),
                    "new_begin": int(new_begin),
                    "new_end": int(new_end),
                }
            )

            original_cursor = token.end

        # Keep the remaining text after the last token.
        tail = original_text[original_cursor:]
        transformed_parts.append(tail)

        normalized_text = "".join(transformed_parts)

        target_view = cas.create_view(self.target_view_name)
        target_view.sofa_string = normalized_text

        if self.copy_tokens:
            self._add_token_annotations(
                cas=cas,
                target_view=target_view,
                token_span_mapping=token_span_mapping,
            )

        if self.copy_sentences:
            self._add_sentence_annotations(
                cas=cas,
                source_view=source_view,
                target_view=target_view,
                token_span_mapping=token_span_mapping,
            )

        return cas

    def _add_token_annotations(
        self,
        *,
        cas: Any,
        target_view: Any,
        token_span_mapping: list[dict[str, int]],
    ) -> None:
        """Create token annotations in the target view."""
        token_type = cas.typesystem.get_type(self.token_type_name)

        for mapping in token_span_mapping:
            new_token = token_type(
                begin=mapping["new_begin"],
                end=mapping["new_end"],
            )
            target_view.add_annotation(new_token)

    def _add_sentence_annotations(
        self,
        *,
        cas: Any,
        source_view: Any,
        target_view: Any,
        token_span_mapping: list[dict[str, int]],
    ) -> None:
        """Create sentence annotations in the target view.

        Sentence boundaries are reconstructed from the tokens that belonged to
        each original sentence.
        """
        sentence_type = cas.typesystem.get_type(self.sentence_type_name)

        sentences = list(source_view.select(self.sentence_type_name))
        sentences = sorted(sentences, key=lambda s: (s.begin, s.end))

        for sentence in sentences:
            sentence_tokens = [
                mapping
                for mapping in token_span_mapping
                if mapping["old_begin"] >= sentence.begin
                and mapping["old_end"] <= sentence.end
            ]

            if not sentence_tokens:
                continue

            new_sentence = sentence_type(
                begin=sentence_tokens[0]["new_begin"],
                end=sentence_tokens[-1]["new_end"],
            )
            target_view.add_annotation(new_sentence)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def get_normalized_corpus_dir(self, corpus_name: str) -> Path:
        """Return cache/normalized/<corpus_name>."""
        safe_corpus_name = self._safe_name(corpus_name)
        return self.normalized_cache_root / safe_corpus_name

    def get_normalized_file_path(
        self,
        *,
        corpus_name: str,
        source_file: Path,
    ) -> Path:
        """Return the target path for a normalized XMI file."""
        normalized_dir = self.get_normalized_corpus_dir(corpus_name)
        return normalized_dir / f"{source_file.stem}.xmi"

    def _safe_name(self, name: str) -> str:
        """Make corpus or view names safe for file paths."""
        return (
            name.replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace(" ", "_")
        )

    # ------------------------------------------------------------------
    # View helpers
    # ------------------------------------------------------------------

    def _get_source_view(self, cas: Any) -> Any:
        """Return the source view.

        If ``_InitialView`` does not explicitly exist, the CAS itself is used
        as a fallback.
        """
        try:
            return cas.get_view(self.source_view_name)
        except Exception:
            if self.source_view_name == "_InitialView":
                return cas
            raise

    def _view_exists(self, cas: Any, view_name: str) -> bool:
        """Check whether a view already exists."""
        try:
            cas.get_view(view_name)
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Token normalization
# ---------------------------------------------------------------------------

SEPARATORS_TO_REMOVE = {
    "-",        # Hyphen-minus
    "‐",        # Unicode hyphen U+2010
    "‑",        # Non-breaking hyphen U+2011
    "‒",        # Figure dash U+2012
    "–",        # En dash U+2013
    "—",        # Em dash U+2014
    "−",        # Minus sign U+2212
    "·",        # Middle dot U+00B7
    "∙",        # Bullet operator U+2219
    "•",        # Bullet U+2022
}


def normalize_token(token_text: str) -> str:
    """Remove hyphens and middle dots inside a token.

    If a separator occurs between two letters, the next segment is lowercased.

    Examples:
        Staats-Anwalt -> Staatsanwalt
        Staats-Anwalt-Gehilfe -> Staatsanwaltgehilfe
        Arbeit·Geber -> Arbeitgeber
        Lehrer‑Zimmer -> Lehrerzimmer
    """
    result: list[str] = []
    lowercase_next_alpha = False

    for i, char in enumerate(token_text):
        if char in SEPARATORS_TO_REMOVE:
            previous_char = result[-1] if result else ""
            next_char = token_text[i + 1] if i + 1 < len(token_text) else ""

            # If the separator occurs between two word parts/letters,
            # lowercase the next letter:
            # Staats-Anwalt -> Staatsanwalt
            if previous_char.isalpha() and next_char.isalpha():
                lowercase_next_alpha = True

            # Remove the separator itself.
            continue

        if lowercase_next_alpha and char.isalpha():
            result.append(char.lower())
            lowercase_next_alpha = False
        else:
            result.append(char)
            lowercase_next_alpha = False

    normalized = "".join(result)

    # Safety fallback:
    # If the token only consists of "-" or "·", normalized would be empty.
    # Empty tokens may cause problematic CAS offsets. In that case, keep the
    # original token.
    return normalized if normalized else token_text


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
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

    corpora = {
        "TIGER_ORIG": TIGER_ORIG,
        "ASGC_ORIG": ASGC_ORIG,
        "DEplain_ORIG": DEplain_ORIG,
        "G4A_ORIG": G4A_ORIG,
        "ASGC_PLAIN": ASGC_PLAIN,
        "DEplain_PLAIN": DEplain_PLAIN,
        "Leiko_PLAIN": Leiko_PLAIN,
        "ASGC_EASY": ASGC_EASY,
        "G4A_EASY": G4A_EASY,
        "Leiko_EASY": Leiko_EASY,
    }

    normalizer = CasNormalizer(
        source_view_name="_InitialView",
        target_view_name="NormalizedView",
        token_function=normalize_token,
        overwrite=True,
        verbose=True,
    )

    print("Normalized cache root:")
    print(normalizer.normalized_cache_root.resolve())

    normalized_paths = normalizer.normalize_corpora(
        corpora=corpora,
        use_input_cache=True,
    )

    print()
    print("Summary:")
    for corpus_name, paths in normalized_paths.items():
        print(f"{corpus_name}: {len(paths)} files")
        for p in paths[:3]:
            print(f"  {p.resolve()} exists={p.exists()}")


if __name__ == "__main__":
    main()