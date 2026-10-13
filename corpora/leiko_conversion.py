"""Utilities to convert CoNLL‑U files from the LeiKo 1.5 corpus to plain text.

The original helper ``conllu_to_plaintext_lib`` converted a single file.  The
new ``convert_all`` function walks the source directory ``LeiKo_1.5_conllu``
recursively, converts every ``*.conllu`` file, and writes the resulting plain‑
text files to the sibling ``Leiko_1.5`` directory while preserving the relative
sub‑directory structure.
"""

from pathlib import Path
import re
from conllu import parse_incr
from conllu.exceptions import ParseException
from tqdm import tqdm


def conllu_to_plaintext_lib(in_path: Path, out_path: Path | None = None) -> str:
    """Convert a single CoNLL‑U file to plain text.

    Parameters
    ----------
    in_path: Path
        Path to the ``.conllu`` source file.
    out_path: Path | None, optional
        If provided, the plain‑text output is written to this location.

    Returns
    -------
    str
        The concatenated plain‑text representation.
    """
    sents: list[str] = []
    # Try the official CoNLL‑U parser first. If it fails (e.g., due to a non‑standard
    # ID format), fall back to a simple line‑based parser that extracts the second
    # column (the token form) for each non‑comment line.
    try:
        with in_path.open("r", encoding="utf-8") as f:
            for tokenlist in parse_incr(f):
                tokens: list[str] = []
                for tok in tokenlist:
                    # Multi‑word tokens appear as a dict without an integer ``id``.
                    tid = tok.get("id")
                    if not isinstance(tid, int):
                        continue
                    tokens.append(tok["form"])
                # Join tokens and then fix spacing before punctuation.
                sentence = " ".join(tokens)
                sentence = re.sub(r"\s+([.,;:!?])", r"\1", sentence)
                sents.append(sentence)
    except ParseException:
        # Simple fallback parser: split on blank lines to separate sentences.
        current_tokens: list[str] = []
        with in_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_tokens:
                        sentence = " ".join(current_tokens)
                        sentence = re.sub(r"\s+([.,;:!?])", r"\1", sentence)
                        sents.append(sentence)
                        current_tokens = []
                    continue
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    # The second column is the token form.
                    current_tokens.append(parts[1])
            # Add the last sentence if file does not end with a blank line.
            if current_tokens:
                sentence = " ".join(current_tokens)
                sentence = re.sub(r"\s+([.,;:!?])", r"\1", sentence)
                sents.append(sentence)

    text = "\n".join(sents)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    return text


def convert_all(src_root: Path = Path(__file__).parent / "LeiKo_1.5_conllu",
                dst_root: Path = Path(__file__).parent / "Leiko_1.5",
                pattern: str = "*.conll") -> None:
    """Convert every ``*.conllu`` file under *src_root* to plain text.

    The directory hierarchy under ``src_root`` is mirrored inside ``dst_root``.
    Files keep their original stem but receive a ``.txt`` extension.
    """
    conllu_files = list(src_root.rglob(pattern))
    for conllu_path in tqdm(conllu_files, desc="Converting CoNLL‑U files"):
        # Determine the relative path to recreate the folder structure.
        rel_path = conllu_path.relative_to(src_root).with_suffix(".txt")
        out_path = dst_root / rel_path
        conllu_to_plaintext_lib(conllu_path, out_path)


if __name__ == "__main__":
    convert_all()