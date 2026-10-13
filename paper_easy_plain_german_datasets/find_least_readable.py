import os
from pathlib import Path
from statistics import mean

import seaborn as sns
import matplotlib.pyplot as plt

from utils import process_folder_with_filenames, _category
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
from py_lift.readability import FE_TextstatWienerSachtextformel_1  # type: ignore


def _min_score_per_corpus(
    folder: Path,
    extractor,  # Callable[[Any], None]
    feature_name: str,
):
    """Return the (file_path, lowest_score) for *folder*.

    The function scans all files in *folder* using ``process_folder_with_filenames``
    and picks the document with the smallest readability score.
    """
    per_file = process_folder_with_filenames(
        folder,
        feature_extractor=extractor,
        feature_names=[feature_name],
    )

    best_file: Path | None = None
    best_score = float("inf")
    for fp, scores_dict in per_file.items():
        scores = scores_dict.get(feature_name, [])
        if not scores:
            continue
        # Each file yields a single score list; take the first element.
        score = scores[0]
        if score < best_score:
            best_score = score
            best_file = fp
    return best_file, best_score


def main() -> None:
    # Extractor for the Wiener‑Sachtextformel readability feature.
    extractor = lambda cas: FE_TextstatWienerSachtextformel_1().extract(cas)
    feat_name = "Readability_Score_WienerSachtextformel-1_de"

    corpora = {
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

    print("Least readable document per corpus:")
    for name, path in corpora.items():
        file_path, score = _min_score_per_corpus(path, extractor, feat_name)
        if file_path is None:
            print(f"{name}: no scores found")
        else:
            print(f"{name}: {score:.2f} → {file_path}")

    # Optional: you could create a plot of the minimum scores similar to the
    # existing ``run_readability.py`` script, but the primary request is to
    # identify the least readable document, which is fulfilled by the console
    # output above.


if __name__ == "__main__":
    main()
