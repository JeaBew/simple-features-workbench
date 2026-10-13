"""
Erstellt eine CSV-Datei mit den Spalten "original" und "simple" aus
einem Ordner mit Textdateien, die paarweise zusammengehören.

Erwartetes Benennungsschema:
    1234.normal
    1234.simple
    5678.normal
    5678.simple
    ...

Die Zahl vor dem Punkt identifiziert zusammengehörende Dateipaare.
"""

import os
import re
import pandas as pd

# ---------------------------------------------------------
# Konfiguration: hier den Pfad zu deinem Ordner anpassen
# ---------------------------------------------------------
INPUT_DIR = "datasets_raw/ASGC/hand_aligned"
OUTPUT_CSV = "/Users/jeanette/PyCharm-after-EZ/paper_easy_plain_german_datasets/datasets_raw/ASGC/to_csv/asgc.csv"


def find_pairs(input_dir):
    """
    Durchsucht den Ordner und gruppiert Dateien nach ihrer ID
    (alles vor der Dateiendung .normal / .simple).
    Gibt ein Dict zurück: {id: {"normal": pfad, "simple": pfad}}
    """
    pairs = {}

    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        if not os.path.isfile(filepath):
            continue

        if filename.endswith(".normal"):
            file_id = filename[: -len(".normal")]
            pairs.setdefault(file_id, {})["normal"] = filepath
        elif filename.endswith(".simple"):
            file_id = filename[: -len(".simple")]
            pairs.setdefault(file_id, {})["simple"] = filepath

    return pairs


def read_text(filepath):
    """Liest eine Textdatei ein und gibt den Inhalt als String zurück."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_dataframe(pairs):
    rows = []
    incomplete = []

    # Sortiert nach ID, damit die Reihenfolge nachvollziehbar/reproduzierbar ist
    for file_id in sorted(pairs.keys()):
        entry = pairs[file_id]

        if "normal" not in entry or "simple" not in entry:
            incomplete.append(file_id)
            continue

        original_text = read_text(entry["normal"])
        simple_text = read_text(entry["simple"])

        rows.append({
            "id": file_id,
            "original": original_text,
            "simple": simple_text
        })

    if incomplete:
        print(f"Warnung: {len(incomplete)} unvollständige Paare gefunden (fehlt .normal oder .simple):")
        for i in incomplete[:10]:
            print(f"  {i}")
        if len(incomplete) > 10:
            print(f"  ... und {len(incomplete) - 10} weitere")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(f"Suche Dateipaare in: {INPUT_DIR}")
    pairs = find_pairs(INPUT_DIR)
    print(f"{len(pairs)} IDs gefunden.")

    df = build_dataframe(pairs)
    print(f"\n{len(df)} vollständige Paare verarbeitet.")
    print(df.head())

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nGespeichert unter: {OUTPUT_CSV}")