from pathlib import Path
from datasets import load_dataset, concatenate_datasets
import pandas as pd


def main():
    # Dataset laden
    german4all_corrected = load_dataset(
        "tum-nlp/German4All-Corpus",
        data_dir="corrected"
    )

    # Alle Splits zusammenführen
    ds_all = concatenate_datasets(list(german4all_corrected.values()))

    print("Splits:", list(german4all_corrected.keys()))
    print("Gesamtanzahl Zeilen:", len(ds_all))
    print("Spalten:", ds_all.column_names)

    # In pandas DataFrame umwandeln
    df = ds_all.to_pandas()

    # Benötigte Spalten prüfen
    required_columns = {"id", "text", "cl_LS"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Fehlende Spalten im Dataset: {missing_columns}")

    # Nur relevante Spalten behalten
    df_out = df[["id", "text", "cl_LS"]].copy()

    # Alles als String behandeln, leere Werte als "" speichern
    df_out = df_out.fillna("")
    df_out["id"] = df_out["id"].astype(str)
    df_out["text"] = df_out["text"].astype(str)
    df_out["cl_LS"] = df_out["cl_LS"].astype(str)

    # CSV speichern
    out_csv = Path("../datasets_raw/german4all_corrected/german4all_corrected_full.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False, encoding="utf-8")

    print(f"Gespeichert als CSV unter: {out_csv}")

    # Zielordner für TXT-Dateien
    base_dir = Path("data/german4all")
    orig_dir = base_dir / "orig"
    easy_dir = base_dir / "easy"

    orig_dir.mkdir(parents=True, exist_ok=True)
    easy_dir.mkdir(parents=True, exist_ok=True)

    # Optional: doppelte IDs abfangen, damit nichts unbemerkt überschrieben wird
    duplicated_ids = df_out[df_out["id"].duplicated()]["id"].unique()
    if len(duplicated_ids) > 0:
        raise ValueError(
            f"Doppelte IDs gefunden. Das würde Dateien überschreiben. "
            f"Beispiele: {duplicated_ids[:10]}"
        )

    # Texte exportieren
    for _, row in df_out.iterrows():
        entry_id = row["id"].strip()

        if not entry_id:
            continue

        original_text = row["text"]
        easy_text = row["cl_LS"]

        orig_path = orig_dir / f"{entry_id}.txt"
        easy_path = easy_dir / f"{entry_id}.txt"

        orig_path.write_text(original_text, encoding="utf-8")
        easy_path.write_text(easy_text, encoding="utf-8")

    print(f"Originaltexte gespeichert in: {orig_dir}")
    print(f"Easy-Language-Texte gespeichert in: {easy_dir}")


if __name__ == "__main__":
    main()