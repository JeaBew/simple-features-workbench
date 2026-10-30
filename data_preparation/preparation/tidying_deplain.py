import re
from cassis import load_cas_from_xmi
from pathlib import Path


def clean_text(text: str) -> str:
    """
    Bereinigt einen Text nach den definierten Regeln.

    Parameters
    ----------
    text : str
        Der ursprüngliche, unbereinigte Text.

    Returns
    -------
    str
        Der bereinigte Text.
    """
    if not isinstance(text, str):
        return text

    cleaned = text

    # Regel 1: "Wir arbeiten an diesem Text." am Anfang entfernen
    # (auch falls danach noch Leerzeichen/Zeilenumbrüche folgen)
    cleaned = re.sub(
        r"^\s*Wir arbeiten an diesem Text\.\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Regel 2: Alles in doppelten Klammern entfernen, z.B. ((blabla))
    # non-greedy (.*?), damit nicht über mehrere Klammerpaare hinweg
    # gematcht wird, und re.DOTALL falls der Inhalt über mehrere Zeilen geht
    cleaned = re.sub(r"\(\(.*?\)\)", "", cleaned, flags=re.DOTALL)

    # Aufräumen: doppelte Leerzeichen, die durchs Entfernen entstehen,
    # auf ein einzelnes Leerzeichen reduzieren
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # Leerzeichen vor Satzzeichen entfernen, die durchs Entfernen von
    # Klammern entstehen können (z.B. "Inhalt ." -> "Inhalt.")
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)

    # Mehrfache Leerzeilen (3+ Zeilenumbrüche) auf maximal 2 reduzieren
    cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)

    # Führende/abschließende Leerzeichen pro Zeile entfernen
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))

    # Gesamten Text noch einmal von außen trimmen
    cleaned = cleaned.strip()

    return cleaned


# ---------------------------------------------------------
# Beispielnutzung / Tests
# ---------------------------------------------------------
if __name__ == "__main__":
    examples = [
        "Wir arbeiten an diesem Text.\nDas ist der eigentliche Inhalt.",

        "Wir arbeiten an diesem Text. Das ist der eigentliche Inhalt ((Hinweis: noch nicht final)).",

        "Das ist ein Satz ((mit einer Anmerkung)) mitten im Text.",

        "Wir arbeiten an diesem Text.\n\nMehrere ((Anmerkungen)) im ((Text)) verteilt.",

        "Kein Hinweis am Anfang, aber ((eine Klammer)) mittendrin.",
    ]

    for i, example in enumerate(examples, start=1):
        print(f"--- Beispiel {i} ---")
        print("Vorher: ", repr(example))
        print("Nachher:", repr(clean_text(example)))
        print()


ordner_plain = Path("data/deplain/plain/")
ordner_orig = Path("data/deplain/orig/")

out_dir_plain = Path("cleaned_data/deplain/plain")
out_dir_orig = Path("cleaned_data/deplain/orig")

for txt_file in ordner_orig.glob("*.txt"):
    print(txt_file)

    # Dateiinhalt in Variable "text" laden
    text = txt_file.read_text(encoding="utf-8")
    print(text[:200])
    cleaned_text = clean_text(text)

    out_path = out_dir_orig / txt_file.name  # gleicher Name wie Original
    out_path.write_text(cleaned_text, encoding="utf-8")

for txt_file in ordner_plain.glob("*.txt"):
    print(txt_file)

    # Dateiinhalt in Variable "text" laden
    text = txt_file.read_text(encoding="utf-8")
    print(text[:200])
    cleaned_text = clean_text(text)

    out_path = out_dir_plain / txt_file.name  # gleicher Name wie Original
    out_path.write_text(cleaned_text, encoding="utf-8")