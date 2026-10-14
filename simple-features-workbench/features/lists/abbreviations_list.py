"""
Liste gängiger deutscher Abkürzungen für die Abkürzungs-Konsistenzprüfung.
"""

from __future__ import annotations


def merge_abbreviation_maps(*maps: dict[str, set[str]]) -> dict[str, set[str]]:
    """Führt mehrere Abkürzungs-Dicts zusammen.

    Kommt eine Kurzform in mehreren Dicts vor, werden ihre Langformen
    vereinigt statt eines das andere zu überschreiben.
    """
    merged: dict[str, set[str]] = {}
    for m in maps:
        for short_form, long_forms in m.items():
            merged.setdefault(short_form, set()).update(long_forms)
    return merged


# Gängige, allgemeine deutsche Abkürzungen - nicht erschöpfend, aber ein
# solider Grundstock. Bei Bedarf einfach weitere Einträge ergänzen.
COMMON_GERMAN_ABBREVIATIONS: dict[str, set[str]] = {
    # Konnektoren/Floskeln
    "z.B.": {"zum Beispiel"},
    "z. B.": {"zum Beispiel"},
    "d.h.": {"das heißt"},
    "d. h.": {"das heißt"},
    "bzw.": {"beziehungsweise"},
    "usw.": {"und so weiter"},
    "u.a.": {"unter anderem"},
    "u. a.": {"unter anderem"},
    "u.Ä.": {"und Ähnliches"},
    "u.U.": {"unter Umständen"},
    "i.d.R.": {"in der Regel"},
    "o.Ä.": {"oder Ähnliches"},
    "s.o.": {"siehe oben"},
    "s.u.": {"siehe unten"},
    "vgl.": {"vergleiche"},
    "ca.": {"circa"},
    "etc.": {"et cetera"},
    "evtl.": {"eventuell"},
    "ggf.": {"gegebenenfalls"},
    "inkl.": {"inklusive"},
    "exkl.": {"exklusive"},
    "insb.": {"insbesondere"},
    "bspw.": {"beispielsweise"},
    "sog.": {"sogenannt"},
    "ggü.": {"gegenüber"},
    "geb.": {"geboren"},
    "gest.": {"gestorben"},
    # Zahlen/Mengen/Maße
    "Nr.": {"Nummer"},
    "Std.": {"Stunde", "Stunden"},
    "Min.": {"Minute", "Minuten"},
    "Sek.": {"Sekunde", "Sekunden"},
    "max.": {"maximal"},
    "min.": {"minimal"},
    "Mio.": {"Million", "Millionen"},
    "Mrd.": {"Milliarde", "Milliarden"},
    "Tsd.": {"Tausend"},
    "kg": {"Kilogramm"},
    "km": {"Kilometer"},
    "km/h": {"Kilometer pro Stunde"},
    "cm": {"Zentimeter"},
    "mm": {"Millimeter"},
    "qm": {"Quadratmeter"},
    # Text-/Publikationsverweise
    "Abb.": {"Abbildung"},
    "Kap.": {"Kapitel"},
    "Bd.": {"Band"},
    "Jh.": {"Jahrhundert"},
    "S.": {"Seite"},
    "Art.": {"Artikel"},
    "Abs.": {"Absatz"},
    "Anz.": {"Anzahl"},
    "Aufl.": {"Auflage"},
    "Hrsg.": {"Herausgeber", "Herausgeberin"},
    # Rechtsformen
    "GmbH": {"Gesellschaft mit beschränkter Haftung"},
    "AG": {"Aktiengesellschaft"},
    "e.V.": {"eingetragener Verein"},
    "KG": {"Kommanditgesellschaft"},
    "OHG": {"Offene Handelsgesellschaft"},
    "mbH": {"mit beschränkter Haftung"},
    "Co.": {"Compagnie"},
    # Institutionen/Organisationen/Länder
    "EU": {"Europäische Union"},
    "BRD": {"Bundesrepublik Deutschland"},
    "NRW": {"Nordrhein-Westfalen"},
    "ADAC": {"Allgemeiner Deutscher Automobil-Club"},
    "ARD": {"Arbeitsgemeinschaft der öffentlich-rechtlichen Rundfunkanstalten der Bundesrepublik Deutschland"},
    "ZDF": {"Zweites Deutsches Fernsehen"},
    "UNO": {"Vereinte Nationen"},
    "NATO": {"Nordatlantikvertrags-Organisation"},
    "WHO": {"Weltgesundheitsorganisation"},
    # Alltag/Verkehr
    "LKW": {"Lastkraftwagen"},
    "PKW": {"Personenkraftwagen"},
    "ÖPNV": {"öffentlicher Personennahverkehr"},
    "Uni": {"Universität", "Hochschule"},
    "Info": {"Information"},
    "Kita": {"Kindertagesstätte"},
    "WG": {"Wohngemeinschaft"},
}

# Projektspezifische/eigene Ergänzungen - hier bei Bedarf weitere eintragen.
CUSTOM_ABBREVIATIONS: dict[str, set[str]] = {
    "FernUni": {"FernUniversität in Hagen", "FernUniversität"},
}

ABBREVIATION_MAP: dict[str, set[str]] = merge_abbreviation_maps(
    COMMON_GERMAN_ABBREVIATIONS,
    CUSTOM_ABBREVIATIONS,
)