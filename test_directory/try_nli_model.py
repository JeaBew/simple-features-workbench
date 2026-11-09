import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

ABBREVIATIONS = ["z. B.", "d. h.", "u. a.", "ca.", "Dr.", "Nr.", "bzw.", "usw."]

def split_sentences(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    protected = text
    for i, abbr in enumerate(ABBREVIATIONS):
        protected = protected.replace(abbr, abbr.replace(".", f"<DOT{i}>"))
    parts = re.split(r"(?<=[.!?])(?:\s+|(?=[A-ZÄÖÜ]))", protected)
    result = []
    for p in parts:
        for i in range(len(ABBREVIATIONS)):
            p = p.replace(f"<DOT{i}>", ".")
        if p.strip():
            result.append(p.strip())
    return result

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

# Modell nur einmal laden, nicht bei jedem Aufruf
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).eval()

# Modellausgabe (Index) -> gewünschter Code, über den Labelnamen
LABEL_TO_CODE = {"entailment": 0, "neutral": 1, "contradiction": 2}
OUTPUT_TO_CODE = {
    idx: LABEL_TO_CODE[label.lower()]
    for idx, label in model.config.id2label.items()
}
def classify(premise: str, hypothesis: str) -> tuple[int, float]:
    """Gibt (Code, Wahrscheinlichkeit) zurück.

    Code: 0 = entailment, 1 = neutral, 2 = contradiction.
    Die Wahrscheinlichkeit gehört zum gewählten Fall.
    """
    inputs = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = logits.softmax(dim=-1)[0]
    best_index = int(probs.argmax())
    return OUTPUT_TO_CODE[best_index], float(probs[best_index])

def find_entailments(split_text: list[str], split_simple_text: list[str]) -> list[tuple[int, list[tuple[int, float]]]]:
    """Prüft jeden Originalsatz gegen jeden vereinfachten Satz.

    Premise = Satz aus split_text, Hypothese = Satz aus split_simple_text.
    Gibt für jeden Originalsatz mit mindestens einem Entailment einen Eintrag
    (index_1, [(index_2, prob), ...]) zurück.
    """
    entailments = []
    for i, original in enumerate(split_text):
        matches = []
        for j, simple in enumerate(split_simple_text):
            code, prob = classify(original, simple)
            if code == 0:  # entailment
                matches.append((j, prob))
        if matches:
            entailments.append((i, matches))
    return entailments

text = 'Ich mag Äpfel. Kartoffeln mag ich auch weil sie so schön weich werden. Ich mag keinen Fisch, aber dafür Reis seht gerne.'
simple_text = 'Ich esse gerne Äpfel. Ich esse gerne Kartoffeln. Ich mag Fisch. Ich mag Reis. Ich mag Äpfel.'

split_text = split_sentences(text)
split_simple_text = split_sentences(simple_text)

found_entailments = find_entailments(split_text, split_simple_text)

print(found_entailments)