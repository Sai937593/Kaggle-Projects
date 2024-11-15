import spacy

nlp = spacy.load("en_core_web_sm")

entity_labels = [
    "DATE",
    "EVENT",
    "FAC",
    "GPE",
    "LANGUAGE",
    "LAW",
    "LOC",
    "NORP",
    "ORG",
    "PERSON",
    "PRODUCT",
    "WORK_OF_ART",
]

import spacy


ent_counts = {ent : [] for ent in entity_labels}
nlp = spacy.load('en_core_web_sm')

def count_entities(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_  in entity_labels:
            ent_counts[ent.label_].append(ent.text)