"""Convert entity annotation from spaCy v2 TRAIN_DATA format to spaCy v3
.spacy format."""
import srsly
import typer
from pathlib import Path

import spacy
from spacy.tokens import DocBin


def convert(lang: str, input_path: Path, output_path: Path):
    nlp = spacy.blank(lang)
    db = DocBin()
    for entry in srsly.read_jsonl(input_path):
        text = entry['data']
        doc = nlp.make_doc(text)
        doc.cats = entry["cats"]
        db.add(doc)
    db.to_disk(output_path)



if __name__ == "__main__":
    typer.run(convert)



