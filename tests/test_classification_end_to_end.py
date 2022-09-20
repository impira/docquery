from typing import Dict, List

import pytest
from pydantic import BaseModel
from transformers.testing_utils import nested_simplify

from docquery import pipeline
from docquery.document import load_document
from docquery.ocr_reader import TesseractReader


CHECKPOINTS = {
    "Donut": "naver-clova-ix/donut-base-finetuned-rvlcdip",
}


class Example(BaseModel):
    name: str
    path: str
    classes: Dict[str, List[str]]


# Use the examples from the DocQuery space (this also solves for hosting)
EXAMPLES = [
    Example(
        name="contract",
        path="https://huggingface.co/spaces/impira/docquery/resolve/2f6c96314dc84dfda62d40de9da55f2f5165d403/contract.jpeg",
        classes={"Donut": ["scientific_report"]},
    ),
    Example(
        name="invoice",
        path="https://huggingface.co/spaces/impira/docquery/resolve/2f6c96314dc84dfda62d40de9da55f2f5165d403/invoice.png",
        classes={"Donut": ["invoice"]},
    ),
    Example(
        name="statement",
        path="https://huggingface.co/spaces/impira/docquery/resolve/2f6c96314dc84dfda62d40de9da55f2f5165d403/statement.png",
        classes={"Donut": ["budget"]},
    ),
]


@pytest.mark.parametrize("example", EXAMPLES)
@pytest.mark.parametrize("model", CHECKPOINTS.keys())
def test_impira_dataset(example, model):
    document = load_document(example.path)
    pipe = pipeline("document-classification", model=CHECKPOINTS[model])
    resp = pipe(top_k=1, **document.context)
    assert resp == [{"label": x} for x in example.classes[model]]
