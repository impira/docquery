import os
import pathlib
from functools import cached_property
from io import BytesIO
from typing import List, Tuple

import requests
from pydantic import validate_arguments

from .ext import transformers


class UnsupportedDocument(Exception):
    def __init__(self, e):
        self.e = e

    def __str__(self):
        return f"unsupported file type: {self.e}"


PIL_AVAILABLE = False
TESSERACT_AVAILABLE = False
PDF_2_IMAGE = False
PDF_PLUMBER = False

try:
    from PIL import Image, UnidentifiedImageError

    PIL_AVAILABLE = True
except ImportError as e:
    pass

try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError as e:
    pass

try:
    import pdf2image

    PDF_2_IMAGE = True
except ImportError as e:
    pass

try:
    import pdfplumber

    PDF_PLUMBER = True
except ImportError as e:
    pass


def use_pil():
    if not PIL_AVAILABLE:
        raise UnsupportedDocument("Unable to import PIL (images will be unavailable): %s", e)


def use_tesseract():
    if not TESSERACT_AVAILABLE:
        raise UnsupportedDocument("Unable to import pytesseract (OCR will be unavailable): %s", e)


def use_pdf2_image():
    if not PDF_2_IMAGE:
        raise UnsupportedDocument("Unable to import pdf2image (OCR will be unavailable for pdfs): %s", e)


def use_pdf_plumber():
    if not PDF_PLUMBER:
        raise UnsupportedDocument("Unable to import pdfplumber (pdfs will be unavailable): %s", e)


def apply_tesseract(*args, **kwargs):
    use_tesseract()
    return transformers.apply_tesseract(*args, **kwargs)


class Document:
    def __init__(self, b):
        self.b = b

    @property
    def context(self) -> Tuple[(str, List[int])]:
        raise NotImplementedError


class PDFDocument(Document):
    @cached_property
    def context(self) -> Tuple[(str, List[int])]:
        # First, try to extract text directly
        use_pdf_plumber()
        pdf = pdfplumber.open(BytesIO(self.b))
        if len(pdf.pages) == 0:
            return []

        word_boxes = []
        for i, page in enumerate(pdf.pages):
            words = page.extract_words()
            if i == 0 and len(words) == 0:
                return self.as_image()

            word_boxes.extend(
                (
                    w["text"],
                    transformers.normalize_box([w["x0"], w["top"], w["x1"], w["bottom"]], page.width, page.height),
                )
                for w in words
            )
        return {"image": None, "word_boxes": word_boxes}

    def as_image(self) -> Tuple[(str, List[int])]:
        use_pdf2_image()
        images = pdf2image.convert_from_bytes(self.b)

        word_boxes = []
        for img in images:
            words, boxes = apply_tesseract(img, lang=None, tesseract_config="")
            word_boxes.extend([x for x in zip(words, boxes)])
        return {"image": None, "word_boxes": word_boxes}


class ImageDocument(Document):
    @cached_property
    def context(self) -> Tuple[(str, List[int])]:
        words, boxes = apply_tesseract(self.b, lang=None, tesseract_config="")
        return {
            "image": self.b,
            "word_boxes": [x for x in zip(words, boxes)],
        }


@validate_arguments
def load_document(fpath: str):
    if fpath.startswith("http://") or fpath.startswith("https://"):
        b = requests.get(fpath, stream=True).raw
    else:
        b = open(fpath, "rb")

    extension = os.path.basename(fpath).rsplit(".", 1)[-1].split("?")[0].strip()
    if extension in ("pdf"):
        return PDFDocument(b.read())
    else:
        use_pil()
        try:
            img = Image.open(b)
        except UnidentifiedImageError as e:
            raise UnsupportedDocument(e)
        return ImageDocument(img)
