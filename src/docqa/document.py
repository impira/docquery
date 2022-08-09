from io import BytesIO
from functools import cached_property
import pathlib
import pdf2image
import pdfplumber
from PIL import Image, UnidentifiedImageError
from pydantic import validate_arguments
from typing import List, Tuple

from .ext import transformers

TESSERACT_AVAILABLE = False


def use_tesseract():
    global TESSERACT_AVAILABLE
    if TESSERACT_AVAILABLE:
        return

    try:
        import pytesseract

        TESSERACT_AVAILABLE = True
    except ImportError as e:
        logging.warning("Unable to import pytesseract (OCR will be unavailable): %s", e)

    return TESSERACT_AVAILABLE


def apply_tesseract(*args, **kwargs):
    if not use_tesseract():
        raise ValueError("Tesseract is required for this file")
    return transformers.apply_tesseract(*args, **kwargs)


class UnsupportedDocument(Exception):
    def __init__(self, e):
        self.e = e

    def __str__(self):
        return f"unsupported file type: {self.e}"


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
def load_document(fpath: pathlib.Path):
    if str(fpath).startswith("http://") or str(fpath).startswith("https://"):
        b = requests.get(fpath, stream=True).raw
    else:
        b = open(fpath, "rb")

    extension = fpath.suffix.split("?")[0].strip()
    if extension in (".pdf"):
        return PDFDocument(b.read())
    else:
        try:
            img = Image.open(b)
        except UnidentifiedImageError as e:
            raise UnsupportedDocument(e)
        return ImageDocument(img)
