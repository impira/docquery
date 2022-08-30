import abc
import logging
import os
from io import BytesIO
from typing import List, Tuple

import requests
from pydantic import validate_arguments

from .ext import transformers


try:
    from functools import cached_property as cached_property
except ImportError:
    # for python 3.7 support fall back to just property
    cached_property = property


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
except ImportError:
    pass

try:
    import pytesseract  # noqa

    pytesseract.get_tesseract_version()
    TESSERACT_AVAILABLE = True
except ImportError:
    pass
except pytesseract.TesseractNotFoundError as e:
    logging.warning("Unable to find tesseract: %s." % (e))
    pass

try:
    import pdf2image

    PDF_2_IMAGE = True
except ImportError:
    pass

try:
    import pdfplumber

    PDF_PLUMBER = True
except ImportError:
    pass


def use_pil():
    if not PIL_AVAILABLE:
        raise UnsupportedDocument("Unable to import PIL (images will be unavailable)")


def use_tesseract():
    if not TESSERACT_AVAILABLE:
        raise UnsupportedDocument(
            "Unable to use pytesseract (OCR will be unavailable). Install tesseract to process images with OCR."
        )


def use_pdf2_image():
    if not PDF_2_IMAGE:
        raise UnsupportedDocument("Unable to import pdf2image (OCR will be unavailable for pdfs)")


def use_pdf_plumber():
    if not PDF_PLUMBER:
        raise UnsupportedDocument("Unable to import pdfplumber (pdfs will be unavailable)")


def apply_tesseract(*args, **kwargs):
    use_tesseract()
    return transformers.apply_tesseract(*args, **kwargs)


class Document(metaclass=abc.ABCMeta):
    def __init__(self, b):
        self.b = b

    @property
    @abc.abstractmethod
    def context(self) -> Tuple[(str, List[int])]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def preview(self) -> "Image":
        raise NotImplementedError


class PDFDocument(Document):
    @cached_property
    def context(self) -> Tuple[(str, List[int])]:
        pdf = self._pdf
        if pdf is None:
            return {}

        images = self._images

        if len(images) != len(pdf.pages):
            raise ValueError(
                f"Mismatch: pdfplumber() thinks there are {len(pdf.pages)} pages and"
                f" pdf2image thinks there are {len(images)}"
            )

        img_words = []
        for i, page in enumerate(pdf.pages):
            words = page.extract_words()

            if len(words) == 0:
                use_tesseract()
                if TESSERACT_AVAILABLE:
                    word_boxes = [x for x in zip(*apply_tesseract(images[i], lang=None, tesseract_config=""))]
            else:
                word_boxes = [
                    (
                        w["text"],
                        transformers.normalize_box([w["x0"], w["top"], w["x1"], w["bottom"]], page.width, page.height),
                    )
                    for w in words
                ]

            img_words.append((images[i], word_boxes))
        return {"image": img_words}

    @cached_property
    def preview(self) -> "Image":
        return self._images

    @cached_property
    def _images(self):
        # First, try to extract text directly
        # TODO: This library requires poppler, which is not present everywhere.
        # We should look into alternatives. We could also gracefully handle this
        # and simply fall back to _only_ extracted text
        return [x.convert("RGB") for x in pdf2image.convert_from_bytes(self.b)]

    @cached_property
    def _pdf(self):
        use_pdf_plumber()
        pdf = pdfplumber.open(BytesIO(self.b))
        if len(pdf.pages) == 0:
            return None
        return pdf


class ImageDocument(Document):
    @cached_property
    def preview(self) -> "Image":
        return [self.b.convert("RGB")]

    @cached_property
    def context(self) -> Tuple[(str, List[int])]:
        words, boxes = apply_tesseract(self.b, lang=None, tesseract_config="")
        return {
            "image": [
                (
                    self.b,
                    [x for x in zip(words, boxes)],
                )
            ]
        }


@validate_arguments
def load_document(fpath: str):
    if fpath.startswith("http://") or fpath.startswith("https://"):
        b = requests.get(fpath, stream=True).raw
    else:
        b = open(fpath, "rb")
    return load_bytes(b, fpath)


def load_bytes(b, fpath):
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
