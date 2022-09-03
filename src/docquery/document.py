import abc
import os
from io import BytesIO
from typing import List, Tuple, Optional

import requests
from pydantic import validate_arguments

from ocr_processor import TesseractProcessor, TESSERACT_AVAILABLE, EasyOCRProcessor, EASYOCR_AVAILABLE, DummyProcessor

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
PDF_2_IMAGE = False
PDF_PLUMBER = False

try:
    from PIL import Image, UnidentifiedImageError

    PIL_AVAILABLE = True
except ImportError:
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


def use_pdf2_image():
    if not PDF_2_IMAGE:
        raise UnsupportedDocument("Unable to import pdf2image (OCR will be unavailable for pdfs)")


def use_pdf_plumber():
    if not PDF_PLUMBER:
        raise UnsupportedDocument("Unable to import pdfplumber (pdfs will be unavailable)")


class Document(metaclass=abc.ABCMeta):
    def __init__(self, b, ocr_processor):
        self.b = b
        self.ocr_processor = ocr_processor

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
                word_boxes = [x for x in zip(*self.ocr_processor.apply_ocr(images[i]))]

            else:
                word_boxes = [
                    (
                        w["text"],
                        self.ocr_processor.normalize_box([w["x0"], w["top"], w["x1"], w["bottom"]],
                                                         page.width, page.height),
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
        words, boxes = self.ocr_processor.apply_ocr(self.b)

        return {
            "image": [
                (
                    self.b,
                    [x for x in zip(words, boxes)],
                )
            ]
        }


@validate_arguments
def load_document(fpath: str, ocr_processor_name=Optional[str]):
    if fpath.startswith("http://") or fpath.startswith("https://"):
        resp = requests.get(fpath, stream=True)
        if not resp.ok:
            raise UnsupportedDocument(f"Failed to download: {resp.content}")
        b = resp.raw
    else:
        b = open(fpath, "rb")
    return load_bytes(b, fpath, ocr_processor_name)


def load_bytes(b, fpath, ocr_processor_name=Optional[str]):
    ocr_processor = get_ocr_processor(ocr_processor_name)
    extension = os.path.basename(fpath).rsplit(".", 1)[-1].split("?")[0].strip()
    if extension in ("pdf"):
        return PDFDocument(b.read(), ocr_processor=ocr_processor)
    else:
        use_pil()
        try:
            img = Image.open(b)
        except UnidentifiedImageError as e:
            raise UnsupportedDocument(e)
        return ImageDocument(img, ocr_processor=ocr_processor)


def get_ocr_processor(ocr_processor_name: Optional[str]):
    if not ocr_processor_name:
        if TESSERACT_AVAILABLE:
            return TesseractProcessor()
        elif EASYOCR_AVAILABLE:
            return EasyOCRProcessor()
        else:
            DummyProcessor()
    elif ocr_processor_name.lower() == "easyocr" and EASYOCR_AVAILABLE:
        return EasyOCRProcessor()
    elif ocr_processor_name.lower() == "tesseract" and TESSERACT_AVAILABLE:
        return TesseractProcessor()
    else:
        DummyProcessor()
