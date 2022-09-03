import abc
import os
from io import BytesIO
from typing import List, Tuple, Optional, Dict, Any

import requests
from pydantic import validate_arguments

from .ocr_reader import TesseractReader, TESSERACT_AVAILABLE, EasyOCRReader, EASYOCR_AVAILABLE, DummyOCRReader

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
    def __init__(self, b, ocr_reader):
        self.b = b
        self.ocr_reader = ocr_reader

    @property
    @abc.abstractmethod
    def context(self) -> Tuple[(str, List[int])]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def preview(self) -> "Image":
        raise NotImplementedError

    @staticmethod
    def _normalize_box(box, width, height) -> List[int]:
        return [
            max(min(c, 1000), 0)
            for c in [
                int(1000 * (box[0] / width)),
                int(1000 * (box[1] / height)),
                int(1000 * (box[2] / width)),
                int(1000 * (box[3] / height)),
            ]
        ]

    @staticmethod
    def _normalize_boxes(image, words, boxes) -> Tuple[List[str], List[List[int]]]:
        image_width, image_height = image.size

        # finally, normalize the bounding boxes
        normalized_boxes = [Document._normalize_box(box, image_width, image_height) for box in boxes]

        assert len(words) == len(normalized_boxes), "Not as many words as there are bounding boxes"
        return words, normalized_boxes


class PDFDocument(Document):
    @cached_property
    def context(self) -> Dict[str, List[tuple["Image", List[Any]]]]:
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
                word_boxes = [x for x in zip(*self._normalize_boxes(images[i],
                                                                    *self.ocr_reader.apply_ocr(images[i])))]

            else:
                word_boxes = [
                    (
                        w["text"],
                        self._normalize_box([w["x0"], w["top"], w["x1"], w["bottom"]],
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
    def context(self) -> Dict[str, List[tuple["Image", List[Any]]]]:
        words, boxes = self._normalize_boxes(self.b,
                                             *self.ocr_reader.apply_ocr(self.b))

        return {
            "image": [
                (
                    self.b,
                    [x for x in zip(words, boxes)],
                )
            ]
        }


@validate_arguments
def load_document(fpath: str, ocr_reader_name=None):
    if fpath.startswith("http://") or fpath.startswith("https://"):
        resp = requests.get(fpath, stream=True)
        if not resp.ok:
            raise UnsupportedDocument(f"Failed to download: {resp.content}")
        b = resp.raw
    else:
        b = open(fpath, "rb")
    return load_bytes(b, fpath, ocr_reader_name=ocr_reader_name)


def load_bytes(b, fpath, ocr_reader_name: Optional[str]):
    ocr_reader = get_ocr_reader(ocr_reader_name)
    extension = os.path.basename(fpath).rsplit(".", 1)[-1].split("?")[0].strip()
    if extension in ("pdf"):
        return PDFDocument(b.read(), ocr_reader=ocr_reader)
    else:
        use_pil()
        try:
            img = Image.open(b)
        except UnidentifiedImageError as e:
            raise UnsupportedDocument(e)
        return ImageDocument(img, ocr_reader=ocr_reader)


tesseract_reader = None
easy_ocr_reader = None


def get_ocr_reader(ocr_reader_name: Optional[str]):
    global tesseract_reader
    global easy_ocr_reader
    if not ocr_reader_name:
        if TESSERACT_AVAILABLE:
            if not tesseract_reader:
                tesseract_reader = TesseractReader()
            return tesseract_reader
        elif EASYOCR_AVAILABLE:
            if not easy_ocr_reader:
                easy_ocr_reader = EasyOCRReader()
            return easy_ocr_reader
        else:
            return DummyOCRReader()
    elif ocr_reader_name.lower() == "easyocr" and EASYOCR_AVAILABLE:
        if not easy_ocr_reader:
            easy_ocr_reader = EasyOCRReader()
        return easy_ocr_reader
    elif ocr_reader_name.lower() == "tesseract" and TESSERACT_AVAILABLE:
        if not tesseract_reader:
            tesseract_reader = TesseractReader()
        return tesseract_reader
    else:
        return DummyOCRReader()
