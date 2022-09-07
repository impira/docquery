import abc
import os
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from pydantic import validate_arguments

from .ocr_reader import NoOCRReaderFound, OCRReader, get_ocr_reader


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
    def _generate_document_output(
        images: List["Image.Image"],
        words_by_page: List[List[str]],
        boxes_by_page: List[List[List[int]]],
        dimensions_by_page: List[Tuple[int, int]],
    ) -> Dict[str, List[Tuple["Image.Image", List[Any]]]]:

        # pages_dimensions (width, height)
        assert len(images) == len(dimensions_by_page)
        assert len(images) == len(words_by_page)
        assert len(images) == len(boxes_by_page)
        processed_pages = []
        for image, words, boxes, dimensions in zip(images, words_by_page, boxes_by_page, dimensions_by_page):
            width, height = dimensions

            """
            box is [x1,y1,x2,y2] where x1,y1 are the top left corner of box and x2,y2 is the bottom right corner
            This function scales the distance between boxes to be on a fixed scale
            It is derived from the preprocessing code for LayoutLM
            """
            normalized_boxes = [
                [
                    max(min(c, 1000), 0)
                    for c in [
                        int(1000 * (box[0] / width)),
                        int(1000 * (box[1] / height)),
                        int(1000 * (box[2] / width)),
                        int(1000 * (box[3] / height)),
                    ]
                ]
                for box in boxes
            ]
            assert len(words) == len(normalized_boxes), "Not as many words as there are bounding boxes"
            word_boxes = [x for x in zip(words, normalized_boxes)]
            processed_pages.append((image, word_boxes))

        return {"image": processed_pages}


class PDFDocument(Document):
    @cached_property
    def context(self) -> Dict[str, List[Tuple["Image.Image", List[Any]]]]:
        pdf = self._pdf
        if pdf is None:
            return {}

        images = self._images

        if len(images) != len(pdf.pages):
            raise ValueError(
                f"Mismatch: pdfplumber() thinks there are {len(pdf.pages)} pages and"
                f" pdf2image thinks there are {len(images)}"
            )

        words_by_page = []
        boxes_by_page = []
        dimensions_by_page = []
        for i, page in enumerate(pdf.pages):
            extracted_words = page.extract_words()

            if len(extracted_words) == 0:
                words, boxes = self.ocr_reader.apply_ocr(images[i])
                words_by_page.append(words)
                boxes_by_page.append(boxes)
                dimensions_by_page.append((images[i].width, images[i].height))

            else:
                words = [w["text"] for w in extracted_words]
                boxes = [[w["x0"], w["top"], w["x1"], w["bottom"]] for w in extracted_words]
                words_by_page.append(words)
                boxes_by_page.append(boxes)
                dimensions_by_page.append((page.width, page.height))

        return self._generate_document_output(images, words_by_page, boxes_by_page, dimensions_by_page)

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
    def context(self) -> Dict[str, List[Tuple["Image.Image", List[Any]]]]:
        words, boxes = self.ocr_reader.apply_ocr(self.b)
        return self._generate_document_output([self.b], [words], [boxes], [(self.b.width, self.b.height)])


@validate_arguments
def load_document(fpath: str, ocr_reader: Optional[Union[str, OCRReader]] = None):
    if fpath.startswith("http://") or fpath.startswith("https://"):
        resp = requests.get(fpath, stream=True)
        if not resp.ok:
            raise UnsupportedDocument(f"Failed to download: {resp.content}")
        b = resp.raw
    else:
        b = open(fpath, "rb")
    return load_bytes(b, fpath, ocr_reader=ocr_reader)


def load_bytes(b, fpath, ocr_reader: Optional[Union[str, Any]]):
    if not ocr_reader or isinstance(ocr_reader, str):
        ocr_reader = get_ocr_reader(ocr_reader)
    elif not isinstance(ocr_reader, OCRReader):
        raise NoOCRReaderFound(f"{ocr_reader} is not a supported OCRReader class")

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
