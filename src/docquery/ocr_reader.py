import abc
import logging
from typing import List, Tuple, Any

import numpy as np


class NoOCRReaderFound(Exception):
    def __init__(self, e):
        self.e = e

    def __str__(self):
        return f"Could not load OCR Reader: {self.e}"


TESSERACT_AVAILABLE = False
EASYOCR_AVAILABLE = False

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
    import easyocr  # noqa

    EASYOCR_AVAILABLE = True
except ImportError:
    pass


class OCRReader(metaclass=abc.ABCMeta):
    def __init__(self):
        # TODO: add device here
        self.check_if_available()

    @abc.abstractmethod
    def apply_ocr(self, image: "Image.Image") -> Tuple[List[Any], List[List[int]]]:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def check_if_available():
        raise NotImplementedError


class TesseractReader(OCRReader):
    def __init__(self):
        super().__init__()

    def apply_ocr(self, image: "Image.Image") -> Tuple[List[Any], List[List[int]]]:
        """
        Applies Tesseract on a document image, and returns recognized words + normalized bounding boxes.
        This was derived from LayoutLM preprocessing code in Huggingface's Transformers library.
        """
        data = pytesseract.image_to_data(image, output_type="dict")
        words, left, top, width, height = data["text"], data["left"], data["top"], data["width"], data["height"]

        # filter empty words and corresponding coordinates
        irrelevant_indices = set(idx for idx, word in enumerate(words) if not word.strip())
        words = [word.strip() for idx, word in enumerate(words) if idx not in irrelevant_indices]
        left = [coord for idx, coord in enumerate(left) if idx not in irrelevant_indices]
        top = [coord for idx, coord in enumerate(top) if idx not in irrelevant_indices]
        width = [coord for idx, coord in enumerate(width) if idx not in irrelevant_indices]
        height = [coord for idx, coord in enumerate(height) if idx not in irrelevant_indices]

        # turn coordinates into (left, top, left+width, top+height) format
        actual_boxes = [[x, y, x + w, y + h] for x, y, w, h in zip(left, top, width, height)]

        return words, actual_boxes

    @staticmethod
    def check_if_available():
        if not TESSERACT_AVAILABLE:
            raise NoOCRReaderFound(
                "Unable to use pytesseract (OCR will be unavailable). Install tesseract to process images with OCR."
            )


class EasyOCRReader(OCRReader):
    def __init__(self):
        super().__init__()
        self.reader = None

    def apply_ocr(self, image: "Image.Image") -> Tuple[List[Any], List[List[int]]]:
        """Applies Easy OCR on a document image, and returns recognized words + normalized bounding boxes."""
        if not self.reader:
            # TODO: expose language currently setting to english
            self.reader = easyocr.Reader(['en'])  # TODO: device here example: gpu=self.device > -1)

        # apply OCR
        data = self.reader.readtext(np.array(image))
        boxes, words, acc = list(map(list, zip(*data)))

        # filter empty words and corresponding coordinates
        irrelevant_indices = set(idx for idx, word in enumerate(words) if not word.strip())
        words = [word.strip() for idx, word in enumerate(words) if idx not in irrelevant_indices]
        boxes = [coords for idx, coords in enumerate(boxes) if idx not in irrelevant_indices]

        # turn coordinates into (left, top, left+width, top+height) format
        actual_boxes = [tl + br for tl, tr, br, bl in boxes]

        return words, actual_boxes

    @staticmethod
    def check_if_available():
        if not EASYOCR_AVAILABLE:
            raise NoOCRReaderFound(
                "Unable to use easyocr (OCR will be unavailable). Install easyocr to process images with OCR."
            )


class DummyOCRReader(OCRReader):
    def __init__(self):
        super().__init__()
        self.reader = None

    def apply_ocr(self, image: "Image.Image") -> Tuple[(List[Any], List[List[int]])]:
        raise NoOCRReaderFound("Unable to find any OCR engine and OCR extraction was requested")

    @staticmethod
    def _check_if_available():
        logging.warning("Falling back to a dummy OCR reader since none were found.")
