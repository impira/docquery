from unittest.mock import patch

import pytest

from docquery.ocr_reader import DummyOCRReader, EasyOCRReader, NoOCRReaderFound, TesseractReader, get_ocr_reader


READER_PERMUTATIONS = [
    {"name": "tesseract", "reader_class": TesseractReader},
    {"name": "easyocr", "reader_class": EasyOCRReader},
    {"name": "dummy", "reader_class": DummyOCRReader},
]


@pytest.mark.parametrize("reader_permutations", READER_PERMUTATIONS)
@patch(
    "docquery.ocr_reader.OCR_AVAILABLE",
    {
        "tesseract": True,
        "easyocr": True,
        "dummy": True,
    },
)
def test_get_ocr_reader(reader_permutations):
    reader = get_ocr_reader(reader_permutations["name"])
    assert isinstance(reader, reader_permutations["reader_class"])


@patch(
    "docquery.ocr_reader.OCR_AVAILABLE",
    {
        "tesseract": True,
        "easyocr": True,
        "dummy": True,
    },
)
def test_wrong_string_ocr_reader():
    with pytest.raises(Exception) as e:
        reader = get_ocr_reader("FAKE_OCR")
    assert (
        "Failed to find: FAKE_OCR in the available ocr libraries. The choices are: ['tesseract', 'easyocr', 'dummy']"
        in str(e.value)
    )
    assert e.type == NoOCRReaderFound


@patch(
    "docquery.ocr_reader.OCR_AVAILABLE",
    {
        "tesseract": False,
        "easyocr": True,
        "dummy": True,
    },
)
def test_choosing_unavailable_ocr_reader():
    with pytest.raises(Exception) as e:
        reader = get_ocr_reader("tesseract")
    assert f"Failed to load: tesseract Please make sure its installed correctly." in str(e.value)
    assert e.type == NoOCRReaderFound


@patch(
    "docquery.ocr_reader.OCR_AVAILABLE",
    {
        "tesseract": False,
        "easyocr": True,
        "dummy": True,
    },
)
def test_assert_fallback():
    reader = get_ocr_reader()
    assert isinstance(reader, EasyOCRReader)


@patch(
    "docquery.ocr_reader.OCR_AVAILABLE",
    {
        "tesseract": False,
        "easyocr": False,
        "dummy": True,
    },
)
def test_assert_fallback_to_dummy():
    reader = get_ocr_reader()
    assert isinstance(reader, DummyOCRReader)


@patch(
    "docquery.ocr_reader.OCR_AVAILABLE",
    {
        "tesseract": False,
        "easyocr": False,
        "dummy": False,
    },
)
def test_fail_to_load_if_called_directly_when_ocr_unavailable():
    EasyOCRReader._instances = {}
    with pytest.raises(Exception) as e:
        reader = EasyOCRReader()
    assert "Unable to use easyocr (OCR will be unavailable). Install easyocr to process images with OCR." in str(
        e.value
    )
    assert e.type == NoOCRReaderFound


def test_ocr_reader_are_singletons():
    reader_a = DummyOCRReader()
    reader_b = DummyOCRReader()
    reader_c = DummyOCRReader()
    assert reader_a is reader_b
    assert reader_a is reader_c
