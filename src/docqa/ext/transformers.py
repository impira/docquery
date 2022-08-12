# XXX CAN WE DELETE THIS FILE?


# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
# Copyright 2022 Impira, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copied and adapted from transformers (https://github.com/huggingface/transformers)

try:
    import pytesseract
    from PIL import Image
except ImportError as e:
    pass  # Handled elsewhere

from typing import Optional


# normalize_bbox() and apply_tesseract() are derived from apply_tesseract in models/layoutlmv3/feature_extraction_layoutlmv3.py.
# However, because the pipeline may evolve from what layoutlmv3 currently does, it's copied (vs. imported) to avoid creating an
# unecessary dependency.
def normalize_box(box, width, height):
    return [
        max(min(c, 1000), 0)
        for c in [
            int(1000 * (box[0] / width)),
            int(1000 * (box[1] / height)),
            int(1000 * (box[2] / width)),
            int(1000 * (box[3] / height)),
        ]
    ]


def apply_tesseract(image: "Image.Image", lang: Optional[str], tesseract_config: Optional[str]):
    """Applies Tesseract OCR on a document image, and returns recognized words + normalized bounding boxes."""
    # apply OCR
    data = pytesseract.image_to_data(image, lang=lang, output_type="dict", config=tesseract_config)
    words, left, top, width, height = data["text"], data["left"], data["top"], data["width"], data["height"]

    # filter empty words and corresponding coordinates
    irrelevant_indices = [idx for idx, word in enumerate(words) if not word.strip()]
    words = [word for idx, word in enumerate(words) if idx not in irrelevant_indices]
    left = [coord for idx, coord in enumerate(left) if idx not in irrelevant_indices]
    top = [coord for idx, coord in enumerate(top) if idx not in irrelevant_indices]
    width = [coord for idx, coord in enumerate(width) if idx not in irrelevant_indices]
    height = [coord for idx, coord in enumerate(height) if idx not in irrelevant_indices]

    # turn coordinates into (left, top, left+width, top+height) format
    actual_boxes = []
    for x, y, w, h in zip(left, top, width, height):
        actual_box = [x, y, x + w, y + h]
        actual_boxes.append(actual_box)

    image_width, image_height = image.size

    # finally, normalize the bounding boxes
    normalized_boxes = []
    for box in actual_boxes:
        normalized_boxes.append(normalize_box(box, image_width, image_height))

    assert len(words) == len(normalized_boxes), "Not as many words as there are bounding boxes"

    return words, normalized_boxes
